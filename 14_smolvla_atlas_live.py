#!/opt/smolvla_npu_test/bin/python
"""Atlas-native SmolVLA NPU inference for the SO-101 follower.

Modes:
  observe      connect, capture, infer, save diagnostics; never send an action
  single-step send one explicitly clipped target, then read the achieved pose
  rollout      repeatedly infer and execute action chunks with per-command limits
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch

PROJECT = Path("/root/lerobot_project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, "/usr/local/Ascend/thirdpart/aarch64/acllite")

from acllite_model import AclLiteModel  # noqa: E402
from acllite_resource import AclLiteResource  # noqa: E402
from atlas_runner import (  # noqa: E402
    FOLLOWER_SERIAL,
    FRONT_CAMERA_PID,
    HANDEYE_CAMERA_PID,
    camera_device,
    environment,
    serial_port,
)

os.environ.update(environment())

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.policies.common.vla_utils import resize_with_pad  # noqa: E402
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402


NPU = PROJECT / "npu_tests"
RUNS = PROJECT / "live_runs"
MOTOR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
STATE_MEAN = np.array([12.68166065, -23.63298035, 23.55857277, 77.20548248, 6.30614901, 17.69643021], np.float32)
STATE_STD = np.array([17.40145302, 57.58153915, 53.76133347, 13.57062626, 10.49081039, 17.13056183], np.float32)
ACTION_MEAN = np.array([12.60392857, -25.58491325, 21.43714333, 77.07484436, 6.25326633, 16.94562912], np.float32)
ACTION_STD = np.array([17.88148689, 57.27727509, 54.75575638, 14.07254219, 10.64185810, 18.56493950], np.float32)
ACTION_MIN = np.array([-55.29670334, -105.80220032, -80.0, 39.69230652, -19.56044006, 0.0], np.float32)
ACTION_MAX = np.array([48.79121017, 61.75824356, 96.96703339, 103.16483307, 42.15384674, 72.86878967], np.float32)
TASK = "Pick up the yellow block, place it inside the black box, release it, then return the arm to its starting position."
JOINT_LABELS = ("Shoulder pan", "Shoulder lift", "Elbow", "Wrist flex", "Wrist roll", "Gripper")
ACTUAL_COLOR = (255, 220, 40)
TARGET_COLOR = (50, 150, 255)
TEXT_COLOR = (232, 232, 236)
GRID_COLOR = (65, 65, 72)


class JointTargetKalmanFilter:
    """Independent scalar Kalman filters for six commanded joint targets.

    The state is deliberately kept across policy chunks so a replanning
    boundary cannot create a discontinuous command.  The gripper has a lower
    measurement noise than the arm joints to preserve grasp/release timing.
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 4.0) -> None:
        if process_noise <= 0 or measurement_noise <= 0:
            raise ValueError("Kalman noise values must be positive")
        self.q = np.full(6, process_noise, dtype=np.float32)
        self.r = np.full(6, measurement_noise, dtype=np.float32)
        self.r[-1] = max(process_noise, measurement_noise * 0.25)
        self.state: np.ndarray | None = None
        self.variance = self.r.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        value = np.asarray(measurement, dtype=np.float32)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError("Kalman target must contain six finite joint values")
        if self.state is None:
            self.state = value.copy()
            return self.state.copy()
        self.variance += self.q
        gain = self.variance / (self.variance + self.r)
        self.state += gain * (value - self.state)
        self.variance *= 1.0 - gain
        return self.state.copy()


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class AtlasDashboard:
    """Headless web dashboard served by the Atlas board itself."""

    def __init__(self, port: int, run_dir: Path) -> None:
        self.port = port
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started_at = time.monotonic()
        self.front: np.ndarray | None = None
        self.handeye: np.ndarray | None = None
        self.actual: list[float | None] = [None] * 6
        self.target: list[float | None] = [None] * 6
        self.history: deque[tuple[float, tuple[float | None, ...], tuple[float | None, ...]]] = deque(maxlen=600)
        self.obs_times: deque[float] = deque(maxlen=60)
        self.status = "Starting Atlas dashboard..."
        self.timings: dict[str, float] = {}
        self.server: ReusableThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.csv_file = None
        self.csv_writer = None

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.run_dir / "telemetry.csv").open("w", newline="", encoding="utf-8-sig")
        fields = ["elapsed_s", *[f"actual.{key}" for key in MOTOR_KEYS], *[f"target.{key}" for key in MOTOR_KEYS]]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.csv_writer.writeheader()
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    page = b"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Atlas SmolVLA NPU Dashboard</title><style>
body{margin:0;background:#101114;color:#eee;font-family:Arial,sans-serif;text-align:center}
.bar{display:flex;align-items:center;justify-content:space-between;padding:10px 18px;background:#1b1d22}
h2{margin:0;font-size:20px}.hint{color:#aeb3bd;font-size:13px}button{border:0;border-radius:7px;padding:9px 18px;background:#c83e3e;color:white;font-weight:bold;cursor:pointer}
img{display:block;width:min(100%,1320px);height:auto;margin:10px auto;background:#121318}
</style></head><body><div class='bar'><div><h2>Atlas SmolVLA NPU</h2><div class='hint'>Live cameras and six-joint telemetry</div></div>
<button onclick=\"fetch('/stop',{method:'POST'}).then(()=>this.innerText='STOP REQUESTED')\">SAFE STOP</button></div>
<img src='/stream.mjpg'></body></html>"""
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if self.path == "/snapshot.jpg":
                    image = dashboard.jpeg()
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(image)))
                    self.end_headers()
                    self.wfile.write(image)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not dashboard.stop_event.is_set():
                        image = dashboard.jpeg()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(image)}\r\n\r\n".encode())
                        self.wfile.write(image + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self) -> None:
                if self.path != "/stop":
                    self.send_error(404)
                    return
                dashboard.set_status("Safe stop requested from web dashboard")
                dashboard.stop_event.set()
                body = b"STOPPING"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        try:
            self.server = ReusableThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as exc:
            raise RuntimeError(f"dashboard port {self.port} is unavailable: {exc}") from exc
        self.server_thread = threading.Thread(target=self.server.serve_forever, name="atlas-dashboard", daemon=True)
        self.server_thread.start()
        print(f"[DASHBOARD] Open http://192.168.0.2:{self.port}")

    def set_status(self, status: str, timings: dict[str, float] | None = None) -> None:
        with self.lock:
            self.status = status
            if timings is not None:
                self.timings = dict(timings)

    def update_target(self, values: np.ndarray | dict[str, float]) -> None:
        if isinstance(values, dict):
            target = [float(values.get(key, np.nan)) for key in MOTOR_KEYS]
        else:
            target = [float(x) for x in values]
        with self.lock:
            self.target = target

    def update_observation(self, observation: dict) -> None:
        now = time.monotonic()
        actual = [float(observation[key]) for key in MOTOR_KEYS]
        with self.lock:
            self.actual = actual
            if isinstance(observation.get("front"), np.ndarray):
                self.front = observation["front"].copy()
            if isinstance(observation.get("handeye"), np.ndarray):
                self.handeye = observation["handeye"].copy()
            self.obs_times.append(now)
            entry = (now - self.started_at, tuple(self.actual), tuple(self.target))
            self.history.append(entry)
            target = tuple(self.target)
        if self.csv_writer is not None:
            row: dict[str, Any] = {"elapsed_s": f"{now - self.started_at:.4f}"}
            row.update({f"actual.{key}": value for key, value in zip(MOTOR_KEYS, actual, strict=True)})
            row.update({f"target.{key}": value for key, value in zip(MOTOR_KEYS, target, strict=True)})
            self.csv_writer.writerow(row)
            self.csv_file.flush()

    def _snapshot(self) -> dict[str, Any]:
        with self.lock:
            hz = 0.0
            if len(self.obs_times) >= 2:
                span = self.obs_times[-1] - self.obs_times[0]
                hz = (len(self.obs_times) - 1) / span if span > 0 else 0.0
            return {
                "front": None if self.front is None else self.front.copy(),
                "handeye": None if self.handeye is None else self.handeye.copy(),
                "actual": tuple(self.actual),
                "target": tuple(self.target),
                "history": tuple(self.history),
                "status": self.status,
                "timings": dict(self.timings),
                "hz": hz,
                "elapsed": time.monotonic() - self.started_at,
            }

    @staticmethod
    def _camera(frame: np.ndarray | None, title: str) -> np.ndarray:
        panel = np.full((310, 620, 3), 24, dtype=np.uint8)
        if frame is not None:
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            scale = min(620 / image.shape[1], 275 / image.shape[0])
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            x = (620 - image.shape[1]) // 2
            y = 34 + (275 - image.shape[0]) // 2
            panel[y : y + image.shape[0], x : x + image.shape[1]] = image
        cv2.rectangle(panel, (0, 0), (619, 309), GRID_COLOR, 1)
        cv2.putText(panel, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2, cv2.LINE_AA)
        return panel

    @staticmethod
    def _joint_chart(canvas: np.ndarray, rect: tuple[int, int, int, int], index: int, snap: dict[str, Any]) -> None:
        x, y, width, height = rect
        cv2.rectangle(canvas, (x, y), (x + width, y + height), GRID_COLOR, 1)
        actual, target = snap["actual"][index], snap["target"][index]
        fmt = lambda value: "--.--" if value is None or not np.isfinite(value) else f"{value:7.2f}"
        cv2.putText(canvas, JOINT_LABELS[index], (x + 9, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"A {fmt(actual)}", (x + 9, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43, ACTUAL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"T {fmt(target)}", (x + 110, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43, TARGET_COLOR, 1, cv2.LINE_AA)
        values = []
        samples = []
        for t, av, tv in snap["history"]:
            pair = (av[index], tv[index])
            samples.append((t, *pair))
            values.extend(v for v in pair if v is not None and np.isfinite(v))
        if len(samples) < 2 or not values:
            return
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.15, 2.0)
        lo, hi = lo - pad, hi + pad
        latest = samples[-1][0]
        earliest = max(samples[0][0], latest - 20.0)
        plot_x, plot_y, plot_w, plot_h = x + 9, y + 50, width - 18, height - 58
        for which, color in ((1, ACTUAL_COLOR), (2, TARGET_COLOR)):
            points = []
            for sample in samples:
                if sample[0] < earliest or sample[which] is None or not np.isfinite(sample[which]):
                    continue
                px = int(plot_x + (sample[0] - earliest) / max(latest - earliest, 0.001) * plot_w)
                py = int(plot_y + (hi - sample[which]) / max(hi - lo, 0.001) * plot_h)
                points.append((px, py))
            if len(points) >= 2:
                cv2.polylines(canvas, [np.asarray(points, np.int32)], False, color, 2, cv2.LINE_AA)

    def render(self) -> np.ndarray:
        snap = self._snapshot()
        canvas = np.full((800, 1280, 3), 18, dtype=np.uint8)
        canvas[55:365, 10:630] = self._camera(snap["front"], "FRONT CAMERA")
        canvas[55:365, 650:1270] = self._camera(snap["handeye"], "HAND-EYE CAMERA")
        timing = snap["timings"]
        total = timing.get("total_ms", 0.0)
        cv2.putText(canvas, f"Atlas SmolVLA NPU | elapsed {snap['elapsed']:.1f}s | observation {snap['hz']:.1f} Hz | inference {total:.0f} ms", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)
        chart_w, chart_h = 410, 190
        for index in range(6):
            row, col = divmod(index, 3)
            self._joint_chart(canvas, (10 + col * 425, 385 + row * 200, chart_w, chart_h), index, snap)
        cv2.putText(canvas, snap["status"][:145], (12, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 190, 220), 1, cv2.LINE_AA)
        return canvas

    def jpeg(self) -> bytes:
        ok, encoded = cv2.imencode(".jpg", self.render(), [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            raise RuntimeError("dashboard JPEG encoding failed")
        return encoded.tobytes()

    def close(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread is not None:
            self.server_thread.join(timeout=2.0)
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None


class AtlasSmolVLANPU:
    def __init__(self) -> None:
        self.resource = AclLiteResource(device_id=0)
        self.resource.init()
        self.vision = AclLiteModel(str(NPU / "vision/vision.om"))
        self.state_proj = AclLiteModel(str(NPU / "state_proj/state_proj.om"))
        self.prefix_model = AclLiteModel(str(NPU / "prefix/prefix_origin.om"))
        self.denoise = AclLiteModel(str(NPU / "denoise/denoise_origin.om"))
        self.prefix_template = np.load(NPU / "prefix/input.npy").astype(np.float32)
        self.noise = np.load(NPU / "denoise/input_xt.npy").astype(np.float32)

    @staticmethod
    def _one(model: AclLiteModel, inputs: list[np.ndarray]) -> list[np.ndarray]:
        output = model.execute(inputs)
        if output is None:
            raise RuntimeError("NPU execution returned no output")
        return output

    @staticmethod
    def prepare_image(rgb: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image = resize_with_pad(image, 512, 512, pad_value=0)
        image = image * 2.0 - 1.0
        return np.ascontiguousarray(image.numpy(), dtype=np.float32)

    def infer(self, observation: dict, num_steps: int = 5) -> tuple[np.ndarray, dict[str, float]]:
        started = time.perf_counter()
        front = self.prepare_image(observation["front"])
        handeye = self.prepare_image(observation["handeye"])
        state_raw = np.array([observation[key] for key in MOTOR_KEYS], dtype=np.float32)[None, :]
        state6 = (state_raw - STATE_MEAN) / STATE_STD
        state = np.zeros((1, 32), dtype=np.float32)
        state[:, :6] = state6
        state = np.ascontiguousarray(state)
        prep_ms = (time.perf_counter() - started) * 1000.0

        timings: dict[str, float] = {"preprocess_ms": prep_ms}
        t0 = time.perf_counter()
        front_emb = self._one(self.vision, [front])[0].astype(np.float32)
        handeye_emb = self._one(self.vision, [handeye])[0].astype(np.float32)
        timings["vision_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        state_emb = self._one(self.state_proj, [state])[0].astype(np.float32)
        prefix = self.prefix_template.copy()
        prefix[:, 0:64, :] = front_emb * np.sqrt(960.0)
        prefix[:, 64:128, :] = handeye_emb * np.sqrt(960.0)
        prefix[:, 176:177, :] = state_emb[:, None, :]
        kv = [np.ascontiguousarray(x, dtype=np.float32) for x in self._one(self.prefix_model, [prefix])]
        timings["prefix_ms"] = (time.perf_counter() - t0) * 1000.0

        x_t = self.noise.copy()
        t0 = time.perf_counter()
        dt = -1.0 / num_steps
        for step in range(num_steps):
            timestep = np.array([1.0 + step * dt], dtype=np.float32)
            velocity = self._one(self.denoise, [x_t, timestep, *kv])[0].astype(np.float32)
            x_t = np.ascontiguousarray(x_t + dt * velocity)
        timings["denoise_ms"] = (time.perf_counter() - t0) * 1000.0
        actions = x_t[0, :, :6] * ACTION_STD + ACTION_MEAN
        actions = np.clip(actions, ACTION_MIN, ACTION_MAX)
        if not np.isfinite(actions).all():
            raise RuntimeError("policy generated NaN/Inf actions")
        timings["total_ms"] = (time.perf_counter() - started) * 1000.0
        return actions.astype(np.float32), timings

    def close(self) -> None:
        for model in (self.denoise, self.prefix_model, self.state_proj, self.vision):
            model.destroy()
        del self.denoise, self.prefix_model, self.state_proj, self.vision, self.resource
        gc.collect()


def make_robot(max_relative_target: float) -> SO101Follower:
    follower_port = serial_port(FOLLOWER_SERIAL, "follower")
    front_path = camera_device(FRONT_CAMERA_PID, "front")
    handeye_path = camera_device(HANDEYE_CAMERA_PID, "hand-eye")
    cameras = {
        "front": OpenCVCameraConfig(index_or_path=Path(front_path), width=640, height=480, fps=30, fourcc="MJPG"),
        "handeye": OpenCVCameraConfig(index_or_path=Path(handeye_path), width=640, height=480, fps=30, fourcc="YUYV"),
    }
    limits = {
        name.removesuffix(".pos"): (max_relative_target * 2.0 if name == "gripper.pos" else max_relative_target)
        for name in MOTOR_KEYS
    }
    return SO101Follower(
        SO101FollowerConfig(
            port=follower_port,
            id="my_awesome_follower_arm",
            cameras=cameras,
            max_relative_target=None if max_relative_target <= 0 else limits,
            disable_torque_on_disconnect=True,
        )
    )


def pose(observation: dict) -> np.ndarray:
    return np.array([observation[key] for key in MOTOR_KEYS], dtype=np.float32)


def action_dict(values: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(MOTOR_KEYS, values, strict=True)}


def save_diagnostics(run_dir: Path, observation: dict, actions: np.ndarray, timings: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "front.jpg"), cv2.cvtColor(observation["front"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(run_dir / "handeye.jpg"), cv2.cvtColor(observation["handeye"], cv2.COLOR_RGB2BGR))
    np.save(run_dir / "predicted_actions.npy", actions)
    (run_dir / "report.json").write_text(
        json.dumps({"task": TASK, "pose": pose(observation).tolist(), "first_action": actions[0].tolist(), "timings": timings}, indent=2) + "\n",
        encoding="utf-8",
    )


def return_to_initial_position(
    robot: SO101Follower,
    initial: np.ndarray,
    dashboard: AtlasDashboard | None,
    duration_s: float = 3.0,
    fps: float = 50.0,
) -> None:
    """Match LeRobot rollout teardown with a smooth three-second return."""
    observation = robot.get_observation()
    current = pose(observation)
    if dashboard is not None:
        dashboard.update_observation(observation)
        dashboard.set_status("Returning smoothly to the initial position...")
    steps = max(1, round(duration_s * fps))
    for step in range(1, steps + 1):
        alpha = step / steps
        target = current * (1.0 - alpha) + initial * alpha
        if dashboard is not None:
            dashboard.update_target(target)
        robot.send_action(action_dict(target))
        time.sleep(1.0 / fps)
    final_observation = robot.get_observation()
    if dashboard is not None:
        dashboard.update_observation(final_observation)
        dashboard.set_status("Returned to initial position; disconnecting safely")
    print("RETURN_TO_INITIAL_POSITION_PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("observe", "single-step", "rollout"), default="observe")
    parser.add_argument("--motion-key", default="", help="must equal ENABLE_ATLAS_MOTION for a motion mode")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=10.0, help="PC rollout-compatible control rate")
    parser.add_argument("--chunk-actions", type=int, default=50, help="PC SmolVLA executes all 50 queued actions")
    parser.add_argument("--num-steps", type=int, default=5, help="flow-matching steps; PC monitor uses 5")
    parser.add_argument("--max-relative-target", type=float, default=10.0, help="body-joint hard clamp in degrees; gripper uses 2x; <=0 matches PC's unlimited setting")
    parser.add_argument("--action-filter", choices=("none", "kalman"), default="none")
    parser.add_argument("--kalman-process-noise", type=float, default=1.0)
    parser.add_argument("--kalman-measurement-noise", type=float, default=4.0)
    parser.add_argument("--return-to-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dashboard-port", type=int, default=8080)
    parser.add_argument("--dashboard-hold", type=float, default=30.0, help="seconds to keep observe dashboard open")
    parser.add_argument("--dashboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dashboard-self-test", action="store_true", help="test web UI without NPU or hardware")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode != "observe" and args.motion_key != "ENABLE_ATLAS_MOTION":
        raise SystemExit("motion refused: pass --motion-key ENABLE_ATLAS_MOTION")
    run_dir = RUNS / time.strftime("%Y%m%d_%H%M%S")
    dashboard = AtlasDashboard(args.dashboard_port, run_dir) if args.dashboard else None
    backend = None
    robot = None
    initial_pose: np.ndarray | None = None
    try:
        if dashboard is not None:
            dashboard.start()
            dashboard.set_status("Loading four SmolVLA OM models on Atlas NPU...")
        if args.dashboard_self_test:
            if dashboard is None:
                raise SystemExit("dashboard self-test requires --dashboard")
            height, width = 480, 640
            xx = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
            for sample in range(30):
                phase = sample / 5.0
                actual = np.array([20.0 * np.sin(phase + joint) for joint in range(6)], np.float32)
                target = actual + np.array([4.0 * np.cos(phase + joint) for joint in range(6)], np.float32)
                observation = {key: float(value) for key, value in zip(MOTOR_KEYS, actual, strict=True)}
                observation["front"] = np.dstack((xx, np.full_like(xx, 90), np.flip(xx, axis=1)))
                observation["handeye"] = np.dstack((np.full_like(xx, 60), np.flip(xx, axis=1), xx))
                dashboard.update_target(target)
                dashboard.update_observation(observation)
            dashboard.set_status("Dashboard self-test: no NPU, camera, serial port, or motor was opened", {"total_ms": 1134.0})
            print("DASHBOARD_SELF_TEST_PASS no hardware was opened")
            dashboard.stop_event.wait(args.dashboard_hold)
            return
        backend = AtlasSmolVLANPU()
        if dashboard is not None:
            dashboard.set_status("Connecting follower and two cameras...")
        robot = make_robot(args.max_relative_target)
        robot.connect()
        observation = robot.get_observation()
        initial_pose = pose(observation).copy()
        if dashboard is not None:
            dashboard.update_observation(observation)
            dashboard.set_status("Running SmolVLA inference on Atlas NPU...")
        actions, timings = backend.infer(observation, num_steps=args.num_steps)
        if dashboard is not None:
            dashboard.update_target(actions[0])
            dashboard.set_status(f"Inference complete: {timings['total_ms']:.1f} ms", timings)
        save_diagnostics(run_dir, observation, actions, timings)
        current = pose(observation)
        print(json.dumps({"mode": args.mode, "current": current.tolist(), "first_prediction": actions[0].tolist(), "delta": (actions[0] - current).tolist(), "timings": timings, "run_dir": str(run_dir)}, indent=2))
        if args.mode == "observe":
            print("OBSERVE_ONLY_PASS no actuator command was sent")
            if dashboard is not None and args.dashboard_hold > 0:
                dashboard.set_status(f"Observe complete; no action sent. Dashboard remains open for {args.dashboard_hold:.0f}s", timings)
                dashboard.stop_event.wait(args.dashboard_hold)
            return

        if args.mode == "single-step":
            max_delta = np.array([2, 2, 2, 2, 2, 5], dtype=np.float32)
            safe_target = current + np.clip(actions[0] - current, -max_delta, max_delta)
            if dashboard is not None:
                dashboard.update_target(safe_target)
                dashboard.set_status("Sending one safety-clipped inference action...", timings)
            sent = robot.send_action(action_dict(safe_target))
            time.sleep(0.75)
            achieved_obs = robot.get_observation()
            if dashboard is not None:
                dashboard.update_target(sent)
                dashboard.update_observation(achieved_obs)
                dashboard.set_status("Single-step motion completed", timings)
            achieved = pose(achieved_obs)
            print(json.dumps({"sent": sent, "achieved": achieved.tolist(), "actual_delta": (achieved - current).tolist()}, indent=2))
            print("SINGLE_STEP_MOTION_PASS")
            return

        deadline = time.monotonic() + args.duration
        chunk = actions
        index = 0
        sent_count = 0
        inference_count = 1
        target_filter = (
            JointTargetKalmanFilter(args.kalman_process_noise, args.kalman_measurement_noise)
            if args.action_filter == "kalman" else None
        )
        if dashboard is not None:
            dashboard.set_status(
                f"PC-sync rollout: {args.fps:.1f} Hz, {args.num_steps} denoise steps, "
                f"{min(args.chunk_actions, len(chunk))} queued actions",
                timings,
            )
        while time.monotonic() < deadline and (dashboard is None or not dashboard.stop_event.is_set()):
            loop_started = time.perf_counter()
            # Match LeRobot BaseStrategy: acquire a fresh observation every
            # control tick, but reuse the policy's queued action chunk until empty.
            observation = robot.get_observation()
            if dashboard is not None:
                dashboard.update_observation(observation)
            if index >= min(args.chunk_actions, len(chunk)):
                if dashboard is not None:
                    dashboard.update_observation(observation)
                    dashboard.set_status("Replanning on Atlas NPU...", timings)
                chunk, timings = backend.infer(observation, num_steps=args.num_steps)
                index = 0
                inference_count += 1
                if dashboard is not None:
                    dashboard.set_status(f"Rollout running; inference {timings['total_ms']:.1f} ms", timings)
                print(f"INFERENCE total_ms={timings['total_ms']:.1f}")
            model_target = chunk[index]
            control_target = target_filter.update(model_target) if target_filter is not None else model_target
            if dashboard is not None:
                dashboard.update_target(control_target)
            sent = robot.send_action(action_dict(control_target))
            if dashboard is not None:
                dashboard.set_status(
                    f"PC-sync action {index + 1}/{min(args.chunk_actions, len(chunk))}; "
                    f"inference {timings['total_ms']:.1f} ms",
                    timings,
                )
            index += 1
            sent_count += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - loop_started)))
        if dashboard is not None:
            reason = "Safe stop requested" if dashboard.stop_event.is_set() else "Rollout duration completed"
            dashboard.set_status(
                f"{reason}; sent {sent_count} actions, completed {inference_count} inference chunks", timings
            )
        print(f"ROLLOUT_PASS sent_actions={sent_count} inference_chunks={inference_count}")
    finally:
        if robot is not None and robot.is_connected:
            if args.mode == "rollout" and args.return_to_initial and initial_pose is not None:
                try:
                    return_to_initial_position(robot, initial_pose, dashboard)
                except Exception as exc:
                    print(f"WARNING return-to-initial failed: {exc}")
            robot.disconnect()
        if backend is not None:
            backend.close()
        if dashboard is not None:
            dashboard.close()


if __name__ == "__main__":
    main()
