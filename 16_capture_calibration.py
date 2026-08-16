#!/opt/lerobot061/bin/python
"""Capture hand-eye calibration dataset on the Atlas board (project 1, B).

Headless-friendly: no GUI window.  A small HTTP server runs on the board so
the user can watch BOTH camera streams and captured samples from any browser:

  http://192.168.0.2:8090

Page shows:
  - live MJPEG preview (left = handeye wrist cam, right = front overhead cam)
  - buttons: [Capture] [Retake] [Quit]  (equivalent to terminal keys)
  - thumbnails of already captured samples (click to open full image)

Terminal keys still work: ENTER=capture  r=retake  q=quit

Usage (on the board):

  cd /root/lerobot_project
  /opt/lerobot061/bin/python 16_capture_calibration.py --poses 15

Output: /root/lerobot_project/calib_data/
  sample_000/handeye.jpg  sample_000/front.jpg  sample_000/joints.json
  samples.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from atlas_runner import (
    FOLLOWER_SERIAL,
    PROJECT_ROOT,
    camera_device,
    environment,
    serial_port,
)

os.environ.update(environment())

from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CALIBRATION_PATH = (
    PROJECT_ROOT
    / "lerobot_home"
    / "calibration"
    / "robots"
    / "so_follower"
    / "my_awesome_follower_arm.json"
)

FRONT_CAMERA_PID = "9221"
HANDEYE_CAMERA_PID = "9005"
HTTP_PORT = 8090


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: MotorCalibration(**values) for name, values in raw.items()}


def read_joints(port: str) -> dict[str, float]:
    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=load_calibration(CALIBRATION_PATH),
    )
    bus.connect()
    try:
        positions = bus.sync_read("Present_Position", normalize=True)
        return {name: float(positions[name]) for name in JOINT_NAMES}
    finally:
        bus.disconnect()


# --------------------------------------------------------------------------- #
# tiny HTTP server (std lib only)
# --------------------------------------------------------------------------- #

STATE = {
    "preview_jpeg": None,        # latest composite preview bytes
    "samples": [],               # list of dicts {index, joints, time}
    "out": None,                 # Path to calib_data
    "capture_cb": None,          # callables wired by main()
    "retake_cb": None,
    "quit_cb": None,
    "pose_target": 0,
    "max_poses": 15,
}


def http_serve() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def _send_bytes(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/preview.jpg":
                img = STATE["preview_jpeg"]
                if img is None:
                    self._send_bytes(b"", "image/jpeg")
                else:
                    self._send_bytes(img, "image/jpeg")
            elif path == "/capture":
                if STATE["capture_cb"]:
                    try:
                        STATE["capture_cb"]()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[HTTP] capture error: {exc}")
                self._send_bytes(b'{"ok":true}', "application/json")
            elif path == "/retake":
                if STATE["retake_cb"]:
                    STATE["retake_cb"]()
                self._send_bytes(b'{"ok":true}', "application/json")
            elif path == "/quit":
                if STATE["quit_cb"]:
                    STATE["quit_cb"]()
                self._send_bytes(b'{"ok":true}', "application/json")
            elif path == "/samples.json":
                self._send_bytes(json.dumps(STATE["samples"], ensure_ascii=False).encode("utf-8"),
                                 "application/json")
            elif path.startswith("/sample/"):
                # /sample/<idx>/<handeye|front>
                parts = path.split("/")
                try:
                    idx = int(parts[2])
                    which = parts[3]
                except (IndexError, ValueError):
                    self._send_bytes(b"bad request", "text/plain")
                    return
                img_path = STATE["out"] / f"sample_{idx:03d}" / f"{which}.jpg"
                if img_path.exists():
                    self._send_bytes(img_path.read_bytes(), "image/jpeg")
                else:
                    self._send_bytes(b"not found", "text/plain")
            else:
                self._send_bytes(b"not found", "text/plain")

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[HTTP] preview server on http://0.0.0.0:{HTTP_PORT}  (PC browser: http://192.168.0.2:{HTTP_PORT})")
    server.serve_forever()


PAGE_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SO-ARM101 calibration capture</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;margin:16px}
.row{display:flex;gap:12px;flex-wrap:wrap}
.cam{flex:1;min-width:300px;background:#000;border:1px solid #333;padding:6px}
.cam img{width:100%;display:block}
.cam .cap{font-size:12px;color:#aaa;margin-top:4px}
button{font-size:16px;padding:10px 22px;margin:4px;cursor:pointer;border:none;border-radius:6px}
#cap{background:#2e7d32;color:#fff}
#retake{background:#b26a00;color:#fff}
#quit{background:#b71c1c;color:#fff}
.samples{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.sample{border:1px solid #444;padding:4px;background:#1c1c1c}
.sample img{height:70px;display:block}
.sample .si{font-size:11px;color:#aaa}
#status{font-size:14px;margin:8px 0}
</style></head><body>
<h2>SO-ARM101 标定拍照</h2>
<div id="status">connecting...</div>
<div class="row">
  <div class="cam"><div class="cap">手眼相机（腕部）</div><img id="ph" src="/preview.jpg"></div>
  <div class="cam"><div class="cap">前置相机（俯视）</div></div>
</div>
<p><button id="cap">📸 拍照 Capture</button>
   <button id="retake">↩️ 重拍 Retake</button>
   <button id="quit">⏹ 结束 Quit</button></p>
<div id="samples" class="samples"></div>
<script>
const img = document.getElementById('ph');
function refreshPreview(){ img.src = '/preview.jpg?t=' + Date.now(); }
setInterval(refreshPreview, 300);
async function loadSamples(){
  const r = await fetch('/samples.json'); const arr = await r.json();
  const box = document.getElementById('samples'); box.innerHTML='';
  for (const s of arr){
    const d = document.createElement('div'); d.className='sample';
    d.innerHTML = '<img src="/sample/'+s.index+'/handeye.jpg"><img src="/sample/'+s.index+'/front.jpg">' +
      '<div class="si">#'+s.index+'</div>';
    box.appendChild(d);
  }
  document.getElementById('status').textContent =
    '已拍 ' + arr.length + ' / ' + arr[0] ? (arr[0].max || '?') : '?';
}
setInterval(loadSamples, 1000);
document.getElementById('cap').onclick = () => fetch('/capture');
document.getElementById('retake').onclick = () => fetch('/retake');
document.getElementById('quit').onclick = () => fetch('/quit');
</script></body></html>"""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=15)
    parser.add_argument("--out", default="/root/lerobot_project/calib_data")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--no-http", action="store_true", help="disable HTTP preview")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    STATE["out"] = out
    STATE["max_poses"] = args.poses

    handeye = camera_device(HANDEYE_CAMERA_PID, "handeye")
    front = camera_device(FRONT_CAMERA_PID, "front")
    cap_h = cv2.VideoCapture(handeye, cv2.CAP_V4L2)
    cap_f = cv2.VideoCapture(front, cv2.CAP_V4L2)
    if not cap_h.isOpened() or not cap_f.isOpened():
        raise SystemExit(f"cannot open cameras: handeye={handeye} front={front}")

    # USB bandwidth: front supports MJPG, handeye is YUYV-only -> lower res.
    cap_f.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap_f.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_f.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_f.set(cv2.CAP_PROP_FPS, 15)
    cap_h.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap_h.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap_h.set(cv2.CAP_PROP_FPS, 10)

    ok_h, frame_h = cap_h.read()
    ok_f, frame_f = cap_f.read()
    if not ok_h or not ok_f:
        cap_h.release()
        cap_f.release()
        raise SystemExit(f"camera read failed: handeye={ok_h} front={ok_f}")
    print(f"handeye frame: {frame_h.shape[1]}x{frame_h.shape[0]}  front frame: {frame_f.shape[1]}x{frame_f.shape[0]}")
    print("Cameras verified readable simultaneously.")

    follower_port = serial_port(FOLLOWER_SERIAL, "follower")
    print(f"Follower port: {follower_port}")

    samples: list[dict] = []
    sample_index = 0
    last_action = 0.0

    def capture() -> None:
        nonlocal sample_index, last_action
        now = time.time()
        if now - last_action < args.interval:
            return
        last_action = now
        joints = read_joints(follower_port)
        sample_dir = out / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "handeye.jpg"), frame_h)
        cv2.imwrite(str(sample_dir / "front.jpg"), frame_f)
        (sample_dir / "joints.json").write_text(json.dumps(joints, indent=2), encoding="utf-8")
        samples.append({"index": sample_index, "joints": joints,
                        "time": time.strftime("%H:%M:%S")})
        print(f"[{sample_index:03d}] captured  joints="
              + json.dumps({k: round(v, 2) for k, v in joints.items()}), flush=True)
        sample_index += 1

    def retake() -> None:
        nonlocal sample_index, last_action
        now = time.time()
        if sample_index == 0 or now - last_action < args.interval:
            return
        last_action = now
        sample_index -= 1
        sample_dir = out / f"sample_{sample_index:03d}"
        for f in ("handeye.jpg", "front.jpg", "joints.json"):
            (sample_dir / f).unlink(missing_ok=True)
        sample_dir.rmdir()
        samples.pop()
        print(f"[{sample_index:03d}] retaken", flush=True)

    def quit_all() -> None:
        print("QUIT requested", flush=True)
        os._exit(0)

    STATE["capture_cb"] = capture
    STATE["retake_cb"] = retake
    STATE["quit_cb"] = quit_all

    if not args.no_http:
        threading.Thread(target=http_serve, daemon=True).start()

    print(f"Target poses: {args.poses}")
    print("Browser:  http://192.168.0.2:8090   (watch both cams + capture buttons)")
    print("Terminal: ENTER=capture  r=retake  q=quit")

    # --- stdin reader thread ---
    def stdin_loop() -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):
                        capture()
                    elif ch == "r":
                        retake()
                    elif ch == "q":
                        quit_all()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=stdin_loop, daemon=True).start()

    # --- main capture loop: read cams, refresh preview, stop at target ---
    try:
        while sample_index < args.poses:
            ok_h, frame_h = cap_h.read()
            ok_f, frame_f = cap_f.read()
            if not ok_h or not ok_f:
                print("camera read failed, retrying...")
                time.sleep(0.2)
                continue
            ph = frame_h if frame_h.shape[1] <= 640 else cv2.resize(frame_h, (640, int(frame_h.shape[0] * 640 / frame_h.shape[1])))
            pf = frame_f if frame_f.shape[1] <= 640 else cv2.resize(frame_f, (640, int(frame_f.shape[0] * 640 / frame_f.shape[1])))
            preview = np.hstack((ph, pf))
            label = f"pose {sample_index}/{args.poses}"
            cv2.putText(preview, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                STATE["preview_jpeg"] = buf.tobytes()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        cap_h.release()
        cap_f.release()

    (out / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"DONE: {len(samples)} samples saved to {out}")
    print("Pack for the PC with:")
    print("  tar -czf /root/lerobot_project/calib_data.tar.gz -C /root/lerobot_project calib_data")


if __name__ == "__main__":
    main()
