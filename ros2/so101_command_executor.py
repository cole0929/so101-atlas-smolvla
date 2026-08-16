#!/opt/lerobot061/bin/python
"""SO-ARM101 command executor: UDP command downlink to the real follower arm.

This process owns the follower serial port.  It keeps streaming read-only
telemetry to the existing ROS bridge (UDP 15000, so RViz keeps showing the real
arm) and additionally listens on UDP 15001 for target-joint commands coming
from the ROS 2 command bridge (MoveIt trajectory execution).

Command protocol (JSON, UDP 15001):
    {"cmd": "trajectory", "seq": int, "t_start": float,
     "points": [{"t": float, "names": [subset of joint names], "positions": [floats]}, ...]}
    positions follow `names`: degrees for the five arm joints, percent for the
    gripper.  Arm and gripper trajectories run in parallel; each point updates
    only its named joints, the remaining joints keep their last target.

    {"cmd": "stop"}            -> clear the active trajectory immediately
    {"cmd": "enable", "on": true/false} -> runtime motion gating

Completion/error reports are sent to UDP 15002:
    {"seq": int, "status": "done"|"aborted"|"error", "reason": str}

Safety: motion only happens while --enable-control is passed on the command
line AND the runtime "enable" flag is on.  All targets are clamped by
max_relative_target inside SO101Follower.send_action.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import time

from atlas_runner import FOLLOWER_SERIAL, environment, serial_port

os.environ.update(environment())

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# Degrees for the five arm joints, percent for the gripper.  These are the
# per-call relative limits enforced by send_action (same values as the SmolVLA
# live runner).  They are the last line of defence, not the primary planner.
MAX_RELATIVE_TARGET = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 10.0,
    "elbow_flex": 10.0,
    "wrist_flex": 10.0,
    "wrist_roll": 10.0,
    "gripper": 20.0,
}
MOTOR_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)
# Absolute last line of defence: the SO-ARM101 URDF mechanical limits
# (radians -> degrees).  MoveIt/URDF limits are the primary constraint applied
# by the planner; this only prevents a corrupted downlink from commanding
# damage.  NOTE: keep these wide enough to cover real planner output.
ARM_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}
GRIPPER_LIMITS_PCT = (0.0, 100.0)

TELEMETRY_PORT = 15000
COMMAND_PORT = 15001
REPORT_PORT = 15002


class CommandExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dry_run = getattr(args, "dry_run", False)
        self.motion_enabled = args.enable_control
        self.running = True
        self.robot: SO101Follower | None = None
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.cmd_sock.bind(("127.0.0.1", COMMAND_PORT))
        self.cmd_sock.settimeout(0.05)
        self.telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.report_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Active trajectory state.
        self.lock = threading.Lock()
        # Multiple trajectories run concurrently (MoveIt sends arm and gripper
        # as separate FollowJointTrajectory goals).  Each is keyed by seq.
        self.trajectories: dict[int, dict] = {}
        self.last_sent: dict[str, float] | None = None
        self.last_telemetry: dict[str, float] | None = None
        # Full 6-DOF target state, merged from parallel arm/gripper trajectories.
        self.target_state: dict[str, float] | None = None

        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, _signum: int, _frame: object) -> None:
        self.running = False

    def connect(self) -> None:
        if self.dry_run:
            print("[CMD] DRY-RUN: serial port not opened; protocol only", flush=True)
            return
        follower_port = serial_port(FOLLOWER_SERIAL, "follower")
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=follower_port,
                id="my_awesome_follower_arm",
                cameras={},
                disable_torque_on_disconnect=True,
                max_relative_target=MAX_RELATIVE_TARGET,
            )
        )
        self.robot.connect()
        print(f"[CMD] Follower connected on {follower_port}", flush=True)
        print(f"[CMD] Motion gated: {'ENABLED' if self.motion_enabled else 'DISABLED (--enable-control missing)'}", flush=True)

    def read_present(self) -> dict[str, float]:
        if self.dry_run:
            return {name: 0.0 for name in JOINT_NAMES}
        assert self.robot is not None
        obs = self.robot.get_observation()
        return {key.removesuffix(".pos"): float(obs[key]) for key in MOTOR_KEYS}

    def _send_telemetry(self) -> None:
        try:
            present = self.read_present()
        except Exception as exc:  # noqa: BLE001 - keep telemetry alive on bus hiccups
            print(f"[CMD] telemetry read error: {exc}", flush=True)
            return
        self.last_telemetry = present
        payload = {
            "stamp_ns": time.time_ns(),
            "names": list(JOINT_NAMES),
            "positions": [float(present[name]) for name in JOINT_NAMES],
        }
        self.telemetry_sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), ("127.0.0.1", TELEMETRY_PORT))

    def _validate_point(self, point: dict) -> str | None:
        names = point.get("names")
        positions = point.get("positions")
        if not isinstance(names, list) or not isinstance(positions, list):
            return "point missing names/positions"
        if len(names) != len(positions) or not names:
            return f"names/positions length mismatch ({len(names)} vs {len(positions)})"
        unknown = [n for n in names if n not in JOINT_NAMES]
        if unknown:
            return f"unknown joints {unknown}"
        for name, value in zip(names, positions):
            if not isinstance(value, (int, float)) or not value == value:  # NaN check
                return f"non-finite target for {name}"
            if name in ARM_LIMITS_DEG:
                lo, hi = ARM_LIMITS_DEG[name]
                if value < lo - 1.0 or value > hi + 1.0:
                    return f"{name} target {value:.1f} outside hard limits [{lo}, {hi}]"
            elif name == "gripper":
                lo, hi = GRIPPER_LIMITS_PCT
                if value < lo - 1.0 or value > hi + 1.0:
                    return f"gripper target {value:.1f} outside hard limits [{lo}, {hi}]"
        return None

    def _handle_command(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[CMD] bad JSON: {exc}", flush=True)
            return
        cmd = msg.get("cmd")
        if cmd == "enable":
            self.motion_enabled = bool(msg.get("on", False))
            print(f"[CMD] motion gate -> {'ENABLED' if self.motion_enabled else 'DISABLED'}", flush=True)
            return
        if cmd == "stop":
            self._abort("stop command")
            return
        if cmd == "trajectory":
            with self.lock:
                self._accept_trajectory(msg)
            return
        print(f"[CMD] unknown command: {cmd!r}", flush=True)

    def _accept_trajectory(self, msg: dict) -> None:
        seq = int(msg.get("seq", -1))
        points = msg.get("points")
        if not isinstance(points, list) or not points:
            self._report(seq, "error", "empty trajectory")
            return
        for point in points:
            err = self._validate_point(point)
            if err:
                self._report(seq, "error", err)
                return
        t_start = float(msg.get("t_start", time.time()))
        if t_start < time.time():
            # Do not start in the past; clamp to now unless the delay is absurd.
            delay = time.time() - t_start
            if delay > 5.0:
                self._report(seq, "error", f"trajectory start {delay:.1f}s in the past")
                return
            t_start = time.time()
        sorted_points = sorted(points, key=lambda p: float(p.get("t", 0.0)))
        # Arm and gripper trajectories have independent seq counters from the
        # bridge, so a seq can be reused by the other controller.  Replace any
        # trajectory with the same seq (re-plan) and keep the rest running.
        self.trajectories[seq] = {
            "points": sorted_points,
            "t_start": t_start,
            "duration": float(sorted_points[-1]["t"]),
            "reported": False,
        }
        print(
            f"[CMD] accepted trajectory seq={seq} points={len(points)} t_start={t_start:.3f} active={sorted(self.trajectories)}",
            flush=True,
        )

    def _report(self, seq: int | None, status: str, reason: str = "") -> None:
        payload = {"seq": seq, "status": status, "reason": reason}
        self.report_sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), ("127.0.0.1", REPORT_PORT))
        print(f"[CMD] report seq={seq} status={status} reason={reason}", flush=True)

    def _abort(self, reason: str) -> None:
        """Emergency stop: clear all trajectories but KEEP torque enabled.

        Disconnecting here would de-energize the arm and let it fall.  The arm
        stays where it is; the process still ends cleanly on Ctrl+C.
        """
        with self.lock:
            seqs = sorted(self.trajectories)
            self.trajectories = {}
        print(f"[CMD] abort: {reason}", flush=True)
        for seq in seqs:
            self._report(seq, "aborted", reason)

    def _command_loop(self) -> None:
        while self.running:
            try:
                raw, _addr = self.cmd_sock.recvfrom(65507)
            except socket.timeout:
                continue
            self._handle_command(raw)

    def run(self) -> None:
        self.connect()
        cmd_thread = threading.Thread(target=self._command_loop, daemon=True)
        cmd_thread.start()
        telemetry_period = 1.0 / max(1.0, float(self.args.telemetry_fps))
        next_telemetry = time.monotonic()
        try:
            while self.running:
                now = time.monotonic()
                if now >= next_telemetry:
                    self._send_telemetry()
                    next_telemetry = now + telemetry_period
                self._execute_loop_tick()
        finally:
            if self.robot is not None and self.robot.is_connected:
                self.robot.disconnect()
            self.cmd_sock.close()
            self.telemetry_sock.close()
            self.report_sock.close()
            print("[CMD] executor stopped, follower disconnected", flush=True)
    def _execute_loop_tick(self) -> None:
        with self.lock:
            trajectories = dict(self.trajectories)
        if (self.robot is None and not self.dry_run) or not self.motion_enabled or not trajectories:
            return
        if self.target_state is None:
            present = self.read_present()
            self.target_state = {name: float(present[name]) for name in JOINT_NAMES}
        now_wall = time.time()
        finished: list[int] = []
        sent_any = False
        for seq, traj in trajectories.items():
            now = now_wall - traj["t_start"]
            target = None
            for point in traj["points"]:
                if float(point.get("t", 0.0)) <= now:
                    target = point
                else:
                    break
            if target is not None:
                names = target.get("names")
                positions = target.get("positions")
                for name, value in zip(names, positions):
                    self.target_state[name] = float(value)
                sent_any = True
            if now >= traj["duration"]:
                if not traj["reported"]:
                    finished.append(seq)
        if sent_any:
            action = {f"{name}.pos": val for name, val in self.target_state.items()}
            try:
                if not self.dry_run:
                    self.robot.send_action(action)
                else:
                    print(
                        f"[CMD][dry] would send: "
                        + ", ".join(f"{k}={v:.1f}" for k, v in self.target_state.items()),
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[CMD] send_action error: {exc}", flush=True)
                with self.lock:
                    failed_seqs = sorted(self.trajectories)
                    self.trajectories = {}
                for seq in failed_seqs:
                    self._report(seq, "error", f"send_action: {exc}")
                return
        for seq in finished:
            with self.lock:
                traj = self.trajectories.get(seq)
                if traj is not None and not traj["reported"]:
                    traj["reported"] = True
            self._report(seq, "done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=30.0, help="command sampling rate")
    parser.add_argument("--telemetry-fps", type=float, default=20.0, help="read-only telemetry rate")
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="allow motion (targets still clamped by max_relative_target)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not open the serial port; only exercise the UDP protocol",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.fps > 100:
        raise SystemExit("--fps must be in (0, 100]")
    executor = CommandExecutor(args)
    executor.dry_run = args.dry_run
    executor.run()


if __name__ == "__main__":
    main()
