#!/usr/bin/env python3
"""Read-only SO-ARM101 motor telemetry source for the LeRobot Python runtime."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from pathlib import Path

from atlas_runner import FOLLOWER_SERIAL, PROJECT_ROOT, environment, serial_port

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


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: MotorCalibration(**values) for name, values in raw.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15000)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero")

    follower_port = serial_port(FOLLOWER_SERIAL, "follower")
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus(
        port=follower_port,
        motors=motors,
        calibration=load_calibration(CALIBRATION_PATH),
    )
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    bus.connect()
    print(f"[SO101] Read-only telemetry from {follower_port} at {args.fps:.1f} Hz", flush=True)
    period = 1.0 / args.fps
    next_tick = time.monotonic()
    packets = 0
    try:
        while running:
            positions = bus.sync_read("Present_Position", normalize=True)
            payload = {
                "stamp_ns": time.time_ns(),
                "names": JOINT_NAMES,
                "positions": [float(positions[name]) for name in JOINT_NAMES],
            }
            udp.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), (args.host, args.port))
            packets += 1
            if packets % max(1, round(args.fps * 5)) == 0:
                values = ", ".join(f"{value:.1f}" for value in payload["positions"])
                print(f"[SO101] deg/gripper: {values}", flush=True)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        udp.close()
        # Preserve the existing torque state; this telemetry process never commands motion.
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

