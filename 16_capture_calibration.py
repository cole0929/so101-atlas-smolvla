#!/opt/lerobot061/bin/python
"""Capture hand-eye calibration dataset on the Atlas board (project 1, B).

For each pose of the arm:
  - reads BOTH cameras (handeye on the wrist + front wide-angle overhead)
  - records the current follower joint positions in DEGREES
  - saves images + joints as a numbered sample

The PC-side calibration script later uses the joint angles to compute the
end-effector pose from the URDF (forward kinematics), then solves hand-eye
calibration for both cameras.

Usage (on the board, arm powered, checkerboard in the workspace):

  cd /root/lerobot_project
  /opt/lerobot061/bin/python 16_capture_calibration.py --poses 15

Controls:
  ENTER        capture current pose
  'r'          retake (delete last sample)
  'q' / Ctrl+C quit

Output: /root/lerobot_project/calib_data/
  sample_000/handeye.jpg  sample_000/front.jpg  sample_000/joints.json
  samples.json
"""

from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=15)
    parser.add_argument("--out", default="/root/lerobot_project/calib_data")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    handeye = camera_device(HANDEYE_CAMERA_PID, "handeye")
    front = camera_device(FRONT_CAMERA_PID, "front")
    cap_h = cv2.VideoCapture(handeye)
    cap_f = cv2.VideoCapture(front)
    if not cap_h.isOpened() or not cap_f.isOpened():
        raise SystemExit(f"cannot open cameras: handeye={handeye} front={front}")
    for cap in (cap_h, cap_f):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    follower_port = serial_port(FOLLOWER_SERIAL, "follower")
    print(f"Follower port: {follower_port}")
    print(f"Cameras: handeye={handeye} front={front}")
    print("Press ENTER to capture, 'r' to retake, 'q'/Ctrl+C to quit.")

    samples: list[dict] = []
    sample_index = 0
    last_key = 0.0

    while sample_index < args.poses:
        ok_h, frame_h = cap_h.read()
        ok_f, frame_f = cap_f.read()
        if not ok_h or not ok_f:
            print("camera read failed, retrying...")
            time.sleep(0.2)
            continue

        preview = np.hstack((frame_h, frame_f))
        label = f"pose {sample_index}/{args.poses}  [ENTER]capture [r]retake [q]quit"
        cv2.putText(preview, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibration capture (left=handeye right=front)", preview)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key in (13, 10) and now - last_key > args.interval:  # ENTER
            last_key = now
            joints = read_joints(follower_port)
            sample_dir = out / f"sample_{sample_index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_dir / "handeye.jpg"), frame_h)
            cv2.imwrite(str(sample_dir / "front.jpg"), frame_f)
            (sample_dir / "joints.json").write_text(
                json.dumps(joints, indent=2), encoding="utf-8"
            )
            samples.append({"index": sample_index, "joints": joints})
            print(f"[{sample_index:03d}] captured")
            print("   joints:", {k: round(v, 2) for k, v in joints.items()})
            sample_index += 1
        elif key == ord("r") and sample_index > 0 and now - last_key > args.interval:
            last_key = now
            sample_index -= 1
            sample_dir = out / f"sample_{sample_index:03d}"
            for f in ("handeye.jpg", "front.jpg", "joints.json"):
                (sample_dir / f).unlink(missing_ok=True)
            sample_dir.rmdir()
            samples.pop()
            print(f"[{sample_index:03d}] retaken")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap_h.release()
    cap_f.release()
    (out / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"DONE: {len(samples)} samples saved to {out}")
    print("Pack for the PC with:")
    print("  tar -czf /root/lerobot_project/calib_data.tar.gz -C /root/lerobot_project calib_data")


if __name__ == "__main__":
    main()
