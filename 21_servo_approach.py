"""Vision-servoed approach: move the arm toward the block using camera
feedback, no absolute IK needed.

Strategy (tolerant to FK/calibration mismatch):
  1. Front camera localizes block -> (bx, by) on table.
  2. Read current joint angles. Compute the Jacobian (d(tip)/dq) with the
     URDF FK - even if the absolute mapping is off, the DIRECTION each joint
     moves the tip is approximately right.
  3. Each iteration: compute tip motion needed to reduce (tip - target)
     horizontal error, pick joint moves that produce that motion, send small
     clamped steps, re-locate the block, repeat until close.
  4. Then descend (all pitch joints together a little) -> close gripper ->
     lift.

This is a coarse approach controller; the fine grasp uses the hand-eye
camera later.  --dry-run prints steps without moving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from atlas_runner import FOLLOWER_SERIAL, FRONT_CAMERA_PID, PROJECT_ROOT, camera_device, environment, serial_port

os.environ.update(environment())

from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from hand_eye_calibration import forward_kinematics, parse_urdf_chain  # noqa: E402

URDF = PROJECT_ROOT / "so101.urdf"
INTRINSICS = PROJECT_ROOT / "front_intrinsics_v2.json"
EXTRINSIC = PROJECT_ROOT / "front_extrinsic_geometric.json"
CALIBRATION = PROJECT_ROOT / "lerobot_home/calibration/robots/so_follower/my_awesome_follower_arm.json"

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110, 110),
    "shoulder_lift": (-100, 100),
    "elbow_flex": (-97, 97),
    "wrist_flex": (-109, 109),
    "wrist_roll": (-178, 178),
}
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])


def load_calib():
    intr = json.loads(INTRINSICS.read_text(encoding="utf-8"))
    ext = json.loads(EXTRINSIC.read_text(encoding="utf-8"))
    return (np.array(intr["camera_matrix"], dtype=float),
            np.array(intr["dist_coeffs"], dtype=float),
            np.array(ext["T"], dtype=float))


def read_joints_deg(port: str) -> dict[str, float]:
    calib = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    cal = {name: MotorCalibration(**v) for name, v in calib.items()}
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
        calibration=cal,
    )
    bus.connect()
    try:
        positions = bus.sync_read("Present_Position", normalize=True)
        return {name: float(positions[name]) for name in ARM + ["gripper"]}
    finally:
        bus.disconnect()


def send_joints(port: str, targets_deg: dict[str, float]) -> None:
    calib = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    cal = {name: MotorCalibration(**v) for name, v in calib.items()}
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
        calibration=cal,
    )
    bus.connect()
    try:
        bus.write("Goal_Position", normalize=True, values=targets_deg)
    finally:
        bus.disconnect()


def locate_block(mtx, dist, T_cam2base):
    dev = camera_device(FRONT_CAMERA_PID, "front")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, frame
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 300:
        return None, frame
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    pts = np.array([[[cx, cy]]], np.float64)
    und = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    uu, vv = und[0, 0]
    fx, fy = mtx[0, 0], mtx[1, 1]
    cxx, cyy = mtx[0, 2], mtx[1, 2]
    dcam = np.array([(uu - cxx) / fx, (vv - cyy) / fy, 1.0])
    dcam = dcam / np.linalg.norm(dcam)
    origin = T_cam2base[:3, 3]
    dbase = T_cam2base[:3, :3] @ dcam
    if abs(dbase[2]) < 1e-6:
        return None, frame
    t = (0.0 - origin[2]) / dbase[2]
    P = origin + t * dbase
    return P, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--step", type=float, default=3.0, help="max joint step per iteration (deg)")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--xy-tol", type=float, default=0.03, help="goal xy tolerance (m)")
    args = parser.parse_args()

    mtx, dist, T_cam2base = load_calib()
    joints = parse_urdf_chain(URDF)
    port = serial_port(FOLLOWER_SERIAL, "follower")

    P, frame = locate_block(mtx, dist, T_cam2base)
    if P is None:
        raise SystemExit("no block detected")
    target = np.array([P[0], P[1], 0.0])
    print(f"block target: ({target[0]*100:.1f}, {target[1]*100:.1f}) cm")

    current = read_joints_deg(port)
    print(f"current joints: { {k: round(v,1) for k,v in current.items()} }")

    # Convert current to radians for FK (gripper ignored).
    def to_fk(q_deg):
        return {k: math.radians(q_deg[k]) if k != "gripper" else q_deg[k] for k in q_deg}

    # Jacobian: how the tip moves when each joint moves +1 deg (from current).
    def tip_pos(q_deg):
        T = forward_kinematics(joints, to_fk(q_deg), "gripper_frame_link")
        return T[:3, 3]

    tip0 = tip_pos(current)
    print(f"tip now (URDF FK): ({tip0[0]*100:.1f}, {tip0[1]*100:.1f}, {tip0[2]*100:.1f}) cm")

    # The FK may be offset from reality; use horizontal direction of target
    # relative to tip, and adjust joints iteratively with camera feedback.
    q = dict(current)
    h = 1.0  # deg
    print("\n--- servo loop ---")
    for it in range(args.iters):
        # jacobian columns: d(tip)/d(qi) per deg
        J = np.zeros((3, 5))
        for i, name in enumerate(ARM):
            qh = dict(q)
            qh[name] += h
            J[:, i] = (tip_pos(qh) - tip0) / h
        # current tip (FK) and horizontal error to target
        tip = tip_pos(q)
        err = target[:2] - tip[:2]
        if np.linalg.norm(err) < args.xy_tol:
            print(f"iter {it}: reached xy error {np.linalg.norm(err)*100:.1f}cm")
            break
        # We want d(tip_xy) = J_xy dq -> dq = pinv(J_xy) * err (small steps)
        Jxy = J[:2, :]
        dq = np.linalg.pinv(Jxy) @ err
        # clamp per-joint step
        dq = np.clip(dq, -args.step, args.step)
        for i, name in enumerate(ARM):
            q[name] = np.clip(q[name] + dq[i], JOINT_LIMITS_DEG[name][0], JOINT_LIMITS_DEG[name][1])
        print(f"iter {it}: err=({err[0]*100:.1f},{err[1]*100:.1f})cm step={ {ARM[i]: round(dq[i],1) for i in range(5)} }")

        if not args.dry_run:
            send_joints(port, q)
            time.sleep(1.0)
        # re-locate block (camera feedback) each few iters
        if it % 3 == 2:
            P2, _ = locate_block(mtx, dist, T_cam2base)
            if P2 is not None:
                target = np.array([P2[0], P2[1], 0.0])
                print(f"  re-located block: ({target[0]*100:.1f}, {target[1]*100:.1f}) cm")

    tip = tip_pos(q)
    print(f"\nfinal FK tip: ({tip[0]*100:.1f}, {tip[1]*100:.1f}, {tip[2]*100:.1f}) cm")
    print(f"final joints: { {k: round(v,1) for k,v in q.items()} }")
    if args.dry_run:
        print("DRY RUN: no motion")


if __name__ == "__main__":
    main()
