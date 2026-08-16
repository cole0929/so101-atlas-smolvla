#!/opt/lerobot061/bin/python
"""Vision-guided vertical grasp of the yellow block (project: RL grasp).

Pipeline (all on the Atlas board):
  1. Front camera localizes the yellow block -> (x, y) on table (z=0).
  2. IK (ik_solver.py) computes the 5 arm joint angles to place the gripper
     above the block, pointing down.
  3. SO101Follower drives the arm: move above block -> open gripper ->
     descend -> close gripper -> lift -> back to a safe pose.

Safety:
  - Target must be inside the workspace (x 0..0.40, y -0.30..0.20).
  - All IK joint angles are checked against URDF limits.
  - robot.send_action clamps per-call relative motion (max_relative_target).
  - --dry-run prints the plan without moving anything.
  - Ctrl+C disconnects and disables torque.

Usage (board):

  cd /root/lerobot_project
  /opt/lerobot061/bin/python 20_grasp_block.py --dry-run          # plan only
  /opt/lerobot061/bin/python 20_grasp_block.py --grasp-height 0.08  # real grasp
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

from atlas_runner import FRONT_CAMERA_PID, camera_device, environment, serial_port, FOLLOWER_SERIAL

os.environ.update(environment())

from atlas_runner import PROJECT_ROOT  # noqa: E402
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from ik_solver import solve_ik, add_downward_wrist  # noqa: E402
from hand_eye_calibration import forward_kinematics, parse_urdf_chain  # noqa: E402

URDF = PROJECT_ROOT / "so101.urdf"
INTRINSICS = PROJECT_ROOT / "front_intrinsics_v2.json"
EXTRINSIC = PROJECT_ROOT / "front_extrinsic_geometric.json"

MOTOR_KEYS = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

# workspace bounds (base_link, meters)
WS_X_MIN, WS_X_MAX = 0.0, 0.40
WS_Y_MIN, WS_Y_MAX = -0.30, 0.20


def load_calib():
    intr = json.loads(INTRINSICS.read_text(encoding="utf-8"))
    ext = json.loads(EXTRINSIC.read_text(encoding="utf-8"))
    return np.array(intr["camera_matrix"], dtype=float), np.array(intr["dist_coeffs"], dtype=float), np.array(ext["T"], dtype=float)


def locate_block(mtx, dist, T_cam2base, table_z=0.0):
    """Return (x, y) of yellow block in base_link or None."""
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
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
    t = (table_z - origin[2]) / dbase[2]
    P = origin + t * dbase
    return P, frame


def make_robot(max_relative_target: float) -> SO101Follower:
    follower_port = serial_port(FOLLOWER_SERIAL, "follower")
    limits = {
        name.removesuffix(".pos"): (max_relative_target * 2.0 if name == "gripper.pos" else max_relative_target)
        for name in MOTOR_KEYS
    }
    return SO101Follower(
        SO101FollowerConfig(
            port=follower_port,
            id="my_awesome_follower_arm",
            cameras={},
            max_relative_target=None if max_relative_target <= 0 else limits,
            disable_torque_on_disconnect=True,
        )
    )


def action_dict(values: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(MOTOR_KEYS, values, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="plan only, no motion")
    parser.add_argument("--grasp-height", type=float, default=0.025,
                        help="gripper height above table at grasp (m)")
    parser.add_argument("--approach-height", type=float, default=0.20,
                        help="height to move above block before descending (m)")
    parser.add_argument("--offset-x", type=float, default=0.0,
                        help="grasp target x offset (m): negative pulls tip toward base")
    parser.add_argument("--offset-y", type=float, default=0.0,
                        help="grasp target y offset (m)")
    parser.add_argument("--offset-z", type=float, default=0.0,
                        help="grasp target z offset (m): negative lowers the jaw "
                             "so it wraps the block center (gripper_frame_link sits "
                             "below the jaw gap)")
    parser.add_argument("--jaw-clearance", type=float, default=0.015,
                        help="distance of the lower jaw OUTSIDE the block's "
                             "lower edge (x_min) in meters; positive = jaw sits "
                             "to the -x side of the block edge so the block can "
                             "slide into the jaw gap")
    parser.add_argument("--handeye-offset-y", type=float, default=0.0,
                        help="extra y compensation for the hand-eye camera "
                             "(m). The wrist camera sits ~5cm left of the jaw "
                             "gap; if its calibrated extrinsic y is wrong, add "
                             "the measured offset here (negative = move jaw "
                             "left, toward the camera side)")
    parser.add_argument("--max-relative-target", type=float, default=30.0,
                        help="per-call joint clamp in degrees (gripper 2x)")
    parser.add_argument("--open-gripper", type=float, default=50.0,
                        help="gripper open percent")
    parser.add_argument("--close-gripper", type=float, default=10.0,
                        help="gripper close percent")
    parser.add_argument("--test-move", action="store_true",
                        help="only execute the first (approach) move for safe verification")
    args = parser.parse_args()

    print(f"grasp height: {args.grasp_height}m  approach: {args.approach_height}m")

    # 1) locate block (bounding box -> edges)
    mtx, dist, T_cam2base = load_calib()
    sys.path.insert(0, str(PROJECT_ROOT))
    from block_bbox import detect_block_bbox  # noqa: E402
    bbox, frame = detect_block_bbox(mtx, dist, T_cam2base)
    if bbox is None:
        raise SystemExit("no yellow block detected in front camera")
    bx, by = float(bbox["center"][0]), float(bbox["center"][1])
    x_min, x_max = bbox["x_min"], bbox["x_max"]
    y_min, y_max = bbox["y_min"], bbox["y_max"]
    print(f"block center: ({bx*100:.1f}, {by*100:.1f}) cm")
    print(f"block x range: [{x_min*100:.1f}, {x_max*100:.1f}] cm  y range: [{y_min*100:.1f}, {y_max*100:.1f}] cm")

    # workspace check
    if not (WS_X_MIN <= bx <= WS_X_MAX and WS_Y_MIN <= by <= WS_Y_MAX):
        raise SystemExit(f"block outside workspace: ({bx*100:.1f}, {by*100:.1f}) cm")
    print("block inside workspace OK")

    # Grasp target: the jaw opens along X with the lower jaw on the -x side.
    # Place the jaw gap center so the lower jaw lands at (x_min - jaw_clearance).
    # jaw_clearance: how far outside the block's lower edge the jaw sits (m).
    jaw_clearance = getattr(args, "jaw_clearance", 0.015)
    gx = x_min - jaw_clearance
    gy = float(bbox["center"][1]) + args.offset_y
    gz = args.grasp_height + args.offset_z
    print(f"grasp target: lower jaw at x={gx*100:.1f}cm (< x_min {x_min*100:.1f}cm), "
          f"center y={gy*100:.1f}cm, z={gz*100:.1f}cm")

    # 2) IK for approach and grasp poses
    joints = parse_urdf_chain(URDF)
    initial = {"shoulder_lift": math.radians(60), "elbow_flex": math.radians(-90), "wrist_flex": math.radians(30)}

    approach_tgt = np.array([gx, gy, args.approach_height])
    grasp_tgt = np.array([gx, gy, gz])
    q_approach = solve_ik(joints, approach_tgt, initial=initial)
    q_approach = add_downward_wrist(joints, q_approach, roll=0.0)
    if q_approach is None:
        raise SystemExit("IK failed for approach pose")

    # 3) plan: approach -> (fine locate) -> grasp -> close -> lift -> home
    home_tgt = np.array([0.25, 0.0, 0.30])
    q_home = solve_ik(joints, home_tgt, initial=initial)
    q_home = add_downward_wrist(joints, q_home, roll=0.0)
    if q_home is None:
        raise SystemExit("IK failed for home pose")

    # approach + home first (approach executed before fine localization)
    plan = []
    plan.append(("move_approach", q_approach, args.open_gripper))
    plan.append(("home", q_home, args.close_gripper))

    print("\n=== PLAN (stage 1) ===")
    for name, q, grip in plan:
        parts = []
        for k in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
            parts.append(f"{k}={math.degrees(q[k]):.0f}deg")
        print(f"  {name}: grip={grip}%  " + "  ".join(parts))
    print("=== END PLAN ===\n")

    if args.dry_run:
        print("DRY RUN: no motion executed")
        return

    # 4) execute with interpolation (respect per-call joint clamps)
    print("Connecting to follower...")
    robot = make_robot(args.max_relative_target)
    robot.connect()
    print("Connected. Executing plan (Ctrl+C to abort).")

    def read_actual() -> np.ndarray:
        obs = robot.get_observation()
        return np.array([obs[k] for k in MOTOR_KEYS], dtype=float)

    def move_to(target_vals: np.ndarray, settle: float = 1.5) -> None:
        """Interpolate from current actual joints to target in small steps."""
        current = read_actual()
        max_step = args.max_relative_target  # deg per call (gripper 2x)
        # number of steps needed for the largest joint delta
        delta = np.abs(target_vals - current)
        steps = int(np.ceil(np.max(delta[:5] / max_step))) if max_step > 0 else 1
        steps = max(1, min(steps, 30))
        print(f"  interpolating in {steps} steps")
        for s in range(1, steps + 1):
            interp = current + (target_vals - current) * (s / steps)
            robot.send_action(action_dict(interp))
            time.sleep(settle / steps)
        # final exact target
        robot.send_action(action_dict(target_vals))
        time.sleep(settle)

    try:
        if args.test_move:
            plan = plan[:1]
            print("TEST MODE: only the approach move will be executed")
        for name, q, grip in plan:
            values = np.zeros(6)
            for i, key in enumerate(MOTOR_KEYS):
                jname = key.removesuffix(".pos")
                if jname == "gripper":
                    values[i] = grip
                else:
                    values[i] = math.degrees(q.get(jname, 0.0))
            print(f"[{name}] target: { {k: round(v,1) for k,v in zip(MOTOR_KEYS, values)} }")
            move_to(values)
            print(f"[{name}] done")

            # After reaching approach pose, fine-localize with the wrist camera.
            if name == "move_approach" and not args.test_move:
                print("\n=== FINE LOCALIZATION (hand-eye camera) ===")
                sys.path.insert(0, str(PROJECT_ROOT))
                from handeye_localize import locate_block_handeye, load_handeye_calib  # noqa: E402

                # current actual joints (calibrated, degrees)
                obs = robot.get_observation()
                actual_deg = {k.removesuffix(".pos"): float(obs[k]) for k in MOTOR_KEYS}
                print("current joints:", {k: round(v, 1) for k, v in actual_deg.items()})

                hmtx, hdist, T_cam2grip = load_handeye_calib()
                Pfine, _frame = locate_block_handeye(hmtx, hdist, T_cam2grip, joints, actual_deg)
                if Pfine is None:
                    print("WARNING: hand-eye camera did not see the block; using front-camera estimate")
                    fx_, fy_ = gx, gy
                else:
                    # use hand-eye bbox: lower jaw at hand-eye x_min - clearance
                    fx_ = Pfine["x_min"] - jaw_clearance
                    fy_ = float(Pfine["center"][1]) + args.handeye_offset_y
                    print(f"hand-eye block x range: [{Pfine['x_min']*100:.1f}, {Pfine['x_max']*100:.1f}] cm")
                    print(f"hand-eye grasp target: lower jaw x={fx_*100:.1f}cm, y={fy_*100:.1f}cm  "
                          f"(front was {gx*100:.1f}, {gy*100:.1f})")

                # recompute grasp pose from fine target
                grasp_tgt = np.array([fx_, fy_, gz])
                q_grasp = solve_ik(joints, grasp_tgt, initial=q_approach)
                if q_grasp is None:
                    raise SystemExit("IK failed for fine grasp pose")
                q_grasp = add_downward_wrist(joints, q_grasp, roll=0.0)
                Tchk = forward_kinematics(joints, q_grasp, "gripper_frame_link")
                print(f"fine grasp IK error: {np.linalg.norm(Tchk[:3,3] - grasp_tgt)*100:.1f} cm")

                # execute: grasp (open) -> close -> lift (back to approach) 
                for sname, sq, sgrip in (
                    ("grasp", q_grasp, args.open_gripper),
                    ("close", q_grasp, args.close_gripper),
                    ("lift", q_approach, args.close_gripper),
                ):
                    svals = np.zeros(6)
                    for i, key in enumerate(MOTOR_KEYS):
                        jn = key.removesuffix(".pos")
                        if jn == "gripper":
                            svals[i] = sgrip
                        else:
                            svals[i] = math.degrees(sq.get(jn, 0.0))
                    print(f"[{sname}] target: { {k: round(v,1) for k,v in zip(MOTOR_KEYS, svals)} }")
                    move_to(svals)
                    print(f"[{sname}] done")
    except KeyboardInterrupt:
        print("\nAborted by user")
    finally:
        robot.disconnect()
        print("Disconnected (torque disabled).")


if __name__ == "__main__":
    main()
