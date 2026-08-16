"""Hand-eye (wrist camera) fine localization of the block.

The wrist camera is calibrated (camera -> gripper_frame_link, 0.46px).  At the
approach pose the camera looks down at the table; the block appears in the
image.  Using the current joint angles (FK: gripper_frame_link -> base) we
compose camera -> base, back-project the block pixel to the table plane z=0,
and get the block's precise (x, y) in base_link.

This is the "fine" stage after the front camera coarse approach.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from atlas_runner import HANDEYE_CAMERA_PID, camera_device, environment, PROJECT_ROOT
import os

os.environ.update(environment())

sys.path.insert(0, str(PROJECT_ROOT))
from hand_eye_calibration import forward_kinematics, parse_urdf_chain  # noqa: E402

URDF = PROJECT_ROOT / "so101.urdf"
HANDEYE_INTRINSICS = PROJECT_ROOT / "calib_out/intrinsics.json"  # may need path fix
HANDEYE_EXTRINSIC = PROJECT_ROOT / "calib_out/extrinsics.json"

YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])
TABLE_Z = 0.0


def load_handeye_calib():
    intr = json.loads(HANDEYE_INTRINSICS.read_text(encoding="utf-8"))
    ext = json.loads(HANDEYE_EXTRINSIC.read_text(encoding="utf-8"))
    mtx = np.array(intr["handeye"]["camera_matrix"], dtype=float)
    dist = np.array(intr["handeye"]["dist_coeffs"], dtype=float)
    T_cam2grip = np.array(ext["handeye"]["T"], dtype=float)
    return mtx, dist, T_cam2grip


def detect_block_px(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None, mask
    M = cv2.moments(c)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"]), mask


def locate_block_handeye(mtx, dist, T_cam2grip, joints, joint_deg, table_z=TABLE_Z):
    """Return (x, y) of block in base via wrist camera, or None."""
    dev = camera_device(HANDEYE_CAMERA_PID, "handeye")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, frame

    px = detect_block_px(frame)
    if px is None:
        return None, frame
    (cx, cy), mask = px

    # bounding box in pixels -> corners
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    bx, by_px, bw, bh = cv2.boundingRect(c)
    corners_px = [(bx, by_px), (bx + bw, by_px), (bx + bw, by_px + bh), (bx, by_px + bh)]

    # camera -> base = gripper->base (FK) @ camera->gripper
    fk = {name: math.radians(joint_deg[name]) if name != "gripper" else joint_deg[name]
          for name in joint_deg}
    T_grip2base = forward_kinematics(joints, fk, "gripper_frame_link")
    T_cam2base = T_grip2base @ T_cam2grip

    def backproj(u, v):
        pts = np.array([[[u, v]]], np.float64)
        und = cv2.undistortPoints(pts, mtx, dist, P=mtx)
        uu, vv = und[0, 0]
        fx, fy = mtx[0, 0], mtx[1, 1]
        cxx, cyy = mtx[0, 2], mtx[1, 2]
        dcam = np.array([(uu - cxx) / fx, (vv - cyy) / fy, 1.0])
        dcam = dcam / np.linalg.norm(dcam)
        origin = T_cam2base[:3, 3]
        dbase = T_cam2base[:3, :3] @ dcam
        if abs(dbase[2]) < 1e-6:
            return None
        t = (table_z - origin[2]) / dbase[2]
        return origin + t * dbase

    center = backproj(cx, cy)
    corners_base = [backproj(u, v) for (u, v) in corners_px]
    if center is None or any(p is None for p in corners_base):
        return None, frame
    xs = [p[0] for p in corners_base]
    ys = [p[1] for p in corners_base]
    result = {
        "center": center,
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "corners_base": corners_base,
    }
    return result, frame


if __name__ == "__main__":
    # self-test with whatever the wrist camera sees right now
    joints = parse_urdf_chain(URDF)
    mtx, dist, T_cam2grip = load_handeye_calib()
    # read current joints from a JSON if provided, else zeros
    import json as _json
    jp = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if jp and jp.exists():
        cur = _json.loads(jp.read_text(encoding="utf-8"))
    else:
        cur = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 10}
    P, frame = locate_block_handeye(mtx, dist, T_cam2grip, joints, cur)
    print("block in base:", None if P is None else np.round(P, 3))
