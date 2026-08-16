#!/usr/bin/env python3
"""SO-ARM101 hand-eye calibration (project 1, stage B) — PC side.

Takes the calibration dataset captured on the Atlas board
(/root/lerobot_project/calib_data/), computes:

  1. Camera intrinsics for both cameras (OpenCV calibrateCamera)
  2. handeye (wrist camera, eye-in-hand):  X = camera -> gripper_frame_link
  3. front  (fixed overhead, eye-to-hand): X = camera -> base_link

The end-effector pose for each sample is computed from the joint angles using
a small URDF forward-kinematics parser (no external robot library needed).

Usage:

  python hand_eye_calibration.py --data <calib_data_dir> --urdf <so101.urdf>
      --board 11 8 0.010 --out <output_dir>

Outputs (written to --out):
  intrinsics.json         camera matrices + distortion for handeye and front
  handeye_extrinsic.json  camera->gripper_frame_link (translation + quaternion)
  front_extrinsic.json    camera->base_link
  report.txt              per-sample corner detection summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# URDF forward kinematics
# --------------------------------------------------------------------------- #


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rpy_to_rot(rpy) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def parse_urdf_chain(path: Path) -> list[dict]:
    """Return ordered joint list: name, parent, child, origin xyz/rpy, axis, type."""
    root = ET.parse(path).getroot()
    joints = []
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype not in ("revolute", "continuous", "prismatic", "fixed"):
            continue
        name = joint.get("name")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        if origin is None:
            xyz, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        else:
            xyz = tuple(map(float, origin.get("xyz", "0 0 0").split()))
            rpy = tuple(map(float, origin.get("rpy", "0 0 0").split()))
        axis = joint.find("axis")
        axis_vec = tuple(map(float, axis.get("xyz", "0 0 1").split())) if axis is not None else (0, 0, 1)
        joints.append(
            {"name": name, "type": jtype, "parent": parent, "child": child,
             "xyz": xyz, "rpy": rpy, "axis": axis_vec}
        )
    return joints


def forward_kinematics(joints: list[dict], joint_positions: dict[str, float],
                       tip_frame: str, base_frame: str = "base_link") -> np.ndarray:
    """Compute T (4x4) of tip_frame in base_frame using chain order."""
    # Build parent->child transforms and resolve the chain.
    transforms: dict[str, np.ndarray] = {base_frame: np.eye(4)}
    # Resolve in order: base already known, iterate until all frames resolved.
    remaining = list(joints)
    progress = True
    while remaining and progress:
        progress = False
        for j in list(remaining):
            if j["parent"] not in transforms:
                continue
            t = np.eye(4)
            t[:3, 3] = j["xyz"]
            t[:3, :3] = rpy_to_rot(j["rpy"])
            if j["type"] == "fixed":
                pass  # static transform only
            elif j["type"] in ("revolute", "continuous"):
                theta = joint_positions.get(j["name"], 0.0)
                axis = np.array(j["axis"], dtype=float)
                # Rotation about arbitrary axis via Rodriguez.
                k = axis / np.linalg.norm(axis)
                kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                rot = np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)
                joint_rot = np.eye(4)
                joint_rot[:3, :3] = rot
                t = t @ joint_rot
            elif j["type"] == "prismatic":
                d = joint_positions.get(j["name"], 0.0)
                axis = np.array(j["axis"], dtype=float) / np.linalg.norm(j["axis"])
                t[:3, 3] += axis * d
            transforms[j["child"]] = transforms[j["parent"]] @ t
            remaining.remove(j)
            progress = True
    if tip_frame not in transforms:
        raise KeyError(f"tip frame {tip_frame} not reachable from {base_frame}")
    return transforms[tip_frame]


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w) quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
            (R[1, 0] - R[0, 1]) / s)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def detect_board(image: np.ndarray, cols: int, rows: int, show: bool = False):
    """Detect checkerboard corners; returns (corners, image_with_overlay) or (None, image)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    display = image.copy()
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, (cols, rows), corners, found)
    return (corners if found else None), display


def build_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return pts


def calibrate_intrinsics(images: list[np.ndarray], cols: int, rows: int, square_m: float,
                         show: bool = False):
    """Return (mtx, dist, reproj_error, usable_images)."""
    objp = build_object_points(cols, rows, square_m)
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    usable: list[int] = []
    for i, img in enumerate(images):
        corners, _ = detect_board(img, cols, rows, show)
        if corners is None:
            continue
        obj_points.append(objp)
        img_points.append(corners)
        usable.append(i)
    if len(usable) < 4:
        raise RuntimeError(f"only {len(usable)} usable images for intrinsics (need >=4)")
    gray = cv2.cvtColor(images[usable[0]], cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    error = 0.0
    for obj_pts, img_pts in zip(obj_points, img_points):
        _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
        error += float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
    error /= len(usable)
    return mtx, dist, error, usable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="board calib_data dir")
    parser.add_argument("--urdf", required=True, type=Path, help="so101.urdf path")
    parser.add_argument("--board", required=True, nargs=3, type=float,
                        help="cols rows square_size_m  (e.g. 11 8 0.010)")
    parser.add_argument("--out", type=Path, default=Path("./calib_out"))
    parser.add_argument("--show", action="store_true", help="show corner detection")
    parser.add_argument("--tip-frame", default="gripper_frame_link")
    args = parser.parse_args()

    cols, rows, square = int(args.board[0]), int(args.board[1]), args.board[2]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    joints = parse_urdf_chain(args.urdf)
    print(f"URDF joints parsed: {[j['name'] for j in joints]}")

    sample_dirs = sorted(args.data.glob("sample_*"))
    if not sample_dirs:
        raise SystemExit(f"no sample_* dirs in {args.data}")
    print(f"{len(sample_dirs)} samples found")

    handeye_images: list[np.ndarray] = []
    front_images: list[np.ndarray] = []
    handeye_joints: list[dict] = []
    front_joints: list[dict] = []
    report: list[str] = []
    handeye_usable_idx: list[int] = []
    front_usable_idx: list[int] = []

    # Pass 1: collect images + joint angles, check board detection per camera.
    for i, sdir in enumerate(sample_dirs):
        joints_file = sdir / "joints.json"
        if not joints_file.exists():
            print(f"[{i}] missing joints.json, skipping")
            continue
        jdata = json.loads(joints_file.read_text(encoding="utf-8"))
        # Convert degrees -> radians for FK (gripper stays 0-100 -> rad via range).
        fk_positions = {}
        for name, value in jdata.items():
            if name == "gripper":
                fk_positions[name] = 0.0  # gripper does not affect tip pose
            else:
                fk_positions[name] = math.radians(value)

        h_img = cv2.imread(str(sdir / "handeye.jpg"))
        f_img = cv2.imread(str(sdir / "front.jpg"))
        if h_img is None or f_img is None:
            print(f"[{i}] image read failed, skipping")
            continue

        h_corners, _ = detect_board(h_img, cols, rows, args.show)
        f_corners, _ = detect_board(f_img, cols, rows, args.show)
        if h_corners is not None:
            handeye_images.append(h_img)
            handeye_joints.append(fk_positions)
            handeye_usable_idx.append(i)
        if f_corners is not None:
            front_images.append(f_img)
            front_joints.append(fk_positions)
            front_usable_idx.append(i)
        report.append(
            f"[{i}] handeye={'OK' if h_corners is not None else '--'} "
            f"front={'OK' if f_corners is not None else '--'}"
        )
        print(report[-1])

    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nhandeye usable: {len(handeye_images)}/{len(sample_dirs)}")
    print(f"front usable:   {len(front_images)}/{len(sample_dirs)}")

    if len(handeye_images) < 4 and len(front_images) < 4:
        raise SystemExit("too few usable images (<4) for either camera")

    # ------------------------------------------------------------------ #
    # Intrinsics
    # ------------------------------------------------------------------ #
    intrinsics: dict = {}
    for cam, images, usable_idx in (
        ("handeye", handeye_images, handeye_usable_idx),
        ("front", front_images, front_usable_idx),
    ):
        if len(images) < 4:
            print(f"[{cam}] skip intrinsics (only {len(images)} usable)")
            continue
        mtx, dist, error, usable = calibrate_intrinsics(images, cols, rows, square, args.show)
        intrinsics[cam] = {
            "usable_samples": [int(i) for i in usable],
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.reshape(-1).tolist(),
            "reprojection_error_px": round(error, 4),
        }
        print(f"[{cam}] intrinsics: reproj error={error:.3f} px")

    # ------------------------------------------------------------------ #
    # Hand-eye calibration
    # ------------------------------------------------------------------ #
    def compute_extrinsic(camera_images: list[np.ndarray], joint_list: list[dict],
                          mtx: np.ndarray, dist: np.ndarray, mode: str) -> dict:
        """mode='eye_in_hand' -> X = camera->gripper; 'eye_to_hand' -> camera->base."""
        objp = build_object_points(cols, rows, square)
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []
        for img, jpos in zip(camera_images, joint_list):
            corners, _ = detect_board(img, cols, rows, args.show)
            if corners is None:
                continue
            _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
            R_target2cam = cv2.Rodrigues(rvec)[0]
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(tvec.reshape(3))
            T_grip2base = forward_kinematics(joints, jpos, args.tip_frame)
            R_gripper2base_list.append(T_grip2base[:3, :3])
            t_gripper2base_list.append(T_grip2base[:3, 3])
        if len(R_target2cam_list) < 4:
            raise RuntimeError("not enough poses for hand-eye")
        if mode == "eye_in_hand":
            R_cam2grip, t_cam2grip = cv2.calibrateHandEye(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                method=cv2.CALIB_HAND_EYE_TSAI,
            )
            T = np.eye(4)
            T[:3, :3] = R_cam2grip
            T[:3, 3] = t_cam2grip.reshape(3)
            return {"mode": "eye_in_hand", "camera_to_tip": T.tolist(),
                    "quaternion": matrix_to_quat(R_cam2grip),
                    "translation_m": t_cam2grip.reshape(3).tolist()}
        else:
            # eye-to-hand: solve for camera->base using AX=ZB formulation.
            # cv2.calibrateHandEye with eye-to-hand flag solves X = base->camera
            # using the same inputs (R_gripper2base, R_target2cam).
            R_base2cam, t_base2cam = cv2.calibrateHandEye(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                method=cv2.CALIB_HAND_EYE_TSAI,
            )
            # Invert to get camera->base.
            R_cam2base = R_base2cam.T
            t_cam2base = -R_cam2base @ t_base2cam.reshape(3)
            T = np.eye(4)
            T[:3, :3] = R_cam2base
            T[:3, 3] = t_cam2base
            return {"mode": "eye_to_hand", "camera_to_base": T.tolist(),
                    "quaternion": matrix_to_quat(R_cam2base),
                    "translation_m": t_cam2base.tolist()}

    handeye_result = front_result = None
    if "handeye" in intrinsics:
        mtx = np.array(intrinsics["handeye"]["camera_matrix"])
        dist = np.array(intrinsics["handeye"]["dist_coeffs"])
        handeye_result = compute_extrinsic(handeye_images, handeye_joints, mtx, dist, "eye_in_hand")
        print("\n[handeye] eye-in-hand result:")
        print(f"  camera->tip  t={np.round(handeye_result['translation_m'], 4)} m")
        print(f"              q={tuple(round(x, 4) for x in handeye_result['quaternion'])}")
        (out / "handeye_extrinsic.json").write_text(
            json.dumps(handeye_result, indent=2), encoding="utf-8")

    if "front" in intrinsics:
        mtx = np.array(intrinsics["front"]["camera_matrix"])
        dist = np.array(intrinsics["front"]["dist_coeffs"])
        front_result = compute_extrinsic(front_images, front_joints, mtx, dist, "eye_to_hand")
        print("\n[front] eye-to-hand result:")
        print(f"  camera->base t={np.round(front_result['translation_m'], 4)} m")
        print(f"               q={tuple(round(x, 4) for x in front_result['quaternion'])}")
        (out / "front_extrinsic.json").write_text(
            json.dumps(front_result, indent=2), encoding="utf-8")

    (out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    print(f"\nAll results written to {out.resolve()}")


if __name__ == "__main__":
    main()
