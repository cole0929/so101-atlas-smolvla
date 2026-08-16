#!/opt/lerobot061/bin/python
"""SO-ARM101 hand-eye calibration runner for the Atlas board.

Computes intrinsics + extrinsics with the board's OpenCV (4.13, which still
has calibrateHandEye).  Input: calibration dataset captured by
16_capture_calibration.py plus the URDF for forward kinematics.

Copy this script and the URDF to the board, then:

  cd /root/lerobot_project
  /opt/lerobot061/bin/python 17_handeye_calibrate.py \
      --data /root/lerobot_project/calib_data \
      --urdf /root/lerobot_project/so101.urdf \
      --board 11 8 0.010 --out /root/lerobot_project/calib_out

Outputs: intrinsics.json, handeye_extrinsic.json, front_extrinsic.json,
report.txt  (same schema as the PC script)
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
# URDF forward kinematics (mirrors the PC script)
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
    transforms: dict[str, np.ndarray] = {base_frame: np.eye(4)}
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
                pass
            elif j["type"] in ("revolute", "continuous"):
                theta = joint_positions.get(j["name"], 0.0)
                axis = np.array(j["axis"], dtype=float)
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


def detect_board(image: np.ndarray, cols: int, rows: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def build_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--board", required=True, nargs=3, type=float,
                        help="cols rows square_size_m (e.g. 11 8 0.010)")
    parser.add_argument("--out", type=Path, default=Path("./calib_out"))
    parser.add_argument("--tip-frame", default="gripper_frame_link")
    args = parser.parse_args()

    cols, rows, square = int(args.board[0]), int(args.board[1]), args.board[2]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    joints = parse_urdf_chain(args.urdf)
    print(f"URDF joints: {[j['name'] for j in joints]}")

    sample_dirs = sorted(args.data.glob("sample_*"))
    if not sample_dirs:
        raise SystemExit(f"no sample_* dirs in {args.data}")
    print(f"{len(sample_dirs)} samples")

    data = {"handeye": {"images": [], "fk": []}, "front": {"images": [], "fk": []}}
    for sdir in sample_dirs:
        jf = sdir / "joints.json"
        if not jf.exists():
            continue
        jdata = json.loads(jf.read_text(encoding="utf-8"))
        fk_positions = {name: (0.0 if name == "gripper" else math.radians(v))
                        for name, v in jdata.items()}
        h = cv2.imread(str(sdir / "handeye.jpg"))
        f = cv2.imread(str(sdir / "front.jpg"))
        if h is None or f is None:
            continue
        data["handeye"]["images"].append(h)
        data["handeye"]["fk"].append(fk_positions)
        data["front"]["images"].append(f)
        data["front"]["fk"].append(fk_positions)

    objp = build_object_points(cols, rows, square)
    report: list[str] = []
    intrinsics: dict = {}
    extrinsics: dict = {}

    for cam in ("handeye", "front"):
        images = data[cam]["images"]
        fk_list = data[cam]["fk"]
        if len(images) < 4:
            print(f"[{cam}] skip: only {len(images)} images")
            continue

        # --- intrinsics ---
        obj_points, img_points = [], []
        usable = []
        for i, img in enumerate(images):
            corners = detect_board(img, cols, rows)
            if corners is None:
                continue
            obj_points.append(objp)
            img_points.append(corners)
            usable.append(i)
        if len(usable) < 4:
            print(f"[{cam}] too few usable for intrinsics: {len(usable)}")
            continue
        gray = cv2.cvtColor(images[usable[0]], cv2.COLOR_BGR2GRAY)
        hgt, wid = gray.shape
        ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (wid, hgt), None, None)
        err = 0.0
        for obj_pts, img_pts in zip(obj_points, img_points):
            _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
            err += float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
        err /= len(usable)
        intrinsics[cam] = {
            "usable_samples": [int(i) for i in usable],
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.reshape(-1).tolist(),
            "reprojection_error_px": round(err, 4),
        }
        print(f"[{cam}] intrinsics reproj error: {err:.4f} px")

        # --- extrinsics ---
        R_g2b, t_g2b = [], []
        R_t2c, t_t2c = [], []
        for i, img in enumerate(images):
            corners = detect_board(img, cols, rows)
            if corners is None:
                continue
            _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
            R_t2c.append(cv2.Rodrigues(rvec)[0])
            t_t2c.append(tvec.reshape(3))
            T = forward_kinematics(joints, fk_list[i], args.tip_frame)
            R_g2b.append(T[:3, :3])
            t_g2b.append(T[:3, 3])
        if len(R_t2c) < 4:
            print(f"[{cam}] too few poses for hand-eye: {len(R_t2c)}")
            continue

        if cam == "handeye":
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                        method=cv2.CALIB_HAND_EYE_TSAI)
            extrinsics[cam] = {
                "mode": "eye_in_hand",
                "translation_m": t.reshape(3).tolist(),
                "quaternion": matrix_to_quat(R),
                "T": np.eye(4).tolist(),
            }
            T4 = np.eye(4)
            T4[:3, :3] = R
            T4[:3, 3] = t.reshape(3)
            extrinsics[cam]["T"] = T4.tolist()
            print(f"[{cam}] camera->tip t={np.round(t.reshape(3), 4).tolist()} "
                  f"q={tuple(round(x, 4) for x in matrix_to_quat(R))}")
        else:
            # eye-to-hand: solve camera->base
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                        method=cv2.CALIB_HAND_EYE_TSAI)
            R_c2b = R.T
            t_c2b = -R_c2b @ t.reshape(3)
            T4 = np.eye(4)
            T4[:3, :3] = R_c2b
            T4[:3, 3] = t_c2b
            extrinsics[cam] = {
                "mode": "eye_to_hand",
                "translation_m": t_c2b.tolist(),
                "quaternion": matrix_to_quat(R_c2b),
                "T": T4.tolist(),
            }
            print(f"[{cam}] camera->base t={np.round(t_c2b, 4).tolist()} "
                  f"q={tuple(round(x, 4) for x in matrix_to_quat(R_c2b))}")

        report.append(f"{cam}: {len(usable)} usable, reproj={err:.3f} px")

    (out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    (out / "extrinsics.json").write_text(json.dumps(extrinsics, indent=2), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nDone. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
