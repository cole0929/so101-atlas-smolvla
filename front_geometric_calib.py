"""Geometric extrinsic calibration for the fixed front camera.

The checkerboard is placed flat on the table (z=0) at a measured position.
Known: board center in base_link, intrinsics (calibrated separately).
Unknown: camera->base transform (4x4), including board yaw (rotation of the
board edges around Z).

We solve solvePnP for several candidate yaw angles and keep the one with the
smallest reprojection error.

Usage:
  python front_geometric_calib.py --image <jpg> --center 0.293 0 0 \
      --intrinsics <front_intrinsics_v2.json> --board 11 8 0.010 --out <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r'F:\robot_arm_atlas')
from hand_eye_calibration import matrix_to_quat


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_board_points(cols: int, rows: int, square: float, center_xyz, yaw: float):
    """Object points in base_link: board grid centered at center_xyz, rotated by yaw."""
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    # Center the grid at origin first.
    cx = (cols - 1) / 2.0 * square
    cy = (rows - 1) / 2.0 * square
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    R = rot_z(yaw)
    R2 = R[:2, :2]
    pts[:, :2] = pts[:, :2] @ R2.T
    pts[:, 0] += center_xyz[0]
    pts[:, 1] += center_xyz[1]
    pts[:, 2] = center_xyz[2]
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--center', required=True, nargs=3, type=float,
                        help='board center x y z in base_link (m)')
    parser.add_argument('--intrinsics', required=True, type=Path)
    parser.add_argument('--board', required=True, nargs=3, type=float)
    parser.add_argument('--out', type=Path, default=Path(r'F:\robot_arm_atlas\calib_out'))
    parser.add_argument('--yaw-candidates', type=int, default=72,
                        help='sample yaw in 360/yaw_candidates degree steps')
    args = parser.parse_args()

    cols, rows, square = int(args.board[0]), int(args.board[1]), args.board[2]
    intr = json.loads(args.intrinsics.read_text(encoding='utf-8'))
    mtx = np.array(intr['camera_matrix'])
    dist = np.array(intr['dist_coeffs'])

    img = cv2.imread(str(args.image))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        raise SystemExit('checkerboard not detected')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners = np.ascontiguousarray(corners.reshape(-1, 1, 2), dtype=np.float32)

    center = tuple(args.center)
    best = None
    results = []
    for i in range(args.yaw_candidates):
        yaw = 2 * np.pi * i / args.yaw_candidates
        objp = build_board_points(cols, rows, square, center, yaw)
        try:
            ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        except cv2.error:
            continue
        if not ok:
            continue
        proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))
        R = cv2.Rodrigues(rvec)[0]
        results.append((err, yaw, R, tvec.reshape(3)))
        if best is None or err < best[0]:
            best = (err, yaw, R, tvec.reshape(3))

    if best is None:
        raise SystemExit('solvePnP failed for all yaws')
    err, yaw, R, t = best
    print(f'best yaw: {np.degrees(yaw):.1f} deg, reproj error: {err:.4f} px')
    print(f'(yaw is board rotation around Z relative to base X axis)')

    # Camera pose in base: solvePnP gives board->camera; camera->base = inv.
    T_cam2board = np.eye(4)
    T_cam2board[:3, :3] = R
    T_cam2board[:3, 3] = t
    T_board2cam = np.linalg.inv(T_cam2board)  # actually rvec/tvec is board in cam
    # rvec/tvec from solvePnP is object (board) pose in camera frame.
    # camera->base = board_in_cam^-1 * board_in_base(=grid known)
    T_board2base = np.eye(4)
    T_board2base[:3, :3] = rot_z(yaw)
    # board origin (corner (0,0)) in base:
    pts0 = build_board_points(cols, rows, square, center, yaw)
    origin_board_in_base = pts0[0]
    T_board2base[:3, 3] = origin_board_in_base
    T_cam2base = T_board2base @ np.linalg.inv(T_board2cam)

    result = {
        'method': 'geometric_single_shot',
        'board_center_base_m': list(center),
        'board_yaw_deg': round(np.degrees(yaw), 2),
        'reprojection_error_px': round(err, 4),
        'translation_m': T_cam2base[:3, 3].tolist(),
        'quaternion': matrix_to_quat(T_cam2base[:3, :3]),
        'T': T_cam2base.tolist(),
        'intrinsics_reproj_px': intr['reprojection_error_px'],
    }
    (args.out / 'front_extrinsic_geometric.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print('\nfront camera->base (geometric):')
    print('  t =', [round(v * 1000, 1) for v in T_cam2base[:3, 3]], 'mm')
    print('  q =', [round(x, 4) for x in matrix_to_quat(T_cam2base[:3, :3])])
    z_axis = T_cam2base[:3, 2]
    print('  camera Z axis in base:', np.round(z_axis, 3), '(should be negative z)')
    print('saved:', args.out / 'front_extrinsic_geometric.json')


if __name__ == '__main__':
    main()
