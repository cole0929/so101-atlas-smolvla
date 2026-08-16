"""Geometric extrinsic calibration v2 - fixed board orientation.

Board is placed FLAT on the table (z=0) with its LONG edge (11 columns,
120mm) along the Y axis and SHORT edge (8 rows, 90mm) along X axis.
Board center measured at (0.293, 0, 0) in base_link.

We build the board 3D points with this exact orientation (yaw=90deg in the
mgrid convention), solve solvePnP once, and construct camera->base.

Usage:
  python front_geometric_calib_v2.py --image <jpg> --center 0.293 0 0 \
      --intrinsics <json> --board 11 8 0.010 --out <dir>
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


def build_board_points_long_along_y(cols: int, rows: int, square: float,
                                    center_xyz) -> np.ndarray:
    """Grid with cols (11) along +Y, rows (8) along +X, centered at center."""
    # local: index c in [0,cols) -> x=0, y=c*square ; r -> x=r*square, y=0
    pts = np.zeros((cols * rows, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            pts[i, 0] = r * square
            pts[i, 1] = c * square
    # center the grid
    pts[:, 0] -= (rows - 1) / 2.0 * square
    pts[:, 1] -= (cols - 1) / 2.0 * square
    pts[:, 0] += center_xyz[0]
    pts[:, 1] += center_xyz[1]
    pts[:, 2] = center_xyz[2]
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--center', required=True, nargs=3, type=float)
    parser.add_argument('--intrinsics', required=True, type=Path)
    parser.add_argument('--board', required=True, nargs=3, type=float)
    parser.add_argument('--out', type=Path, default=Path(r'F:\robot_arm_atlas\calib_out'))
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
    objp = build_board_points_long_along_y(cols, rows, square, center)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.reshape(3)

    # rvec/tvec = camera pose in board frame (OpenCV returns cam in object frame
    # for the standard solvePnP when using object points; verify both ways).
    T_board2cam = np.eye(4)
    T_board2cam[:3, :3] = R
    T_board2cam[:3, 3] = t

    # board->base: board frame origin (corner 0,0) in base + orientation
    # board X axis along +X, board Y along +Y (no rotation, since we built
    # points directly in base frame with long edge along Y).
    T_board2base = np.eye(4)
    T_board2base[:3, 3] = objp[0]  # corner (0,0) position in base

    # Two conventions:
    T_cam2base_A = T_board2base @ np.linalg.inv(T_board2cam)  # rvec = board in cam
    T_cam2base_B = T_board2cam @ T_board2base  # rvec = cam in board

    # reprojection
    proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))

    for name, T in (('A(inv)', T_cam2base_A), ('B(direct)', T_cam2base_B)):
        z_axis = T[:3, 2]
        cam_z = T[:3, 3][2]
        print(f'{name}: cam_t={np.round(T[:3,3],3)} z_axis={np.round(z_axis,3)} cam_z={cam_z:.3f}')
        print(f'   reproj={err:.3f}px  cam_above_table={cam_z > 0.05}')

    # Pick the physically valid one (camera above table).
    chosen = None
    for name, T in (('A', T_cam2base_A), ('B', T_cam2base_B)):
        if T[:3, 3][2] > 0.05:
            chosen = (name, T)
            break
    if chosen is None:
        raise SystemExit('no physically valid solution')
    name, T_final = chosen
    print(f'\nchosen: {name}')
    print('T_final:')
    print(np.round(T_final, 4))
    print('cam z axis:', np.round(T_final[:3, 2], 3), '(should be negative z for overhead)')
    print('cam t mm:', [round(v*1000, 1) for v in T_final[:3, 3]])

    result = {
        'method': 'geometric_v2_fixed_yaw90',
        'board_center_base_m': list(center),
        'convention': name,
        'reprojection_error_px': round(err, 4),
        'translation_m': T_final[:3, 3].tolist(),
        'quaternion': matrix_to_quat(T_final[:3, :3]),
        'T': T_final.tolist(),
    }
    (args.out / 'front_extrinsic_geometric_v2.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print('saved:', args.out / 'front_extrinsic_geometric_v2.json')


if __name__ == '__main__':
    main()
