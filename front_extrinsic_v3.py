"""Recompute front camera (eye-to-hand) extrinsic CORRECTLY.

OpenCV calibrateHandEye always solves eye-in-hand form: given
  A_i = T_gripper^base(i)  (FK),  B_i = T_board^cam(i)  (solvePnP)
it returns X = T_cam^gripper satisfying  A_i X = X B_i.

For eye-to-hand data (board fixed on gripper, camera fixed in base) this
equation still holds with X = T_cam^gripper (constant).  The desired
camera->base is then  T_cam^base = T_gripper^base(0) @ X  (same for all i).

Usage:
  python front_extrinsic_v3.py --data <dir> --urdf <urdf> \
      --intrinsics <front_intrinsics_v2.json> --board 11 8 0.010
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r'F:\robot_arm_atlas')
from hand_eye_calibration import (
    build_object_points,
    detect_board,
    forward_kinematics,
    matrix_to_quat,
    parse_urdf_chain,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--urdf', required=True, type=Path)
    parser.add_argument('--intrinsics', required=True, type=Path)
    parser.add_argument('--board', required=True, nargs=3, type=float)
    parser.add_argument('--out', type=Path, default=Path(r'F:\robot_arm_atlas\calib_out'))
    args = parser.parse_args()

    cols, rows, square = int(args.board[0]), int(args.board[1]), args.board[2]
    intr = json.loads(args.intrinsics.read_text(encoding='utf-8'))
    mtx = np.array(intr['camera_matrix'])
    dist = np.array(intr['dist_coeffs'])
    print(f'intrinsics reproj={intr["reprojection_error_px"]} px')

    joints = parse_urdf_chain(args.urdf)
    objp = build_object_points(cols, rows, square)

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    usable = []
    for sdir in sorted(args.data.glob('sample_*')):
        jf, img = sdir / 'joints.json', sdir / 'front.jpg'
        if not jf.exists() or not img.exists():
            continue
        jdata = json.loads(jf.read_text(encoding='utf-8'))
        fk = {name: (0.0 if name == 'gripper' else math.radians(v))
              for name, v in jdata.items()}
        corners, _ = detect_board(cv2.imread(str(img)), cols, rows)
        if corners is None:
            print(f'{sdir.name}: board FAIL, skip')
            continue
        corners = np.ascontiguousarray(corners.reshape(-1, 1, 2), dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        R_t2c.append(cv2.Rodrigues(rvec)[0])
        t_t2c.append(tvec.reshape(3))
        T = forward_kinematics(joints, fk, 'gripper_frame_link')
        R_g2b.append(T[:3, :3])
        t_g2b.append(T[:3, 3])
        usable.append(sdir.name)

    print(f'usable poses: {len(usable)}')
    if len(usable) < 4:
        raise SystemExit('too few poses')

    # Step 1: eye-in-hand solve -> X = T_cam^gripper.
    R_cam2grip, t_cam2grip = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    X = np.eye(4)
    X[:3, :3] = R_cam2grip
    X[:3, 3] = t_cam2grip.reshape(3)

    # Step 2: camera->base = T_gripper^base(0) @ X ; check consistency across poses.
    T_g2b_0 = np.eye(4)
    T_g2b_0[:3, :3] = R_g2b[0]
    T_g2b_0[:3, 3] = t_g2b[0]
    T_cam2base = T_g2b_0 @ X

    # Consistency: recompute for every pose, they should all agree.
    positions = []
    for i in range(len(R_g2b)):
        T = np.eye(4)
        T[:3, :3] = R_g2b[i]
        T[:3, 3] = t_g2b[i]
        Tc = T @ X
        positions.append(Tc[:3, 3])
    positions = np.array(positions)
    spread = positions.std(axis=0)
    print('camera->base position spread (m):', np.round(spread, 4))
    print('  (small = consistent calibration)')

    result = {
        'mode': 'eye_to_hand',
        'translation_m': T_cam2base[:3, 3].tolist(),
        'quaternion': matrix_to_quat(T_cam2base[:3, :3]),
        'T': T_cam2base.tolist(),
        'intrinsics_reproj_px': intr['reprojection_error_px'],
        'usable_poses': usable,
        'position_spread_m': spread.tolist(),
        'notes': 'eye-to-hand via calibrateHandEye + base transform; v3',
    }
    (args.out / 'front_extrinsic_v3.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')

    print('\nfront camera->base (v3):')
    print('  t =', [round(v * 1000, 1) for v in T_cam2base[:3, 3]], 'mm')
    print('  q =', [round(x, 4) for x in matrix_to_quat(T_cam2base[:3, :3])])
    # Sanity: camera Z axis in base should point DOWN (negative z) for overhead cam.
    z_axis = T_cam2base[:3, 2]
    print('  camera Z axis in base:', np.round(z_axis, 3), '(should be negative z)')
    print('saved:', args.out / 'front_extrinsic_v3.json')


if __name__ == '__main__':
    main()
