"""Recompute front camera (eye-to-hand) extrinsic using the NEW intrinsics.

Data condition is now correct: checkerboard mounted on the wrist moves with
the arm while the front camera stays fixed.

Usage:
  python front_extrinsic_v2.py --data <dir> --urdf <so101.urdf> \
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
    print(f'using intrinsics reproj={intr["reprojection_error_px"]} px')

    joints = parse_urdf_chain(args.urdf)
    objp = build_object_points(cols, rows, square)

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    usable = []
    for sdir in sorted(args.data.glob('sample_*')):
        jf = sdir / 'joints.json'
        img = sdir / 'front.jpg'
        if not jf.exists() or not img.exists():
            continue
        jdata = json.loads(jf.read_text(encoding='utf-8'))
        fk = {name: (0.0 if name == 'gripper' else math.radians(v))
              for name, v in jdata.items()}
        frame = cv2.imread(str(img))
        corners, _ = detect_board(frame, cols, rows)
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

    # cv2.calibrateHandEye in eye-to-hand mode solves camera->base (AX=ZB).
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                method=cv2.CALIB_HAND_EYE_TSAI)
    # Invert to camera->base.
    R_c2b = R.T
    t_c2b = -R_c2b @ t.reshape(3)
    T4 = np.eye(4)
    T4[:3, :3] = R_c2b
    T4[:3, 3] = t_c2b

    result = {
        'mode': 'eye_to_hand',
        'translation_m': t_c2b.tolist(),
        'quaternion': matrix_to_quat(R_c2b),
        'T': T4.tolist(),
        'intrinsics_reproj_px': intr['reprojection_error_px'],
        'usable_poses': usable,
        'notes': 'checkerboard on wrist, camera fixed; new intrinsics v2',
    }
    (args.out / 'front_extrinsic_v2.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(f'\nfront camera->base:')
    print(f'  t = {t_c2b.tolist()} m  ({[round(v*1000,1) for v in t_c2b]} mm)')
    print(f'  q = {[round(x,4) for x in matrix_to_quat(R_c2b)]}')
    print(f'saved: {args.out / "front_extrinsic_v2.json"}')


if __name__ == '__main__':
    main()
