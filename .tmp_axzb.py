"""Independent eye-to-hand (camera->base) solver: AX = ZB.

Given N poses with:
  A_i = T_base^gripper(i)   (from URDF FK, gripper pose in base)
  B_i = T_cam^board(i)      (from solvePnP, board pose in camera)

we solve for X = T_cam^base (camera fixed in base frame) such that
  A_i @ B_i_inv @ X ... derive: board is fixed on the gripper, so
  T_base^board = A_i @ T_grip^board  (constant)
  T_cam^board  = B_i
  T_base^cam   = X        =>  T_base^board = X @ B_i
  => A_i @ T_grip^board = X @ B_i   for all i
  => X = A_i @ T_grip^board @ B_i^-1  (should be the same X for all i)

Two-step estimation:
  1. rotation: from A_i @ K = X @ B_i  with unknown K = T_grip^board.
     Eliminate K: X = A_i K B_i^-1 ;  use pairs -> X = A_i K B_i^-1 = A_j K B_j^-1
     => B_j B_i^-1 X ... standard hand-eye AX=XB form after re-arrangement:
        (A_j^-1 A_i) X = X (B_j^-1 B_i)   [eye-to-hand Tsai form]
     Solved with a linear system on rotation.
  2. translation: least squares over all i.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/root/lerobot_project")
from hand_eye_calibration import (  # noqa: E402
    forward_kinematics,
    matrix_to_quat,
    parse_urdf_chain,
)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def ax_xb_rotation(RA: list[np.ndarray], RB: list[np.ndarray]) -> np.ndarray:
    """Solve R_X from R_A R_X = R_X R_B using Tsai's linear method."""
    A = []
    b = []
    n = len(RA)
    for i in range(n):
        for j in range(i + 1, n):
            # RA_ij = R_Aj * R_Ai^T ; RB_ij = R_Bj * R_Bi^T
            RA_ij = RA[j] @ RA[i].T
            RB_ij = RB[j] @ RB[i].T
            # (RA_ij - I) r = skew(RB_ij) ... build 9x9 system for r (rodrigues of RX)
            # Tsai: P * r = 0  where P = [[RA_ij - I], [skew(RA_ij + I)]] stacked
            P = np.vstack([RA_ij - np.eye(3), skew(RA_ij + np.eye(3))])
            Q = np.hstack([np.zeros((6, 3)), np.zeros((6, 3))])  # placeholder
            # Use the classic: P r = 0 with r = axis*angle (Rodrigues vector)
            # We use the formulation: (RA_ij + I) r = 2*rod(RA_ij) ... skip, use SVD directly
            # Simpler robust route: use cv2 calibration per-pair then average? No.
            # Instead solve via quaternion (Horn): build 4x4 for each pair.
            # quaternion q satisfies R_A q = q R_B  =>  (Q_A - Q_B) q = 0
            pass
    # Fallback: use cv2.calibrateHandEye on eye-in-hand form by symmetry:
    # eye-to-hand AX=XB with A=gripper2base,B=board2cam is SAME math as
    # eye-in-hand, output is camera->gripper... not directly usable.
    raise NotImplementedError


def main() -> None:
    print("use the dedicated solver script instead")
