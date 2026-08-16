"""Robust numerical IK for SO-ARM101 (position-only, damped).

SO-ARM101 has 5 arm DOF but a vertical grasp only needs the tip position
(3 constraints).  With 2 redundant DOF the Jacobian is rank-deficient, so we
use damped least squares (Levenberg-Marquardt) and add small joint-limit
pull-back terms.  After position IK, wrist_flex/wrist_roll are set so the
gripper points down.

The key fix vs the earlier attempt: NO orientation weight (it caused
oscillation), and clamping joints each step.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from hand_eye_calibration import forward_kinematics, parse_urdf_chain

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP = "gripper_frame_link"  # calibrated reference frame (hand-eye extrinsic anchor)

# URDF joint limits (radians) from so101.urdf
LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.9, 1.9),
    "wrist_roll": (-3.1, 3.1),
}


def solve_ik(joints, target_xyz, initial=None, max_iter=300, tol=1e-5, damping=1e-4):
    names = [j["name"] for j in joints if j["name"] in ARM_JOINTS and j["type"] in ("revolute", "continuous")]
    q = {n: 0.0 for n in names}
    if initial:
        q.update({n: initial[n] for n in names if n in initial})

    target = np.asarray(target_xyz, dtype=float)
    h = 1e-5

    for it in range(max_iter):
        T = forward_kinematics(joints, q, TIP)
        err = target - T[:3, 3]
        cost = np.linalg.norm(err)
        if cost < tol:
            break
        J = np.zeros((3, len(names)))
        for i, n in enumerate(names):
            qh, ql = dict(q), dict(q)
            qh[n] += h
            ql[n] -= h
            Th = forward_kinematics(joints, qh, TIP)
            Tl = forward_kinematics(joints, ql, TIP)
            J[:, i] = (Th[:3, 3] - Tl[:3, 3]) / (2 * h)
        # Damped least squares: dq = J^T (J J^T + lam I)^-1 e
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        # Step with line search to avoid overshoot
        alpha = 1.0
        best_q = dict(q)
        best_err = cost
        for _ in range(12):
            q_try = {n: q[n] + alpha * dq[i] for i, n in enumerate(names)}
            # clamp to limits
            for n in names:
                lo, hi = LIMITS[n]
                q_try[n] = max(lo, min(hi, q_try[n]))
            T_try = forward_kinematics(joints, q_try, TIP)
            e_try = np.linalg.norm(target - T_try[:3, 3])
            if e_try < best_err:
                best_err = e_try
                best_q = q_try
            alpha *= 0.5
        q = best_q

    T = forward_kinematics(joints, q, TIP)
    err = np.linalg.norm(T[:3, 3] - target)
    if err > 0.01:
        return None
    return q


def add_downward_wrist(joints, q, roll=0.0):
    """After position IK, set wrist_flex so gripper points down (-Z)."""
    # Keep shoulder_pan/lift/elbow; adjust wrist_flex so tip Z axis = -Z
    names = [j["name"] for j in joints if j["name"] in ARM_JOINTS]
    # current tip Z axis from the pitch chain: the wrist_flex rotates the
    # gripper; we solve wrist_flex such that z_axis = [0,0,-1].
    # Brute force: scan wrist_flex over range, pick best alignment.
    best = None
    for wf in np.linspace(LIMITS["wrist_flex"][0], LIMITS["wrist_flex"][1], 2000):
        q_t = dict(q)
        q_t["wrist_flex"] = wf
        q_t["wrist_roll"] = roll
        T = forward_kinematics(joints, q_t, TIP)
        z = T[:3, 2]
        align = np.dot(z, np.array([0.0, 0.0, -1.0]))
        if best is None or align > best[0]:
            best = (align, wf)
    q["wrist_flex"] = best[1]
    q["wrist_roll"] = roll
    return q


def solve_grasp_pose(urdf_path, target_xyz, roll=0.0, initial=None):
    joints = parse_urdf_chain(urdf_path)
    q = solve_ik(joints, target_xyz, initial)
    if q is None:
        return None, None
    q = add_downward_wrist(joints, q, roll)
    T = forward_kinematics(joints, q, TIP)
    return q, T


if __name__ == "__main__":
    urdf = Path(sys.argv[1] if len(sys.argv) > 1 else "so101.urdf")
    for tgt in [(0.25, 0.0, 0.20), (0.25, 0.10, 0.18), (0.20, -0.10, 0.15), (0.15, 0.0, 0.10)]:
        q, T = solve_grasp_pose(str(urdf), tgt)
        if q is None:
            print(f"{tgt}: FAIL")
        else:
            err = np.linalg.norm(T[:3, 3] - np.array(tgt))
            z = T[:3, 2]
            print(f"{tgt}: err={err*100:.1f}cm z_axis={np.round(z,2)} q={ {k: round(math.degrees(v),1) for k,v in q.items()} }")
