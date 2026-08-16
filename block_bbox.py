"""Detect the yellow block's bounding box in base_link via the front camera.

Returns the block's x_min/x_max/y_min/y_max on the table (z=0) by
back-projecting the four corners of the detected contour's bounding box.

Grasp strategy for a jaw that opens along X with the fixed (lower) jaw on the
-x side: the gripper center should be placed so that the lower jaw lands at
x < block_x_min.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from atlas_runner import FRONT_CAMERA_PID, camera_device

PROJECT = Path("/root/lerobot_project")
INTRINSICS = PROJECT / "front_intrinsics_v2.json"
EXTRINSIC = PROJECT / "front_extrinsic_geometric.json"
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])


def detect_block_bbox(mtx, dist, T_cam2base):
    """Return dict with block center and bbox corners in base, or None."""
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
    x, y, w, h = cv2.boundingRect(c)  # pixel bbox
    corners_px = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    M = cv2.moments(c)
    centroid_px = (M["m10"] / M["m00"], M["m01"] / M["m00"])  # contour centroid (more robust than bbox center)

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
        t = (0.0 - origin[2]) / dbase[2]
        return origin + t * dbase

    corners_base = [backproj(u, v) for (u, v) in corners_px]
    if any(p is None for p in corners_base):
        return None, frame
    xs = [p[0] for p in corners_base]
    ys = [p[1] for p in corners_base]
    M = cv2.moments(c)
    cx_px, cy_px = M["m10"] / M["m00"], M["m01"] / M["m00"]
    center = backproj(cx_px, cy_px)
    return {
        "center": center,
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "corners_base": corners_base,
    }, frame


if __name__ == "__main__":
    intr = json.loads(INTRINSICS.read_text(encoding="utf-8"))
    ext = json.loads(EXTRINSIC.read_text(encoding="utf-8"))
    mtx = np.array(intr["camera_matrix"], dtype=float)
    dist = np.array(intr["dist_coeffs"], dtype=float)
    T = np.array(ext["T"], dtype=float)
    result, frame = detect_block_bbox(mtx, dist, T)
    if result is None:
        print("no block detected")
    else:
        print(f"center: {np.round(result['center']*100, 1)} cm")
        print(f"x range: [{result['x_min']*100:.1f}, {result['x_max']*100:.1f}] cm  (len { (result['x_max']-result['x_min'])*100:.1f} cm)")
        print(f"y range: [{result['y_min']*100:.1f}, {result['y_max']*100:.1f}] cm")
