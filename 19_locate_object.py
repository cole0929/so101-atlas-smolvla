#!/opt/lerobot061/bin/python
"""Front-camera object localization demo (project: RL grasp foundation).

Detects a yellow object in the front overhead camera image, back-projects its
center pixel to a 3D point on the table plane (z=0 in base_link), and
publishes the result for RViz.

Pipeline:
  front camera pixel (u,v)
    -> undistort with calibrated intrinsics
    -> camera frame ray
    -> transform ray into base_link with calibrated extrinsic (camera->base)
    -> intersect ray with table plane z=0  ->  3D position (base_link)

Usage (board, front camera on workspace):

  cd /root/lerobot_project
  /opt/lerobot061/bin/python 19_locate_object.py --loop --show

Publishes:
  /located_object  (visualization_msgs/Marker, SPHERE at the object in base_link)
  /located_text    (Marker TEXT_VIEW_FACING, coordinate text)

The marker topics only exist while ROS2 (domain 42) is reachable; without ROS
the script still prints coordinates to the terminal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from atlas_runner import FRONT_CAMERA_PID, camera_device, environment

os.environ.update(environment())

CALIB = {
    "intrinsics": "/root/lerobot_project/front_intrinsics_v2.json",
    "extrinsic": "/root/lerobot_project/front_extrinsic_geometric.json",
}

# Yellow object HSV range (tune for your object).
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

TABLE_Z = 0.0  # table plane height in base_link frame (meters)


def load_calib() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intr = json.loads(Path(CALIB["intrinsics"]).read_text(encoding="utf-8"))
    mtx = np.array(intr["camera_matrix"], dtype=float)
    dist = np.array(intr["dist_coeffs"], dtype=float)

    ext = json.loads(Path(CALIB["extrinsic"]).read_text(encoding="utf-8"))
    T_cam2base = np.array(ext["T"], dtype=float)  # 4x4 camera -> base_link
    return mtx, dist, T_cam2base, intr.get("reprojection_error_px")


def find_yellow(frame: np.ndarray):
    """Return (center_px, contour_area, mask) of largest yellow blob or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200:  # ignore tiny blobs
        return None
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy), area, mask


def pixel_to_3d(u: float, v: float, mtx, dist, T_cam2base, table_z: float):
    """Back-project pixel ray, intersect with plane z=table_z in base_link."""
    # Undistort the pixel.
    points = np.array([[[u, v]]], dtype=np.float64)
    undist = cv2.undistortPoints(points, mtx, dist, P=mtx)
    uu, vv = undist[0, 0]
    # Ray direction in camera frame (z=1 convention).
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    dir_cam = np.array([(uu - cx) / fx, (vv - cy) / fy, 1.0])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    # Camera origin and ray in base_link.
    origin_cam = T_cam2base[:3, 3]
    rot = T_cam2base[:3, :3]
    dir_base = rot @ dir_cam

    # Ray: P = origin + t*dir ; intersect z = table_z.
    if abs(dir_base[2]) < 1e-6:
        return None  # ray parallel to table
    t = (table_z - origin_cam[2]) / dir_base[2]
    if t <= 0:
        return None  # intersection behind camera
    P = origin_cam + t * dir_base
    return P


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="continuous mode")
    parser.add_argument("--show", action="store_true", help="show debug window")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--table-z", type=float, default=TABLE_Z)
    parser.add_argument("--udp", action="store_true",
                        help="send located object JSON over UDP (for WSL RViz bridge)")
    parser.add_argument("--udp-host", default="192.168.0.1",
                        help="WSL host that runs located_object_bridge.py")
    parser.add_argument("--udp-port", type=int, default=15002)
    parser.add_argument("--file", default=None,
                        help="write latest object JSON to this file (board side)")
    args = parser.parse_args()

    udp_sock = None
    if args.udp:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP send enabled -> {args.udp_host}:{args.udp_port}")
    if args.file:
        print(f"file output enabled -> {args.file}")

    mtx, dist, T_cam2base, reproj = load_calib()
    print(f"intrinsics reproj: {reproj} px")
    print(f"camera->base translation: {[round(v*1000,1) for v in T_cam2base[:3,3]]} mm")

    # Optional ROS2 publisher (only when domain 42 reachable).
    marker_pub = text_pub = None
    try:
        import rclpy
        from visualization_msgs.msg import Marker

        rclpy.init()
        node = rclpy.create_node("object_locator")
        marker_pub = node.create_publisher(Marker, "/located_object", 10)
        text_pub = node.create_publisher(Marker, "/located_text", 10)
        print("ROS2 publisher ready (domain 42)")
    except Exception as exc:  # noqa: BLE001
        print(f"ROS2 unavailable ({exc}); terminal-only mode")

    dev = camera_device(FRONT_CAMERA_PID, "front")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def publish(P: np.ndarray, text: str) -> None:
        if marker_pub is None or not rclpy.ok():
            return
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = node.get_clock().now().to_msg()
        m.ns = "object"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.8, 0.0, 0.9
        m.pose.position.x, m.pose.position.y, m.pose.position.z = P[0], P[1], P[2]
        m.pose.orientation.w = 1.0
        marker_pub.publish(m)

        t = Marker()
        t.header.frame_id = "base_link"
        t.header.stamp = m.header.stamp
        t.ns = "object_text"
        t.id = 0
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.scale.z = 0.03
        t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
        t.pose.position.x, t.pose.position.y, t.pose.position.z = P[0], P[1], P[2] + 0.04
        t.pose.orientation.w = 1.0
        t.text = text
        text_pub.publish(t)

    print("Detecting yellow object on table plane z=0... (Ctrl+C to stop)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            result = find_yellow(frame)
            if result is None:
                print("no yellow object in view", flush=True)
                if args.show:
                    cv2.imshow("front", frame)
                    cv2.waitKey(1)
                time.sleep(args.interval)
                continue
            (cx, cy), area, mask = result
            P = pixel_to_3d(cx, cy, mtx, dist, T_cam2base, args.table_z)
            if P is None:
                print("intersection failed", flush=True)
            else:
                text = f"obj: ({P[0]*100:.1f}, {P[1]*100:.1f}, {P[2]*100:.1f}) cm"
                print(text + f"  area={area:.0f}px  px=({cx:.0f},{cy:.0f})", flush=True)
                payload = {
                    "x": float(P[0]), "y": float(P[1]), "z": float(P[2]),
                    "text": text, "time": time.time(),
                }
                if udp_sock is not None:
                    udp_sock.sendto(json.dumps(payload).encode("utf-8"),
                                    (args.udp_host, args.udp_port))
                if args.file:
                    Path(args.file).write_text(json.dumps(payload), encoding="utf-8")

            if args.show:
                vis = frame.copy()
                if result is not None:
                    cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.imshow("front", vis)
                cv2.waitKey(1)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if marker_pub is not None and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
