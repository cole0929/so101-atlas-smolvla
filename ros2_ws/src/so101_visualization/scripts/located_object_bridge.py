#!/usr/bin/env python3
"""WSL-side bridge: publish located-object markers for RViz.

Reads the latest located-object JSON written by the board's
19_locate_object.py (--file) and publishes RViz markers.

The board writes /root/located_object.json; a Windows-side loop scp-pulls it
to F:\\robot_arm_atlas\\.tmp\\located_object.json, which WSL sees at
/mnt/f/robot_arm_atlas/.tmp/located_object.json.

Run in WSL (domain 42):

  ros2 run so101_visualization located_object_bridge.py --file /mnt/f/robot_arm_atlas/.tmp/located_object.json

RViz: add MarkerArray display on /located_object.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class LocatedObjectBridge(Node):
    def __init__(self, file_path: str, publish_rate: float) -> None:
        super().__init__("located_object_bridge")
        self.pub = self.create_publisher(MarkerArray, "/located_object", 10)
        self.file_path = Path(file_path)
        self._last_mtime = None
        self.create_timer(1.0 / publish_rate, self.poll)
        self.get_logger().info(f"watching {self.file_path}")

    def poll(self) -> None:
        try:
            if not self.file_path.exists():
                return
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            x, y, z = float(payload["x"]), float(payload["y"]), float(payload["z"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"read failed: {exc}")
            return

        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        sphere = Marker()
        sphere.header.frame_id = "base_link"
        sphere.header.stamp = now
        sphere.ns = "object"
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.02
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.8, 0.0, 0.95
        sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, z
        sphere.pose.orientation.w = 1.0
        arr.markers.append(sphere)

        text = Marker()
        text.header.frame_id = "base_link"
        text.header.stamp = now
        text.ns = "object_text"
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.scale.z = 0.03
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
        text.pose.position.x, text.pose.position.y, text.pose.position.z = x, y, z + 0.04
        text.pose.orientation.w = 1.0
        text.text = payload.get("text", f"({x:.2f}, {y:.2f}, {z:.2f})")
        arr.markers.append(text)

        self.pub.publish(arr)


def main() -> None:
    argv = sys.argv[1:]
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/mnt/f/robot_arm_atlas/.tmp/located_object.json")
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args(argv)
    rclpy.init()
    node = LocatedObjectBridge(args.file, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
