#!/usr/bin/env python3
"""Convert local SO-ARM101 UDP telemetry into ROS 2 JointState messages."""

from __future__ import annotations

import json
import math
import socket
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
GRIPPER_LOWER = -0.174533
GRIPPER_UPPER = 1.74533


class SO101JointBridge(Node):
    def __init__(self) -> None:
        super().__init__("so101_joint_bridge")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind(("127.0.0.1", 15000))
        self.last_packet = time.monotonic()
        self.last_warning = 0.0
        self.create_timer(0.01, self.poll)
        self.get_logger().info("Listening for SO-ARM101 telemetry on udp://127.0.0.1:15000")

    def poll(self) -> None:
        latest: bytes | None = None
        while True:
            try:
                latest, _address = self.sock.recvfrom(4096)
            except BlockingIOError:
                break

        if latest is None:
            now = time.monotonic()
            if now - self.last_packet > 2.0 and now - self.last_warning > 5.0:
                self.get_logger().warning("Waiting for SO-ARM101 hardware telemetry")
                self.last_warning = now
            return

        try:
            payload = json.loads(latest.decode("utf-8"))
            values = [float(value) for value in payload["positions"]]
            if payload.get("names") != JOINT_NAMES or len(values) != 6 or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid joint payload")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Discarding malformed telemetry: {exc}")
            return

        gripper_percent = min(100.0, max(0.0, values[5]))
        positions = [math.radians(value) for value in values[:5]]
        positions.append(GRIPPER_LOWER + (GRIPPER_UPPER - GRIPPER_LOWER) * gripper_percent / 100.0)

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = JOINT_NAMES
        message.position = positions
        self.publisher.publish(message)
        self.last_packet = time.monotonic()

    def destroy_node(self) -> bool:
        self.sock.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SO101JointBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
