#!/usr/bin/env python3
"""Mock joint-state source for offline testing of the trajectory pipeline.

Publishes sinusoidal joint positions on /joint_states that approximate a
circular end-effector motion, so the recorder + TF + statistics pipeline can
be exercised without hardware.

Usage (WSL):

  ros2 run so101_visualization mock_joint_source.py --rate 30 --circles 3

Then in another shell:

  ros2 launch so101_visualization trajectory_viz.launch.py

The mock node publishes on /joint_states; the retime node re-stamps it to
/joint_states_local, robot_state_publisher turns it into TF and the recorder
produces /ee_path, /ee_trail and /ee_stats.
"""

from __future__ import annotations

import argparse
import math

import rclpy
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


class MockJointSource(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("mock_joint_source")
        self.rate = args.rate
        self.circles = args.circles
        self.period = args.period
        self.amplitude = args.amplitude
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.timer = self.create_timer(1.0 / self.rate, self.tick)
        self.t = 0.0

    def tick(self) -> None:
        self.t += 1.0 / self.rate
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        # Base pose: shoulder_lift slightly forward, elbow flexed.
        base = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.5,
            "elbow_flex": -1.0,
            "wrist_flex": 0.5,
            "wrist_roll": 0.0,
            "gripper": 0.05,
        }
        # Drive shoulder_pan + shoulder_lift with a phase offset to trace an
        # ellipse in the horizontal plane (approximate circle).
        w = 2.0 * math.pi / self.period
        n = int(self.t / self.period)  # current circle number
        if n >= self.circles:
            # Hold still after the requested circles.
            phase = 2.0 * math.pi
        else:
            phase = w * self.t
        positions = []
        for name in JOINT_NAMES:
            if name == "shoulder_pan":
                positions.append(base[name] + self.amplitude * math.sin(phase))
            elif name == "shoulder_lift":
                positions.append(base[name] + self.amplitude * 0.6 * math.sin(phase + math.pi / 2.0))
            else:
                positions.append(base[name])
        msg.position = positions
        msg.velocity = [0.0] * len(positions)
        msg.effort = [0.0] * len(positions)
        self.publisher.publish(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--circles", type=int, default=3, help="number of circles to trace")
    parser.add_argument("--period", type=float, default=4.0, help="seconds per circle")
    parser.add_argument("--amplitude", type=float, default=0.15, help="radians joint amplitude")
    return parser.parse_args()


def main() -> None:
    rclpy.init()
    node = MockJointSource(parse_args())
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
