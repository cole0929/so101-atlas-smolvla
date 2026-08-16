#!/usr/bin/env python3
"""Restamp remote JointState messages with the visualization host clock."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateRetime(Node):
    def __init__(self) -> None:
        super().__init__("so101_joint_state_retime")
        self.publisher = self.create_publisher(JointState, "/joint_states_local", 10)
        self.subscription = self.create_subscription(JointState, "/joint_states", self.callback, 10)

    def callback(self, incoming: JointState) -> None:
        outgoing = JointState()
        outgoing.header = incoming.header
        outgoing.header.stamp = self.get_clock().now().to_msg()
        outgoing.name = incoming.name
        outgoing.position = incoming.position
        outgoing.velocity = incoming.velocity
        outgoing.effort = incoming.effort
        self.publisher.publish(outgoing)


def main() -> None:
    rclpy.init()
    node = JointStateRetime()
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

