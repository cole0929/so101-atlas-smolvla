#!/usr/bin/env python3
"""SO-ARM101 digital-twin overlay layers (project 1, software part).

Publishes static and live visualization layers on top of the robot model in
RViz:

  /dt_boundary (Marker)         workspace boundary (semi-transparent box)
  /dt_forbidden (MarkerArray)   forbidden / collision zones (red volumes)
  /dt_joint_labels (MarkerArray) live joint-angle text labels (TEXT_VIEW_FACING)
  /dt_stats (std_msgs/String)   JSON snapshot of current joint angles

Static geometry is configured via YAML parameters passed on the ROS parameter
server (see config/dt_layers.yaml).  Run inside the WSL ROS2 environment with
robot_state_publisher up:

  ros2 run so101_visualization digital_twin_layers.py \
    --ros-args --params-file <pkg>/config/dt_layers.yaml
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _parse_volume_param(raw) -> list:
    """Decode a JSON-string volume list; tolerate already-decoded structures."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return raw
    return []


def _marker_base(ns: str, marker_id: int, frame_id: str, mtype: int) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


class DigitalTwinLayers(Node):
    def __init__(self) -> None:
        super().__init__("digital_twin_layers")

        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("boundary", "")  # JSON string: [{x,y,z,size_x,size_y,size_z,r,g,b,a,label}]
        self.declare_parameter("forbidden", "")  # JSON string, same schema
        self.declare_parameter("joint_angle_label", True)
        self.declare_parameter("joints", DEFAULT_JOINTS)

        self.frame_id = self.get_parameter("frame_id").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.boundary = _parse_volume_param(self.get_parameter("boundary").value)
        self.forbidden = _parse_volume_param(self.get_parameter("forbidden").value)
        self.joint_angle_label = self.get_parameter("joint_angle_label").value
        self.joints = self.get_parameter("joints").value or DEFAULT_JOINTS

        self.boundary_pub = self.create_publisher(Marker, "/dt_boundary", 10)
        self.forbidden_pub = self.create_publisher(MarkerArray, "/dt_forbidden", 10)
        self.label_pub = self.create_publisher(MarkerArray, "/dt_joint_labels", 10)
        self.stats_pub = self.create_publisher(String, "/dt_stats", 10)

        self.latest_positions: dict[str, float] = {name: 0.0 for name in self.joints}
        self.sub = self.create_subscription(JointState, "/joint_states_local", self.joint_state_cb, 10)

        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_loop)

    # ------------------------------------------------------------------ #
    def joint_state_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_positions:
                self.latest_positions[name] = pos

    # ------------------------------------------------------------------ #
    def publish_loop(self) -> None:
        now = self.get_clock().now().to_msg()

        # Boundary marker.
        if self.boundary:
            cfg = self.boundary[0]  # single boundary volume for now
            m = _marker_base("boundary", 0, self.frame_id, Marker.CUBE)
            m.header.stamp = now
            m.pose.position.x = cfg.get("x", 0.0)
            m.pose.position.y = cfg.get("y", 0.0)
            m.pose.position.z = cfg.get("z", 0.0)
            m.scale.x = cfg.get("size_x", 0.5)
            m.scale.y = cfg.get("size_y", 0.5)
            m.scale.z = cfg.get("size_z", 0.5)
            m.color.r = cfg.get("r", 0.2)
            m.color.g = cfg.get("g", 0.8)
            m.color.b = cfg.get("b", 0.3)
            m.color.a = cfg.get("a", 0.12)
            if cfg.get("label"):
                m.text = cfg["label"]
            self.boundary_pub.publish(m)

        # Forbidden zones.
        if self.forbidden:
            arr = MarkerArray()
            for i, cfg in enumerate(self.forbidden):
                m = _marker_base("forbidden", i, self.frame_id, Marker.CUBE)
                m.header.stamp = now
                m.pose.position.x = cfg.get("x", 0.0)
                m.pose.position.y = cfg.get("y", 0.0)
                m.pose.position.z = cfg.get("z", 0.0)
                m.scale.x = cfg.get("size_x", 0.2)
                m.scale.y = cfg.get("size_y", 0.2)
                m.scale.z = cfg.get("size_z", 0.2)
                m.color.r = cfg.get("r", 1.0)
                m.color.g = cfg.get("g", 0.1)
                m.color.b = cfg.get("b", 0.1)
                m.color.a = cfg.get("a", 0.35)
                arr.markers.append(m)
            self.forbidden_pub.publish(arr)

        # Joint angle labels.
        if self.joint_angle_label:
            arr = MarkerArray()
            for i, name in enumerate(self.joints):
                deg = self.latest_positions.get(name, 0.0) * 180.0 / 3.141592653589793
                m = _marker_base("joint_labels", i, self.frame_id, Marker.TEXT_VIEW_FACING)
                m.header.stamp = now
                # Place the label at the joint's own frame origin; the marker is
                # attached to the frame so it follows the link.  Offset slightly
                # so text is readable.
                m.pose.position.z = 0.02
                m.scale.z = 0.02
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.4, 1.0
                m.text = f"{name}: {deg:+.1f} deg"
                arr.markers.append(m)
            self.label_pub.publish(arr)

        # Stats JSON.
        stats = {name: round(self.latest_positions.get(name, 0.0), 6) for name in self.joints}
        msg = String()
        msg.data = json.dumps(stats, ensure_ascii=False)
        self.stats_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = DigitalTwinLayers()
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
