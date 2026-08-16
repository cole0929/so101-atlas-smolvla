#!/usr/bin/env python3
"""SO-ARM101 command bridge: MoveIt trajectory action server -> UDP downlink.

This node runs in the board's ROS 2 environment (/usr/bin/python3 + rclpy).
It exposes two FollowJointTrajectory action servers that match the MoveIt
controller names in moveit_controllers.yaml:

    /so101_arm_controller/follow_joint_trajectory     (5 arm joints)
    /so101_gripper_controller/follow_joint_trajectory (gripper)

Every accepted trajectory is converted from ROS radians (arm) / radians
(gripper) back into LeRobot degrees (arm) / percent (gripper) and streamed to
the LeRobot-side executor (so101_command_executor.py) over UDP 15001.  The
executor reports completion/errors back on UDP 15002.

The node also subscribes to /joint_states (published by so101_joint_bridge)
and publishes trajectory feedback so the MoveIt MotionPlanning panel shows
real progress while the arm moves.

Units: MoveIt trajectories are radians for all six joints.  The gripper joint
in the URDF maps 0..100 percent onto [-0.174533, 1.74533] rad, mirroring the
read-only bridge.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]

GRIPPER_LOWER = -0.174533
GRIPPER_UPPER = 1.74533

COMMAND_PORT = 15001
REPORT_PORT = 15002
REPORT_HOST = "127.0.0.1"


def radians_to_lerobot(joint_name: str, value: float) -> float:
    """ROS radians -> LeRobot units: degrees for arm joints, percent for gripper."""
    if joint_name == GRIPPER_JOINT:
        percent = (value - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 100.0
        return min(100.0, max(0.0, percent))
    return math.degrees(value)


class SO101CommandBridge(Node):
    def __init__(self) -> None:
        super().__init__("so101_command_bridge")

        self._seq_lock = threading.Lock()
        self._seq = 0
        self._pending: dict[int, float] = {}  # seq -> deadline for completion check
        self._reports: dict[int, str] = {}  # seq -> error reason from executor
        self._latest_state: JointState | None = None

        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.report_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.report_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.report_sock.bind((REPORT_HOST, REPORT_PORT))
        self.report_sock.setblocking(False)

        self.joint_states_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10
        )

        arm_cb = MutuallyExclusiveCallbackGroup()
        gripper_cb = MutuallyExclusiveCallbackGroup()

        self._arm_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/so101_arm_controller/follow_joint_trajectory",
            self._arm_execute,
            callback_group=arm_cb,
        )
        self._gripper_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/so101_gripper_controller/follow_joint_trajectory",
            self._gripper_execute,
            callback_group=gripper_cb,
        )

        self._report_timer = self.create_timer(0.05, self._drain_reports)
        self.get_logger().info(
            "SO101 command bridge ready: arm + gripper FollowJointTrajectory servers -> UDP %d" % COMMAND_PORT
        )

    # ------------------------------------------------------------------ state
    def _on_joint_states(self, msg: JointState) -> None:
        self._latest_state = msg

    def _state_radians(self, joints: list[str]) -> dict[str, float] | None:
        if self._latest_state is None:
            return None
        names = list(self._latest_state.name)
        positions = list(self._latest_state.position)
        try:
            return {j: float(positions[names.index(j)]) for j in joints}
        except ValueError:
            return None

    # ------------------------------------------------------------- trajectory
    def _common_execute(self, joints: list[str], goal_handle):
        trajectory = goal_handle.request.trajectory
        points: list[JointTrajectoryPoint] = list(trajectory.points)
        if not points:
            goal_handle.abort()
            self.get_logger().error("Empty trajectory rejected")
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result

        point_positions = [list(p.positions) for p in points]
        point_times = [float(p.time_from_start.sec) + float(p.time_from_start.nanosec) * 1e-9 for p in points]
        # ROS trajectory order follows the controller joint list; map by name.
        traj_names = list(trajectory.joint_names)
        seq = self._next_seq()

        self.get_logger().info(
            "Received trajectory seq=%d joints=%s points=%d duration=%.2fs"
            % (seq, joints, len(points), point_times[-1])
        )

        # Reject trajectories that start far from the current state.
        current = self._state_radians(joints)
        if current is None:
            # No joint state yet: safer to reject until real state is available.
            goal_handle.abort()
            self.get_logger().error("No current joint state; trajectory rejected")
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result
        if point_positions:
            first = point_positions[0]
            if len(first) == len(joints):
                for j, name in enumerate(joints):
                    if name in traj_names:
                        idx = traj_names.index(name)
                        delta = abs(float(first[idx]) - current[name])
                        if delta > math.radians(45.0):
                            goal_handle.abort()
                            self.get_logger().error(
                                "Trajectory starts %.1f deg from current pose on %s; rejected"
                                % (math.degrees(delta), name)
                            )
                            result = FollowJointTrajectory.Result()
                            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                            return result

        # Convert every point into LeRobot units, keeping only the joints that
        # this trajectory owns.  The executor merges partial updates into its
        # full 6-DOF target state (arm and gripper trajectories run in parallel).
        converted = []
        for p_idx, pos in enumerate(point_positions):
            names_in_point = []
            values_in_point = []
            for name in joints:
                if name in traj_names:
                    raw = float(pos[traj_names.index(name)])
                    names_in_point.append(name)
                    values_in_point.append(radians_to_lerobot(name, raw))
            converted.append({"t": point_times[p_idx], "names": names_in_point, "positions": values_in_point})

        start_delay = 0.5
        t_start = time.time() + start_delay
        payload = {"cmd": "trajectory", "seq": seq, "t_start": t_start, "points": converted}
        self.command_sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), (REPORT_HOST, COMMAND_PORT))
        self.get_logger().info("Sent trajectory seq=%d to executor (UDP %d)" % (seq, COMMAND_PORT))

        with self._seq_lock:
            self._pending[seq] = time.time() + point_times[-1] + start_delay + 5.0

        deadline = time.time() + point_times[-1] + start_delay + 10.0
        while time.time() < deadline:
            if not rclpy.ok():
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                return result
            # Abort if the executor reported an error for this trajectory.
            with self._seq_lock:
                failed = self._reports.pop(seq, None)
            if failed is not None:
                self.get_logger().error("Executor failed trajectory seq=%d: %s" % (seq, failed))
                goal_handle.abort()
                with self._seq_lock:
                    self._pending.pop(seq, None)
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = str(failed)
                return result
            # Publish feedback from the latest real joint state.
            feedback = FollowJointTrajectory.Feedback()
            current = self._state_radians(joints)
            if current is not None and points:
                last = points[-1]
                feedback.joint_names = list(trajectory.joint_names)
                feedback.actual = last
                feedback.desired = last
                goal_handle.publish_feedback(feedback)
            time.sleep(0.05)

        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        with self._seq_lock:
            self._pending.pop(seq, None)
        self.get_logger().info("Trajectory seq=%d completed" % seq)
        return result

    def _arm_execute(self, goal_handle):
        self._common_execute(ARM_JOINTS, goal_handle)

    def _gripper_execute(self, goal_handle):
        self._common_execute([GRIPPER_JOINT], goal_handle)

    # ------------------------------------------------------------- reporting
    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _drain_reports(self) -> None:
        while True:
            try:
                raw, _addr = self.report_sock.recvfrom(8192)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                report = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            seq = report.get("seq")
            status = report.get("status")
            reason = report.get("reason", "")
            self.get_logger().info("Executor report seq=%s status=%s reason=%s" % (seq, status, reason))
            if status == "error" and seq is not None:
                with self._seq_lock:
                    self._reports[int(seq)] = str(reason)

    def destroy_node(self) -> bool:
        self.command_sock.close()
        self.report_sock.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SO101CommandBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
