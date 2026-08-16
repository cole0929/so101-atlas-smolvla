"""Launch end-effector trajectory visualization for SO-ARM101.

Brings up the joint-state retime node, robot_state_publisher, the trajectory
recorder/statistics node and RViz2 with a dedicated trajectory view.

Usage (inside WSL ROS2, board read-only bridge running for live data):

  ros2 launch so101_visualization trajectory_viz.launch.py

Options:

  with_rviz   (default true) start RViz2
  with_rsp    (default true) start robot_state_publisher (disable when another
                             launch already publishes /tf)
  with_retime (default true) start the joint-state retime node (disable when a
                             display/real-arm launch already provides
                             /joint_states_local)
  target_frame (default gripper_frame_link)  tracked end-effector frame
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    desc_share = Path(get_package_share_directory("so101_description"))
    viz_share = Path(get_package_share_directory("so101_visualization"))
    robot_description = (desc_share / "urdf" / "so101.urdf").read_text(encoding="utf-8")
    rviz_config = str(viz_share / "rviz" / "trajectory.rviz")

    retime_arg = LaunchConfiguration("with_retime")
    rsp_arg = LaunchConfiguration("with_rsp")
    rviz_arg = LaunchConfiguration("with_rviz")
    target_frame = LaunchConfiguration("target_frame")

    return LaunchDescription(
        [
            DeclareLaunchArgument("with_rviz", default_value="true"),
            DeclareLaunchArgument("with_rsp", default_value="true"),
            DeclareLaunchArgument("with_retime", default_value="true"),
            DeclareLaunchArgument("target_frame", default_value="gripper_frame_link"),
            Node(
                package="so101_description",
                executable="joint_state_retime.py",
                name="so101_joint_state_retime",
                output="screen",
                condition=IfCondition(retime_arg),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="so101_robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[("/joint_states", "/joint_states_local")],
                condition=IfCondition(rsp_arg),
            ),
            Node(
                package="so101_visualization",
                executable="ee_trajectory_recorder.py",
                name="ee_trajectory_recorder",
                output="screen",
                arguments=["--target-frame", target_frame],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="so101_rviz2_trajectory",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(rviz_arg),
            ),
        ]
    )
