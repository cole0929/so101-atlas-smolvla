from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("so101_description"))
    robot_description = (package_share / "urdf" / "so101.urdf").read_text(encoding="utf-8")
    rviz_config = str(package_share / "rviz" / "so101.rviz")

    return LaunchDescription(
        [
            Node(
                package="so101_description",
                executable="joint_state_retime.py",
                name="so101_joint_state_retime",
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="so101_robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[("/joint_states", "/joint_states_local")],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="so101_rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
