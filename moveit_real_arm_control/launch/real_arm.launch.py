from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

# Real-arm MoveIt: planning runs on this PC (WSL), trajectory execution is
# performed by the Atlas board.  The board runs so101_command_bridge.py which
# exposes /so101_arm_controller/follow_joint_trajectory and
# /so101_gripper_controller/follow_joint_trajectory action servers; the
# moveit_controllers.yaml below already points at those names, so move_group
# will execute real trajectories through the board over domain 42 / Zenoh.
#
# Start the board side first:
#   systemctl start so101-rviz-bridge.service          # /joint_states source
#   /usr/bin/python3 /root/lerobot_project/so101_command_bridge.py
#   /opt/lerobot061/bin/python /root/lerobot_project/so101_command_executor.py --enable-control
#
# Then launch this file on the PC (with the VcXsrv display while WSLg is broken).


def generate_launch_description() -> LaunchDescription:
    moveit_config = (
        MoveItConfigsBuilder("so101", package_name="so101_moveit_config")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    package_share = Path(get_package_share_directory("so101_description"))
    moveit_share = Path(get_package_share_directory("so101_moveit_config"))
    rviz_config = str(moveit_share / "config" / "moveit.rviz")

    # Restamp board /joint_states with the PC clock so TF does not blink.
    retime_node = Node(
        package="so101_description",
        executable="joint_state_retime.py",
        name="so101_joint_state_retime",
        output="screen",
    )

    # Publish TF from the restamped joint states.
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="so101_robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
        remappings=[("/joint_states", "/joint_states_local")],
    )

    # MoveIt planning; the controller manager connects to the board action servers.
    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": True,
        "capabilities": moveit_config.move_group_capabilities["capabilities"],
        "disable_capabilities": moveit_config.move_group_capabilities["disable_capabilities"],
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "monitor_dynamics": False,
    }
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), move_group_configuration],
        remappings=[("/joint_states", "/joint_states_local")],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="so101_rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            retime_node,
            rsp_node,
            move_group_node,
            rviz_node,
        ]
    )
