#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /mnt/f/robot_arm_atlas/ros2_ws/install/setup.bash

WINDOWS_HOST="$(ip route show default | awk '{print $3; exit}')"
export DISPLAY="${WINDOWS_HOST}:1.0"
export XAUTHORITY=/mnt/f/robot_arm_atlas/.x11/vcxsrv.Xauthority

export ROS_DOMAIN_ID=43
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset ZENOH_CONFIG_OVERRIDE

export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR="${QT_SCALE_FACTOR:-1.5}"
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1

# VcXsrv handles the X11 window while Mesa renders with the WSL GPU driver.
# Do not force indirect GL: VcXsrv exposes only OpenGL 1.4 in that mode, below
# RViz/Ogre's minimum. The direct context exposes OpenGL 4.x.
unset LIBGL_ALWAYS_INDIRECT
unset LIBGL_ALWAYS_SOFTWARE
unset GALLIUM_DRIVER
unset MESA_D3D12_DEFAULT_ADAPTER_NAME
unset OGRE_RTT_MODE

exec ros2 launch so101_moveit_config demo.launch.py
