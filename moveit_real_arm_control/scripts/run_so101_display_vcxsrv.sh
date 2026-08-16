#!/usr/bin/env bash
set -eo pipefail

# SO-ARM101 display.launch.py on VcXsrv instead of WSLg.
# WSLg's GPU-PV path is currently broken on this PC (dxg ioctl failures make
# every WSLg window render black in COPY MODE), so reuse the proven VcXsrv
# display that the MoveIt demo already uses.

source /opt/ros/humble/setup.bash
source /mnt/f/robot_arm_atlas/ros2_ws/install/setup.bash

WINDOWS_HOST="$(ip route show default | awk '{print $3; exit}')"
export DISPLAY="${WINDOWS_HOST}:1.0"
export XAUTHORITY=/mnt/f/robot_arm_atlas/.x11/vcxsrv.Xauthority

# Same ROS network profile as ~/ros2_humble_env.sh (domain 42, Zenoh client
# to the Atlas board), so /joint_states still arrives from the board.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=10
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/192.168.0.2:7447"]'

# VcXsrv display settings (same as the MoveIt VcXsrv script).
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR="${QT_SCALE_FACTOR:-1.5}"
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1

# VcXsrv exposes only OpenGL 1.4 with indirect GL; keep the direct context so
# Mesa renders with the WSL GPU driver at OpenGL 4.x.
unset LIBGL_ALWAYS_INDIRECT
unset LIBGL_ALWAYS_SOFTWARE
unset GALLIUM_DRIVER
unset MESA_D3D12_DEFAULT_ADAPTER_NAME
unset OGRE_RTT_MODE

exec ros2 launch so101_description display.launch.py
