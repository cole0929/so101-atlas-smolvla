#!/usr/bin/env bash
# Source this file in Ubuntu 22.04 under WSL2.

source /opt/ros/humble/setup.bash
if [[ -f /mnt/f/robot_arm_atlas/ros2_ws/install/setup.bash ]]; then
    source /mnt/f/robot_arm_atlas/ros2_ws/install/setup.bash
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=10
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/192.168.0.2:7447"]'

# WSLg does not always inherit Windows high-DPI scaling for Qt 5 applications.
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR="${QT_SCALE_FACTOR:-2}"

# RViz2 can flicker through WSLg's D3D12 OpenGL path on some Windows GPUs.
# The SO-ARM101 scene is light enough for Mesa's stable software renderer.
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
