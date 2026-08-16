#!/usr/bin/env bash
# Source this file in a dedicated ROS 2 shell on the Atlas board.
# Keep it separate from the CANN/SmolVLA environment to avoid library conflicts.

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=10
