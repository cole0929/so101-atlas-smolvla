#!/usr/bin/env bash
set -e
source /root/ros2_humble_env.sh
exec /usr/bin/python3 /root/lerobot_project/so101_command_bridge.py
