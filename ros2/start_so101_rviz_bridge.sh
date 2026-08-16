#!/usr/bin/env bash
set -eo pipefail

source /root/ros2_humble_env.sh
set -u

/usr/bin/python3 /root/lerobot_project/so101_joint_bridge.py \
  >>/tmp/so101_joint_bridge.log 2>&1 &
bridge_pid=$!

cleanup() {
  kill "$bridge_pid" 2>/dev/null || true
  wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd /root/lerobot_project
/opt/lerobot061/bin/python /root/lerobot_project/so101_joint_source.py "$@"
