#!/usr/bin/env bash
# Read-only SO-ARM101 joint state bridge: ROS UDP receiver + serial telemetry.
# The serial handshake sometimes drops a motor on the first try; the source is
# retried a few times before giving up.
set -eo pipefail

source /root/ros2_humble_env.sh

# 1) ROS-side UDP -> /joint_states receiver (stays up as long as the service runs).
/usr/bin/python3 /root/lerobot_project/so101_joint_bridge.py \
  >>/tmp/so101_joint_bridge.log 2>&1 &
bridge_pid=$!

cleanup() {
  kill "$bridge_pid" 2>/dev/null || true
  wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 2) LeRobot-side serial telemetry with handshake retry.
cd /root/lerobot_project
for attempt in 1 2 3 4 5; do
  echo "[BRIDGE] joint_source attempt $attempt at $(date +%H:%M:%S)" >> /tmp/so101_joint_source.log
  /opt/lerobot061/bin/python so101_joint_source.py --fps 20 >> /tmp/so101_joint_source.log 2>&1
  code=$?
  echo "[BRIDGE] joint_source attempt $attempt exited code=$code" >> /tmp/so101_joint_source.log
  if [ "$code" -eq 0 ] || [ "$code" -eq 130 ] || [ "$code" -eq 143 ]; then
    exit 0
  fi
  sleep 2
done
echo "[BRIDGE] joint_source gave up after 5 attempts" >> /tmp/so101_joint_source.log
exit 1
