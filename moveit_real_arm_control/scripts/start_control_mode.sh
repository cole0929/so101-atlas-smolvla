#!/usr/bin/env bash
# ONE-CLICK: enter MoveIt control mode on the Atlas board.
# Usage: bash /root/lerobot_project/start_control_mode.sh
set -e

echo "[1/5] Stopping read-only bridge (frees the follower serial port)..."
systemctl stop so101-rviz-bridge.service 2>/dev/null || true
pkill -9 -f so101_command_executor.py 2>/dev/null || true
pkill -9 -f so101_joint_bridge.py 2>/dev/null || true
sleep 2

echo "[2/5] Starting joint state forwarder (UDP 15000 -> /joint_states)..."
setsid /root/lerobot_project/start_so101_joint_bridge.sh > /tmp/so101_joint_bridge.log 2>&1 < /dev/null &
sleep 3

echo "[3/5] Starting command executor (writes follower serial port)..."
setsid /opt/lerobot061/bin/python /root/lerobot_project/so101_command_executor.py > /tmp/executor_real.log 2>&1 < /dev/null &
sleep 6
if ! pgrep -f so101_command_executor.py > /dev/null; then
  echo "[ERROR] executor failed to start; last log:"; tail -5 /tmp/executor_real.log; exit 1
fi
tail -2 /tmp/executor_real.log

echo "[4/5] Enabling motion gate (required for movement)..."
python3 -c "import socket,json;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.sendto(json.dumps({\"cmd\":\"enable\",\"on\":True}).encode(),(\"127.0.0.1\",15001))"
sleep 1
grep "motion gate" /tmp/executor_real.log | tail -1

echo "[5/5] Verifying /joint_states..."
source /root/ros2_humble_env.sh
timeout 5 ros2 topic echo /joint_states --once 2>/dev/null | grep -A2 "position:" | head -4 || echo "(echo timeout, check bridge log)"

echo ""
echo "=== DONE: board is in MoveIt control mode. ==="
echo "Now on the PC run:"
echo "  powershell -ExecutionPolicy Bypass -File F:\\robot_arm_atlas\\ros2\\start_so101_real_arm.ps1"
