#!/usr/bin/env bash
set -e

mapfile -t launch_pids < <(
  pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch so101_moveit_config demo.launch.py$' || true
)

if ((${#launch_pids[@]} == 0)); then
  echo "SO-ARM101 MoveIt demo is not running."
  exit 0
fi

for launch_pid in "${launch_pids[@]}"; do
  process_group="$(ps -o pgid= -p "$launch_pid" | tr -d ' ')"
  if [[ -n "$process_group" ]]; then
    echo "Stopping MoveIt demo process group $process_group"
    kill -INT -- "-$process_group"
  fi
done
