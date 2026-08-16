# Atlas ROS 2 deployment

完整的中文部署、启动、停止和故障排查文档见：

```text
F:\robot_arm_atlas\docs\Atlas_ROS2_SO101部署与使用技术文档_2026-08-15.md
```

## Board installation

- Board: Atlas 200I DK A2 / Ascend 310B1
- OS: Ubuntu 22.04 arm64
- ROS distribution: ROS 2 Humble
- Installed variant: `ros-humble-ros-base`
- Network middleware: Zenoh (`rmw_zenoh_cpp`)
- Default project domain: `ROS_DOMAIN_ID=42`

The board uses the Tsinghua ROS 2 mirror:

```text
https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu
```

Start a dedicated ROS shell with:

```bash
source /root/ros2_humble_env.sh
```

Do not source the CANN environment and the ROS environment globally from the
same shell profile. Source only the runtime needed by each process. The current
SmolVLA environment is Python 3.12, while Humble `rclpy` uses system Python
3.10; integrate them as separate processes unless a compatible bridge is built.

## Smoke test

Terminal 1:

```bash
source /root/ros2_humble_env.sh
ros2 run demo_nodes_cpp talker
```

Terminal 2:

```bash
source /root/ros2_humble_env.sh
ros2 run demo_nodes_py listener
```

The deployment backup of the original APT configuration is stored under
`/root/ros2_deploy_backup_20260815_122324` on the board.

## Visualization PC

The visualization host is Windows 11 with WSL2 Ubuntu 22.04, ROS 2 Humble
Desktop, RViz2, and Gazebo 11. Its WSL virtual disk is stored at:

```text
F:\robot_arm_atlas\.wsl\Ubuntu-22.04\ext4.vhdx
```

WSL uses NAT networking because the Atlas USB/RNDIS interface is not exposed in
mirrored mode on this PC. A Zenoh TCP router runs on the Atlas board at
`192.168.0.2:7447`; WSL ROS nodes connect to it in client mode, so both discovery
and topic data cross WSL NAT reliably.

Start a WSL ROS shell with:

```bash
source ~/ros2_humble_env.sh
```

The environment applies `QT_SCALE_FACTOR=2` so RViz2 and Gazebo are readable
on the Windows high-DPI display. Override it before launch if needed, for
example `QT_SCALE_FACTOR=2 rviz2`.

RViz2 defaults to Mesa software rendering and integer 2x Qt scaling in WSLg.
Remote joint states are restamped on the WSL host before TF generation so board
and PC clock drift cannot make the robot model blink in and out.

## Live SO-ARM101 visualization

The description package uses the official SO-ARM101 new-calibration URDF. On
the Atlas board, start the read-only motor telemetry and ROS bridge:

```bash
systemctl start so101-rviz-bridge.service
```

This process reads the follower motor positions without sending goal positions
or changing the existing torque state. It cannot share the follower serial port
with a LeRobot rollout process. Stop the bridge before running SmolVLA control:

```bash
systemctl stop so101-rviz-bridge.service
```

In WSL, start the model and RViz2:

```bash
source ~/ros2_humble_env.sh
ros2 launch so101_description display.launch.py
```

For concurrent Atlas leader/follower teleoperation and RViz2, leave the WSL
launch above running and execute this on the board:

```bash
cd /root/lerobot_project
/opt/lerobot061/bin/python 03_teleoperate_with_rviz.py
```

The teleoperation process is the sole serial-port owner and forwards the same
follower observation to ROS 2. Press Ctrl+C on the board to stop motion and the
temporary ROS bridge safely.

## SO-ARM101 MoveIt/RViz on Windows

WSLg is not used for this MoveIt window because its D3D12/copy-mode path caused
flashing or an unresponsive blank RViz window on this PC. VcXsrv is installed at
`F:\VcXsrv` and runs on the dedicated X display `:1`. The launcher creates a
per-run Xauthority cookie, so X11 access control remains enabled.

Since 2026-08-15 evening, WSLg is fully broken on this PC (GPU-PV dxg ioctl
failures make every WSLg window render black in COPY MODE; a Windows reboot is
the permanent fix). Until then, use the VcXsrv launcher for the plain
`display.launch.py` visualization as well:

```powershell
powershell -ExecutionPolicy Bypass -File F:\robot_arm_atlas\ros2\start_so101_display_vcxsrv.ps1
```

This starts VcXsrv `:1` and `ros2 launch so101_description display.launch.py`
with the domain-42 Zenoh client profile, so `/joint_states` from the Atlas board
still arrives. Start the read-only bridge on the board first if you need live
joint data:

```bash
systemctl start so101-rviz-bridge.service
```

From Windows PowerShell, start the complete mock-planning demo with:

```powershell
powershell -ExecutionPolicy Bypass -File F:\robot_arm_atlas\ros2\start_so101_moveit_vcxsrv.ps1
```

This starts VcXsrv, `robot_state_publisher`, the mock controllers, `move_group`,
and RViz. It uses ROS domain 43 and does not command the physical arm. Keep the
PowerShell window open while using MoveIt.

Stop the demo and its dedicated VcXsrv instance with:

```powershell
powershell -ExecutionPolicy Bypass -File F:\robot_arm_atlas\ros2\stop_so101_moveit_vcxsrv.ps1
```

If Windows Firewall prompts for VcXsrv access, allow only private networks.
