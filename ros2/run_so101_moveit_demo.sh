#!/usr/bin/env bash
set -e

# Local simulation uses its own ROS domain and never connects to the Atlas board.
source /opt/ros/humble/setup.bash
source /mnt/f/robot_arm_atlas/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=43
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset ZENOH_CONFIG_OVERRIDE

export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR="${QT_SCALE_FACTOR:-2}"

# WSLg's D3D12 path opens a blank RViz window in COPY MODE on this machine.
# Use llvmpipe, but force OGRE to copy render textures instead of using the
# framebuffer-object path that alternated with a black 3D viewport.
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
export QT_OPENGL=software
export QT_XCB_GL_INTEGRATION=xcb_glx
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
unset LIBGL_DRI3_DISABLE
unset MESA_D3D12_DEFAULT_ADAPTER_NAME
export OGRE_RTT_MODE=Copy

exec ros2 launch so101_moveit_config demo.launch.py
