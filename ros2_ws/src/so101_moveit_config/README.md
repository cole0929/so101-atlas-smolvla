# SO-ARM101 MoveIt 2 demo

This package is a simulation-only MoveIt 2 configuration. It uses
`mock_components/GenericSystem`; it never opens the SO-ARM101 serial ports and
cannot move the real robot.

Run in WSL:

```bash
bash /mnt/f/robot_arm_atlas/ros2/run_so101_moveit_demo.sh
```

The launcher uses ROS domain 43 and Fast DDS so its simulated `/joint_states`
cannot mix with the Atlas hardware graph on domain 42.

## RViz controls

1. Open the `MotionPlanning` panel.
2. Select planning group `arm`.
3. Set the start state to `current`.
4. Choose named goal `home` or `ready`, or drag the end-effector marker.
5. Click `Plan` to preview the purple trajectory.
6. Click `Execute` to move the simulated robot, or `Plan & Execute` to do both.
7. Select group `gripper` and named goal `open` or `closed` to test the jaw.

The arm has five positioning joints and the jaw is the sixth actuator. A full
six-dimensional end-effector pose is not always reachable, so small marker
movements and named joint targets are the best first tests.

Stop all processes belonging to this demo with:

```bash
bash /mnt/f/robot_arm_atlas/ros2/stop_so101_moveit_demo.sh
```
