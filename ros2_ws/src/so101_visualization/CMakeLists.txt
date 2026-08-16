cmake_minimum_required(VERSION 3.8)
project(so101_visualization)

find_package(ament_cmake REQUIRED)

install(PROGRAMS scripts/ee_trajectory_recorder.py scripts/mock_joint_source.py scripts/digital_twin_layers.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
