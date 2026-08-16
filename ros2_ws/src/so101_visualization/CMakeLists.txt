cmake_minimum_required(VERSION 3.8)
project(so101_visualization)

find_package(ament_cmake REQUIRED)

install(PROGRAMS scripts/ee_trajectory_recorder.py scripts/mock_joint_source.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
