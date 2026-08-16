cmake_minimum_required(VERSION 3.8)
project(so101_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/joint_state_retime.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
