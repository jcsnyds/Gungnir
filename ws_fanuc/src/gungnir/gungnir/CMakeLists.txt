cmake_minimum_required(VERSION 3.22)
project(gungnir)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config worlds DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/sim_camera_adapter.py
  scripts/sim_gripper_stub.py
  scripts/l515_node.py
  scripts/camera_viz.py
  scripts/world_markers.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
