cmake_minimum_required(VERSION 3.8)
project(gripper_3d)

find_package(ament_cmake REQUIRED)

install(FILES package.xml README.md
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch urdf rviz meshes config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/robot_description_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS scripts/gripper_io_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS scripts/gripper_command_adapter
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
