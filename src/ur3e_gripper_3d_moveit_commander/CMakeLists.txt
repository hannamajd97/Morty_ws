cmake_minimum_required(VERSION 3.8)
project(ur3e_gripper_3d_moveit_commander)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(morty_interfaces REQUIRED)
find_package(ur_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(grasp_executor src/grasp_executor.cpp)
ament_target_dependencies(grasp_executor
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
  geometry_msgs
  control_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(grasp_executor_pilz src/grasp_executor_pilz.cpp)
ament_target_dependencies(grasp_executor_pilz
  rclcpp
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(grasp_executor_lin src/grasp_executor_lin.cpp)
ament_target_dependencies(grasp_executor_lin
  sensor_msgs
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
  geometry_msgs
  control_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  morty_interfaces
)

# Voice-driven variant: same LIN pick logic but with an interactive
# drop pause after Step 3 (close gripper). Waits for a destination
# string on resume_topic before continuing with lift + drop.
add_executable(grasp_executor_lin_llm src/grasp_executor_lin_llm.cpp)
ament_target_dependencies(grasp_executor_lin_llm
  sensor_msgs
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
  geometry_msgs
  control_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  morty_interfaces
  ur_msgs
)

add_executable(observation_ptp_commander src/observation_ptp_commander.cpp)
ament_target_dependencies(observation_ptp_commander
  rclcpp
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# High-level OperateGripper action server (C++ port of the original Python
# gripper_action_server). Translates voice/intent OPEN/CLOSE/SET_WIDTH goals
# into low-level control_msgs/GripperCommand goals on the ros2_control
# gripper controller.
add_executable(gripper_action_server src/gripper_action_server.cpp)
ament_target_dependencies(gripper_action_server
  rclcpp
  rclcpp_action
  control_msgs
  morty_interfaces
)

install(TARGETS
  grasp_executor
  grasp_executor_pilz
  grasp_executor_lin
  grasp_executor_lin_llm
  observation_ptp_commander
  gripper_action_server
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  # Standalone check of the probe-plan reuse comparison logic. No ROS,
  # no robot: it compiles the SHIPPED src/plan_reuse_tolerance.hpp and
  # exercises the pose / joint-vector tolerances with synthetic values.
  # The robot cannot be driven to verify the reuse path, so this is the
  # part of it that CAN be verified, and it is verified on every build.
  add_executable(plan_reuse_tolerance_check
    test/plan_reuse_tolerance_check.cpp)
  add_test(NAME plan_reuse_tolerance_check
    COMMAND plan_reuse_tolerance_check)

  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
