cmake_minimum_required(VERSION 3.8)
project(morty_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

set(msg_files
  msg/Detection.msg
  msg/Grasp.msg
)

set(srv_files
  srv/Stop.srv
  srv/GetSnapshot.srv
)

set(action_files
  # Robot motion primitives (low-level — served by the executor package)
  action/MoveToPose.action
  action/MoveToNamedPose.action
  action/OperateGripper.action

  # Robot perception / planning (served by grasp_pipeline)
  action/DetectObjects.action
  action/ScanScene.action
  action/GenerateGrasps.action

  # Robot high-level orchestration (served by grasp_pipeline)
  action/Pick.action
  action/Place.action
  action/GoHome.action
  action/ExecuteGrasp.action

  # Speech I/O (served by communication_llm)
  action/Speak.action
  action/ListenForChoice.action
  action/SelectObject.action
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES
    action_msgs
    std_msgs
    geometry_msgs
    sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
