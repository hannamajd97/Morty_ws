"""
move_group.launch.py
====================

Launches MoveIt's ``move_group`` node on its own. Normally you do NOT call
this directly — ``ur3e_gripper_3d_moveit.launch.py`` includes it as part of
the full robot bringup, after the URDF/SRDF + ros2_control controllers are
up. Provided as a separate file mostly so it can be tested in isolation
during MoveIt config debugging.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    use_ur_driver = LaunchConfiguration("use_ur_driver")
    move_group_log_level = LaunchConfiguration("move_group_log_level")
    use_sim_time = LaunchConfiguration("use_sim_time")
    camera_enabled = LaunchConfiguration("camera_enabled")

    moveit_config_fake = (
        MoveItConfigsBuilder(
            "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
        )
        .robot_description(
            file_path="urdf/ur3e_gripper_wrapper.urdf.xacro",
            mappings={"use_fake_hardware": use_fake_hardware, "camera_enabled": camera_enabled},
        )
        # Controllers are spawned/activated by launch; avoid MoveIt controller-switching
        # assumptions (especially since the gripper is adapted via an action bridge).
        .trajectory_execution(file_path="config/moveit_controllers.yaml", moveit_manage_controllers=False)
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="pilz_industrial_motion_planner",   # Pilz sequence service (blended LIN+LIN) plans with the DEFAULT pipeline in Humble (2026-09-16); every other caller names its pipeline explicitly
            load_all=False,
        )
        .to_moveit_configs()
    )

    moveit_config_driver = (
        MoveItConfigsBuilder(
            "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
        )
        .robot_description(
            file_path="urdf/ur3e_gripper_wrapper.urdf.xacro",
            mappings={"use_fake_hardware": use_fake_hardware, "camera_enabled": camera_enabled},
        )
        .trajectory_execution(
            file_path="config/ur_driver_moveit_controllers.yaml", moveit_manage_controllers=False
        )
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="pilz_industrial_motion_planner",   # Pilz sequence service (blended LIN+LIN) plans with the DEFAULT pipeline in Humble (2026-09-16); every other caller names its pipeline explicitly
            load_all=False,
        )
        .to_moveit_configs()
    )

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument("debug", default_value="false"))
    ld.add_action(DeclareLaunchArgument("use_sim_time", default_value="false"))
    ld.add_action(DeclareLaunchArgument("use_fake_hardware", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_ur_driver", default_value="false"))
    ld.add_action(DeclareLaunchArgument("camera_enabled", default_value="false"))
    ld.add_action(
        DeclareLaunchArgument(
            "move_group_log_level",
            default_value="info",
            description="ROS log level for the MoveIt move_group node (e.g. debug, info, warn).",
        )
    )
    ld.add_action(DeclareLaunchArgument("allow_trajectory_execution", default_value="true"))
    ld.add_action(DeclareLaunchArgument("publish_monitored_planning_scene", default_value="true"))
    ld.add_action(DeclareLaunchArgument("capabilities", default_value=""))
    ld.add_action(DeclareLaunchArgument("disable_capabilities", default_value=""))
    ld.add_action(DeclareLaunchArgument("monitor_dynamics", default_value="false"))

    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        "capabilities": ParameterValue(LaunchConfiguration("capabilities"), value_type=str),
        "disable_capabilities": ParameterValue(
            LaunchConfiguration("disable_capabilities"), value_type=str
        ),
        "publish_planning_scene": LaunchConfiguration("publish_monitored_planning_scene"),
        "publish_geometry_updates": LaunchConfiguration("publish_monitored_planning_scene"),
        "publish_state_updates": LaunchConfiguration("publish_monitored_planning_scene"),
        "publish_transforms_updates": LaunchConfiguration("publish_monitored_planning_scene"),
        "monitor_dynamics": False,
    }

    move_group_params_fake = [
        moveit_config_fake.to_dict(),
        move_group_configuration,
        {"use_sim_time": use_sim_time},
    ]
    move_group_params_driver = [
        moveit_config_driver.to_dict(),
        move_group_configuration,
        {"use_sim_time": use_sim_time},
    ]

    move_group_node_fake = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        arguments=[
            "--ros-args",
            "--log-level",
            move_group_log_level,
        ],
        parameters=move_group_params_fake,
        additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
        condition=UnlessCondition(use_ur_driver),
    )

    move_group_node_driver = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        arguments=[
            "--ros-args",
            "--log-level",
            move_group_log_level,
        ],
        parameters=move_group_params_driver,
        additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
        condition=IfCondition(use_ur_driver),
    )

    ld.add_action(move_group_node_fake)
    ld.add_action(move_group_node_driver)
    return ld
