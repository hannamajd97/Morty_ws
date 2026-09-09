"""
observation_ptp_commander.launch.py
===================================

Launches the ``observation_ptp_commander`` C++ node: moves the arm through
the configured observation poses using PTP planning. Used when you want to
sweep through the multi-view observation positions WITHOUT using the fast
named-pose replay path — for example during initial pose teaching, or when
the named-pose registry is empty.

For the fast (non-MoveIt) version of the same idea, see the
``ur3e_fast_observation`` package's ``replay.launch.py``.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    camera_enabled = LaunchConfiguration("camera_enabled")

    moveit_config = (
        MoveItConfigsBuilder(
            "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
        )
        .robot_description(
            file_path="urdf/ur3e_gripper_wrapper.urdf.xacro",
            mappings={
                "use_fake_hardware": use_fake_hardware,
                "camera_enabled": camera_enabled,
            },
        )
        .to_moveit_configs()
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_fake_hardware", default_value="true"),
        DeclareLaunchArgument("camera_enabled", default_value="true"),
        DeclareLaunchArgument("planning_group", default_value="ur3e_manipulator"),
        DeclareLaunchArgument("end_effector_link", default_value="gripper_tcp"),
        DeclareLaunchArgument("target_name", default_value="home"),
        DeclareLaunchArgument("use_pose_target", default_value="false"),
        DeclareLaunchArgument("reference_frame", default_value="base_link"),
        DeclareLaunchArgument("pose_x", default_value="0.0"),
        DeclareLaunchArgument("pose_y", default_value="0.0"),
        DeclareLaunchArgument("pose_z", default_value="0.0"),
        DeclareLaunchArgument("pose_qx", default_value="0.0"),
        DeclareLaunchArgument("pose_qy", default_value="0.0"),
        DeclareLaunchArgument("pose_qz", default_value="0.0"),
        DeclareLaunchArgument("pose_qw", default_value="1.0"),
        DeclareLaunchArgument("execute", default_value="false"),
        DeclareLaunchArgument("one_shot", default_value="true"),
        DeclareLaunchArgument("planning_time", default_value="5.0"),
        DeclareLaunchArgument("num_attempts", default_value="10"),
        DeclareLaunchArgument("velocity_scaling", default_value="0.15"),
        DeclareLaunchArgument("acceleration_scaling", default_value="0.15"),
        DeclareLaunchArgument("planning_pipeline", default_value="pilz_industrial_motion_planner"),
        DeclareLaunchArgument("planner_id", default_value="PTP"),
        DeclareLaunchArgument("allow_lin_fallback", default_value="true"),
        DeclareLaunchArgument("fallback_planning_pipeline", default_value="pilz_industrial_motion_planner"),
        DeclareLaunchArgument("fallback_planner_id", default_value="LIN"),
        Node(
            package="ur3e_gripper_3d_moveit_commander",
            executable="observation_ptp_commander",
            name="observation_ptp_commander",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "planning_group": LaunchConfiguration("planning_group"),
                    "end_effector_link": LaunchConfiguration("end_effector_link"),
                    "target_name": LaunchConfiguration("target_name"),
                    "use_pose_target": LaunchConfiguration("use_pose_target"),
                    "reference_frame": LaunchConfiguration("reference_frame"),
                    "pose_x": LaunchConfiguration("pose_x"),
                    "pose_y": LaunchConfiguration("pose_y"),
                    "pose_z": LaunchConfiguration("pose_z"),
                    "pose_qx": LaunchConfiguration("pose_qx"),
                    "pose_qy": LaunchConfiguration("pose_qy"),
                    "pose_qz": LaunchConfiguration("pose_qz"),
                    "pose_qw": LaunchConfiguration("pose_qw"),
                    "execute": LaunchConfiguration("execute"),
                    "one_shot": LaunchConfiguration("one_shot"),
                    "planning_time": LaunchConfiguration("planning_time"),
                    "num_attempts": LaunchConfiguration("num_attempts"),
                    "velocity_scaling": LaunchConfiguration("velocity_scaling"),
                    "acceleration_scaling": LaunchConfiguration("acceleration_scaling"),
                    "planning_pipeline": LaunchConfiguration("planning_pipeline"),
                    "planner_id": LaunchConfiguration("planner_id"),
                    "allow_lin_fallback": LaunchConfiguration("allow_lin_fallback"),
                    "fallback_planning_pipeline": LaunchConfiguration("fallback_planning_pipeline"),
                    "fallback_planner_id": LaunchConfiguration("fallback_planner_id"),
                },
            ],
        ),
    ])
