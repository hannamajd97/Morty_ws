#!/usr/bin/env python3

"""
Display-only launch: UR3e + gripper_3d model in RViz.
"""

import os
import subprocess
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_path = PathJoinSubstitution(
        [FindPackageShare("gripper_3d"), "urdf", "ur3e_with_gripper_3d.urdf.xacro"]
    )
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("gripper_3d"), "rviz", "ur3e_with_gripper_3d.rviz"]
    )

    use_gui = LaunchConfiguration("use_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    urdf_file = LaunchConfiguration("urdf_file")
    camera_enabled = LaunchConfiguration("camera_enabled")

    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    "xacro",
                    " ",
                    urdf_path,
                    " ",
                    "camera_enabled:=",
                    camera_enabled,
                ]
            ),
            value_type=str,
        )
    }

    def _write_urdf_file(context, *_args, **_kwargs):
        urdf_path_value = urdf_path.perform(context)
        urdf_text = subprocess.check_output(
                [
                    "xacro",
                    urdf_path_value,
                    f"camera_enabled:={camera_enabled.perform(context)}",
                ],
                text=True,
            )
        tmp_path = os.path.join(tempfile.gettempdir(), "gripper_3d_ur3e_with_gripper.urdf")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(urdf_text)
        return [SetLaunchConfiguration("urdf_file", tmp_path)]

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        arguments=[urdf_file],
        condition=IfCondition(use_gui),
    )
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        arguments=[urdf_file],
        condition=UnlessCondition(use_gui),
    )
    robot_description_publisher = Node(
        package="gripper_3d",
        executable="robot_description_publisher",
        arguments=[urdf_file],
        output="screen",
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(launch_rviz),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_gui", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("urdf_file", default_value=""),
            DeclareLaunchArgument("camera_enabled", default_value="false"),
            OpaqueFunction(function=_write_urdf_file),
            joint_state_publisher_gui,
            joint_state_publisher,
            robot_description_publisher,
            robot_state_publisher,
            rviz,
        ]
    )
