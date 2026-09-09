#!/usr/bin/env python3

"""
Hardware-only: bridge UR tool IO to gripper finger joints in RViz.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("gripper_control_params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gripper_control_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("gripper_3d"), "config", "gripper_control.yaml"]
                ),
                description="Single source of truth for gripper IO/adapter parameters.",
            ),
            Node(
                package="gripper_3d",
                executable="gripper_io_bridge",
                parameters=[params_file],
                output="screen",
            )
        ]
    )
