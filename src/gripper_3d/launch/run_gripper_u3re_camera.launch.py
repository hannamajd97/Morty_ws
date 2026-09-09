#!/usr/bin/env python3

"""
Run UR3e + gripper_3d display, optionally start the real RealSense camera.

Default: `use_real_camera:=false` (simulation / visualization only).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("gripper_3d")
    use_real_camera = LaunchConfiguration("use_real_camera")

    display = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "ur3e_gripper_3d.launch.py")
        ),
        launch_arguments={
            "camera_enabled": "true",
            "launch_rviz": "true",
        }.items(),
    )

    real_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "realsense_d345_hardware.launch.py")
        ),
        condition=IfCondition(use_real_camera),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_real_camera",
                default_value="false",
                description="Start RealSense driver (real hardware).",
            ),
            display,
            real_camera,
        ]
    )

