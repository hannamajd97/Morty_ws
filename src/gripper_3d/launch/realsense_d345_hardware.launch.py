#!/usr/bin/env python3

"""
Real hardware launch: start a RealSense D345/D435 using `realsense2_camera`.
This launch only starts the camera driver; it does not publish robot/gripper TF.

The defaults here prioritize reliable streaming on typical USB links.
If you want higher resolution later, override the launch arguments.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    color_profile = LaunchConfiguration("color_profile")
    depth_profile = LaunchConfiguration("depth_profile")
    pointcloud_enable = LaunchConfiguration("pointcloud_enable")
    align_depth_enable = LaunchConfiguration("align_depth_enable")
    initial_reset = LaunchConfiguration("initial_reset")

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            # ---- RealSense (standard frames) ----
            "publish_tf": "true",
            # Must match the camera link used in the robot description / static TF so the
            # RealSense TF tree connects to the robot TF tree.
            "base_frame_id": "camera_link",
            "depth_frame_id": "camera_depth_frame",
            "depth_optical_frame_id": "camera_depth_optical_frame",
            "color_frame_id": "camera_color_frame",
            "color_optical_frame_id": "camera_color_optical_frame",

            # Streams
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.profile": color_profile,
            "depth_module.profile": depth_profile,
            "initial_reset": initial_reset,

            # Pointcloud + alignment
            "pointcloud.enable": pointcloud_enable,
            "pointcloud.texture_stream": "RS2_STREAM_COLOR",
            # 2026-09-02 (#306): best-effort QoS for the raw cloud so a
            # slow subscriber (RViz) simply drops frames instead of
            # forcing RELIABLE retransmits and history queues — the
            # mechanism that both backlogged RViz and inflated this
            # node's CPU when RViz fell behind (#298, #303).
            "pointcloud.pointcloud_qos": "SENSOR_DATA",
            "align_depth.enable": align_depth_enable,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("color_profile", default_value="640x480x30"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("pointcloud_enable", default_value="true"),
            DeclareLaunchArgument("align_depth_enable", default_value="true"),
            DeclareLaunchArgument("initial_reset", default_value="true"),
            realsense,
        ]
    )
