"""
open3d_cloud_viewer.launch.py
=============================

Launches the standalone Open3D point-cloud viewer (debug tool). Subscribes
to a PointCloud2 topic (selected/fused object cloud) and shows it in an
interactive Open3D window. Useful for inspecting what the perception side
is producing without bringing up the rest of the pipeline.

WSL caveat: Open3D's interactive window can be flaky over WSLg; consider
running headlessly via ``tools/grasp_debug_viewer.py --save out.png`` if
the window will not open.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("topic", default_value="/camera/camera/depth/color/points_downsampled"),
        DeclareLaunchArgument("point_size", default_value="2.0"),
        DeclareLaunchArgument("stride", default_value="1"),
        DeclareLaunchArgument("voxel_size_m", default_value="0.0"),
        DeclareLaunchArgument("max_points", default_value="120000"),
        DeclareLaunchArgument("update_hz", default_value="30.0"),
        DeclareLaunchArgument("use_rgb", default_value="true"),
        DeclareLaunchArgument("show_axes", default_value="true"),
        DeclareLaunchArgument("width", default_value="800"),
        DeclareLaunchArgument("height", default_value="520"),
        DeclareLaunchArgument("status_log_period_s", default_value="2.0"),
        Node(
            package="grasp_pipeline",
            executable="open3d_cloud_viewer",
            name="open3d_cloud_viewer",
            output="screen",
            parameters=[{
                "topic": LaunchConfiguration("topic"),
                "point_size": ParameterValue(LaunchConfiguration("point_size"), value_type=float),
                "stride": ParameterValue(LaunchConfiguration("stride"), value_type=int),
                "voxel_size_m": ParameterValue(LaunchConfiguration("voxel_size_m"), value_type=float),
                "max_points": ParameterValue(LaunchConfiguration("max_points"), value_type=int),
                "update_hz": ParameterValue(LaunchConfiguration("update_hz"), value_type=float),
                "use_rgb": ParameterValue(LaunchConfiguration("use_rgb"), value_type=bool),
                "show_axes": ParameterValue(LaunchConfiguration("show_axes"), value_type=bool),
                "width": ParameterValue(LaunchConfiguration("width"), value_type=int),
                "height": ParameterValue(LaunchConfiguration("height"), value_type=int),
                "status_log_period_s": ParameterValue(LaunchConfiguration("status_log_period_s"), value_type=float),
            }],
        ),
    ])
