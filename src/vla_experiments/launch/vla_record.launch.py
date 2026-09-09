"""Start the VLA episode recorder (+ Astra static TF).

Sets FASTRTPS_DEFAULT_PROFILES_FILE to the package's fastdds_shuttle_peer.xml so the
recorder discovers the shuttle's Astra topics over the dock link (see README).
    ros2 launch vla_experiments vla_record.launch.py                 # manual start/stop via vla_episode
    ros2 launch vla_experiments vla_record.launch.py auto_mode:=true # one episode per Morty voice trial
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("vla_experiments")
    cfg = os.path.join(share, "config", "recorder.yaml")
    profile = os.path.join(share, "config", "fastdds_shuttle_peer.xml")
    args = [
        DeclareLaunchArgument("config", default_value=cfg),
        DeclareLaunchArgument("dataset_name", default_value="morty_ur3e"),
        DeclareLaunchArgument("auto_mode", default_value="false"),
        DeclareLaunchArgument("with_astra_tf", default_value="true"),
        DeclareLaunchArgument("require_exterior", default_value="true"),
        DeclareLaunchArgument("require_wrist", default_value="true"),
    ]
    env = [
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", profile),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "0"),
    ]
    recorder = Node(
        package="vla_experiments", executable="episode_recorder", name="vla_episode_recorder", output="screen",
        parameters=[LaunchConfiguration("config"), {
            "dataset_name": LaunchConfiguration("dataset_name"),
            "auto_mode": LaunchConfiguration("auto_mode"),
            "require_exterior": LaunchConfiguration("require_exterior"),
            "require_wrist": LaunchConfiguration("require_wrist"),
        }],
    )
    astra_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "astra_tf.launch.py")),
        condition=IfCondition(LaunchConfiguration("with_astra_tf")),
    )
    return LaunchDescription(args + env + [astra_tf, recorder])
