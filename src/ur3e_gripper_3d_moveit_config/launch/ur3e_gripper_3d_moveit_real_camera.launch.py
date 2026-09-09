from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    this_package_share = FindPackageShare("ur3e_gripper_3d_moveit_config")

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([this_package_share, "launch", "ur3e_gripper_3d_moveit.launch.py"])
        ),
        launch_arguments={
            # Real hardware defaults for this lab setup
            "use_ur_driver": "true",
            "use_fake_hardware": "false",
            "robot_ip": "192.168.1.102",
            "reverse_ip": "192.168.1.101",
            # Enable camera model/TF and start the RealSense driver
            "camera_enabled": "true",
            "use_real_camera": "true",
        }.items(),
    )

    return LaunchDescription([base_launch])
