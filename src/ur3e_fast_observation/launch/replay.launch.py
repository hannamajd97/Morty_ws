"""Launch the replay_node for executing taught observation poses."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('ur3e_fast_observation')
    default_poses = os.path.join(pkg_share, 'config', 'observation_poses.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'poses_file',
            default_value=default_poses,
            description='YAML file of taught observation poses.',
        ),
        DeclareLaunchArgument(
            'controller_action_ns',
            default_value='/scaled_joint_trajectory_controller',
            description='Action namespace of the trajectory controller.',
        ),
        DeclareLaunchArgument(
            'move_duration_s',
            default_value='2.0',
            description='Time allotted for the single-waypoint motion.',
        ),
        DeclareLaunchArgument(
            'default_pose_name',
            default_value='',
            description='Optional pose label to replay once on startup.',
        ),
        Node(
            package='ur3e_fast_observation',
            executable='replay_node',
            name='ur3e_fast_observation_replay',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'poses_file': LaunchConfiguration('poses_file'),
                'controller_action_ns': LaunchConfiguration(
                    'controller_action_ns'
                ),
                'move_duration_s': LaunchConfiguration('move_duration_s'),
                'default_pose_name': LaunchConfiguration('default_pose_name'),
            }],
        ),
    ])
