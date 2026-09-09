"""Launch the teach_node for recording UR3e observation poses."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('ur3e_fast_observation')
    default_output = os.path.join(pkg_share, 'config', 'observation_poses.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'output_file',
            default_value=default_output,
            description='YAML file to record taught poses into.',
        ),
        DeclareLaunchArgument(
            'joint_states_topic',
            default_value='/joint_states',
            description='Topic for sensor_msgs/JointState.',
        ),
        Node(
            package='ur3e_fast_observation',
            executable='teach_node',
            name='ur3e_fast_observation_teach',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'output_file': LaunchConfiguration('output_file'),
                'joint_states_topic': LaunchConfiguration('joint_states_topic'),
            }],
        ),
    ])
