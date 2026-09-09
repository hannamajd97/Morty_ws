"""
grasp_executor_pilz.launch.py
=============================

Launches the Pilz-planner variant of the grasp executor. Uses the
``pilz_industrial_motion_planner`` (PTP / LIN / CIRC primitives) instead of
the OMPL sampling-based planner, which gives more predictable,
industrial-style trajectories at the cost of less flexibility around
obstacles. Useful for repeatable timing in experiments.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
 
    moveit_config = (
        MoveItConfigsBuilder(
            "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
        )
        .robot_description(
            file_path="urdf/ur3e_gripper_wrapper.urdf.xacro",
            mappings={"use_fake_hardware": use_fake_hardware},
        )
        .to_moveit_configs()
    )
 
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),
            DeclareLaunchArgument("planning_group", default_value="ur3e_manipulator"),
            DeclareLaunchArgument("end_effector_link", default_value="gripper_tcp"),
            DeclareLaunchArgument(
                "execute",
                default_value="false",
                description="Set to true to actually execute motions",
            ),
            DeclareLaunchArgument(
                "grasp_pose_topic",
                default_value="/grasp_pose",
                description="Topic to subscribe for grasp poses",
            ),
            DeclareLaunchArgument(
                "pregrasp_distance",
                default_value="0.03",
                description="Distance to back off for pre-grasp (meters)",
            ),
            DeclareLaunchArgument(
                "target_frame",
                default_value="world",
                description="Frame to transform grasp poses into",
            ),
            DeclareLaunchArgument(
                "planning_time",
                default_value="10.0",
                description="Maximum planning time (seconds)",
            ),
            DeclareLaunchArgument(
                "num_attempts",
                default_value="10",
                description="Number of planning attempts",
            ),
            DeclareLaunchArgument(
                "tf_timeout",
                default_value="2.0",
                description="TF lookup timeout (seconds)",
            ),
            Node(
                package="ur3e_gripper_3d_moveit_commander",
                executable="grasp_executor_pilz",
                name="grasp_executor_pilz",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {
                        "planning_group": LaunchConfiguration("planning_group"),
                        "end_effector_link": LaunchConfiguration("end_effector_link"),
                        "execute": LaunchConfiguration("execute"),
                        "grasp_pose_topic": LaunchConfiguration("grasp_pose_topic"),
                        "pregrasp_distance": LaunchConfiguration("pregrasp_distance"),
                        "target_frame": LaunchConfiguration("target_frame"),
                        "planning_time": LaunchConfiguration("planning_time"),
                        "num_attempts": LaunchConfiguration("num_attempts"),
                        "tf_timeout": LaunchConfiguration("tf_timeout"),
                    },
                ],
            ),
        ]
    )

