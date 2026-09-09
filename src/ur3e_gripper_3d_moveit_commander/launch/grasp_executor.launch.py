"""
grasp_executor.launch.py
========================

Launches the default ``grasp_executor`` node: the C++ MoveIt-based pick
executor that consumes grasps produced by ``grasp_pipeline`` and drives the
arm + gripper through them using PTP (point-to-point) planning.

Sister launchers in this package:

* ``grasp_executor_lin.launch.py``     — same, but with LINear Cartesian
                                         approach/retreat.
* ``grasp_executor_lin_llm.launch.py`` — LIN + an interactive drop-pause
                                         (used by ``morty_full``: the arm
                                         holds the object until the intent
                                         router pushes a destination).
* ``grasp_executor_pilz.launch.py``    — Pilz industrial motion planner
                                         backend.

Loads the MoveIt config for the UR3e + custom gripper (``moveit_configs_utils``)
and passes ``use_fake_hardware`` through so the same launcher works against
the simulated robot (``true``) or the real one (``false``).
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
            # Hardware simulation
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),

            # Planning configuration
            DeclareLaunchArgument("planning_group", default_value="ur3e_manipulator"),
            DeclareLaunchArgument("end_effector_link", default_value="gripper_tcp"),

            # Execution control
            DeclareLaunchArgument("execute", default_value="false",
                description="Set to true to actually execute motions"),

            # Grasp parameters
            DeclareLaunchArgument("grasp_pose_topic", default_value="/grasp_pose",
                description="Topic to subscribe for grasp poses"),
            DeclareLaunchArgument("pregrasp_distance", default_value="0.03",
                description="Distance to back off for pre-grasp (meters)"),
            DeclareLaunchArgument("target_frame", default_value="world",
                description="Frame to transform grasp poses into"),

            # Cartesian path parameters
            DeclareLaunchArgument("cartesian_step", default_value="0.005",
                description="Step size for Cartesian path (meters)"),

            # Planning parameters
            DeclareLaunchArgument("planning_time", default_value="10.0",
                description="Maximum planning time (seconds)"),
            DeclareLaunchArgument("num_attempts", default_value="10",
                description="Number of planning attempts"),

            # TF timeout
            DeclareLaunchArgument("tf_timeout", default_value="2.0",
                description="TF lookup timeout (seconds)"),

            Node(
                package="ur3e_gripper_3d_moveit_commander",
                executable="grasp_executor",
                name="grasp_executor",
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
                        "cartesian_step": LaunchConfiguration("cartesian_step"),
                        "planning_time": LaunchConfiguration("planning_time"),
                        "num_attempts": LaunchConfiguration("num_attempts"),
                        "tf_timeout": LaunchConfiguration("tf_timeout"),
                    },
                ],
            ),
        ]
    )
