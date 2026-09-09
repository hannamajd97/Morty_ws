"""
moveit_rviz.launch.py
=====================

Launches RViz with the MoveIt motion-planning panel configured for the UR3e
+ custom gripper. Usually included by the main bringup
(``ur3e_gripper_3d_moveit.launch.py``); kept as a separate file so RViz can
be restarted independently from the rest of the robot stack (handy when the
GUI dies but the controllers / move_group are still healthy).

WSL caveat: RViz needs an OpenGL context via WSLg, which can be flaky.
If the window won't open, restart Windows / WSL, or work from the topic
side (``ros2 topic echo /joint_states`` etc.) until it recovers.
"""
import os
import subprocess
import tempfile

from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    camera_enabled = LaunchConfiguration("camera_enabled")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    moveit_config = (
        MoveItConfigsBuilder(
            "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
        )
        .robot_description(
            file_path="urdf/ur3e_gripper_wrapper.urdf.xacro",
            mappings={"camera_enabled": camera_enabled, "use_fake_hardware": use_fake_hardware},
        )
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl",
            load_all=False,
        )
        .to_moveit_configs()
    )

    urdf_xacro_path = PathJoinSubstitution(
        [FindPackageShare("ur3e_gripper_3d_moveit_config"), "urdf", "ur3e_gripper_wrapper.urdf.xacro"]
    )

    def _launch_rviz_with_urdf(context, *_args, **_kwargs):
        # Generate URDF *before* RViz starts. RViz MotionPlanning reads robot_description as a
        # parameter and will fail if it starts before the URDF is available.
        urdf_text = subprocess.check_output(
            [
                "xacro",
                urdf_xacro_path.perform(context),
                f"camera_enabled:={camera_enabled.perform(context)}",
                f"use_fake_hardware:={use_fake_hardware.perform(context)}",
            ],
            text=True,
        )
        tmp_path = os.path.join(tempfile.gettempdir(), "ur3e_gripper_3d_moveit.urdf")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(urdf_text)

        # RViz RobotModel display expects /robot_description by default (std_msgs/String).
        # Publish it so RobotModel and MotionPlanning visuals stay in sync.
        robot_description_publisher = Node(
            package="gripper_3d",
            executable="robot_description_publisher",
            arguments=[tmp_path],
            output="screen",
        )

        use_sim_time_str = str(use_sim_time.perform(context)).strip().lower()
        use_sim_time_bool = use_sim_time_str in {"true", "1", "yes", "on"}

        rviz_parameters = [
            {"robot_description": urdf_text},
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            {"use_sim_time": use_sim_time_bool},
        ]

        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_moveit",
            output="screen",
            arguments=[
                "-d",
                LaunchConfiguration("rviz_config").perform(context),
                "--ros-args",
                "--log-level",
                "moveit_ros.planning_scene_monitor:=error",
            ],
            parameters=rviz_parameters,
            # Pre-load the URDF XML parser plugin to work around a class_loader
            # namespace collision inside the RViz process that prevents pluginlib
            # from instantiating urdf_xml_parser/URDFXMLParser at runtime.
            additional_env={
                "LD_PRELOAD": "/opt/ros/humble/lib/liburdf_xml_parser.so",
            },
        )
        return [robot_description_publisher, rviz_node]

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument("use_sim_time", default_value="false"))
    ld.add_action(DeclareLaunchArgument("camera_enabled", default_value="false"))
    ld.add_action(DeclareLaunchArgument("use_fake_hardware", default_value="true"))
    ld.add_action(
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(moveit_config.package_path / "config/moveit.rviz"),
        )
    )

    ld.add_action(OpaqueFunction(function=_launch_rviz_with_urdf))
    return ld
