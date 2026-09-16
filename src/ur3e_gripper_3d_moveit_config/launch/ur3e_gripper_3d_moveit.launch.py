"""
ur3e_gripper_3d_moveit.launch.py
================================

The main robot bringup. Starts the UR3e + custom gripper with MoveIt and
ros2_control: ``robot_state_publisher``, ``controller_manager``, the
arm and gripper controllers, ``move_group``, the environment / planning-scene
publishers, and RViz with the MoveIt motion-planning panel.

The default is **simulation** (``use_fake_hardware:=true``) so the launcher
works out of the box without a real robot. The two thin wrappers in this
package launch the same description with different defaults:

* ``ur3e_gripper_3d_moveit_real.launch.py``        — connects to the physical
                                                     UR3e (``use_fake_hardware:=false``).
* ``ur3e_gripper_3d_moveit_real_camera.launch.py`` — real robot AND RealSense.

The mock arm boots at the joint pose defined in ``config/home_positions.yaml``
(currently the ``home`` SRDF pose) — keep that pose collision-free; the
default ``init`` pose Creo / mock_components would otherwise pick can put
the arm into the workspace walls and MoveIt then refuses to execute.

Built-in timing: there is a ``moveit_start_delay`` (~12 s) so MoveIt waits
for the UR driver + controllers to settle before loading, and an
``rviz_start_delay`` for the RViz GUI. Tweak via the launch args.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import AndSubstitution, NotSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    use_ur_driver = LaunchConfiguration("use_ur_driver")
    camera_enabled = LaunchConfiguration("camera_enabled")
    use_real_camera = LaunchConfiguration("use_real_camera")
    # RealSense point cloud + downsampler (2026-09-09): the driver's own
    # point cloud cost a full core (99 % CPU) and its only consumer, the
    # downsampler, had no subscribers — the pipeline builds its clouds
    # from the depth image and the GUI shows those. Off by default; pass
    # camera_pointcloud:=true for an RViz camera cloud.
    camera_pointcloud = LaunchConfiguration("camera_pointcloud")
    use_joint_state_publisher_gui = LaunchConfiguration("use_joint_state_publisher_gui")
    use_standalone_ros2_control = LaunchConfiguration("use_standalone_ros2_control")
    robot_ip = LaunchConfiguration("robot_ip")
    reverse_ip = LaunchConfiguration("reverse_ip")
    ur_type = LaunchConfiguration("ur_type")
    kinematics_params_file = LaunchConfiguration("kinematics_params_file")
    publish_environment_markers = LaunchConfiguration("publish_environment_markers")
    publish_moveit_environment = LaunchConfiguration("publish_moveit_environment")
    gripper_control_params_file = LaunchConfiguration("gripper_control_params_file")
    ur_start_delay = LaunchConfiguration("ur_start_delay")
    moveit_start_delay = LaunchConfiguration("moveit_start_delay")
    rviz_start_delay = LaunchConfiguration("rviz_start_delay")

    # Prevent `ros2 launch` from failing when ~/.ros/log is not writable (e.g. root-owned).
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/ros_log")

    moveit_config = MoveItConfigsBuilder(
        "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
    ).robot_description(
        file_path="urdf/ur3e_gripper_wrapper.urdf.xacro",
        mappings={"use_fake_hardware": use_fake_hardware, "camera_enabled": camera_enabled},
    ).to_moveit_configs()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, {"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(
            AndSubstitution(LaunchConfiguration("use_standalone_rsp"), NotSubstitution(use_ur_driver))
        ),
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        # Required so the GUI knows which joints exist; without this RViz will show TFs but no meshes.
        parameters=[moveit_config.robot_description, {"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(
            AndSubstitution(
                AndSubstitution(use_joint_state_publisher_gui, NotSubstitution(use_ur_driver)),
                NotSubstitution(use_standalone_ros2_control),
            ),
        ),
    )

    # Publish a *single* /joint_states stream by merging ros2_control states with any
    # auxiliary joints (e.g. gripper IO fallback) and mimic joints.
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[
            moveit_config.robot_description,
            {
                "use_sim_time": use_sim_time,
                "rate": 50,
                "source_list": [
                    "/joint_state_broadcaster/joint_states",
                    "/gripper_controller/joint_states",
                ],
                "use_mimic_tags": True,
            },
        ],
        output="screen",
        condition=IfCondition(
            AndSubstitution(
                LaunchConfiguration("use_standalone_ros2_control"), NotSubstitution(use_ur_driver)
            )
        ),
    )

    this_package_share = FindPackageShare("ur3e_gripper_3d_moveit_config")
    environment_params = PathJoinSubstitution([this_package_share, "config", "environment.yaml"])

    environment_markers_node = Node(
        package="ur3e_gripper_3d_moveit_config",
        executable="environment_markers_publisher",
        output="screen",
        parameters=[environment_params, {"frame_id": "world", "use_sim_time": use_sim_time}],
        condition=IfCondition(publish_environment_markers),
    )

    moveit_environment_node = Node(
        package="ur3e_gripper_3d_moveit_config",
        executable="planning_scene_environment_publisher",
        output="screen",
        parameters=[environment_params, {"frame_id": "world", "use_sim_time": use_sim_time}],
        condition=IfCondition(publish_moveit_environment),
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            PathJoinSubstitution([this_package_share, "config", "ros2_controllers.yaml"]),
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
        condition=IfCondition(
            AndSubstitution(
                LaunchConfiguration("use_standalone_ros2_control"), NotSubstitution(use_ur_driver)
            )
        ),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
        condition=IfCondition(
            AndSubstitution(
                LaunchConfiguration("use_standalone_ros2_control"), NotSubstitution(use_ur_driver)
            )
        ),
    )

    ur3e_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ur3e_arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
        condition=IfCondition(
            AndSubstitution(
                LaunchConfiguration("use_standalone_ros2_control"), NotSubstitution(use_ur_driver)
            )
        ),
    )

    gripper_hw_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_hw_controller", "--controller-manager", "/controller_manager"],
        output="screen",
        condition=IfCondition(
            AndSubstitution(
                LaunchConfiguration("use_standalone_ros2_control"), NotSubstitution(use_ur_driver)
            )
        ),
    )

    # Adapter that exposes a stable MoveIt gripper action (/gripper_controller/gripper_cmd)
    # and forwards to either ros2_control (fake) or UR tool I/O (real) depending on availability.
    # Adapter that exposes a stable MoveIt gripper action (/gripper_controller/gripper_cmd).
    #
    # For fake/standalone ros2_control we avoid publishing on /joint_states directly and instead
    # merge with a joint_state_publisher node so there is only one /joint_states stream.
    #
    # For real UR hardware, the UR driver publishes arm joints on /joint_states, but it does not
    # know about the added gripper joint. In that case we publish the gripper joint state on
    # /joint_states as a partial JointState message so MoveIt can track the full model.
    gripper_command_adapter_node_fake = Node(
        package="gripper_3d",
        executable="gripper_command_adapter",
        namespace="gripper_controller",
        output="screen",
        parameters=[
            gripper_control_params_file,
            {
                # Keep fake mode off /joint_states and merge centrally via joint_state_publisher.
                "joint_states_topic_alt": "/gripper_controller/joint_states_alt",
            }
        ],
        condition=UnlessCondition(use_ur_driver),
    )

    gripper_command_adapter_node_real = Node(
        package="gripper_3d",
        executable="gripper_command_adapter",
        namespace="gripper_controller",
        output="screen",
        parameters=[
            gripper_control_params_file,
            {
                # Publish the gripper joint directly into the global /joint_states stream.
                "joint_states_topic_alt": "/joint_states",
            }
        ],
        condition=IfCondition(use_ur_driver),
    )

    # When using the UR driver, the driver publishes the main TF tree. Since we don't want to patch
    # `ur_robot_driver` launch files, publish only the *camera* fixed frames here so TF can include
    # the camera when `camera_enabled:=true`.
    #
    # These values mirror the defaults in `gripper_3d/urdf/gripper_3d.xacro`.
    camera_mount_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_mount_static_tf",
        output="screen",
        arguments=[
            "--x",
            "0.0001",
            "--y",
            "-0.09405",
            "--z",
            "-0.01335",
            "--roll",
            "0.0",
            "--pitch",
            "-1.3963",
            "--yaw",
            "1.5708",
            "--frame-id",
            "tool0",
            "--child-frame-id",
            "camera_mount",
        ],
        condition=IfCondition(AndSubstitution(use_ur_driver, camera_enabled)),
    )

    camera_link_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_link_static_tf",
        output="screen",
        arguments=[
            "--x",
            "0.0042",
            "--y",
            "0.0175",
            "--z",
            "0.0125",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "camera_mount",
            "--child-frame-id",
            "camera_camera_link",
        ],
        condition=IfCondition(AndSubstitution(use_ur_driver, camera_enabled)),
    )

    # Point cloud downsampler for RViz performance
    # Subscribes to full-res point cloud and publishes downsampled version
    # This keeps RViz fast while grasp generation uses full resolution
    pointcloud_downsampler_node = Node(
        package="ur3e_gripper_3d_moveit_config",
        executable="pointcloud_downsampler",
        name="pointcloud_downsampler",
        output="screen",
        parameters=[
            {
                "voxel_size": 0.01,  # 1cm voxels
                "stride": 2,  # Process every 2nd point
                "max_publish_hz": 1.0,  # 1 Hz — slow enough to avoid RViz2 memory leak
                "input_topic": "/camera/camera/depth/color/points",
                "output_topic": "/camera/camera/depth/color/points_downsampled",
            }
        ],
        condition=IfCondition(AndSubstitution(use_real_camera, camera_pointcloud)),
    )

    ur_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "reverse_ip": reverse_ip,
            "description_package": "ur3e_gripper_3d_moveit_config",
            "description_file": "ur3e_gripper_wrapper.urdf.xacro",
            "kinematics_params_file": kinematics_params_file,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": "false",
            # Give the robot/controller more time to settle during cold starts.
            "controller_spawner_timeout": "30",
        }.items(),
        condition=IfCondition(use_ur_driver),
    )

    ur_driver_launch_delayed = TimerAction(period=ur_start_delay, actions=[ur_driver_launch])

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("gripper_3d"), "launch", "realsense_d345_hardware.launch.py"]
            )
        ),
        # Only start the camera driver when explicitly requested.
        # (camera_enabled only affects the robot model / TF, not the hardware driver.)
        condition=IfCondition(use_real_camera),
        launch_arguments={"pointcloud_enable": camera_pointcloud}.items(),
    )

    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([this_package_share, "launch", "move_group.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_fake_hardware": use_fake_hardware,
            "use_ur_driver": use_ur_driver,
            "camera_enabled": camera_enabled,
            # Pilz sequence planning (2026-09-16): the grasp executor plans
            # home -> pre-grasp -> grasp as ONE blended LIN+LIN trajectory.
            "capabilities": "pilz_industrial_motion_planner/MoveGroupSequenceAction "
                            "pilz_industrial_motion_planner/MoveGroupSequenceService",
        }.items(),
    )

    move_group_launch_delayed = TimerAction(period=moveit_start_delay, actions=[move_group_launch])

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([this_package_share, "launch", "moveit_rviz.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "camera_enabled": camera_enabled,
            "use_fake_hardware": use_fake_hardware,
        }.items(),
    )

    # start_rviz gate (2026-08-07, for the Morty GUI system launcher):
    # RViz was previously unconditional.
    rviz_launch_delayed = TimerAction(
        period=rviz_start_delay,
        actions=[rviz_launch],
        condition=IfCondition(LaunchConfiguration("start_rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument(
                "camera_enabled",
                default_value="true",
                description="Add the camera mount + camera link to the URDF (visualization/TF; does not start any camera driver).",
            ),
            DeclareLaunchArgument(
                "use_real_camera",
                default_value="false",
                description="Start the RealSense driver (gripper_3d realsense_d345_hardware.launch.py).",
            ),
            DeclareLaunchArgument(
                "camera_pointcloud",
                default_value="false",
                description="Let the RealSense driver compute its point cloud (+ downsampler) for RViz. "
                            "Costs a full core; the pipeline and GUI do not use it.",
            ),
            DeclareLaunchArgument(
                "use_ur_driver",
                default_value="false",
                description="Use `ur_robot_driver` (real robot / URSim / fake via ur_control.launch.py).",
            ),
            DeclareLaunchArgument(
                "use_fake_hardware",
                default_value="true",
                description="Use fake hardware (mock) instead of connecting to a robot.",
            ),
            DeclareLaunchArgument(
                "robot_ip",
                default_value="192.168.1.102",
                description="Robot/URSim IP (required when use_ur_driver:=true and use_fake_hardware:=false).",
            ),
            DeclareLaunchArgument(
                "ur_start_delay",
                default_value="5.0",
                description="Delay before launching the UR driver to avoid cold-start handshake races.",
            ),
            DeclareLaunchArgument(
                "moveit_start_delay",
                default_value="12.0",
                description="Delay before starting MoveIt so the UR driver and controllers can settle.",
            ),
            DeclareLaunchArgument(
                "rviz_start_delay",
                default_value="14.0",
                description="Delay before starting RViz so controller and TF startup have stabilized.",
            ),
            DeclareLaunchArgument(
                "reverse_ip",
                default_value="192.168.1.101",
                description="PC IP address reachable by the robot (UR reverse interface).",
            ),
            DeclareLaunchArgument("ur_type", default_value="ur3e"),
            DeclareLaunchArgument(
                "kinematics_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("ur3e_gripper_3d_moveit_config"), "config", "ur3e_calibration.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "use_joint_state_publisher_gui",
                default_value="false",
                description="Show joint sliders (leave false when using ros2_control to avoid multiple joint_states publishers).",
            ),
            DeclareLaunchArgument(
                "publish_environment_markers",
                default_value="true",
                description="Publish wall/flower markers like in `gripper_3d` demo RViz.",
            ),
            DeclareLaunchArgument(
                "publish_moveit_environment",
                default_value="true",
                description="Publish wall/flower as MoveIt collision objects on /planning_scene.",
            ),
            DeclareLaunchArgument(
                "use_standalone_ros2_control",
                default_value="true",
                description="Start a local ros2_control mock + controllers (ignored when use_ur_driver:=true).",
            ),
            DeclareLaunchArgument(
                "use_standalone_rsp",
                default_value="true",
                description="Start robot_state_publisher locally (ignored when use_ur_driver:=true).",
            ),
            DeclareLaunchArgument(
                "gripper_control_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("gripper_3d"), "config", "gripper_control.yaml"]
                ),
                description="Single source of truth for gripper IO/adapter parameters.",
            ),
            joint_state_publisher_gui_node,
            joint_state_publisher_node,
            robot_state_publisher_node,
            environment_markers_node,
            moveit_environment_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            ur3e_arm_controller_spawner,
            gripper_hw_controller_spawner,
            gripper_command_adapter_node_fake,
            gripper_command_adapter_node_real,
            camera_mount_tf,
            camera_link_tf,
            ur_driver_launch_delayed,
            realsense_launch,
            pointcloud_downsampler_node,
            move_group_launch_delayed,
            rviz_launch_delayed,
        ]
    )
