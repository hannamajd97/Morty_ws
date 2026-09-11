"""
grasp_executor_lin_llm.launch.py
================================

Launches the voice-driven variant of the LIN grasp executor: same LIN pick
logic as ``grasp_executor_lin``, plus an **interactive drop-pause** after
Step 3 (close gripper). The arm holds the object at the grasp pose until
``intent_router`` publishes a destination on
``/grasp_executor_lin_llm/resume`` (the chat layer sets this from the user's
follow-up "put it on the table" / "drop it in the box" instruction).

This is the executor ``morty_full.launch.py`` spawns — it pairs with the
chat / intent-routing flow. For non-voice usage where the drop is fixed,
prefer ``grasp_executor_lin.launch.py`` instead.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    camera_enabled = LaunchConfiguration("camera_enabled")

    moveit_config = (
        MoveItConfigsBuilder(
            "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
        )
        .robot_description(
            file_path="urdf/ur3e_gripper_wrapper.urdf.xacro",
            mappings={
                "use_fake_hardware": use_fake_hardware,
                "camera_enabled": camera_enabled,
            },
        )
        .to_moveit_configs()
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),
            DeclareLaunchArgument(
                "camera_enabled",
                default_value="true",
                description="Include camera_mount/camera_camera_link in robot_description.",
            ),
            DeclareLaunchArgument("planning_group", default_value="ur3e_manipulator"),
            DeclareLaunchArgument("end_effector_link", default_value="gripper_tcp"),
            DeclareLaunchArgument(
                "best_pose",
                default_value="false",
                description="When true, subscribe to /graspgen/best_pose_base_link instead of centroid.",
            ),
            DeclareLaunchArgument(
                "execute",
                default_value="true",
                description="Set to true to actually execute motions",
            ),
            DeclareLaunchArgument(
                "grasp_pose_topic",
                default_value=PythonExpression(
                    [
                        "'/graspgen/best_pose_base_link' if '",
                        LaunchConfiguration("best_pose"),
                        "' == 'true' else '/graspgen/centroid_base'"
                    ]
                ),
                description="Topic to subscribe for grasp poses",
            ),
            DeclareLaunchArgument(
                "use_candidate_pose_array",
                default_value="true",
                description="When true, subscribe to grasp candidate PoseArray and choose first LIN-feasible pose.",
            ),
            DeclareLaunchArgument(
                "candidate_pose_topic",
                default_value="/graspgen/candidate_poses_base_link",
                description="Topic to subscribe for grasp candidate poses (geometry_msgs/PoseArray).",
            ),
            DeclareLaunchArgument(
                "completion_topic",
                default_value="/grasp_executor_lin_llm/cycle_done",
                description="Status topic published by executor after each cycle ('done' or 'failed').",
            ),
            DeclareLaunchArgument(
                "selection_cycle_id_topic",
                default_value="/graspgen/selection_cycle_id",
                description="Cycle-id topic published by selector; used to tag completion status.",
            ),
            DeclareLaunchArgument(
                "execution_gate_topic",
                default_value="/grasp_executor_lin_llm/execution_enabled",
                description=(
                    "Latched std_msgs/Bool gate. False buffers a prepared "
                    "manual grasp; true permits execution."
                ),
            ),
            DeclareLaunchArgument(
                "max_candidate_tries",
                default_value="10",
                description="Maximum number of candidate poses to test with LIN planning.",
            ),
            DeclareLaunchArgument(
                "candidate_try_all_batches",
                default_value="true",
                description="When true, keep testing next candidate batches of size max_candidate_tries.",
            ),
            DeclareLaunchArgument(
                "pregrasp_distance",
                default_value="0.03",
                description="Distance to back off for pre-grasp (meters)",
            ),
            # Staged approach (2026-09-11): a joint (PTP) move to a stand-off
            # point in front of the object along the grasp's approach axis,
            # then the short straight LIN in. One long straight line from home
            # crosses the elbow singularity for far objects (Pilz: "Joint
            # velocity limit of elbow_joint violated"); the short line does not.
            # GUI-only sessions (2026-09-11): with both false a pick runs to
            # the end on its own — grasp, lift, home, drop at drop_target_name,
            # open, home — and the pipeline is free for the next click. The
            # voice router still sets wait_for_destination per pick when it
            # has a question to ask, so voice picks keep their pauses.
            DeclareLaunchArgument(
                "wait_for_destination",
                default_value="true",
                description="Pause after the grasp until a destination arrives on the resume topic.",
            ),
            DeclareLaunchArgument(
                "wait_for_release",
                default_value="true",
                description="Pause at the drop pose until a release confirmation arrives.",
            ),
            DeclareLaunchArgument(
                "use_pregrasp_staging",
                default_value="false",
                description="PTP to a stand-off in front of the grasp, then LIN for the last stretch.",
            ),
            DeclareLaunchArgument(
                "pregrasp_approach_distance",
                default_value="0.10",
                description="Stand-off distance along the grasp approach axis (meters).",
            ),
            DeclareLaunchArgument(
                "lift_height_m",
                default_value="0.07",
                description="Lift distance after grasp (meters).",
            ),
            DeclareLaunchArgument(
                "lift_table_yaw_deg",
                default_value="45.0",
                description="Fallback yaw (deg) for table +Y direction in base frame.",
            ),
            DeclareLaunchArgument(
                "lift_table_frame",
                default_value="",
                description="Optional table frame. If set, lift uses this frame's +Y axis.",
            ),
            DeclareLaunchArgument(
                "target_frame",
                default_value=PythonExpression(
                    [
                        "'base_link' if '",
                        LaunchConfiguration("best_pose"),
                        "' == 'true' else 'world'"
                    ]
                ),
                description="Frame to transform grasp poses into",
            ),
            DeclareLaunchArgument(
                "planning_time",
                default_value="5.0",
                description="Maximum planning time (seconds) for execution plans",
            ),
            DeclareLaunchArgument(
                "lin_feasibility_planning_time",
                default_value="1.5",
                description=(
                    "Tight per-candidate LIN feasibility budget (s). "
                    "Caps the time the LIN planner can churn on a hard pose "
                    "before reporting failure. Lower = faster failure loop."
                ),
            ),
            DeclareLaunchArgument(
                "lin_feasibility_attempts",
                default_value="1",
                description="Planning attempts per LIN feasibility check.",
            ),
            DeclareLaunchArgument(
                "sort_candidates_by_reachability",
                default_value="false",
                description=(
                    "Sort grasp candidates by Euclidean distance from current "
                    "TCP before LIN evaluation. The first feasible (closest) "
                    "grasp is usually found in the first 1-3 attempts."
                ),
            ),
            DeclareLaunchArgument(
                "enable_wrist_twin_rewrite",
                default_value="false",
                description=(
                    "Allow the executor to rotate candidate TCP orientations "
                    "180 degrees about their approach axis. Disabled by default "
                    "so execution preserves the pipeline's filtered pose."
                ),
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
            DeclareLaunchArgument(
                "one_shot",
                default_value="false",
                description="Exit after first execution",
            ),
            DeclareLaunchArgument(
                "velocity_scaling",
                default_value="0.2",
                description="Max velocity scaling factor (0.0-1.0)",
            ),
            DeclareLaunchArgument(
                "acceleration_scaling",
                default_value="0.1",
                description="Max acceleration scaling factor (0.0-1.0)",
            ),
            DeclareLaunchArgument(
                "drop_hold_s",
                default_value="5.0",
                description="Time to hold at drop target (seconds)",
            ),
            DeclareLaunchArgument(
                "candidate_wait_timeout_s",
                default_value="0.35",
                description=(
                    "When candidate mode is enabled, wait this long for PoseArray "
                    "before falling back to single grasp pose."
                ),
            ),
            DeclareLaunchArgument(
                "stop_after_grasp",
                default_value="false",
                description=(
                    "Manual-button mode: grasp and HOLD at the grasp pose; "
                    "no lift/transport/drop/reset."
                ),
            ),
            DeclareLaunchArgument(
                "drop_target_name",
                default_value="drop_right_side",
                description="Named arm target used as drop pose",
            ),
            DeclareLaunchArgument(
                "reset_target_name",
                default_value="home",
                description="Named arm target used after drop sequence",
            ),
            DeclareLaunchArgument(
                "gripper_open_name",
                default_value="open",
                description="Named gripper state for opening",
            ),
            DeclareLaunchArgument(
                "gripper_close_name",
                default_value="close",
                description="Named gripper state for closing",
            ),
            DeclareLaunchArgument(
                "override_gripper_pulse_from_executor",
                default_value="false",
                description="When true, set gripper adapter pulse_hold_s_open/close from this launch.",
            ),
            DeclareLaunchArgument(
                "gripper_adapter_node_name",
                default_value="/gripper_controller/gripper_command_adapter",
                description="Target adapter node name for runtime pulse override.",
            ),
            DeclareLaunchArgument(
                "executor_pulse_hold_s_open",
                default_value="-1.0",
                description="Open pulse override in seconds (>0 enables open override).",
            ),
            DeclareLaunchArgument(
                "executor_pulse_hold_s_close",
                default_value="-1.0",
                description="Close pulse override in seconds (>0 enables close override).",
            ),
            DeclareLaunchArgument(
                "retreat_base_minus_z_m",
                default_value="0.10",
                description="Linear retreat distance along base-frame -Z before moving to drop pose (meters)",
            ),
            Node(
                package="ur3e_gripper_3d_moveit_commander",
                executable="grasp_executor_lin_llm",
                # A dead executor must take its launch down with it: the GUI
                # watches the launch process, and a wrapper that outlives a
                # crashed node hid the crash (2026-09-11: exit -11, "no
                # logs, nothing moves") and blocked the GUI's self-heal restart.
                on_exit=Shutdown(),
                # name= REMOVED (2026-08-26): a launch name override is a
                # GLOBAL __node remap — it renamed the in-process helper
                # node too, creating a bare twin "/grasp_executor_lin_llm"
                # that hijacked set_parameters (wait_for_destination
                # "not declared" → one-shot picks silently kept the pause).
                # The binary names its main node itself.
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {
                        "planning_group": LaunchConfiguration("planning_group"),
                        "end_effector_link": LaunchConfiguration("end_effector_link"),
                        "execute": LaunchConfiguration("execute"),
                        "grasp_pose_topic": LaunchConfiguration("grasp_pose_topic"),
                        "use_candidate_pose_array": LaunchConfiguration("use_candidate_pose_array"),
                        "candidate_pose_topic": LaunchConfiguration("candidate_pose_topic"),
                        "completion_topic": LaunchConfiguration("completion_topic"),
                        "selection_cycle_id_topic": LaunchConfiguration("selection_cycle_id_topic"),
                        "execution_gate_topic": LaunchConfiguration("execution_gate_topic"),
                        "max_candidate_tries": LaunchConfiguration("max_candidate_tries"),
                        "candidate_try_all_batches": LaunchConfiguration("candidate_try_all_batches"),
                        "pregrasp_distance": LaunchConfiguration("pregrasp_distance"),
                        "wait_for_destination": LaunchConfiguration("wait_for_destination"),
                        "wait_for_release": LaunchConfiguration("wait_for_release"),
                        "use_pregrasp_staging": LaunchConfiguration("use_pregrasp_staging"),
                        "pregrasp_approach_distance": LaunchConfiguration("pregrasp_approach_distance"),
                        "lift_height_m": LaunchConfiguration("lift_height_m"),
                        "lift_table_yaw_deg": LaunchConfiguration("lift_table_yaw_deg"),
                        "lift_table_frame": LaunchConfiguration("lift_table_frame"),
                        "target_frame": LaunchConfiguration("target_frame"),
                        "planning_time": LaunchConfiguration("planning_time"),
                        "lin_feasibility_planning_time": LaunchConfiguration("lin_feasibility_planning_time"),
                        "lin_feasibility_attempts": LaunchConfiguration("lin_feasibility_attempts"),
                        "sort_candidates_by_reachability": LaunchConfiguration("sort_candidates_by_reachability"),
                        "enable_wrist_twin_rewrite": LaunchConfiguration("enable_wrist_twin_rewrite"),
                        "num_attempts": LaunchConfiguration("num_attempts"),
                        "tf_timeout": LaunchConfiguration("tf_timeout"),
                        "one_shot": LaunchConfiguration("one_shot"),
                        "velocity_scaling": LaunchConfiguration("velocity_scaling"),
                        "acceleration_scaling": LaunchConfiguration("acceleration_scaling"),
                        "drop_hold_s": LaunchConfiguration("drop_hold_s"),
                        "candidate_wait_timeout_s": LaunchConfiguration("candidate_wait_timeout_s"),
                        "stop_after_grasp": LaunchConfiguration("stop_after_grasp"),
                        "drop_target_name": LaunchConfiguration("drop_target_name"),
                        "reset_target_name": LaunchConfiguration("reset_target_name"),
                        "gripper_open_name": LaunchConfiguration("gripper_open_name"),
                        "gripper_close_name": LaunchConfiguration("gripper_close_name"),
                        "override_gripper_pulse_from_executor": LaunchConfiguration("override_gripper_pulse_from_executor"),
                        "gripper_adapter_node_name": LaunchConfiguration("gripper_adapter_node_name"),
                        "executor_pulse_hold_s_open": LaunchConfiguration("executor_pulse_hold_s_open"),
                        "executor_pulse_hold_s_close": LaunchConfiguration("executor_pulse_hold_s_close"),
                        "retreat_base_minus_z_m": LaunchConfiguration("retreat_base_minus_z_m"),
                    },
                ],
            ),
        ]
    )
