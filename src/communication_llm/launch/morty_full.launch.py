"""Single-command bringup for the whole Morty stack.

Brings up — in one terminal — everything needed for voice-driven picks:

  - voice_robot.launch.py         (TTS + STT + LLM + intent router +
                                    named-pose server + replay node +
                                    gripper bridge + mic visualizer)
  - grasp_pipeline                (YOLO detection + segmentation + grasp
                                    generation client). Starts with
                                    registration DISABLED — single view
                                    is the default for snappy picks; say
                                    "enable multi-view" to flip it on
                                    at runtime without restarting.
  - grasp_executor_lin            (MoveIt LIN executor that picks the
                                    object and drops it).

This launch does NOT bring up the UR3e driver / MoveIt; run your usual
robot bringup in a separate terminal first.

Examples:

    # Default — single-view + voice
    ros2 launch communication_llm morty_full.launch.py

    # Start with multi-view already on
    ros2 launch communication_llm morty_full.launch.py registration_enabled:=true

    # No voice (mouse-click only); useful for headless / debugging
    ros2 launch communication_llm morty_full.launch.py speech_input_enabled:=false \\
                                                       mic_visualizer:=false

NOTE: We pass with_replay:=false into grasp_pipeline because
voice_robot.launch.py is the sole owner of the replay node. Spawning
two would collide on the controller action client and one would
cancel the other.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # ── launch-time toggles ─────────────────────────────────────────────
    registration_enabled = LaunchConfiguration("registration_enabled")
    speech_input_enabled = LaunchConfiguration("speech_input_enabled")
    mic_visualizer = LaunchConfiguration("mic_visualizer")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")

    # Both voice_robot.launch.py and run_grasp_pipeline.launch.py used to
    # try to spawn the replay node and we relied on `with_replay` flags to
    # arbitrate. In practice the per-include LaunchConfigurations leaked
    # across siblings on some launch versions and the replay node ended up
    # not starting at all — silent no-op named-pose moves. The robust fix
    # is to OWN the replay node here in the umbrella and tell BOTH children
    # `with_replay:=false`.
    voice_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("communication_llm"),
                 "launch", "voice_robot.launch.py"]
            )
        ),
        launch_arguments={
            "speech_input_enabled": speech_input_enabled,
            "mic_visualizer": mic_visualizer,
            "kill_stragglers": "true",
            "with_replay": "false",
        }.items(),
    )

    grasp_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("grasp_pipeline"),
                 "launch", "run_grasp_pipeline.launch.py"]
            )
        ),
        launch_arguments={
            "registration_enabled": registration_enabled,
            "with_replay": "false",
            # Silence the open3d viewer's repeating "WAITING for first cloud"
            # spam. 1e9 seconds = effectively never. The viewer still logs
            # on actual status TRANSITIONS, just not periodically.
            "registration_viewer_status_log_period_s": "1000000000.0",
            # KEEP YOLO running while waiting for a selection. The default
            # pauses inference once the pipeline enters "waiting for user
            # to pick" mode — but that starves chat_node of fresh scene
            # context, so "what do you see?" mid-conversation gets "I
            # don't have detections". With this off, inference keeps
            # ticking and /yolo/all_detections stays live, while the
            # selection thread independently waits for a click/voice.
            "suspend_inference_while_selecting": "false",
            # Make the pipeline re-enter its selection cycle every ~1s while
            # waiting for input. Default is -1 (infinite), which means once
            # the manual-wait loop is entered, auto_select_label changes are
            # ignored until something else (click, finger-point) ends the
            # cycle. With 1s, the loop times out, the outer cycle restarts,
            # and auto_select_label is re-evaluated — so a voice-set label
            # actually lands within ~1s instead of hanging forever.
            "selection_manual_wait_s": "1.0",
            # The umbrella runs grasp_executor_lin_llm (interactive
            # variant), which publishes on /grasp_executor_lin_llm/...
            # instead of /grasp_executor_lin/.... Tell the pipeline to
            # listen on the new topic — otherwise the selection thread
            # never sees 'cycle_done', never resets, and detection
            # silently stops after the first pick.
            "executor_cycle_done_topic": "/grasp_executor_lin_llm/cycle_done",
            # Per-class vitamin_c override. Combined with the new
            # MULTI_INSTANCE_LABELS filter in grasp_selection_node
            # (vitamin_tube keeps all detections, every other class
            # collapses to best-per-class), this surfaces the 2nd
            # vitamin tube at 0.02 confidence WITHOUT polluting other
            # classes with ghost detections — those are filtered out
            # by the best-per-class collapse before reaching the LLM.
            "per_class_min_conf_labels": "['vitamin_c']",
            "per_class_min_conf_values": "[0.02]",
        }.items(),
    )

    # Pre-kill any stale replay_node from a previous launch. Uses pgrep so
    # the kill pattern doesn't match this bash itself (SELF=$$ excluded).
    # Removed from voice_robot's pre_kill list since the umbrella now owns
    # the replay node — having BOTH kill it caused a race where voice_robot's
    # pre_kill SIGKILLed the freshly-spawned replay_node 200ms after birth.
    replay_pre_kill = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            (
                "SELF=$$; "
                "for pid in $(pgrep -f ur3e_fast_observation_replay 2>/dev/null); do "
                "  if [ \"$pid\" != \"$SELF\" ]; then "
                "    kill -9 \"$pid\" 2>/dev/null || true; "
                "  fi; "
                "done; "
                "echo 'morty_full: replay pre-kill done.'; "
                "exit 0"
            ),
        ],
        output="screen",
    )

    replay_node = Node(
        package="ur3e_fast_observation",
        executable="replay_node",
        name="ur3e_fast_observation_replay",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "poses_file": PathJoinSubstitution([
                    FindPackageShare("ur3e_fast_observation"),
                    "config", "observation_poses.yaml",
                ]),
                "controller_action_ns": "/scaled_joint_trajectory_controller",
                "move_duration_s": ParameterValue(2.0, value_type=float),
                "default_pose_name": "",
            }
        ],
    )

    # Spawn replay_node only AFTER replay_pre_kill has exited, otherwise the
    # pkill races with the brand-new process and kills it at birth.
    replay_after_kill = RegisterEventHandler(
        OnProcessExit(
            target_action=replay_pre_kill,
            on_exit=[replay_node],
        )
    )

    # Voice-driven variant of grasp_executor_lin. Same LIN pick logic
    # plus an interactive drop pause after Step 3 (close gripper) — the
    # arm holds the object at the grasp pose until intent_router pushes
    # a destination on /grasp_executor_lin_llm/resume. The plain
    # grasp_executor_lin binary is still built for backward compat but
    # is NOT spawned by morty_full.launch.py.
    grasp_executor = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur3e_gripper_3d_moveit_commander"),
                 "launch", "grasp_executor_lin_llm.launch.py"]
            )
        ),
        launch_arguments={
            "use_fake_hardware": use_fake_hardware,
            # The communication side already routes voice/intents through
            # /yolo/select_index and /yolo_live_select_grasp params; let
            # the executor consume the grasp poses produced by the
            # pipeline.
            "best_pose": "true",
            "use_candidate_pose_array": "true",
            "execute": "true",
            # Bump the executor's PoseArray wait timeout. Default 0.35s
            # is far too short: the pipeline takes ~10-15s to segment +
            # generate grasps + run the filter chain. With the default,
            # the executor times out, falls back to the SINGLE grasp
            # pose, and that single pose often fails LIN pre-check with
            # NO_IK_SOLUTION (especially when picking from the far side
            # of the workspace, e.g. wooden_maze_puzzle on the left
            # while arm is at detect_right). 20s gives the pipeline
            # plenty of time to publish all 50 candidates so the
            # executor can pick the first feasible one.
            "candidate_wait_timeout_s": "20.0",
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "registration_enabled",
                default_value="false",
                description="Start with multi-view registration OFF "
                "(single-view picks). Flip on at runtime by voice "
                "('enable multi-view') or with `ros2 param set "
                "/yolo_live_select_grasp registration_enabled true`.",
            ),
            DeclareLaunchArgument(
                "speech_input_enabled",
                default_value="true",
                description="Enable the speech_input_node (faster-whisper).",
            ),
            DeclareLaunchArgument(
                "mic_visualizer",
                default_value="true",
                description="Show the OpenCV mic visualiser window.",
            ),
            DeclareLaunchArgument(
                "use_fake_hardware",
                default_value="false",
                description="Pass through to grasp_executor_lin; "
                "set true if you're running a fake UR3e.",
            ),
            voice_robot,
            grasp_pipeline,
            grasp_executor,
            replay_pre_kill,
            replay_after_kill,
        ]
    )
