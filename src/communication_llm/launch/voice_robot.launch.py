"""Single-command bringup for the communication side of the voice-robot.

Brings up the seven processes that translate user speech into robot actions:
  - speak_action_server          (Kokoro TTS)
  - speech_input_node            (Vosk or faster-whisper STT)
  - gripper_action_server        (OperateGripper bridge to the gripper driver)
  - move_to_named_pose_server    (MoveToNamedPose bridge to the replay node)
  - ur3e_fast_observation_replay (the actual driver of named-pose moves)
  - intent_router_node           (routes /robot/intent to the right action)
  - chat_node                    (LLM conversation + intent extraction)

This launch does NOT bring up the UR driver / MoveIt or perception
(grasp_pipeline). Run the UR driver / MoveIt yourself for any physical
motion. The communication side here covers chat, gripper open/close, and
named-pose moves (home, ready, look_left/right/up). Add grasp_pipeline +
grasp_executor_lin separately only when you want full picks.

Examples:

    # Default — everything voice + typing (chat_node has terminal input)
    ros2 launch communication_llm voice_robot.launch.py

    # Voice only — no chat_node terminal stdin (e.g. you want to type at a
    # SEPARATE terminal). Then run `ros2 run communication_llm chat_node`
    # yourself in that terminal.
    ros2 launch communication_llm voice_robot.launch.py with_chat:=false

    # Pick a different home pose
    ros2 launch communication_llm voice_robot.launch.py home_pose_name:=detect_higher

    # Disable TTS (everything still routes; you just won't hear narration)
    ros2 launch communication_llm voice_robot.launch.py tts_enabled:=false

OPENAI_API_KEY-style env var: chat_node reads the api_key from os.environ
inside llm_service.py. Make sure your shell has the key exported BEFORE
running this launch, otherwise chat_node will fail to start.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # ── launch arguments ────────────────────────────────────────────────
    tts_enabled = LaunchConfiguration("tts_enabled")
    speech_input_enabled = LaunchConfiguration("speech_input_enabled")
    mic_visualizer = LaunchConfiguration("mic_visualizer")
    with_chat = LaunchConfiguration("with_chat")
    auto_execute_grasp = LaunchConfiguration("auto_execute_grasp")
    home_pose_name = LaunchConfiguration("home_pose_name")
    ready_pose_name = LaunchConfiguration("ready_pose_name")
    voice = LaunchConfiguration("voice")
    speech_input_mic_index = LaunchConfiguration("mic_index")
    mic_device_name = LaunchConfiguration("mic_device_name")
    known_labels = LaunchConfiguration("known_labels")
    mute_during_tts = LaunchConfiguration("mute_during_tts")
    vad_enabled = LaunchConfiguration("vad_enabled")
    vad_aggressiveness = LaunchConfiguration("vad_aggressiveness")
    wake_word_enabled = LaunchConfiguration("wake_word_enabled")
    wake_words = LaunchConfiguration("wake_words")
    stt_engine = LaunchConfiguration("stt_engine")
    whisper_model = LaunchConfiguration("whisper_model")
    whisper_language = LaunchConfiguration("whisper_language")
    # Voice-filter tunables exposed for the GUI (2026-08-11).
    whisper_min_rms = LaunchConfiguration("whisper_min_rms")
    whisper_silence_ms = LaunchConfiguration("whisper_silence_ms")
    whisper_max_utterance_s = LaunchConfiguration("whisper_max_utterance_s")
    min_chars = LaunchConfiguration("min_chars")
    with_replay = LaunchConfiguration("with_replay")
    with_named_pose_server = LaunchConfiguration("with_named_pose_server")
    replay_poses_file = LaunchConfiguration("replay_poses_file")
    replay_controller_ns = LaunchConfiguration("replay_controller_ns")
    replay_move_duration_s = LaunchConfiguration("replay_move_duration_s")
    auto_aec = LaunchConfiguration("auto_aec")
    pulse_source = LaunchConfiguration("pulse_source")
    pulse_sink = LaunchConfiguration("pulse_sink")
    kill_stragglers = LaunchConfiguration("kill_stragglers")
    collision_check = LaunchConfiguration("collision_check")

    # ── nodes ───────────────────────────────────────────────────────────
    force_pyttsx3 = LaunchConfiguration("force_pyttsx3")

    speak_node = Node(
        package="communication_llm",
        executable="speak_action_server",
        name="speak_action_server",
        output="screen",
        parameters=[{"voice": voice, "force_pyttsx3": force_pyttsx3}],
    )

    speech_input = Node(
        package="communication_llm",
        executable="speech_input_node",
        name="speech_input_node",
        output="screen",
        parameters=[
            {
                "mic_index": speech_input_mic_index,
                "mic_device_name": mic_device_name,
                "vad_enabled": vad_enabled,
                "vad_aggressiveness": vad_aggressiveness,
                "wake_word_enabled": wake_word_enabled,
                "wake_words": wake_words,
                "stt_engine": stt_engine,
                "whisper_model": whisper_model,
                "whisper_language": whisper_language,
                "whisper_device": LaunchConfiguration("whisper_device"),
                "whisper_compute_type": LaunchConfiguration("whisper_compute_type"),
                "whisper_min_rms": ParameterValue(whisper_min_rms, value_type=float),
                "whisper_silence_ms": ParameterValue(whisper_silence_ms, value_type=int),
                "whisper_max_utterance_s": ParameterValue(
                    whisper_max_utterance_s, value_type=float
                ),
                "min_chars": ParameterValue(min_chars, value_type=int),
            }
        ],
        condition=IfCondition(speech_input_enabled),
    )

    # Small OpenCV window with a live level bar + history + state badge
    # so the user can see the STT is actually hearing them. Subscribes to
    # /speech/level and /speech/state. Disable with mic_visualizer:=false.
    mic_viz = Node(
        package="communication_llm",
        executable="mic_visualizer_node",
        name="mic_visualizer_node",
        output="screen",
        condition=IfCondition(mic_visualizer),
    )

    gripper = Node(
        package="ur3e_gripper_3d_moveit_commander",
        executable="gripper_action_server",
        name="gripper_action_server",
        output="screen",
    )

    # with_named_pose_server:=false when the GUI already runs the server
    # with the driver (2026-09-16): the grasp executor needs it for every
    # drop, voice or not, and two instances would race for the replay node.
    named_pose = Node(
        package="ur3e_fast_observation",
        executable="move_to_named_pose_server",
        name="move_to_named_pose_server",
        output="screen",
        parameters=[{"collision_check": collision_check}],
        condition=IfCondition(with_named_pose_server),
    )

    router = Node(
        package="intent_router",
        executable="intent_router_node",
        name="intent_router_node",
        output="screen",
        parameters=[
            {
                "home_pose_name": home_pose_name,
                "ready_pose_name": ready_pose_name,
                "known_labels": known_labels,
            }
        ],
    )

    chat = Node(
        package="communication_llm",
        executable="chat_node",
        name="communication_llm_chat_node",
        output="screen",
        emulate_tty=True,  # so chat_node's "User: " prompt + input() work
        parameters=[
            {
                "tts_enabled": tts_enabled,
                "auto_execute_grasp": auto_execute_grasp,
                "speech_input_enabled": speech_input_enabled,
                "mute_during_tts": mute_during_tts,
            }
        ],
        condition=IfCondition(with_chat),
    )

    # The "replay" node executes named-pose moves (home, ready, look_left/right/up).
    # Brought up here so named-pose voice commands work WITHOUT having to also
    # launch grasp_pipeline. Toggle with with_replay:=false if you already
    # launch run_grasp_pipeline.launch.py separately (which also starts a
    # replay_node — running two would cause a name clash).
    replay_node = Node(
        package="ur3e_fast_observation",
        executable="replay_node",
        name="ur3e_fast_observation_replay",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "poses_file": replay_poses_file,
                "controller_action_ns": replay_controller_ns,
                "move_duration_s": ParameterValue(
                    replay_move_duration_s, value_type=float
                ),
                "default_pose_name": "",
            }
        ],
        condition=IfCondition(with_replay),
    )

    # Auto-load PulseAudio AEC at launch start. Idempotent — if it's already
    # loaded, this is a no-op. Skipped if auto_aec:=false.
    # Pre-flight cleanup: kill any stale instances of our voice-robot nodes
    # from a previous launch. Always runs (so the OnProcessExit chain
    # below has a target_action that actually exits), but its body is
    # conditional on the `kill_stragglers` launch arg.
    pre_kill = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            (
                # Exclude our own PID from pkill so the pattern can't match
                # the very bash that runs it. Previously this script would
                # SIGKILL itself (exit code -9), which broke the OnProcessExit
                # chain — the replay node and other follow-on nodes never
                # started.
                "if [ \"$KILL_STRAGGLERS\" = \"true\" ]; then "
                "  SELF=$$; "
                "  for name in speak_action_server speech_input_node chat_node "
                "  intent_router_node gripper_action_server "
                "  move_to_named_pose_server "
                "  mic_visualizer_node; do "
                "    for pid in $(pgrep -f \"$name\" 2>/dev/null); do "
                "      if [ \"$pid\" != \"$SELF\" ]; then "
                "        kill -9 \"$pid\" 2>/dev/null || true; "
                "      fi; "
                "    done; "
                "  done; "
                "  echo 'voice_robot.launch: stragglers cleared.'; "
                "  sleep 0.5; "
                "else "
                "  echo 'voice_robot.launch: kill_stragglers=false, skipping cleanup.'; "
                "fi"
            ),
        ],
        output="screen",
        additional_env={"KILL_STRAGGLERS": kill_stragglers},
    )

    aec_loader = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            (
                # Idempotent AEC load. Uses the minimal arg set that's
                # known to work on PulseAudio across versions; advanced
                # tuning (noise suppression, AGC) can be added later if
                # needed via `pactl unload-module module-echo-cancel`
                # then a custom load with extra `aec_args=...`.
                "if ! command -v pactl >/dev/null 2>&1; then "
                "  echo 'voice_robot.launch: pactl not found, skipping AEC.'; "
                "elif pactl list short modules | grep -q module-echo-cancel; then "
                "  echo 'voice_robot.launch: AEC already loaded.'; "
                "elif pactl load-module module-echo-cancel "
                "       aec_method=webrtc "
                "       source_name=echocancel_mic "
                "       sink_name=echocancel_spk >/dev/null; then "
                "  echo 'voice_robot.launch: AEC module loaded.'; "
                "else "
                "  echo 'voice_robot.launch: AEC load failed (carrying on without AEC).'; "
                "fi"
            ),
        ],
        output="screen",
        condition=IfCondition(auto_aec),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tts_enabled",
                default_value="true",
                description="Have chat_node speak its replies via /speak.",
            ),
            DeclareLaunchArgument(
                "force_pyttsx3",
                default_value="false",
                description="Skip Kokoro and use pyttsx3 only. ~10x faster on "
                "CPU at the cost of a robotic voice. Set true for debugging "
                "on a VM with no GPU.",
            ),
            DeclareLaunchArgument(
                "speech_input_enabled",
                default_value="true",
                description="Run the Vosk speech-input node and wire chat_node to it.",
            ),
            DeclareLaunchArgument(
                "mic_visualizer",
                default_value="true",
                description="Show a small OpenCV window with a live mic "
                "level bar + waveform + state badge so the user can SEE "
                "the STT is hearing them. Subscribes to /speech/level + "
                "/speech/state published by speech_input_node.",
            ),
            DeclareLaunchArgument(
                "with_chat",
                default_value="true",
                description="Include chat_node in this launch. Set false to run "
                "chat_node yourself in a separate terminal (cleaner for typing).",
            ),
            DeclareLaunchArgument(
                "auto_execute_grasp",
                default_value="false",
                description="Have chat_node directly fire ExecuteGrasp on pick "
                "intents. Leave false when using intent_router + grasp_pipeline.",
            ),
            DeclareLaunchArgument(
                "home_pose_name",
                default_value="home",
                description="Named pose for 'go home' intents. Default 'home' "
                "is the working / ready-to-grasp pose (formerly 'initgrasp').",
            ),
            DeclareLaunchArgument(
                "ready_pose_name",
                default_value="home",
                description="Named pose for 'go to ready' / 'start position' "
                "intents. Defaults to the same 'home' pose; override if you "
                "want them distinct.",
            ),
            DeclareLaunchArgument(
                "voice",
                default_value="bf_emma",
                description="Kokoro voice id used by speak_action_server.",
            ),
            DeclareLaunchArgument(
                "mic_index",
                default_value="-1",
                description="PyAudio mic device index. -1 = default mic.",
            ),
            DeclareLaunchArgument(
                "known_labels",
                default_value=(
                    "['tape_dispenser','cd_case','toy_faucet',"
                    "'kitchen_towel','nutmeg_jar','retail_box',"
                    "'small_bottle','wooden_maze_puzzle','vitamin_c']"
                ),
                description="YOLO class labels intent_router will match against.",
            ),
            DeclareLaunchArgument(
                "mute_during_tts",
                default_value="false",
                description="If true, mute the mic while the AI speaks. "
                "Default false enables barge-in (interrupt the AI mid-speech). "
                "The echo content filter prevents the AI from hearing itself "
                "when AEC isn't available.",
            ),
            DeclareLaunchArgument(
                "mic_device_name",
                default_value="",
                description="Optional: substring of the mic device name "
                "(PortAudio level) to select. Leave empty when using "
                "PulseAudio (the PULSE_SOURCE env var below routes through "
                "the AEC source automatically). Set this only for direct "
                "USB array selection, e.g. 'ReSpeaker'.",
            ),
            DeclareLaunchArgument(
                "vad_enabled",
                default_value="true",
                description="Layer 2: gate Vosk with WebRTC VAD so silence "
                "and steady noise don't get transcribed. Requires webrtcvad.",
            ),
            DeclareLaunchArgument(
                "vad_aggressiveness",
                default_value="2",
                description="WebRTC VAD aggressiveness 0-3 (3 filters most).",
            ),
            DeclareLaunchArgument(
                "wake_word_enabled",
                default_value="false",
                description="Layer 3: only forward transcripts that start "
                "with a wake word OR are in the always_allow set. The wake "
                "prefix is stripped from the forwarded text.",
            ),
            DeclareLaunchArgument(
                "wake_words",
                default_value="['morty','hey morty']",
                description="List of wake-word phrases (case-insensitive). "
                "The robot's name is Morty.",
            ),
            DeclareLaunchArgument(
                "stt_engine",
                default_value="whisper",
                description="STT backend: 'whisper' (faster-whisper, default, "
                "much better accuracy, VAD-segmented) or 'vosk' (Kaldi-based, "
                "streaming, lower accuracy). Whisper downloads its model on "
                "first run; once cached, startup is fast.",
            ),
            DeclareLaunchArgument(
                "whisper_min_rms",
                default_value="0.025",
                description="Minimum RMS energy for a buffer to count as speech.",
            ),
            DeclareLaunchArgument(
                "whisper_silence_ms",
                default_value="300",
                description="Silence (ms) that finalises an utterance.",
            ),
            DeclareLaunchArgument(
                "whisper_max_utterance_s",
                default_value="12.0",
                description="Maximum utterance length in seconds.",
            ),
            DeclareLaunchArgument(
                "min_chars",
                default_value="2",
                description="Minimum transcript length to accept.",
            ),
            DeclareLaunchArgument(
                "whisper_language",
                default_value="en",
                description="Transcription language for whisper ('en', 'de', ...). "
                "Needs a multilingual model (whisper_model:=base) for non-English.",
            ),
            DeclareLaunchArgument(
                "whisper_device",
                default_value="auto",
                description="faster-whisper device: auto / cpu / cuda. "
                "The GUI passes cuda (GTX 1080 has ~6.7 GB free next to "
                "YOLO; large-v3-turbo int8 needs ~1.5 GB).",
            ),
            DeclareLaunchArgument(
                "whisper_compute_type",
                default_value="auto",
                description="CTranslate2 compute type (int8 recommended "
                "on Pascal GPUs — fp16 is crippled there).",
            ),
            DeclareLaunchArgument(
                "whisper_model",
                default_value="base",
                description="faster-whisper model size. 'base' is multilingual "
                "(handles German + English + others). For English-only and "
                "smaller, use 'base.en'. Sizes: tiny (~75MB), base (~140MB, "
                "DEFAULT, multilingual), small (~470MB), medium (~1.5GB), "
                "large-v3 (~2.9GB, GPU recommended).",
            ),
            DeclareLaunchArgument("with_named_pose_server", default_value="true",
                              description="Start move_to_named_pose_server (false when the GUI runs it with the driver)."),
            DeclareLaunchArgument(
                "with_replay",
                default_value="true",
                description="Start the ur3e_fast_observation_replay node so "
                "named-pose moves (home / ready / look_*) work WITHOUT "
                "needing run_grasp_pipeline.launch.py. Set false if you "
                "launch grasp_pipeline separately, to avoid a duplicate node.",
            ),
            DeclareLaunchArgument(
                "replay_poses_file",
                default_value=PathJoinSubstitution([
                    FindPackageShare("ur3e_fast_observation"),
                    "config", "observation_poses.yaml",
                ]),
                description="YAML file with named joint poses for the replay node.",
            ),
            DeclareLaunchArgument(
                "replay_controller_ns",
                default_value="/scaled_joint_trajectory_controller",
                description="Namespace of the joint trajectory controller "
                "the replay node sends FollowJointTrajectory goals to.",
            ),
            DeclareLaunchArgument(
                "replay_move_duration_s",
                default_value="2.0",
                description="Default move duration the replay node uses if "
                "the pose YAML doesn't specify one.",
            ),
            DeclareLaunchArgument(
                "auto_aec",
                default_value="true",
                description="Auto-load PulseAudio's WebRTC AEC at launch "
                "start (idempotent). Set false if you manage AEC yourself "
                "or your system doesn't have PulseAudio.",
            ),
            DeclareLaunchArgument(
                "kill_stragglers",
                default_value="true",
                description="Pre-flight: pkill any stale Morty nodes from a "
                "prior launch before starting fresh. Prevents duplicate-server "
                "issues. Set false only if you intentionally have external "
                "instances of these nodes running.",
            ),
            DeclareLaunchArgument(
                "collision_check",
                default_value="true",
                description="Pre-flight collision check on every named-pose "
                "move: linearly interpolate the joint trajectory and ask "
                "MoveIt's /check_state_validity service whether each waypoint "
                "is in collision against the planning scene (table, walls, "
                "mount). Refuses the move if it would crash. Set false to "
                "disable (e.g. when running without MoveIt).",
            ),
            DeclareLaunchArgument(
                "pulse_source",
                default_value="echocancel_mic",
                description="Force PulseAudio source for mic input via the "
                "PULSE_SOURCE env var. Routes the mic through this exact "
                "source regardless of pavucontrol's default. Set to '' to "
                "leave the system default.",
            ),
            DeclareLaunchArgument(
                "pulse_sink",
                default_value="echocancel_spk",
                description="Force PulseAudio sink for TTS output via the "
                "PULSE_SINK env var. Routes Kokoro audio through this exact "
                "sink. Set to '' to leave the system default.",
            ),
            # Set the env vars BEFORE any pyaudio process starts so they
            # take effect. Empty values are still set; PulseAudio falls back
            # to system default in that case.
            SetEnvironmentVariable("PULSE_SOURCE", pulse_source),
            SetEnvironmentVariable("PULSE_SINK", pulse_sink),
            pre_kill,
            # Defer launching the actual nodes until pre_kill finishes,
            # otherwise the pkill -9 -f races with the freshly-spawned
            # processes (its `-f` pattern matches their cmdlines) and
            # kills them at birth. With this handler, the kill is fully
            # complete before any node starts.
            RegisterEventHandler(
                OnProcessExit(
                    target_action=pre_kill,
                    on_exit=[
                        aec_loader,
                        speak_node,
                        speech_input,
                        gripper,
                        named_pose,
                        replay_node,
                        router,
                        chat,
                        mic_viz,
                    ],
                )
            ),
        ]
    )
