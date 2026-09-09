#!/usr/bin/env bash
# Morty launcher — one command, guaranteed clean shutdown.
#
# Daily usage:
#   cd ~/morty_ws
#   ./morty.sh                  # OpenAI gpt-5.4-mini + REAL hardware (production)
#   ./morty.sh --debug          # Bedrock Claude Haiku 4.5 + faster STT/TTS
#   ./morty.sh --fake           # Same provider as above, but fake hardware
#   ./morty.sh --voice-only     # Same provider, voice/chat only (no grasp)
#
# Flags can be combined freely, e.g.
#   ./morty.sh --debug --fake             # Claude + fake robot (today's setup)
#   ./morty.sh --debug --voice-only       # Just chat with Claude
#
# Ctrl-C kills EVERYTHING — no orphan processes left behind.

# NOTE: do not use `set -u` here — ROS2 setup scripts reference unset env
# vars (e.g. AMENT_TRACE_SETUP_FILES) and would crash this wrapper.

# Handle --help / unknown flags BEFORE the cleanup/source block so the user
# gets help even without ROS sourced.
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            cat <<EOF
Usage: $0 [option] [option ...]

LLM provider (pick one):
  (no arg)      OpenAI direct + gpt-5.4-mini (default, production).
                Requires OPENAI_API_KEY_DIRECT in ~/morty_ws/.morty_env.
  --debug       AWS Bedrock + Claude Haiku 4.5 (cheap, fast dev path).

Voice quality (pick one):
  (no arg)      Kokoro neural TTS + whisper-base.en STT (natural voice).
  --fast-voice  pyttsx3 (espeak) TTS + whisper-tiny.en STT.
                Voice is robotic but TTS is instant; STT is sloppier.

Hardware (combine with anything):
  --fake        Use fake hardware (offline / no robot)
  --voice-only  Skip grasp pipeline + executor — just chat layer

  --help        This message

Examples:
  ./morty.sh                          OpenAI + real robot + Kokoro voice
  ./morty.sh --debug --fake           Claude + fake robot + Kokoro voice
  ./morty.sh --debug --voice-only     Claude + chat-only + Kokoro voice
  ./morty.sh --debug --fast-voice     Claude + real robot + robotic voice
                                      (use when you want max speed for testing)

Ctrl-C cleanly kills every Morty process.
EOF
            exit 0
            ;;
        --fake|--voice-only|--debug|--fast-voice)
            ;;
        *:=*)
            # ros2-launch style overrides (e.g. mic_visualizer:=false) are
            # forwarded to the launch untouched (2026-08-11, for morty_gui).
            ;;
        *)
            echo "[morty.sh] unknown option: $arg (try --help)"
            exit 1
            ;;
    esac
done

# All Morty processes we know about. Used both for pre-launch cleanup and
# post-Ctrl-C teardown. Add new node names here if you ever spawn more.
MORTY_PROCS=(
    speak_action_server
    speech_input_node
    chat_node
    intent_router_node
    gripper_action_server
    move_to_named_pose_server
    ur3e_fast_observation_replay
    mic_visualizer_node
    grasp_executor_lin_llm
    grasp_executor_lin
    grasp_executor_pilz
    grasp_executor
    open3d_cloud_viewer
    yolo_detector
    yolo_live_select_grasp
    run_grasp_pipeline_venv
    run_grasp_pipeline
)

cleanup() {
    echo ""
    # Same scoping as the pre-launch cleanup: a --voice-only run must
    # not take the pipeline/executor down with it (2026-08-21).
    local procs=("${CLEANUP_PROCS[@]:-${MORTY_PROCS[@]}}")
    echo "[morty.sh] Ctrl-C received — killing ${#procs[@]} Morty process patterns…"
    for p in "${procs[@]}"; do
        pkill -TERM -f "$p" 2>/dev/null || true
    done
    sleep 2
    for p in "${procs[@]}"; do
        pkill -KILL -f "$p" 2>/dev/null || true
    done
    if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
        kill -KILL "$LAUNCH_PID" 2>/dev/null || true
    fi
    echo "[morty.sh] done. Clean exit."
    exit 0
}

trap cleanup INT TERM

# ── Parse flags first so we know which provider to source ────────────────
USE_DEBUG=false
USE_FAKE=false
USE_VOICE_ONLY=false
USE_FAST_VOICE=false
EXTRA_LAUNCH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --debug)       USE_DEBUG=true ;;
        --fake)        USE_FAKE=true ;;
        --voice-only)  USE_VOICE_ONLY=true ;;
        --fast-voice)  USE_FAST_VOICE=true ;;
        *:=*)          EXTRA_LAUNCH_ARGS+=("$arg") ;;
    esac
done

# ── Pre-launch cleanup: stragglers from last session ─────────────────────
# --voice-only must ONLY clean up the VOICE processes. Killing the full
# list here murdered a freshly started pipeline+executor whenever Majd
# pressed "Start pipeline" then "Start voice" in the GUI (2026-08-21).
VOICE_PROCS=(
    speak_action_server
    speech_input_node
    chat_node
    intent_router_node
    mic_visualizer_node
)
if $USE_VOICE_ONLY; then
    CLEANUP_PROCS=("${VOICE_PROCS[@]}")
    echo "[morty.sh] clearing VOICE stragglers only (--voice-only)…"
else
    CLEANUP_PROCS=("${MORTY_PROCS[@]}")
    echo "[morty.sh] clearing stragglers from any previous launch…"
fi
for p in "${CLEANUP_PROCS[@]}"; do
    pkill -KILL -f "$p" 2>/dev/null || true
done
sleep 1

# ── Source ROS + workspace ───────────────────────────────────────────────
# WS_DIR resolves to the directory this script lives in, so the launcher
# works whatever the workspace folder is called. On the Alienware the name
# ~/new_ur_gripper_ws was already taken by a pre-existing workspace that must
# not be touched, so this copy lives in ~/morty_ws. (Edited 2026-08-06.)
# Exported, not just set: ROS nodes are child processes and read WS_DIR from the
# environment to locate workspace-root assets (observation_poses.yaml, models/,
# the Vosk model). Without the export they silently fall back to the old
# hardcoded ~/morty_ws. (2026-08-06)
export WS_DIR="${WS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
# Fast-DDS: UDP-only transport, shared memory OFF (2026-09-02, see
# config/fastdds_udp_only.xml and SYSTEM_STATE_LOG #291). Same export lives
# in morty_gui/config.py and ~/.bashrc so every launch path agrees.
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$WS_DIR/config/fastdds_udp_only.xml}"

# ── Source the right provider based on --debug flag ─────────────────────
if $USE_DEBUG; then
    # Bedrock + Claude (cheap dev path).
    source "$WS_DIR/scripts/use_bedrock.sh"
else
    # OpenAI direct + gpt-5.4-mini (production default).
    # First load .morty_env directly so we can sanity-check the key
    # BEFORE use_openai.sh prints its own banner / returns nonzero.
    source "$WS_DIR/.morty_env"
    if [[ -z "${OPENAI_API_KEY_DIRECT:-}" ]]; then
        cat <<EOF >&2

[morty.sh] ERROR: no OpenAI API key found.

The default provider is OpenAI direct (gpt-5.4-mini), which needs a real
'sk-...' key from https://platform.openai.com/api-keys pasted into
~/morty_ws/.morty_env (variable: OPENAI_API_KEY_DIRECT).

Options:
  1. Paste the key into ~/morty_ws/.morty_env and retry, OR
  2. Use the Bedrock + Claude development path instead:
       $0 --debug $*

EOF
        exit 1
    fi
    source "$WS_DIR/scripts/use_openai.sh"
fi

# ── Decide launch file + args based on hardware flags ────────────────────
LAUNCH_PKG="communication_llm"
LAUNCH_FILE="morty_full.launch.py"

if $USE_VOICE_ONLY; then
    LAUNCH_FILE="voice_robot.launch.py"
    LAUNCH_ARGS=()
else
    if $USE_FAKE; then
        LAUNCH_ARGS=("use_fake_hardware:=true")
    else
        LAUNCH_ARGS=("use_fake_hardware:=false")
    fi
fi

# ── Voice-quality flag (independent of --debug) ─────────────────────────
# Without --fast-voice: defaults apply (Kokoro neural TTS + whisper base).
# With --fast-voice: pyttsx3 + tiny.en for max speed but robotic voice.
if $USE_FAST_VOICE; then
    LAUNCH_ARGS+=(
        "stt_engine:=whisper"
        "whisper_model:=tiny.en"
        "force_pyttsx3:=true"
    )
else
    # Force whisper backend even without --fast-voice, but at base.en for
    # decent accuracy. (The vosk default is too quirky for command voice.)
    LAUNCH_ARGS+=(
        "stt_engine:=whisper"
        "whisper_model:=base.en"
    )
fi

# On WSL the AEC virtual devices echocancel_mic / echocancel_spk don't exist
# because PulseAudio's module-echo-cancel won't load against WSLg's RDP-bridged
# sources/sinks. Route mic + speakers at the raw RDP devices instead.
if grep -qi 'microsoft\|WSL' /proc/version 2>/dev/null; then
    LAUNCH_ARGS+=(
        "pulse_source:=RDPSource"
        "pulse_sink:=RDPSink"
        "mic_device_name:=default"
    )
    echo "[morty.sh] WSL detected -> routing audio through RDPSource/RDPSink (no AEC)."
fi

echo "[morty.sh] launching: ros2 launch $LAUNCH_PKG $LAUNCH_FILE ${LAUNCH_ARGS[*]}"
echo "[morty.sh] press Ctrl-C to stop everything cleanly."
echo ""

LAUNCH_ARGS+=("${EXTRA_LAUNCH_ARGS[@]}")
ros2 launch "$LAUNCH_PKG" "$LAUNCH_FILE" "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!
wait "$LAUNCH_PID"
EXIT_CODE=$?

cleanup
exit $EXIT_CODE
