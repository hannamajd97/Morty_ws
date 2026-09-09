#!/usr/bin/env bash
# Enable PulseAudio's WebRTC echo canceller. After this runs, two virtual
# devices appear:
#   source: echocancel_mic   (mic with speaker signal subtracted)
#   sink:   echocancel_spk   (speaker, routed through AEC)
#
# Point speech_input_node at echocancel_mic via:
#   ros2 launch communication_llm voice_robot.launch.py mic_device_name:=echocancel_mic
# and route TTS playback through echocancel_spk in pavucontrol → Playback tab.
#
# Usage:
#   bash scripts/setup_aec.sh           # load
#   bash scripts/setup_aec.sh unload    # remove

set -e

ACTION="${1:-load}"

if ! command -v pactl >/dev/null 2>&1; then
    echo "pactl not found. AEC requires PulseAudio (or Pipewire's pulse-server)."
    exit 1
fi

case "$ACTION" in
    load)
        # Avoid loading twice
        if pactl list short modules | grep -q module-echo-cancel; then
            echo "module-echo-cancel already loaded. Showing devices:"
        else
            pactl load-module module-echo-cancel \
                aec_method=webrtc \
                aec_args="analog_gain_control=0 digital_gain_control=1 noise_suppression=1 voice_detection=1" \
                source_name=echocancel_mic \
                sink_name=echocancel_spk \
                use_master_format=1
            echo "Loaded module-echo-cancel."
        fi
        echo
        echo "=== Sources (mics) ==="
        pactl list short sources | grep -E "echocancel_mic|RUNNING|IDLE" || true
        echo
        echo "=== Sinks (speakers) ==="
        pactl list short sinks | grep -E "echocancel_spk|RUNNING|IDLE" || true
        echo
        echo "Next steps:"
        echo "  1. Open pavucontrol → Configuration / Output / Input tabs."
        echo "     Set 'Echo-Cancel ...' as default for mic AND speaker output."
        echo "  2. Run the launch with mic_device_name:=echocancel_mic"
        echo "  3. To remove AEC again: bash scripts/setup_aec.sh unload"
        ;;
    unload)
        ID="$(pactl list short modules | awk '/module-echo-cancel/ {print $1; exit}')"
        if [ -n "$ID" ]; then
            pactl unload-module "$ID"
            echo "Unloaded module-echo-cancel (id $ID)."
        else
            echo "module-echo-cancel was not loaded."
        fi
        ;;
    *)
        echo "Usage: $0 [load|unload]"
        exit 2
        ;;
esac
