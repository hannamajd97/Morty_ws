# communication_llm

Morty's ears, brain and mouth: speech-to-text, the language-model chat and
intent layer, text-to-speech. The voice command reference is in
[MORTY.md](MORTY.md).

## Nodes

| node | role | topics / actions |
| --- | --- | --- |
| `speech_input_node` | microphone → transcript (Vosk or faster-whisper, VAD, echo-cancel aware) | `/speech/transcript`, `/speech/state`, `/speech/level`, `/speech/waveform`, `/speech/muted`, `/speech/rejected`, listens to `/speech/tts_active` so it does not hear Morty |
| `chat_node` | the conversation: deterministic fast paths for pick / stop / home / drop / gripper and pause answers, the LLM for everything else; knows the robot state | in: `/speech/transcript`, `/yolo/all_detections`, `/grasp_executor_lin_llm/interactive_status`, `/robot/held_object`, `/robot/current_pose`, `/grasp_pipeline/selection_prompt`; out: `/robot/intent`, `/robot/assistant_text`, `/chat/state` |
| `speak_action_server` | text → audio (Kokoro or pyttsx3) | action `/speak`, `/speech/tts_active`, `/speech/ai_said` |
| `mic_visualizer_node` | the level meter / state widget the GUI embeds | reads the `/speech/*` and `/chat/*` topics |

`llm_service.py` selects the backend from the model id: `anthropic.` /
`us.anthropic.` / `eu.anthropic.` / `meta.` / … → AWS Bedrock Converse via
boto3; `gpt-` with `OPENAI_BASE_URL` → Bedrock's OpenAI-compatible endpoint;
`gpt-` alone → OpenAI. Keys and model ids come from `.morty_env` through
`llm_provider/use_*.sh`.

## Launch

`voice_robot.launch.py` starts the four nodes above plus the pieces they
call: `intent_router_node`, `gripper_action_server`,
`move_to_named_pose_server` and `replay_node`. `morty_full.launch.py` adds
the grasp pipeline and executor for terminal-only use.

Normally started by the GUI ("Start ALL" / voice), or by hand:

```bash
./morty.sh --voice-only          # OpenAI
./morty.sh --debug --voice-only  # Bedrock
```

## Models

`models/vosk-model-small-en-us-0.15/` for Vosk is in the repository;
faster-whisper and Kokoro download their weights on first use.

## Audio setup

The launch loads PulseAudio's echo-cancel module (`scripts/setup_aec.sh`).
Once per user, in `pavucontrol`, set the "Echo-Cancel" entry as default on
both the Input and Output tabs; otherwise the microphone hears the speakers.
