# 🤖 Morty — Voice Command Reference

> *Your embodied robot assistant. LLM brain, Whisper/Vosk ears, Kokoro voice, UR3e body.*

---

## 🚀 The fast start

The GUI starts the voice stack ("Start ALL", or the voice button) and stops
it cleanly; nothing has to be killed by hand. From a terminal instead:

```bash
cd ~/morty_ws
./morty.sh --voice-only          # OpenAI provider
./morty.sh --debug --voice-only  # AWS Bedrock provider
```

Either way seven processes start and the echo-cancel module auto-loads.
Ctrl-C in `morty.sh` stops all of them.

> ⚠️ **One manual step the first time on a machine:** open `pavucontrol` and on the **Input Devices** and **Output Devices** tabs pick the "Echo-Cancel…" entry as default.

---

## 🧱 What you DO and DO NOT need running

| What you want | Required also running |
|---|---|
| Just chat with Morty | Nothing else (just the launch) |
| **Open / close gripper** | Your UR driver / MoveIt launch (provides `/gripper_controller/gripper_cmd`) |
| **Go home / ready / look left / look right / look up** | UR driver / MoveIt (provides controllers + joint_states) |
| **Pick the X** (full grasp) | **All of**: UR driver, `grasp_executor_lin`, `run_grasp_pipeline.launch.py` |

> 💡 The replay node (named-pose moves) is now bundled inside this launch, so you no longer need `run_grasp_pipeline.launch.py` for "go home" / "look left" / etc. You only need it for picking.

---

## 💬 Conversation

| 🎙️ You say | 🔊 What happens |
|---|---|
| *"hello"* / *"hi morty"* | Free-form chat |
| *"what's your name?"* | *"I'm Morty, your robot assistant."* |
| *"tell me a joke"* | LLM tells one |
| *"how does X work?"* | Knowledge Q&A |

---

## 👀 Scene awareness *(needs `grasp_pipeline` running)*

| 🎙️ You say | 🔊 What happens |
|---|---|
| *"what do you see?"* | Lists visible objects with positions |
| *"is there a vitamin tube?"* | Yes/no, grounded in real detections |
| *"how many towels?"* | Counts what's in view |
| *"what's on the left?"* | Names the leftmost object |

---

## ✊ Gripper control

| 🎙️ You say | ⚙️ Action |
|---|---|
| *"open the gripper"* / *"open it"* / *"release"* / *"let go"* | 🟢 **Opens** |
| *"close the gripper"* / *"close it"* / *"grip"* / *"clamp"* | 🔴 **Closes** |

If the robot driver isn't running:
> *"I can't move the gripper because the robot driver isn't running."*

---

## 🏠 Move to named pose

Works **without** grasp_pipeline (replay node is now part of this launch).

| 🎙️ You say | 📍 Pose |
|---|---|
| *"go home"* / *"return home"* | `home_pose_name` (default `initgrasp`) |
| *"go to ready"* / *"ready position"* | `ready_pose_name` (default `initgrasp`) |
| *"go to start"* / *"start position"* | same as ready |
| *"look left"* / *"view from left"* | `detect_left` |
| *"look right"* / *"view from right"* | `detect_right` |
| *"look up"* / *"look higher"* | `detect_higher` |

If the robot driver / motion server isn't running:
> *"I can't move the arm because the motion server isn't running."*

---

## 🦾 Picking *(needs full robot stack)*

### Basic
| 🎙️ You say | 🎯 What it does |
|---|---|
| *"pick the vitamin tube"* | Picks a vitamin tube |
| *"grab the kitchen towel"* | Picks a kitchen towel |
| *"take the retail box"* | Picks a retail box |

### 📐 Spatial picks
| 🎙️ You say | 🎯 What it does |
|---|---|
| *"pick the leftmost vitamin tube"* | Grabs the **leftmost** one |
| *"pick the rightmost one"* | Grabs the **rightmost** match |
| *"pick the one in the middle"* | Median by horizontal position |

### 📏 Size picks
| 🎙️ You say | 🎯 What it does |
|---|---|
| *"pick the biggest"* | Largest bbox (≈ **closest**) |
| *"pick the smallest one"* | Smallest bbox |

If grasp_pipeline isn't running:
> *"I can't pick anything because the grasp pipeline isn't running."*

---

## ⏪ Memory and control

| 🎙️ You say | ⚙️ Effect |
|---|---|
| *"again"* / *"do it again"* / *"repeat"* | 🔁 Re-fires the last action intent (skips LLM, instant) |
| *"do that again"* / *"redo"* / *"once more"* | Same |
| *"stop"* / *"cancel"* / *"abort"* | 🛑 Cancels in-flight action + clears the queue (instant, before LLM) |
| *"halt"* / *"freeze"* | Same as stop |

> ⚠️ **Soft-cancel only**: stops the things Morty owns (Speak, gripper, named-pose, ExecuteGrasp). Does **not** halt the C++ executor mid-trajectory.

---

## 🔗 Compound commands (multi-action)

Chain commands in one utterance — Morty queues them and runs each as the previous completes.

| 🎙️ You say | 📋 Sequence |
|---|---|
| *"pick the vitamin tube and then go home"* | Pick → wait → go home |
| *"open the gripper, then look right"* | Open → arrive → look right |
| *"go to ready, then close the gripper"* | Move → arrive → close |
| *"look right, then look up, then go home"* | Three pose moves |

---

## 🗣️ Wake-word mode (off by default)

```bash
ros2 launch communication_llm voice_robot.launch.py wake_word_enabled:=true
```

| 🎙️ You say | ⚙️ Effect |
|---|---|
| *"Morty, pick the vitamin tube"* | ✅ Acts |
| *"Hey Morty, what do you see?"* | ✅ Acts |
| *"Hey John, lunch?"* | ❌ Ignored |
| *"stop"* | ✅ Always allowed |

---

## ⚙️ Default config (what you get with bare `ros2 launch`)

| Setting | Default | Effect |
|---|---|---|
| `tts_enabled` | `true` | Morty speaks his replies |
| `mute_during_tts` | `false` | Mic stays open while Morty talks (barge-in) |
| `auto_aec` | `true` | PulseAudio AEC auto-loads at launch start |
| `mic_device_name` | `echocancel_mic` | Uses the AEC mic if available |
| `vad_enabled` | `true` | Background silence/noise filtered |
| `wake_word_enabled` | `false` | No "Morty" prefix required |
| `with_replay` | `true` | Replay node bundled (named-pose moves work) |
| `with_chat` | `true` | chat_node included in this launch |
| `stt_engine` | `vosk` | Streaming Vosk (fast, lower accuracy) |
| `whisper_model` | `base.en` | If you switch to whisper, this is the size |
| `auto_execute_grasp` | `false` | chat_node only routes intent (no direct ExecuteGrasp) |

---

## 🎛️ Useful launch overrides

```bash
# 🎤 Switch to faster-whisper STT (better accuracy, especially for "Morty")
ros2 launch communication_llm voice_robot.launch.py stt_engine:=whisper

# 🎤 Tiny whisper if your CPU struggles
ros2 launch communication_llm voice_robot.launch.py stt_engine:=whisper whisper_model:=tiny.en

# 🗣️ Different Kokoro voice
ros2 launch communication_llm voice_robot.launch.py voice:=am_michael

# 🏠 Different home pose (when you add one to observation_poses.yaml)
ros2 launch communication_llm voice_robot.launch.py home_pose_name:=safe_stow

# 🎙️ Different mic
ros2 launch communication_llm voice_robot.launch.py mic_device_name:=ReSpeaker
ros2 launch communication_llm voice_robot.launch.py mic_index:=2

# 🔇 Mute the AI (intents still route)
ros2 launch communication_llm voice_robot.launch.py tts_enabled:=false

# 🗣️ Wake word required (demo / multi-person mode)
ros2 launch communication_llm voice_robot.launch.py \
  wake_word_enabled:=true \
  wake_words:='["morty","hey morty"]'

# 🚫 Don't bundle the replay node (because you launch grasp_pipeline yourself)
ros2 launch communication_llm voice_robot.launch.py with_replay:=false

# 🚫 Don't auto-load AEC (you manage it yourself)
ros2 launch communication_llm voice_robot.launch.py auto_aec:=false

# ⌨️ Headless / type-only mode
ros2 launch communication_llm voice_robot.launch.py \
  speech_input_enabled:=false tts_enabled:=false

# 🔧 Direct ExecuteGrasp from chat_node (skip grasp_pipeline)
ros2 launch communication_llm voice_robot.launch.py auto_execute_grasp:=true
```

---

## 🩺 Common issues & how to fix

| Symptom | Cause | Fix |
|---|---|---|
| Morty hears himself talk | AEC not routed in `pavucontrol` | Open `pavucontrol`, set "Echo-Cancel..." as default for both Input + Output. Or use headphones. |
| `mic_device_name 'echocancel_mic' not found` | AEC didn't auto-load | Check `pactl list short modules` for `module-echo-cancel`; pass `auto_aec:=true` (default) or run `setup_aec.sh` manually |
| *"I can't move the arm because the motion server isn't running"* | `move_to_named_pose_server` didn't come up | Look for `move_to_named_pose_server` in the launch log — should say "MoveToNamedPose server ready"; otherwise the bundled replay was disabled |
| *"The replay node is not running"* | UR driver up, but `ur3e_fast_observation_replay` isn't | Pass `with_replay:=true` (default), or launch `run_grasp_pipeline.launch.py` |
| *"I can't pick anything because the grasp pipeline isn't running"* | grasp_pipeline isn't launched | `ros2 launch grasp_pipeline run_grasp_pipeline.launch.py enable_tts:=false` |
| Vosk mishears "Morty" as "morpeth" | Vosk small model is weak | Switch to whisper: `stt_engine:=whisper` |
| Multiple servers / duplicate output | Stragglers from an earlier session | Stop voice in the GUI (it adopts and stops leftovers), or Ctrl-C the `morty.sh` that started them |
| LLM says nothing | OpenAI key missing | `export ...=YOUR_KEY` before launch |

---

## 🧪 Debug from a second terminal

```bash
# What's being heard
ros2 topic echo /speech/transcript

# What intent the LLM extracted
ros2 topic echo /robot/intent

# Manual gripper test (no LLM needed)
ros2 action send_goal -f /gripper morty_interfaces/action/OperateGripper "{command: 0}"   # open
ros2 action send_goal -f /gripper morty_interfaces/action/OperateGripper "{command: 1}"   # close

# Manual go-home test
ros2 action send_goal -f /move_to_named_pose morty_interfaces/action/MoveToNamedPose \
  "{pose_name: 'initgrasp'}"

# Check scene awareness
ros2 topic echo /yolo/all_detections --once

# Check mic mute state
ros2 topic echo /speech/muted

# AEC module status
pactl list short modules | grep echo
```

---

## 🧠 What's under the hood

```
🎙️  microphone
     ↓
[ speech_input_node ]  ← Vosk OR faster-whisper, WebRTC VAD, wake-word gate
     ↓ /speech/transcript
[ chat_node ]          ← OpenAI LLM with scene + spatial awareness
     ↓ /robot/intent
[ intent_router_node ] ← multi-action queue + lifecycle narration
     ↓
   ┌─┴─┬───────────┬──────────────┬───────────────┐
   ↓   ↓           ↓              ↓               ↓
gripper_action  move_to_named  /yolo/select_index  Speak
   ↓                ↓                                ↓
🦾 robot      replay_node                        🔊 speakers
```

---

## ✅ Today's checklist

1. ☐ Pendant: program with External Control loaded, pendant in Remote mode.
2. ☐ GUI: `ros2 run morty_gui morty_gui`, then **Start ALL** (driver, pipeline, executor, voice).
3. ☐ Lights: driver, realsense, robot, play and voice green.
4. ☐ First time on a machine: `pavucontrol`, "Echo-Cancel" default on Input and Output.
5. ☐ Test: *"Morty, go home"* — the arm should move.

---

*Morty is yours. Take care of him.* 🤖💛
