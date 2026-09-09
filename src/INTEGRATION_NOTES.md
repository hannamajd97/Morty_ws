# Morty workspace — integration notes for a fresh PC

This document is the briefing a new machine (and the agent helping the
operator on it) needs **before** dropping this `src/` into another
`~/ur_ros2_ws/` and trying to build / run it.

The codebase ships as `src/` only. Several things that the running
system needs **do not live in `src/`**: system apt packages, Python
packages installed to user-site, configuration files in the workspace
root, a symlink in `$HOME`, fresh API keys, and a clean build. If any
of these is missing the system will start with cryptic errors. The
checklist below covers them all in the right order.

---

## 0. ROS distribution

Built and tested against **ROS 2 Humble** on Ubuntu 22.04. Other ROS 2
distributions are not supported.

```bash
source /opt/ros/humble/setup.bash
```

---

## 1. System packages (apt)

```bash
sudo apt update
sudo apt install -y \
    portaudio19-dev \
    espeak-ng \
    pulseaudio-utils \
    pavucontrol \
    ros-humble-tf-transformations
```

| Package | Why it is needed |
|---|---|
| `portaudio19-dev` | C library that `pyaudio` and `sounddevice` link against. Without it both fail to import. |
| `espeak-ng` | Fallback TTS engine; also used by Kokoro for words not in its phoneme dictionary. |
| `pulseaudio-utils` | Provides `pactl`. The voice launch loads PulseAudio's echo-cancel module via `pactl load-module`. |
| `pavucontrol` | The PulseAudio GUI. One-time-per-machine setup: set "Echo-Cancel" as default on Input + Output tabs (see §8). |
| `ros-humble-tf-transformations` | Used by `grasp_pipeline/grasp_pipeline/geometry_utils.py` for quaternion math. |

---

## 2. Python packages — install to **system user-site**, NOT a venv

ROS 2 Humble's launch wrappers carry a hard-coded `#!/usr/bin/python3`
shebang. They will import from `~/.local/lib/python3.10/site-packages`
(system user-site) but **not** from any venv. Do not use the workspace's
`.venv/` for runtime — only for tooling.

```bash
/usr/bin/python3 -m pip install --user \
    "openai>=1.68.0" \
    pydantic \
    "numpy==1.26.4" \
    "opencv-python<5" \
    sounddevice \
    webrtcvad \
    pyaudio \
    pyttsx3 \
    vosk \
    faster-whisper \
    kokoro \
    "transformers==4.45.0" \
    "tokenizers>=0.20,<0.21" \
    typeguard \
    boto3 \
    ultralytics \
    pyzmq \
    open3d \
    "transforms3d>=0.4"
```

### Why the version pins are non-negotiable

| Pin | Reason |
|---|---|
| `numpy==1.26.4` | Kokoro and Ubuntu's apt `python3-transforms3d` both use `np.float`, removed in NumPy 2.x. |
| `opencv-python<5` | Newer wheels silently bump NumPy to 2.x. |
| `transformers==4.45.0` | Kokoro 0.9.x calls `from transformers import AlbertModel` at the top level; 4.46+ removed that. |
| `tokenizers>=0.20,<0.21` | Pinned-transformers' hard requirement. |
| `transforms3d>=0.4` | Apt ships 0.3 which calls `np.float` and crashes under any NumPy 1.20+. The user-site install shadows the apt one. |

### If a venv exists at `~/ur_ros2_ws/.venv/`

Delete it, or make sure its `pyvenv.cfg` has `include-system-site-packages = true`.
Otherwise sourcing it will hide rclpy from the launch wrappers.

```bash
# Either:
rm -rf ~/ur_ros2_ws/.venv

# Or:
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' \
    ~/ur_ros2_ws/.venv/pyvenv.cfg
```

---

## 3. Workspace-root files that live OUTSIDE `src/`

These four files are not part of the package tree. They must be created
or copied into `~/ur_ros2_ws/` on the new machine.

| File | Source | Purpose |
|---|---|---|
| `~/ur_ros2_ws/.morty_env` | Manually authored (template below) | API keys, region, model selection |
| `~/ur_ros2_ws/use_bedrock.sh` | Copy from old machine | Wires Bedrock+Claude as the active LLM |
| `~/ur_ros2_ws/use_openai.sh` | Copy from old machine | Wires OpenAI direct as the active LLM |
| `~/ur_ros2_ws/morty.sh` | Copy from old machine | Single-command launcher with cleanup on Ctrl-C |

Lock the secrets file:

```bash
chmod 600 ~/ur_ros2_ws/.morty_env
chmod 700 ~/ur_ros2_ws/use_bedrock.sh ~/ur_ros2_ws/use_openai.sh
```

### `.morty_env` template

```bash
# ── 1. AWS Bedrock + Claude Haiku 4.5 ────────────────────────────────────
export AWS_BEARER_TOKEN_BEDROCK="ABSK..."        # paste fresh key
export AWS_REGION=us-east-1
export BEDROCK_MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ── 2. OpenAI direct + gpt-5.4-mini ──────────────────────────────────────
export OPENAI_API_KEY_DIRECT="sk-..."            # paste fresh key
export OPENAI_MODEL="gpt-5.4-mini"
```

**Do NOT reuse the keys from the original developer machine.** Rotate
each one in its provider console (AWS / OpenAI) and paste the new value
on the new machine. Keys pasted into chat logs / commit history must be
treated as compromised.

---

## 4. Workspace location

All launch files and scripts resolve the workspace from `$WS_DIR`, falling
back to `~/morty_ws`. If the workspace lives elsewhere, export `WS_DIR`
before launching (the GUI does this for everything it spawns). The YOLO
model file `yolo26s-seg_24_04.pt` (in `src/grasp_pipeline/`) is found
through that path at launch time.

---

## 5. Add the user to `dialout` for UR driver serial / E-stop (real robot only)

```bash
sudo usermod -aG dialout $USER
# log out and back in for the group to take effect
```

Skip on machines that will only ever run with `--fake`.

---

## 6. Clean build

The previous developer had a different set of packages (`moveit_resources`,
`grasp_executor_action_server`, `gripper_action_server` as a standalone
package, `move_to_pose_action_server` as a standalone package). The
current workspace has all of those merged or deleted. A stale
`install/` from before the cleanup will confuse colcon, so always:

```bash
cd ~/ur_ros2_ws
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Expected: **16 packages finished** with zero failures. A few may print
informational stderr ("did not request a specific build type, selecting
RelWithDebInfo") — those are harmless.

---

## 7. First run

```bash
cd ~/ur_ros2_ws
./morty.sh --help               # see all flags
./morty.sh --debug --voice-only # quickest end-to-end smoke test
```

`--help` documents every flag. Quick guide:

| Combination | Use it for |
|---|---|
| `./morty.sh` | OpenAI gpt-5.4-mini + real UR3e (production) |
| `./morty.sh --debug` | Bedrock Claude Haiku 4.5 + real robot |
| `./morty.sh --fake` | OpenAI + fake hardware (no UR driver needed) |
| `./morty.sh --debug --fake` | Bedrock + fake hardware (cheapest dev path) |
| `./morty.sh --debug --voice-only` | Just chat, no robot / grasp pipeline / executor |
| add `--fast-voice` | Replace Kokoro+whisper-base with pyttsx3+tiny.en (snappy but robotic) |

Ctrl-C from `morty.sh` cleanly kills every Morty process — no leftover
PIDs need to be hunted with `pkill`. The script preserves that contract
for every flag combination.

---

## 8. One-time PulseAudio routing (any machine that uses voice)

The launch loads PulseAudio's echo-cancel module automatically, but the
module is only an *available* sink/source until the user picks it as
**default** in `pavucontrol`. Without this, the microphone hears the
speakers and the chat path collects echo-corrupted transcripts.

While Morty is running:

```bash
pavucontrol &
```

- **Output Devices** tab → find the row whose name contains
  "Echo-Cancel" → click the green check on the right (default).
- **Input Devices** tab → find the row whose name contains
  "Echo-Cancel" → click the green check on the right (default).

Done once per user account. Settings persist across reboots.

---

## 9. Slip-detection behaviour (real hardware only)

### How the gripper itself works

The custom pneumatic gripper is **intelligent on its own** — it carries
FSLP (force-sensing linear potentiometer) sensors on each finger and its
own logic that runs independent of ROS. It broadcasts its state on two
tool digital inputs:

| Signal | What the gripper reports | What the gripper does on its own |
|---|---|---|
| **TI0 HIGH** | "I am holding an object" (FSLPs report finger pressure above threshold) | Nothing — informational only |
| **TI0 LOW** | "Nothing in grasp" | Nothing — informational only |
| **TI1 HIGH** | "I detected slip" (the object is moving inside the fingers) | **The gripper autonomously squeezes harder to recover** — no software intervention required |
| **TI1 LOW** | "Grip stable / slip recovered" | Nothing |

The key thing to understand: when TI1 goes HIGH, **the gripper is already
self-correcting**. Software's job is not to fix the slip, it's to *react*
to the broadcast — slow the arm a bit and tell the human what's
happening so the gripper has a chance to recover without the arm
shaking the object loose.

### The software watchdog

The LIN grasp executor is the **only** grasp executor in the workspace —
the previous "vitamin-c-specific executor" no longer exists. One
executor handles every pick. The slip watchdog inside it runs for
**every object**, not just the vitamin tube: it watches TI1 the entire
time the executor is active, and reacts the same way regardless of
which label was picked.

The watchdog subscribes to `/io_and_status_controller/io_states` and
looks only at TI1. Object identity is not part of the decision.

| Behaviour | Default | Override |
|---|---|---|
| Watchdog enabled | **ON for every pick** | `slip_watchdog_enabled:=false` to ignore TI1 entirely |
| Action when TI1 goes HIGH | Drop UR speed slider to `slip_slow_slider` (0.05 = 5%) and keep moving | `slip_slow_slider:=0.0` to fully stop, higher values for less cautious creep |
| Action when TI1 returns LOW | Restore `nominal_speed_slider` (1.0) | Same param exists if you want a non-1.0 nominal |
| Narration | "Slip detected, slowing down." / "Grip restored, resuming." spoken via Morty TTS | Subscribed via `intent_router_node._on_slip_event`; topic configurable via `slip_event_topic` |

Why slow-down instead of stop: stopping the arm freezes the geometry
mid-motion and gives the gripper longer to recover, but it also leaves
the operator wondering "did the system crash?" A 5% creep keeps the
motion visibly alive and gives the gripper its recovery window in the
same gesture. The trajectory is not cancelled or re-planned — the UR
speed slider just scales the parameterisation, so when the slip clears
the arm resumes the same path at full speed.

### Optional: the vitamin-tube close-pulse mode

Separate from slip handling. OFF by default. Pass
`enable_vitamin_mode:=true` to `intent_router_node` to opt in. When
enabled AND the user is picking `vitamin_c` (vitamin tube), the
executor uses a shorter first close pulse, skips the post-close TI0
no-grasp check, and fires a second harder close pulse later in the
path. This is a knob that controls **how the close is performed for
that specific object**; it is **NOT** the slip detection. Slip detection
runs for every object regardless of this flag.

---

## 10. Project structure quick reference

```
~/ur_ros2_ws/
├── .morty_env                       (chmod 600 — secrets)
├── morty.sh                         (single-command launcher)
├── use_bedrock.sh / use_openai.sh   (provider switch scripts)
├── src/
│   ├── communication_llm/           (chat, STT, TTS, mic visualizer)
│   ├── intent_router/               (routes /robot/intent to actions, narrates lifecycle)
│   ├── ur3e_fast_observation/       (replay_node + named_pose_server + teach)
│   ├── ur3e_gripper_3d_moveit_commander/
│   │     ├── grasp_executor_lin_llm.cpp   (the main pick executor)
│   │     ├── grasp_executor_lin.cpp       (non-voice variant)
│   │     ├── grasp_executor.cpp / _pilz.cpp
│   │     ├── observation_ptp_commander.cpp
│   │     └── gripper_action_server.cpp    (C++ OperateGripper bridge)
│   ├── ur3e_gripper_3d_moveit_config/
│   ├── gripper_3d/                  (URDF + I/O bridge + gripper command adapter)
│   ├── grasp_pipeline/              (YOLO detection + grasp generation)
│   ├── morty_interfaces/            (action + msg definitions)
│   ├── Universal_Robots_ROS2_Description
│   └── Universal_Robots_ROS2_Driver
├── build/  install/  log/           (created by colcon build)
└── .venv/                           (optional; not used at runtime)
```

`gripper_action_server` and `move_to_named_pose_server` used to be
standalone packages; they were merged into
`ur3e_gripper_3d_moveit_commander` (as C++) and `ur3e_fast_observation`
(as Python) respectively. Action names and node names are unchanged so
callers see no difference.

---

## 11. LLM backend selection

Two providers are supported. The code path is selected by the model ID
in `BEDROCK_MODEL` or `OPENAI_MODEL`. `llm_service.py` routes
automatically:

- Model starting with `anthropic.` / `us.anthropic.` / `meta.` / `mistral.` /
  `cohere.` / `ai21.` / `amazon.` → native AWS Bedrock Converse via
  `boto3` (no proxy needed).
- Model starting with `openai.` or `gpt-` together with a non-empty
  `OPENAI_BASE_URL` → Bedrock's OpenAI-compatible REST endpoint via the
  `openai` Python SDK.
- Default (no `OPENAI_BASE_URL`) → OpenAI direct via the `openai` SDK.

GPT-OSS models (`openai.gpt-oss-*`) automatically get
`reasoning_effort=low` injected to skip the chain-of-thought delay; the
adapter also strips any leftover `<reasoning>...</reasoning>` text from
TTS output.

---

## 12. Troubleshooting cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: openai` (and friends) | Python deps installed to a venv that ROS isn't using | Reinstall to `--user` (§2) |
| `cannot import name 'Inf' from 'numpy'` | NumPy 2.x got pulled in | `pip install --user --force-reinstall "numpy==1.26.4"` |
| `cannot import name 'AlbertModel' from 'transformers'` | Transformers 4.46+ | `pip install --user --force-reinstall "transformers==4.45.0" "tokenizers<0.21"` |
| `No such file or directory: '/home/.../morty_ws/...'` | Workspace not at `~/morty_ws` | `export WS_DIR=/path/to/workspace` (§4) |
| `Bedrock 'on-demand throughput isn't supported'` for Claude | Wrong model ID — direct Claude IDs need cross-region inference | Prefix the model ID with `us.` (e.g. `us.anthropic.claude-haiku-4-5-...`) |
| Morty hears himself / says "I didn't catch that" loop | PulseAudio echo-cancel not the default sink/source | Set defaults in `pavucontrol` (§8) — or use headphones |
| `[morty.sh] ERROR: no OpenAI API key found` | `OPENAI_API_KEY_DIRECT` empty | Paste a fresh key in `.morty_env`, or use `--debug` |
| ALSA underrun spam | Audio buffer can't keep up — usually a VM with no GPU | Cosmetic on a workstation; on a real PC with audio passthrough it stops |

---

## 13. What changed vs the previous developer's workspace

Useful when asking an LLM agent to help — these are the structural
deltas a fresh agent should know about:

- Workspace root layout: `src_13_05_2026/src/` was renamed to `src/`.
- Deleted packages: `moveit_resources/*`, `grasp_executor_action_server`
  (dead code), `gripper_action_server` (merged into the C++ commander),
  `move_to_pose_action_server` (merged into `ur3e_fast_observation`).
- `gripper_action_server` is now a **C++** executable inside
  `ur3e_gripper_3d_moveit_commander`. Same node name, same action name.
- `move_to_named_pose_server` is now an entry point of
  `ur3e_fast_observation`. Same node name, same action name.
- `llm_service.py` gained a Bedrock Converse backend (boto3) in addition
  to the OpenAI SDK path, plus a `<reasoning>` stripper for gpt-oss
  models.
- `chat_node.py` SYSTEM_PROMPT was trimmed from ~272 lines to ~54 and
  now explicitly permits non-robot chat.
- `morty.sh` is a single-command launcher with guaranteed cleanup on
  Ctrl-C. It auto-sources the right provider based on flags and aborts
  with a friendly message if the OpenAI key is missing.
- There is now exactly ONE grasp executor (`grasp_executor_lin_llm`)
  that handles every pick. There used to be a separate vitamin-c
  executor; that has been removed. The remaining vitamin_c knob is just
  a close-pulse pattern variant inside the same executor, gated by a
  parameter (see below).
- Slip handling in the LIN executor: runs for EVERY object (not just
  vitamin_c). Was full-stop on TI1=HIGH (and disabled by default), now
  is slow-down (5% slider) on TI1=HIGH AND enabled by default.
  Publishes `slip_start` / `slip_clear` on `/executor/slip_event` so
  `intent_router_node` can have Morty narrate.
- Vitamin-tube close-pulse variant: was auto-activated for any
  `vitamin_c` pick, now requires opting in via
  `intent_router_node`'s `enable_vitamin_mode` parameter (default
  false). This is independent of slip handling and only changes the
  close-pulse pattern for that specific object.
- The interface package was renamed from `robot_interfaces` to
  `morty_interfaces`. Same action / msg / srv definitions, same field
  names, same downstream behaviour — only the package name changed.
  Every caller (`#include <morty_interfaces/...>`, `from
  morty_interfaces.action import ...`, `<depend>morty_interfaces</depend>`,
  `find_package(morty_interfaces REQUIRED)`) was updated atomically.
  If a fresh checkout still has any reference to `robot_interfaces` it
  is stale — do a clean wipe (`rm -rf build install log`) before
  rebuilding or the old generated headers will mask the rename.

---

## 14. Final sanity check

After a clean build + first launch, in a second terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/ur_ros2_ws/install/setup.bash

ros2 node list                  # expect ~10 nodes for --voice-only, ~14 for full
ros2 action list                # expect /gripper, /move_to_named_pose, /speak, ...
ros2 action info /gripper       # server: /gripper_action_server (C++), client: /intent_router_node
```

If all three return sensible output, the workspace is functional.
