# Morty — voice- and pointing-guided grasping on a UR3e

ROS 2 Humble workspace for the Morty robot: a UR3e with a 3D-printed
parallel gripper, a wrist-mounted RealSense camera, a fixed Astra camera on
a second PC for recordings, and a speech/LLM layer. A person points at an
object or names it; the robot detects it, generates grasps, checks them
against the scene, and picks it.

## Hardware

| part | role |
| --- | --- |
| UR3e (e-Series, driver over Ethernet) | arm; pendant program with the External Control node |
| gripper_3d | 3D-printed two-finger gripper, driven through the UR tool I/O |
| Intel RealSense (eye-in-hand) | colour + aligned depth for detection, pointing and grasp clouds |
| Orbbec Astra Pro on the shuttle PC | fixed third-person camera for demonstration recordings |
| Alienware laptop (12 cores, GTX 1080) | runs everything below |

## Packages (`src/`)

| package | what it does |
| --- | --- |
| `grasp_pipeline` | YOLO detection, MediaPipe finger-pointing selection, object segmentation, GraspGen candidate generation, the geometric filter chain, neighbour obstacles, reach warning overlay |
| `ur3e_gripper_3d_moveit_commander` | the grasp executor: probes candidates with straight-line (Pilz LIN) plans, executes pick → lift → home → drop, reports why candidates were rejected |
| `ur3e_gripper_3d_moveit_config` | MoveIt configuration for the arm + gripper, driver launch, RealSense launch |
| `gripper_3d` | URDF/xacro, meshes (visual + convex collision hulls), gripper command adapter |
| `ur3e_fast_observation` | named observation poses and fast replay of recorded motions |
| `communication_llm`, `intent_router`, `morty_interfaces` | speech-to-text, the LLM chat/intent layer (OpenAI or AWS Bedrock), text-to-speech, intent messages |
| `morty_gui` | the operator GUI: starts/stops the stack, camera and 3D views, experiment (MMHRI) and VLA recording, pendant/driver status |
| `vla_experiments` | synchronised demonstration recorder (both cameras, joints, TCP pose, gripper, commands) |
| `Universal_Robots_ROS2_Driver`, `Universal_Robots_ROS2_Description` | vendor driver and robot description, built from source |

Other folders: `llm_provider/` (API-key and provider selection sourced by
`morty.sh`) and `config/` (Fast DDS profiles). How the parts are wired, what a fresh machine needs and a
troubleshooting table are in [INTEGRATION_NOTES.md](INTEGRATION_NOTES.md).

## What is not in the repository

Downloaded or generated on each machine, excluded by `.gitignore`:

- `GraspGen/` — NVIDIA GraspGen checkout; `models/graspgen/` holds its
  generator/discriminator checkpoints and the gripper config.
- `src/grasp_pipeline/yolo26s-seg_24_04.pt` — the trained YOLO model.
- `src/communication_llm/models/` — the Vosk speech model.
- `build/`, `install/`, `log/`, `logs/`, `experiments*/`, `backups/`.
- `.morty_env` — API keys (see below).

## Setting up a machine

```bash
# ROS 2 Humble + MoveIt 2 + ros2_control installed from apt
git clone git@github.com:hannamajd97/Morty_ws.git ~/morty_ws
cd ~/morty_ws
pip install --user -r requirements.txt      # no venv: ROS 2 Python paths need --user
colcon build --symlink-install
source install/setup.bash
```

Then put the GraspGen checkout, its checkpoints and the model files in the
places listed above. The workspace resolves its own location; if it is not
at `~/morty_ws`, export `WS_DIR` before launching.

### API keys (`.morty_env`)

Create `~/morty_ws/.morty_env` (mode 600, never committed) with one or both
providers:

```bash
# OpenAI direct
export OPENAI_API_KEY_DIRECT="sk-..."
export OPENAI_MODEL="gpt-5.4-mini"              # optional; the default

# AWS Bedrock (used with ./morty.sh --debug)
export AWS_BEARER_TOKEN_BEDROCK="..."
export AWS_REGION="us-east-1"                  # or eu-central-1 with the eu.* model id
export BEDROCK_MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
```

### Robot side

The pendant runs a saved program containing the External Control node
(host = this laptop's IP on the robot link, port 50002). With the pendant in
Remote Control mode the GUI starts that program itself whenever it is
stopped; in Local mode press Play on the pendant.

## Running

Everything is started from the GUI:

```bash
source ~/morty_ws/install/setup.bash
ros2 run morty_gui morty_gui
```

`Start ALL` brings up the driver + MoveIt + RealSense, the grasp pipeline,
the executor and the voice stack, and the status lights show driver,
cameras, robot, pendant program and voice. `./morty.sh --help` lists the
command-line variants (`--debug` for Bedrock, `--voice-only`, `--fake`).

A pick cycle: detection → selection (pointing, click or voice) →
segmentation and grasp cloud → GraspGen → filter chain → candidates to the
executor → LIN probe of each candidate → pick, lift, home, drop.

## Recording experiments

The GUI's experiment window records MMHRI trials (`trial_NNN_LX_SN` folders
with logs, CSV summary, 3D view) and, in the same session, VLA
demonstration episodes through `vla_experiments`. Logs of the running
stack are in `logs/`.
