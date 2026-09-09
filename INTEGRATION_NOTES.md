# Morty — integration notes

How the parts of this workspace fit together and what a fresh machine
needs before the system runs. The README describes what the system is;
this file describes how it is wired.

## 1. Platform

ROS 2 Humble on Ubuntu 22.04. Installed from apt: `ros-humble-moveit`,
`ros-humble-pilz-industrial-motion-planner`, `ros-humble-ros2-control`,
`ros-humble-ros2-controllers`, `ros-humble-realsense2-camera`,
`ros-humble-tf2-tools`, `ros-humble-rosbag2`, plus the audio libraries
`portaudio19-dev`, `espeak-ng`, `pulseaudio-utils`, `pavucontrol`.

The UR driver and description are built from source in `src/` (vendor
packages at version 2.9), so no `ros-humble-ur` binary is required for the
build; the apt one may be present without harm.

## 2. Python packages

Install to the user site, never into a venv: the ROS launch wrappers use
`/usr/bin/python3` and only see `~/.local/lib/python3.10/site-packages`.

```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

Pins that must not be relaxed: `numpy==1.26.4` (Kokoro and transforms3d
use `np.float`), `opencv-python<5` (newer wheels pull NumPy 2),
`transformers==4.45.0` with `tokenizers<0.21` (Kokoro imports
`AlbertModel`), `transforms3d>=0.4` (the apt version crashes on NumPy 1.20+).

## 3. Files outside `src/`

| path | what | comes from |
| --- | --- | --- |
| `.morty_env` | API keys, region, model ids (mode 600) | written by hand, template in README |
| `GraspGen/` | NVIDIA GraspGen source checkout | github.com/NVlabs/GraspGen |
| `models/graspgen/` | `graspgen_robotiq_2f_140.yml`, `_gen.pth`, `_dis.pth` | GraspGen release |
| `src/grasp_pipeline/yolo26s-seg_24_04.pt` | trained YOLO detector | copied from the lab machine |
| `src/communication_llm/models/vosk-model-small-en-us-0.15/` | Vosk speech model | alphacephei.com/vosk/models |
| `observation_poses.yaml` | named arm poses (home, drop sides, observation) | in the repository |
| `config/fastdds_udp_only.xml` | Fast DDS profile used by every launch | in the repository |

Everything resolves the workspace from `$WS_DIR`, falling back to
`~/morty_ws`; the GUI exports `WS_DIR` for every process it starts.

## 4. Build

```bash
cd ~/morty_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` matters: Python packages run from `src/`, so an edit
takes effect at the next node start without rebuilding. C++ changes (the
executor) need `colcon build --packages-select ur3e_gripper_3d_moveit_commander`.
After renaming or removing packages wipe `build/ install/ log/` first.

## 5. The robot side

- The pendant runs a **saved** program with the External Control node
  (host = this laptop's IP on the robot link, 192.168.1.101, port 50002).
  An unsaved program is lost on reboot and the dashboard then reports
  `<unnamed>` with nothing to start.
- **Remote Control mode** on the pendant lets the GUI start the program
  through the dashboard whenever it finds it stopped. In Local mode the
  dashboard refuses Play and a human presses it on the pendant.
- Stopping the program deactivates `scaled_joint_trajectory_controller`;
  the GUI re-activates it while the program runs (bridge watchdog).
- `/tf` runs at ~830 Hz here (joint states at 800 Hz). Python nodes must
  not use the stock tf2 listener; `grasp_pipeline/tf_listener_lite.py`
  drains it at 50 Hz on its own thread.

## 6. What runs, and who starts it

The GUI (`ros2 run morty_gui morty_gui`) owns every process. Its four
launches, in the order `Start ALL` uses:

| launch | contents |
| --- | --- |
| `ur3e_gripper_3d_moveit_config/ur3e_gripper_3d_moveit_real_camera.launch.py` | UR driver, controllers, robot_state_publisher, MoveIt `move_group` (Pilz + OMPL), RealSense, gripper command adapter, point-cloud downsampler, planning-scene environment (table, mount, wall) |
| `grasp_pipeline/run_grasp_pipeline.launch.py` | the `yolo_live_select_grasp` node (detection, pointing, segmentation, GraspGen in-process, filter chain, neighbour obstacles, debug image) |
| `ur3e_gripper_3d_moveit_commander/grasp_executor_lin_llm.launch.py` | the grasp executor (persistent; one cycle per selection) |
| `communication_llm/voice_robot.launch.py` via `morty.sh --voice-only` | speech input, chat/LLM node, intent router, speech output |

`morty.sh` is the terminal alternative for the voice stack (`--debug`
selects Bedrock, `--fake` fake hardware). Stack logs go to `logs/*_gui.log`.

## 7. How a pick flows between the nodes

```text
camera ──► yolo_live_select_grasp ──/yolo/all_detections──► GUI, voice
                │  selection: /yolo/select_index (click/voice) or pointing
                │  publishes /graspgen/selection_cycle_id, then
                │  /graspgen/candidate_poses_base_link (kept grasps, ranked)
                ▼
        grasp_executor_lin_llm ── probes each candidate with a Pilz LIN plan,
                │  names the blocker of a rejected one (/check_state_validity),
                │  executes pick → lift 10 cm → home → drop_left/right
                │  status: /grasp_executor_lin_llm/interactive_status, cycle_done
                ▼
        pipeline waits on cycle_done, then re-arms for the next selection
```

Voice: `speech_input_node` → `chat_node` (LLM, deterministic fast paths for
pick/stop/home/drop) → `/robot/intent` → `intent_router` → executor
parameters and `/grasp_executor_lin_llm/abort`; answers go back through
`/robot/assistant_text` to `speak_action_server`. Held object and pose are
published on `/robot/held_object` and `/robot/current_pose` so the chat
knows the robot's state.

GUI: subscribes to the debug image, candidates, executor status, driver and
dashboard state; publishes selections, abort, E-Stop (dashboard stop),
experiment markers; records MMHRI trials and drives the VLA recorder.

## 8. Executor behaviour worth knowing

- **LIN-only protocol:** a candidate is accepted only if a straight-line
  approach exists; candidates without a quick IK solution are probed last,
  not dropped. Rejections say why: the colliding pair with penetration
  depth, "mid-approach" when the goal is free but the line is blocked,
  reach/IK when no joint solution exists.
- **Bounded execution:** every trajectory goes through the ExecuteTrajectory
  action with a deadline; a controller that never completes cannot wedge
  the node.
- **Slip watchdog:** the gripper reports "holding" on tool input TI0 and
  "slip" on TI1 and squeezes harder on its own; the executor slows the arm
  to 5 % on TI1 and restores speed when it clears, narrated through the
  voice stack. `slip_watchdog_enabled:=false` disables it.
- **Reach warning:** the detection overlay shows `edge N% @A deg` or
  `too far` per object before any planning, from the arm's reach envelope
  (`grasp_pipeline/perception/reach_envelope.py`).

## 9. Audio

The voice launch loads PulseAudio's echo-cancel module. Once per user
account, in `pavucontrol`, set the "Echo-Cancel" entry as default on both
the Output and Input tabs; otherwise the microphone hears the speakers.

## 10. LLM backend

`llm_service.py` picks the backend from the model id: `anthropic.`,
`us.anthropic.`, `eu.anthropic.`, `meta.`, `mistral.`, `cohere.`, `ai21.`,
`amazon.` go to AWS Bedrock Converse through boto3; `gpt-` with
`OPENAI_BASE_URL` set goes to Bedrock's OpenAI-compatible endpoint; `gpt-`
without it goes to OpenAI directly. `llm_provider/use_openai.sh` and
`llm_provider/use_bedrock.sh` export the matching variables from `.morty_env`.

## 11. Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| CONTROL_FAILED right after a good plan | pendant program stopped, trajectory controller inactive | Remote mode: wait for auto-Play; Local: press Play on the pendant |
| dashboard answers `Failed to execute: play` | program already running, or no program loaded (`<unnamed>`), or protective stop | load/save the program; unlock the protective stop on the pendant |
| `Attempt to write on a non-connected socket` in the driver log | dashboard socket dropped after a Local/Remote toggle | the GUI reconnects it; manually `ros2 service call /dashboard_client/connect std_srvs/srv/Trigger` |
| driver dies at start with `Could not get configuration package within timeout` | started too soon after a driver stop, or heavy load | Stop driver, wait 5 s, Start again |
| GraspGen takes seconds, YOLO > 500 ms per frame | the pipeline process is starved (interpreter lock or machine load) | check `[perf] infer_ms_avg` in the pipeline log and `uptime`; look for orphaned rviz2/move_group |
| `ModuleNotFoundError` for a Python package | installed into a venv | reinstall with `--user` |
| `cannot import name 'Inf' from 'numpy'` | NumPy 2 pulled in | `pip install --user --force-reinstall numpy==1.26.4` |
| Morty hears himself | echo-cancel not default | §9 |
| Astra light grey | the Astra driver on the shuttle PC froze | `ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh'` |
