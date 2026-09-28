# vla_experiments

Demonstration recording and dataset export for fine-tuning vision-language-action
models (**openpi** π0 / π0-FAST / π0.5 and **OpenVLA**) on the Morty UR3e.
Everything VLA-related lives here and in `~/morty_ws/vla_experiments/` so it never
mixes with the MMHRI / grasp-pipeline experiments in `experiments/`.

## What the models need (checked against the upstream repos, 2026-09-02)

| Item | openpi (LeRobot v2.1) | OpenVLA (RLDS) |
|---|---|---|
| Dataset format | LeRobot v2.1, built with `LeRobotDataset.create/add_frame/save_episode` at lerobot commit `0cf8648` | RLDS/TFDS via `rlds_dataset_builder` |
| Images | `image` (fixed cam) + `wrist_image`, uint8 HxWx3; UR5 example uses `base_rgb`/`wrist_rgb` names in its repack transform | `observation/image` (primary) + `observation/wrist_image`; model input is 224 px |
| State | `state` float32; UR5 example = 6 joints + gripper (7) | `observation/state`, POS_EULER = xyz rpy pad gripper (8) or JOINT = 7 joints(pad) + gripper (8) |
| Action | `actions` float32 (7); UR5 example stores absolute joint targets and lets `DeltaActions` in the data config turn them into deltas | `action` (7) = EEF delta xyz, delta rpy, gripper {0,1}  (ActionEncoding.EEF_POS) |
| Language | `task` string per frame (becomes `prompt` with `prompt_from_task=True`) | `language_instruction` per step |
| Rate | 10 Hz is what the LIBERO / UR5 examples use | any; 5–10 Hz typical |
| Compute | LoRA > 22.5 GB GPU, full > 70 GB; python ≥ 3.11 | LoRA ~72 GB at batch 16 on A100 (smaller batch works); 8×A100 for full |
| Register | new `DataConfig` + `TrainConfig` in `src/openpi/training/config.py`, then `compute_norm_stats.py` | entries in `configs.py`, `transforms.py`, `mixtures.py` |

So per step we must capture, with one shared clock: both RGB images, joint positions,
end-effector pose in `base_link`, gripper state, the commanded next target, and the
instruction. **Depth is not used by either model.** The laptop's GTX 1080 (8 GB) can
record and convert but cannot fine-tune; training goes on a big GPU box.

What to collect: 50–150 successful demonstrations per instruction, objects moved
between episodes, a few distractors, same camera mount and lighting for every episode
and at deployment. The camera pose is baked into the policy, so do not move the Astra
after you start recording.

## Raw format (recorder output)

Since 2026-09-24 a trial's episode is written INTO its trial folder, next to the bag
and the MMHRI files, so one experiment is one folder; the episode id is the folder
name and `episode_index` the trial number (trial.json names it as `vla_episode`):

```
~/morty_ws/experiments/experiments_24_09_2026/trial_070_LA_S4/
    meta.json        task, instructions (one per motion intent, with stamps), outcome, events, quality, intrinsics, Astra TF
    steps.npz        joint_pos/vel/cmd + joint_stamp, tcp/tool0 pose + tcp_stamp, ur_tcp_* (driver TCP), gripper,
                     gripper_cmd + gripper_cmd_age, grasp_detected, img_stamp_*/img_age_*, instruction_idx, phase, segment
    images/exterior/000000.jpg ...    Astra colour
    images/wrist/000000.jpg ...       RealSense colour
    bag/ audio.wav camera.mp4 trial.json ...
~/morty_ws/vla_experiments/datasets/morty_ur3e/episode_000000/   # episodes started from the CLI, same files
```
Actions are derived at conversion time from consecutive states (or the controller
reference when present), so one recording feeds both joint-space and EEF-delta targets.

Tools (2026-09-28): `ros2 run vla_experiments vla_view <trial dir> ...` writes `viewer.html` into the
folder (both frames per step with the step's time, joints, gripper, TCP, instruction, a timeline of joint
speed vs image change, the instruction list and every recorded event with its time; `viewer.html#<step>`
opens at a step). `ros2 run vla_experiments vla_relabel [--apply] <trial dir> ...` rebuilds instructions
and step labels from the saved events with the current rules (dry run without --apply; originals kept
as *.orig_<date>; v1 episodes are skipped). `ros2 run vla_experiments vla_joints <trial dir> ...` reads the
trial's bag and writes `joints_full.npz` / `joints_full.csv` (arm joints at ~500 Hz: position, velocity,
acceleration = d(velocity)/dt with a 20 ms centred average, motor current; plus the trajectory controller's
commanded position/velocity/acceleration, the latter only partly filled) and adds `joint_acc` /
`joint_current` to steps.npz. The MMHRI window runs both (vla_experiments.postprocess) after every saved
trial; output in postprocess.log. The window also waits for the bag to be subscribed before it says
"speak now" (the bag needs 2-6 s; earlier trials miss their first seconds in the bag).

What the v2 recorder guarantees (see `raw_dataset.py` for the exact keys):

* **Original stamps.** Every stream keeps its own header stamp next to the value
  (`joint_stamp`, `img_stamp_exterior/wrist`, `tcp_stamp`, `ur_tcp_stamp`, gripper
  command `stamp`), plus the tick time `stamp_ros`; delays are measurable afterwards.
* **No stale poses.** The ee/tool0 pose from TF is rejected (NaN, counted as
  `tf_stale`) when its stamp is older than `max_tf_age_s`; the driver's own
  `/tcp_pose_broadcaster/pose` is stored as `ur_tcp_*` (expressed in `base_link`)
  as an independent witness. The status topic lists every required stream and
  `stale` names the missing ones; an episode is **refused** while one is stale.
* **Gripper commands as events.** The gripper adapter publishes every open/close
  it executes on `/gripper_controller/command_event` with its stamp; the recorder
  stores the events (`events.gripper_cmd`), the last command per step
  (`gripper_cmd`) and its age, separately from the finger joint (`gripper`) and
  the grasp pin (`grasp_detected`). The quality report measures the state lag.
  **Do not use `grasp_detected`** as an input feature or a success label: the pin
  is HIGH whenever the fingers stall, on an object or on each other, so an empty
  closed gripper reads 1 (2026-09-28). It is recorded as raw data only.
* **Instructions with stamps.** Every accepted transcript and every intent is an
  event; each motion intent becomes an instruction (transcript + canonical text
  from the resolved intent, e.g. "pick up the bottle", "drop it on the right side",
  "go back home"), and every step carries `instruction_idx`.
* **Segments.** `segment` is idle / task / reset per step: task from the intent
  until the executor's verdict (or the next instruction), reset after a failure
  or an explicit reset-home. Holds inside an instruction (waiting at the hand-over)
  are task. The executor's phase marks are stored per step as `phase`.
* **Outcome** is `success` / `failure` / `unknown` (never silently true): the human
  verdict from the trial form, else the executor's `cycle_done` of that episode,
  else unknown.
* **Quality** (`meta.quality`): step gaps > `max_step_gap_s`, dropped ticks, image
  age, stale/frozen poses, gripper state lag, missing command events, missing
  instruction. The converters skip flagged episodes unless told otherwise.

## Recording alongside the MMHRI experiments (from the GUI — the normal way)

Open **Experiments ▸** in the GUI. The recorder node is launched with that window (no terminal
command), and the trial recorder's own buttons drive it:

| you press | MMHRI record | VLA episode |
|---|---|---|
| ⏺ Start trial | `experiments/experiments_DD_MM_YYYY/trial_NNN_LX_SN/` opens (the VLA episode is written into it) | `trial <fallback task>` → episode opens **provisional** |
| (during the trial) | video/audio/events | confirmed the moment an arm joint moves > 0.1 rad (or a motion intent arrives); the last command heard before that becomes the task |
| ⏹ Stop & save | folder + CSV row written | `stop ok/fail` if you filled *Full Task Success?*, else `stop auto` (the executor's `cycle_done` of this episode decides, otherwise outcome **unknown**) |
| ✖ discard | nothing saved | `discard` |

A trial in which the arm never moves (S1 "what do you see", S2 an absent object) is discarded by the
recorder itself — asking about objects never becomes training data — while a trial driven by a GUI
click (no speech, no intent) is kept because the arm moved. The **Record:** row in the window chooses
MMHRI evidence, VLA episode, or both (default); MMHRI off is a pure demonstration-collection session
that writes no trial folder. The trial window's status line says at ⏺ whether a VLA episode was
opened; the status bar's `astra` LED turns red if the fixed camera dies while the recorder is armed.

`meta.json` records `bound_to_trial`, `confirmed_by` (`motion` / `intent`), `task_from`
(`transcript` / `trial form`), every `transcripts` line, and `mmhri_trial_hint` (the trial folder).

The older transcript-driven **auto mode** still exists for headless use
(`ros2 launch vla_experiments vla_record.launch.py auto_mode:=true`), but do not run it next to the
GUI: both would try to own the episode.

## Recording workflow (manual)

1. Robot + wrist camera as usual (GUI "driver" or `ur3e_gripper_3d_moveit_real_camera.launch.py`).
2. Astra driver on the shuttle: `ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh'`
   (driver + relay; the recorder reads the relayed `/astra/color/compressed_slow` at 10 Hz, frames under `astra_link`).
3. Recorder + Astra static TF:
   ```bash
   ros2 launch vla_experiments vla_record.launch.py                      # both cameras required
   ros2 launch vla_experiments vla_record.launch.py require_exterior:=false   # wrist only, for dry runs
   ```
4. Per demonstration:
   ```bash
   vla_episode start "pick up the vitamin c and put it in the box"
   # ... drive the robot (voice command, GUI, replay poses, or freedrive) ...
   vla_episode gripper close        # optional: log the commanded gripper target explicitly
   vla_episode stop                 # or: vla_episode stop fail   /   vla_episode discard
   vla_episode status
   ```
   With `task_from_transcript:=true` the first accepted voice transcript becomes the task.
5. Inspect: `vla_dataset_info ~/morty_ws/experiments/experiments_24_09_2026 --all` (any root: a
   day folder, one trial folder, or the standalone dataset dir; without `--all` only successes)

Ticks are dropped (not written) when any required source is older than
`max_image_age_s` / `max_joint_age_s`; `meta.json` records how many and why, so a
stale camera is visible immediately.

## ROS bags (one per trial, since 2026-09-09)

Next to the recorder's own files, the GUI records a standard ROS 2 bag for
the span of every trial (`Record: ROS bag`, on by default): inside the
trial folder as `bag/`, or under `~/morty_ws/vla_experiments/bags/` for a
VLA-only trial. Topics (`morty_gui/config.py`, `BAG_TOPICS`):

| group | topics |
| --- | --- |
| cameras | `/camera/camera/color/image_raw/compressed` (wrist, 30 Hz), `/astra/color/compressed_slow` (fixed, 10 Hz), both `camera_info` |
| robot state | `/joint_states` (500 Hz), `/tcp_pose_broadcaster/pose`, `/scaled_joint_trajectory_controller/controller_state` (commanded vs actual), `/speed_scaling_state_broadcaster/speed_scaling`, `/io_and_status_controller/robot_program_running` |
| gripper | `/io_and_status_controller/io_states` (TI0 holding, TI1 slip, tool outputs), `/gripper_controller/joint_states` |
| commands / language | `/speech/accepted_transcript`, `/robot/intent`, `/yolo/all_detections`, `/graspgen/selection_cycle_id`, `/grasp_executor_lin_llm/selected_pose`, `/grasp_executor_lin_llm/interactive_status`, `/grasp_executor_lin_llm/cycle_done` |
| frames | `/tf`, `/tf_static` |

Every message keeps the ROS time it was published, so the streams align by
timestamp. About 2.5 MB/s; a 30 s demonstration is under 100 MB. Depth is
not recorded (18 MB/s raw, unused by the models). Read with `ros2 bag info`,
`ros2 bag play`, or `rosbag2_py` / the `rosbags` Python package on a
machine without ROS.

## Export

`<raw dir>` is any root holding episodes: `~/morty_ws/experiments` (every campaign day),
one day folder, one trial folder, or the standalone dataset dir. Both exports emit **one
training episode per instruction** from its task segment only (conversation, waiting and
resets are left out), with the canonical instruction as the language input; failures,
unknown outcomes and recorder-flagged episodes are skipped unless asked for.

openpi:
```bash
# on the training machine, inside openpi's uv env (lerobot pinned there)
python -m vla_experiments.convert_to_lerobot --raw <raw dir> --repo-id <user>/morty_ur3e --image-size 256
#   --include-failed --include-unknown --allow-quality-issues --keep-idle  (whole episode under one task)
uv run scripts/compute_norm_stats.py --config-name pi0_morty_ur3e_lora   # after adding the config
```
OpenVLA:
```bash
convert_to_rlds --raw <raw dir> --out ~/rlds/morty_ur3e
cd ~/rlds/morty_ur3e && MORTY_VLA_RAW=<raw dir> tfds build --overwrite   # in the rlds_dataset_builder conda env
#   MORTY_VLA_ALL=1 keeps failures/unknowns/flagged episodes and exports whole episodes
```
The builder docstring contains the exact `configs.py` entry for OpenVLA.

## Autostart

Both sides start by themselves after a reboot via `@reboot` cron jobs (`crontab -l`): the shuttle runs
`start_astra_on_shuttle.sh`, the laptop runs `scripts/astra_restart_laptop.sh` (logs in `~/astra_ws/logs/` and
`vla_experiments/logs/autostart.log`).

## Astra camera pose

`launch/astra_tf.launch.py` publishes `base_link -> astra_link`. Initial guess
(0, 0.709, 0) m with rpy (0, -1.3963, 1.5708); adjust with launch args until the Astra
cloud overlays the RealSense cloud in RViz, then update the defaults in the launch
file and `config/recorder.yaml` (the pose is stamped into every episode's meta.json).
