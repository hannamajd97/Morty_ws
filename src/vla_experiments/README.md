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

```
~/morty_ws/vla_experiments/datasets/morty_ur3e/
  episode_000000/
    meta.json        task, fps, success, intrinsics of both cameras, Astra TF, drop stats
    steps.npz        joint_pos/vel/cmd, tcp_pos/quat, tool0 pose, gripper, gripper_cmd, grasp_detected, ages
    images/exterior/000000.jpg ...    Astra colour
    images/wrist/000000.jpg ...       RealSense colour
```
Actions are derived at conversion time from consecutive states (or the controller
reference when present), so one recording feeds both joint-space and EEF-delta targets.

## Recording alongside the MMHRI experiments (from the GUI — the normal way)

Open **Experiments ▸** in the GUI. The recorder node is launched with that window (no terminal
command), and the trial recorder's own buttons drive it:

| you press | MMHRI record | VLA episode |
|---|---|---|
| ⏺ Start trial | `experiments/mmhri/trial_NNN_LX_SN/` opens | `trial <fallback task>` → episode opens **provisional** |
| (during the trial) | video/audio/events | confirmed the moment an arm joint moves > 0.1 rad (or a motion intent arrives); the last command heard before that becomes the task |
| ⏹ Stop & save | folder + CSV row written | `stop ok/fail` if you filled *Full Task Success?*, else `stop auto` (the executor's `cycle_done` decides) |
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
5. Inspect: `vla_dataset_info ~/morty_ws/vla_experiments/datasets/morty_ur3e`

Ticks are dropped (not written) when any required source is older than
`max_image_age_s` / `max_joint_age_s`; `meta.json` records how many and why, so a
stale camera is visible immediately.

## Export

openpi:
```bash
# on the training machine, inside openpi's uv env (lerobot pinned there)
python -m vla_experiments.convert_to_lerobot --raw <raw dir> --repo-id <user>/morty_ur3e --image-size 256
uv run scripts/compute_norm_stats.py --config-name pi0_morty_ur3e_lora   # after adding the config
```
OpenVLA:
```bash
convert_to_rlds --raw <raw dir> --out ~/rlds/morty_ur3e
cd ~/rlds/morty_ur3e && MORTY_VLA_RAW=<raw dir> tfds build --overwrite   # in the rlds_dataset_builder conda env
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
