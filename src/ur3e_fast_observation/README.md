# ur3e_fast_observation

Fast observation-pose motion for the UR3e arm.

> **Background reading:** [ARCHITECTURE.md](ARCHITECTURE.md) — what the cache
> actually stores, the cubic-spline math the controller runs, the hybrid
> observation/grasp design, on-robot performance numbers, and the safe way
> to frame this work in a paper.

The package provides three executables:

| Executable | Purpose | Motion? |
| --- | --- | --- |
| `import_srdf` | Import named `group_state` entries from a MoveIt SRDF into the YAML registry. **Recommended** path: reuses poses already defined in `ur3e_with_gripper_3d.srdf` (e.g. `initgrasp`, `detect_left`, `detect_right`, `detect_higher`). | No. |
| `teach_node` | Optional: snapshot the current `/joint_states` into the YAML registry for poses that are *not* in the SRDF. | **No** — read only. |
| `replay_node` | Replay a labelled pose by sending a single-waypoint `FollowJointTrajectory` goal directly to the `scaled_joint_trajectory_controller`, bypassing MoveIt for taught, collision-free observation viewpoints. | Yes. |

Grasp execution still goes through MoveIt (LIN). Only **observation** moves between hand-taught viewpoints are accelerated by this package.

## Why

Planning a PTP move with the active planning scene currently takes ~25–40 s
because every `move_group` request re-integrates the published collision
geometry. Hand-taught observation viewpoints are known to be collision-free,
so the planning step is pure overhead — replaying the joint trajectory
directly drops the move time well below one second.

## Quick start

### 1. Build

```bash
cd ~/morty_ws
colcon build --packages-select ur3e_fast_observation --symlink-install
source install/setup.bash
```

### 2. Populate the YAML registry

**Recommended — import from the SRDF you already maintain:**

```bash
ros2 run ur3e_fast_observation import_srdf \
  --srdf $HOME/morty_ws/src/ur3e_gripper_3d_moveit_config/config/ur3e_with_gripper_3d.srdf \
  --group ur3e_manipulator \
  --output $HOME/morty_ws/observation_poses.yaml \
  --names initgrasp detect_left detect_right detect_higher
```

This pulls existing taught poses (no re-teaching needed) into the YAML
schema consumed by `replay_node`. Re-run after editing the SRDF.

**Optional — teach extra poses interactively:**

```bash
ros2 launch ur3e_fast_observation teach.launch.py \
  output_file:=$HOME/morty_ws/observation_poses.yaml
```

| Command | Effect |
| --- | --- |
| `<Enter>` | capture current pose with auto label `view_NN` |
| `name=seed_view` | capture with explicit label |
| `list` | list stored entries |
| `drop=seed_view` | remove an entry |
| `quit` | exit |

The YAML file is rewritten atomically after every capture.

### 3. Replay

```bash
ros2 launch ur3e_fast_observation replay.launch.py \
  poses_file:=$HOME/morty_ws/observation_poses.yaml \
  default_pose_name:=seed_view
```

Or trigger replays from any node:

```bash
ros2 topic pub --once \
  /ur3e_fast_observation_replay/replay std_msgs/msg/String "{data: 'seed_view'}"
```

Watch the result on `/ur3e_fast_observation_replay/status`.

## Safety notes

- This package never plans Cartesian motion and never commands the gripper.
- Joint-limit and velocity safety is enforced by the running
  `scaled_joint_trajectory_controller` and the URDF — this node only sends
  joint targets.
- Only replay poses you have actually taught with `teach_node`. Editing the
  YAML by hand is supported but the operator owns the responsibility for
  collision-free targets.
