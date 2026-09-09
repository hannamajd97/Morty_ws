# morty_gui

The operator window. It owns every process of the stack and is the place
experiments are run and recorded from.

```bash
source ~/morty_ws/install/setup.bash
ros2 run morty_gui morty_gui
```

## What it does

- **Starts and stops the stack**: driver + MoveIt + RealSense, the
  GraspGen service, grasp pipeline, executor, voice stack
  (`managed_launch.py`, commands in `config.py`). Adopts processes left over from a previous GUI session and
  retires duplicate executors.
- **Status lights**: driver, RealSense, Astra, shuttle PC, RViz, robot link,
  pendant program (Play), voice. Pendant state and Remote mode are read
  from the UR dashboard every 5 s; in Remote mode a stopped program is
  restarted automatically, and a dropped dashboard socket is reconnected.
- **Camera and 3D views** (`camera_panel.py`, `gl_view.py`): the detection
  overlay with pointing, selection by click, RealSense or Astra view; the
  object cloud with kept / rejected / best / executed grasp glyphs.
- **Pick control**: execute / stop, auto-execute, move-to named poses,
  gripper open/close, E-Stop (cancels goals, stops the pendant program).
  Refuses execute and jog while the pendant program is stopped, and keeps
  `scaled_joint_trajectory_controller` active while it runs.
- **Experiments** (`experiments_window.py`, `mmhri_window.py`,
  `scenario_experiments.py`): MMHRI trial recorder — trial folders
  `trial_NNN_LX_SN` with logs, CSV summary, 3D view, modifiers M1–M3;
  starts and stops a VLA episode with each trial (`vla_link.py`) and a
  ROS bag for the span of the trial (`bag_link.py`, topic list in
  `config.BAG_TOPICS`). The Record row chooses MMHRI evidence, VLA
  episode, ROS bag; all three on by default.
- **Motion window** (`motion_window.py`): jog and named poses.
- **Advanced**: settings profiles (`profiles.py`), executor variant, view
  options.

## Files

| file | role |
| --- | --- |
| `app.py` | main window, toolbar, bottom bar, tick loop |
| `ros_bridge.py` | the ROS node behind the GUI: subscriptions, publishers, dashboard and controller-manager clients, watchdogs |
| `config.py` | launch commands, IPs, topic names, colours |
| `managed_launch.py` | start/stop/adopt of `ros2 launch` processes with per-process logs in `logs/*_gui.log` |
| `html_viewer.py` | writes `experiments/latest/view3d.html` |

## Recorded data

Trials go to `experiments/<session>/trial_NNN_LX_SN/`; the latest cloud and
grasps are always in `experiments/latest/`. VLA episodes are written by
`vla_experiments` (see its README).
