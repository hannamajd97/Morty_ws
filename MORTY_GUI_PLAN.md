# Morty GUI — Plan & Status

*Created 2026-08-07. Package: `src/morty_gui/`. This is the plan of record —
update it as phases land.*

## How to run

```bash
# Terminal A (robot+camera, as always):
ros2 launch ur3e_gripper_3d_moveit_commander ur3e_gripper_3d_moveit_real_camera.launch.py

# Terminal B (pipeline, as always):
ros2 launch grasp_pipeline run_grasp_pipeline.launch.py

# Terminal C (the new GUI):
ros2 run morty_gui morty_gui
```

The GUI is **additive**: it runs alongside the old windows and changes nothing
in the pipeline. It talks to `grasp_pipeline` only via topics.

## Window 1 — "Morty Operator" (opens on start)

```
┌─ Morty Operator ────────────────────────────────────────────┐
│ [Advanced ▸] [Reset 3D view]                                │
│ System: [Start driver] ☑camera ☑RViz  [Start pipeline]      │
│         ☐multiview                                          │
│ ┌───────────────────┐ ┌───────────────────────────────────┐ │
│ │ live camera with  │ │ 3D FINAL RESULT (drag to orbit)   │ │
│ │ YOLO boxes        │ │  · colored object cloud           │ │
│ │                   │ │  · green = kept grasps            │ │
│ │ CLICK an object   │ │  · red   = filtered out           │ │
│ │ to select it      │ │  · blue  = GraspGen best          │ │
│ │                   │ │  · ORANGE = robot's executed pick │ │
│ └───────────────────┘ └───────────────────────────────────┘ │
│ [▶ EXECUTE]  Send robot to: [home ▼] [GO]                   │
│              ☑ green kept ☑ red rejected ☑ blue best        │
│ status bar: last pipeline event / U value / move status     │
└─────────────────────────────────────────────────────────────┘
```

- **Click to select** publishes on `/yolo/debug_image_mouse_left` — identical
  to the old click window, so voice/pointing/click all still work.
- **▶ EXECUTE** spawns the executor launch (mode chosen in Advanced; LIN by
  default with the verified argument set). Button becomes **■ STOP** (SIGINT
  to the launch group) while running. Publishes `/morty/executor_active`.
- **Send robot to + GO**: all SRDF arm poses — home, init, drop,
  drop_left_side, drop_right_side, detect_left, detect_right, detect_higher,
  old_initgrasp, sleep_home — via `observation_ptp_commander` (PTP, vel 0.2).
- **Checkboxes** toggle green/red/blue glyphs; orange is always shown.
- **Reset 3D view** re-frames the camera on the current cloud.
- **System bar** (top row): launches/stops the whole stack from the GUI.
  - **Start driver** — with **☑ camera** it runs
    `ur3e_gripper_3d_moveit_real_camera.launch.py`, without it the
    no-camera `..._real.launch.py`; **☑ RViz** maps to the new
    `start_rviz:=true/false` launch argument (added 2026-08-07 — RViz used
    to be unconditional).
  - **Start pipeline** — `run_grasp_pipeline.launch.py`, with **☐ multiview**
    adding `registration_enabled:=true`.
  - Stop buttons use **escalating shutdown**: SIGINT → 6 s → SIGTERM →
    4 s → SIGKILL, because move_group can segfault mid-shutdown on Humble
    and wedge `ros2 launch` forever (observed live).
  - Closing the GUI **never** stops the driver/pipeline — they run in their
    own sessions; only the System bar buttons stop them.

## RESTRUCTURED 2026-08-07 evening: now THREE windows

Toolbar buttons: **Motion ▸** and **Experiments ▸** (each opens/closes its
window; the old single "Advanced" window is gone).

### Window 2 — "Morty Motion" (RViz-MotionPlanning style)

- **Planning Library**: pilz_industrial_motion_planner (PTP/LIN/CIRC) or
  OMPL (RRTConnect, RRT, RRT*, EST, KPIECE, BKPIECE, LBKPIECE, PRM, PRM*,
  SBL, TRRT) — applies to Send-robot-to and joint goals.
- **Options**: planning time, attempts, velocity & accel scaling.
- **⚡ Fast replay** checkbox: Send-robot-to first tries the pipeline's
  cached-trajectory replay node (`/ur3e_fast_observation_replay/replay`) and
  falls back to the planner automatically on failure/timeout (needs the
  pipeline running; cache file `observation_poses.yaml`).
- **Executor mode** (LIN verified / PTP / PILZ) — moved here.
- **Gripper pulse holds** — moved here.
- **Joints panel** like RViz's Joints tab: 6 sliders/spinboxes in degrees,
  "Read current joints" from the live robot, "Plan & Execute joint goal"
  (via `use_joint_target` + `joint_target_0..5` on the commander).

### Window 3 — "Morty Experiments"

- Tab **Live debug**: filter-chain table, U(v) plot, pipeline log tail.
- Tab **Filter parameters**: reads the 21 grasp-filter parameters live off
  the pipeline node (finger-penetration geometry, behind/side rejection,
  centroid gates, orientation filters/reweights, cloud pre-filters),
  editable table + Apply → `set_parameters` on the running node. Values
  take effect next cycle; reset on pipeline restart.
- Experiment recorder bar (as before).

### System bar additions

- **● camera** LED (`camera_info` freshness) and **● rviz** LED (process
  check every 2 s), joining driver/robot/play.
- Camera and RViz checkboxes are **live**: while the driver runs, unticking
  kills that component (RViz via pkill; camera via pkill of the RealSense
  node), re-ticking starts it standalone (`moveit_rviz.launch.py` /
  `gripper_3d realsense_d345_hardware.launch.py`) without touching the arm.

## OLD Window 2 layout (superseded, kept for history)

```
┌─ Morty Advanced — debug ────────────────────────────────────┐
│ Executor mode:  (•) LIN (verified)  ( ) PTP  ( ) PILZ       │
│ ┌──────────────────────┐ ┌────────────────────────────────┐ │
│ │ filter stage │ left  │ │  U(v) utility plot per cycle   │ │
│ │ graspgen raw │ 1275  │ │      ·─·                       │ │
│ │ no-under     │ 1275  │ │   ·─·   ·                      │ │
│ │ finger-pen.  │ 1243  │ │                                │ │
│ │ ...          │ ...   │ └────────────────────────────────┘ │
│ └──────────────────────┘                                    │
│ pipeline log tail (live, filtered to yolo_live_select_grasp)│
└─────────────────────────────────────────────────────────────┘
```

- **Executor mode** feeds the EXECUTE button: LIN is the verified pick
  executor; PTP and PILZ launch with their default arguments — experiments.
- Filter table, U(v) plot and log tail are parsed live from `/rosout` —
  **zero pipeline modifications needed**.

## Architecture (why it's built this way)

- `ros_bridge.py` — one rclpy node spun in a background thread; everything
  lands in lock-guarded holders; a 10 Hz Qt timer polls them. GUI never
  blocks ROS, ROS never touches Qt (the Day-1 threading lesson).
- `gl_view.py` — pyqtgraph OpenGL panel; gripper glyph geometry + flip_z
  conventions copied 1:1 from the verified `grasp_viz_viewer`.
- `app.py` — the two windows; forces `QT_QPA_PLATFORM=xcb` (GLX via
  XWayland) and scrubs pip-opencv's Qt plugin path (both known landmines).
- Extra deps: `pyqtgraph` + `PyOpenGL`, installed `--user --no-deps` so the
  numpy/torch pins stay untouched.

## FAQ

- **What motion does "Send robot to" use?** `observation_ptp_commander`:
  PILZ industrial planner in **PTP** mode (joint motion), falling back to
  PILZ LIN, then OMPL if allowed — at velocity/acceleration scaling 0.2.
  Not the grasp executor; EXECUTE is a separate LIN pipeline.

## What's MISSING (the roadmap)

**Phase 2 — operator completeness**
- [x] System launcher: driver ± camera ± RViz, pipeline ± multiview (2026-08-07)
- [x] Offline mode: "real robot / offline (fake)" selector — base launch with
      `use_fake_hardware:=true use_ur_driver:=false` (2026-08-07)
- [x] Health LEDs in the System bar (2026-08-07): **● driver** green when
      `/joint_states` flows (fake or real), red if launched but silent, grey
      when off; **● robot** green when the real UR link publishes
      `io_and_status_controller/robot_program_running`; **● play** green =
      pendant program RUNNING, orange = STOPPED (press Play), grey = unknown.
      Hover any LED for the explanation.
- [x] GO exit codes translated in the status bar (0 OK / 1 setup / 2 planning
      failed / 3 execution failed → check pendant Play) (2026-08-07)
- [ ] Camera-pose-anchored initial 3D view (face the object as the camera
      saw it, like the Open3D window does; needs `/graspgen/viz/camera_pose`)
- [ ] "Morty Frames" as a second tab/panel in the 3D area (per-frame
      captured cloud during sweeps, from `/yolo/selected_cloud_rgb`)
- [ ] Mode switch in the GUI: single view ↔ multiview, frames count —
      needs a small pipeline param service call (`set_parameters`)
- [x] Gripper open/close buttons (2026-08-07): Operator bar, via the same
      `/gripper_controller/gripper_cmd` GripperCommand action MoveIt uses;
      pulse holds editable in Advanced (open 0.5 s default / 1.0 max, close
      1.2 s default / 1.5 max — adapter defaults) and pushed to the adapter
      as live parameters on every press.
- [x] "Send robot to" motion settings in Advanced (2026-08-07): planner
      choice (PILZ PTP default / PILZ LIN / OMPL RRTConnect), velocity &
      accel scaling (0.20), planning time (5 s), planning attempts (10) —
      the RViz MotionPlanning panel's knobs, applied to every GO press.
- [ ] Launch-flag to stop opening the old click/Open3D windows once trusted

**Phase 3 — the thesis debug suite (approved "item 5" design)**
- [x] Experiment recorder v1 (2026-08-07): "● Record experiments" button in
      the Advanced window → `~/morty_ws/experiments/exp_<stamp>.csv`, one row
      per cycle (cycle id, best score, U, full filter chain) + executor exit
      events. Missing still: per-stage timings, object label column.
- [ ] Per-stage filter toggles WITH their parameter values (enable/disable a
      filter from the GUI — needs pipeline params per stage)
- [ ] Session timeline (list of cycles with outcome badges)
- [ ] Utility-stop markers on the U(v) plot (τ, K, drop threshold events)
- [ ] Voice/LLM status panel (what Morty heard / replied)

**Parked by Majd's priority call (2026-08-11: "voice is more important now")**
- [ ] Multiview cycle shows NO grasp glyphs in the Operator 3D panel (seen
      2026-08-11; suspicion: multiview viz path publishes poses that don't
      land near the TSDF cloud — multiview viz was never live-tested before
      because all earlier sweeps captured 0 frames). Investigate `[viz]`
      log frame vs cloud frame on the next multiview run.
- [ ] "Choose the grasp": show ALL grasps in the Scene/3D view and let the
      user CLICK one to execute exactly that candidate (needs an executor
      override input — real feature, not a toggle).

**Known limits right now**
- PTP/PILZ executor modes run with launch defaults, not the tuned LIN args.
- The 3D view re-frames on every new final cloud (once per cycle — fine, but
  it forgets your orbit between cycles).
- No experiment persistence yet — everything shown is live-only.
