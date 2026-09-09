# ur3e_gripper_3d_moveit_commander

The grasp executor and the arm-side helpers, in C++ on MoveIt 2.

## grasp_executor_lin_llm — the executor Morty uses

Persistent node, one pick cycle per selection. Started by the GUI:

```bash
ros2 launch ur3e_gripper_3d_moveit_commander grasp_executor_lin_llm.launch.py \
    best_pose:=true use_candidate_pose_array:=true execute:=true \
    candidate_wait_timeout_s:=20.0 enable_wrist_twin_rewrite:=false
```

Cycle:

1. Waits for `/graspgen/selection_cycle_id` and the candidate array on
   `/graspgen/candidate_poses_base_link`.
2. Pre-filter: candidates without a quick IK solution are moved to the end
   of the probe order, never dropped.
3. Probes each candidate with a Pilz **LIN** plan from the current pose to
   the pre-grasp and along the straight approach. A rejection names its
   cause: the colliding pair with penetration depth (from
   `/check_state_validity`), "mid-approach" when the goal is free but a
   sample on the straight line collides, IK failure, or a planner failure.
   Per-candidate verdicts go to `/grasp_executor_lin_llm/candidate_status`.
4. Executes the first feasible one: open gripper, LIN approach, close, lift
   10 cm, PTP to `home`, then the drop pose (`drop_left_side` /
   `drop_right_side`), release. Pauses for a voice destination or release
   when `wait_for_destination` / `wait_for_release` are set.
5. Publishes `/grasp_executor_lin_llm/cycle_done` so the pipeline re-arms.

Every trajectory goes through the `/execute_trajectory` action with a
deadline; a controller that never completes cannot wedge the node. The abort
topic `/grasp_executor_lin_llm/abort` stops the current motion and the cycle.

Slip watchdog: the gripper signals "holding" on tool input TI0 and "slip" on
TI1 (it re-squeezes on its own); the executor drops the UR speed slider to
`slip_slow_slider` on TI1 and restores it when the slip clears, and reports
on `/executor/slip_event`. `slip_watchdog_enabled:=false` turns it off.

Key parameters: `pregrasp_approach_distance`, `approach_autopsy_samples`,
`candidate_wait_timeout_s`, `execute`, `use_candidate_pose_array`,
`wait_for_destination`, `wait_for_release`, `drop_target_name`,
`skip_reset_before_drop` (keep it: the canonical flow goes through home),
`slip_watchdog_enabled`, `slip_slow_slider`, `enable_wrist_twin_rewrite`.

Topics used: `/joint_states`, `/gripper_controller/gripper_cmd` (action),
`/move_to_named_pose` (action, fast replay of named poses),
`/io_and_status_controller/io_states`, `/io_and_status_controller/set_speed_slider`,
`/check_state_validity`, `/execute_trajectory`.

## Other executables

| executable | role |
| --- | --- |
| `grasp_executor`, `grasp_executor_pilz`, `grasp_executor_lin` | earlier executor variants (PTP / Pilz / LIN without the voice integration); selectable in the GUI's Advanced window, not used in the experiments |
| `observation_ptp_commander` | PTP moves to joint targets or named poses; the GUI's "Move to" buttons |
| `gripper_action_server` | `OperateGripper` action bridge used by the voice stack ("open/close the gripper") |
| `plan_reuse_tolerance_check` | unit check for the probe-plan reuse predicates |

## Build

```bash
colcon build --packages-select ur3e_gripper_3d_moveit_commander --cmake-args -DCMAKE_BUILD_TYPE=Release
```

About 2.5 minutes; the running executor must be restarted afterwards.
