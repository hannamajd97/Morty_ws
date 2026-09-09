# Architecture & Design Notes — `ur3e_fast_observation`

This document complements the [README](README.md) by explaining **what** the
package actually does under the hood, **why** the design choices were made,
and **how** to describe the technique honestly in a paper or thesis. It is
intentionally scoped: this package is engineering infrastructure for the
grasp-pipeline experiments, not a research contribution by itself.

---

## 1. Problem Statement

The grasp pipeline performs **two fundamentally different motion classes**:

| Class | Where the target comes from | Path constraint | Collision-free a priori? |
| --- | --- | --- | --- |
| **Observation moves** | Hand-taught named SRDF poses (`initgrasp`, `detect_left`, …) | None — any joint path is acceptable | **Yes** — operator placed them |
| **Grasp moves** | Computed at runtime from a neural network (GraspGen) on a fused point cloud | Must be a **straight Cartesian line** (LIN) so the fingers do not strike the object on approach | **No** — depends on object position |

Both motion classes used to be routed through `move_group` (MoveIt 2) with
the Pilz industrial planner. With the active planning scene populated by
the perception pipeline, every PTP request to `move_group` was paying
~25–40 s of planning-scene integration overhead, even though the
observation viewpoints are *known to be collision-free* (we taught them).

**Goal of this package:** keep the safe, planner-backed path for grasping,
but make observation transitions sub-second by skipping `move_group`.

---

## 2. The "Cache" — What Is Actually Stored

The cache is a flat YAML file. There is no learned model, no graph, no
roadmap. Each entry stores **six floating-point joint angles** in radians,
in the canonical UR3e arm order:

```yaml
joint_names:
  - shoulder_pan_joint
  - shoulder_lift_joint
  - elbow_joint
  - wrist_1_joint
  - wrist_2_joint
  - wrist_3_joint
poses:
  detect_left:
    positions: [3.5022, -2.8762, 0.7025, 1.3927, 1.2212, -1.2280]
    captured_at: 1777314776.523
    source: srdf:.../ur3e_with_gripper_3d.srdf
```

That is the entire state of the cache. A pose is fully described by a
vector `q_target ∈ ℝ⁶`.

The YAML can be populated two ways:

1. **`import_srdf`** — parses the existing `<group_state>` entries from
   `ur3e_with_gripper_3d.srdf`. This is the recommended path because
   MoveIt and the cache then share the **same source of truth**: edit the
   SRDF, re-import, both stay consistent.
2. **`teach_node`** — read-only subscriber to `/joint_states`; on operator
   command, it snapshots the current joint vector and writes it to YAML.
   Used for poses not present in the SRDF.

Neither tool ever commands the robot.

---

## 3. The "Math" — Where Smoothing Actually Happens

There is no novel mathematics in this package. The replay path is:

1. Read `q_target` from YAML (microseconds, dictionary lookup).
2. Build a `trajectory_msgs/JointTrajectory` containing **one waypoint**:
   - `positions = q_target`
   - `velocities = 0`
   - `time_from_start = T` (default 2.0 s)
3. Send it to
   `/scaled_joint_trajectory_controller/follow_joint_trajectory` via the
   standard `control_msgs/FollowJointTrajectory` action.

All trajectory generation happens **inside `ros2_control`**, specifically
inside the `JointTrajectoryController` (the scaled UR variant). For each
joint, given the current state `q_current`, target `q_target`, time horizon
`T`, and zero boundary velocities, the controller fits a **cubic Hermite
spline**:

```
q(t) = a0 + a1·t + a2·t² + a3·t³        for  t ∈ [0, T]

with boundary conditions
    q(0)   = q_current
    q(T)   = q_target
    q̇(0)   = 0
    q̇(T)   = 0
```

Solving the four constraints yields

```
a0 = q_current
a1 = 0
a2 =  3·(q_target − q_current) / T²
a3 = −2·(q_target − q_current) / T³
```

The controller streams these interpolated setpoints to the UR driver at
500 Hz. Joint-limit and velocity-limit safety is enforced by the
`scaled_joint_trajectory_controller` itself plus the URDF.

**This package writes none of this math.** It is standard `ros2_control`
behavior, available out of the box. The package's only contribution to
the runtime is steps 1–3 above.

> A reviewer who asks "what is the trajectory generator?" should be
> pointed at `ros2_control`'s `JointTrajectoryController` documentation;
> the relevant math is the cubic-spline interpolation reproduced above.

---

## 4. Why This Is Fast (and MoveIt Was Slow)

| Stage | MoveIt PTP via `move_group` | Direct trajectory replay |
| --- | --- | --- |
| RPC / action overhead | ~50 ms | ~10 ms |
| Planning-scene integration | **20–35 s** with active obstacles | n/a (skipped) |
| Path search (Pilz / OMPL) | 0.1–2 s | n/a (skipped) |
| Trajectory time parameterization | 50–200 ms | ~5 ms (cubic spline, 6 DoF) |
| Execution | **`T` seconds** (the actual move) | **`T` seconds** (the actual move) |

The execution duration `T` is identical between the two paths — both
ultimately stream the same controller setpoints. **The savings come
entirely from skipping a problem we have already solved manually**: the
operator vouched for the path's collision-freedom by recording the pose,
so no search is needed.

This is the correct trade-off for **observation** viewpoints (offline,
fixed, infrequently changed). It would be the **wrong** trade-off for
**grasp** approaches, where the target depends on the object position
detected at runtime.

---

## 5. Hybrid Motion Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Grasp pipeline orchestrator                 │
└───────────────┬───────────────────────────────────┬─────────────┘
                │                                   │
                │ named pose                        │ Cartesian target
                │ (initgrasp, detect_left, …)       │ (computed grasp)
                ▼                                   ▼
   ┌────────────────────────┐          ┌──────────────────────────┐
   │  ur3e_fast_observation │          │ grasp_executor_lin       │
   │      replay_node       │          │   (MoveIt + Pilz LIN)    │
   │                        │          │                          │
   │  YAML → JointTraj      │          │  Cartesian goal →        │
   │  → FollowJointTraj.    │          │  IK + collision-aware    │
   │                        │          │  straight-line plan      │
   │  ~2 s, no MoveIt       │          │  ~3–5 s w/o scene churn  │
   └───────────┬────────────┘          └─────────────┬────────────┘
               │                                     │
               └─────────────┬───────────────────────┘
                             ▼
              /scaled_joint_trajectory_controller
                             │
                             ▼
                       UR3e (500 Hz)
```

The replay node and the LIN executor share the **same low-level
controller**. They never compete for it because the orchestrator is
strictly sequential: an observation move completes before a grasp request
is issued, and vice-versa.

---

## 6. Safety Considerations

This package does not weaken any existing safety mechanism:

- **Joint limits:** enforced by the URDF and by `scaled_joint_trajectory_controller`.
- **Velocity limits:** enforced by the controller's `time_from_start`
  reasoning and by the UR's internal checks. A short `move_duration_s`
  on a long-distance request will be rejected by the controller, not
  silently sped through.
- **Collision avoidance:** the **operator** vouches for collision-freedom
  when teaching a pose. This is identical to the assumption made by every
  named SRDF `<group_state>` already in `ur3e_with_gripper_3d.srdf`.
- **Cartesian approach to objects:** never done by this package. Grasp
  approach motion stays on MoveIt + LIN, where collision-aware IK and
  planning-scene checks remain in force.
- **Gripper control:** never touched here. Pulse durations and open/close
  commands continue to flow through the existing `grasp_executor_lin`.

If a taught pose later becomes unsafe (e.g. furniture moved, object
fixture changed), the correct response is to re-teach it — exactly as
you would re-edit the SRDF entry. The cache is intentionally easy to
update.

---

## 7. How to Describe This in a Paper

The honest framing — what to claim and what **not** to claim:

**Safe claims (engineering, not research):**

> *"Observation viewpoints are pre-recorded joint configurations, replayed
> via the `JointTrajectoryController` action interface using a cubic
> spline parameterization (default 2 s per move). This bypasses online
> motion planning for taught, collision-free transitions while preserving
> MoveIt + Pilz LIN for the collision-aware Cartesian grasp approach."*

> *"This hybrid motion strategy reduces inter-view latency from
> approximately 27 s (PTP via `move_group` with an active perception
> planning scene) to roughly 2 s, dominated by the controller-side
> trajectory duration rather than planning overhead."*

**Avoid:**

- ❌ "novel motion planner" — this is `ros2_control`'s built-in spline.
- ❌ "outperforms MoveIt" — apples-to-oranges; MoveIt solves a harder
  problem (search) that we have eliminated by construction (taught poses).
- ❌ "online trajectory optimization" — we do offline pre-recording and
  online single-waypoint streaming. There is no optimization at runtime.

**The actual research contribution lives in `grasp_pipeline`** — the
utility-driven active perception loop. This package is the engineering
substrate that lets those experiments run at a usable cadence.

---

## 8. References & Vocabulary

The technique implemented here is widely known under multiple names:

- **"Cached / pre-recorded trajectories"** — most common informal term.
- **"Taught-pose replay" / "pose replay"** — industrial robotics.
- **"Direct trajectory streaming"** — sending joint goals straight to the
  low-level controller.
- **"Offline-planned, online-executed"** — academic phrasing.
- **"MoveIt bypass for taught configurations"** — descriptive shorthand.

Industrial controllers have done this for decades:

- Universal Robots: PolyScope `MoveJ` on a saved waypoint.
- ABB RAPID: `MoveJ jt_pose`.
- KUKA KRL: `PTP P1` on a recorded `JointPos`.

Our contribution is bringing the same pattern into the ROS 2 +
`ros2_control` stack alongside MoveIt-based grasping, so research on the
*perception* side is not held back by motion-planning latency on a path
that did not need to be re-planned.

---

## 9. Performance Verified On-Robot (2026-04-27)

Measured end-to-end times (`starting:LABEL` → `done:LABEL`), UR3e on real
hardware, `move_duration_s = 2.0`, `velocity_scaling` and
`acceleration_scaling` left at their controller defaults:

| Transition | Old (MoveIt PTP) | New (replay) |
| --- | --- | --- |
| `initgrasp` ↔ `initgrasp` (zero move) | n/a | 2.06 s |
| `initgrasp` → `detect_left` | ~27 s | 2.06 s |
| `detect_left` → `detect_right` | ~27 s | 2.06 s |
| `detect_right` → `detect_higher` | ~27 s | 2.06 s |
| `detect_higher` → `initgrasp` | ~27 s | 2.06 s |

Position error at the end of every move: **0.0000 rad on all 6 joints**.

The 2.06 s figure is dominated by `move_duration_s`; the replay-side
overhead (YAML lookup + message construction + spline fit + action
round-trip) measures ~60 ms.

---

## 10. Where to Go Next

- **Phase 3:** integrate `replay_node` into `grasp_pipeline` behind a
  launch flag (`use_fast_observation_moves:=true`, default `false`).
  The pipeline already requests transitions by name (`detect_left`,
  `detect_right`, `detect_higher`); rerouting those calls to the topic
  publisher of `replay_node` is the only change required. The MoveIt
  path stays as the documented fallback.
- **Pose hygiene:** if SRDF entries are edited, re-run `import_srdf` to
  refresh the cache. Treat the SRDF as authoritative; treat the YAML as
  derived.
- **`move_duration_s` tuning:** 2 s is a conservative default. Once
  `use_fast_observation_moves` is in production, this can be measured per
  transition and tightened toward the controller's velocity-limited
  minimum.
