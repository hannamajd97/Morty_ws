"""
named_pose_server.py
====================

Action server for ``morty_interfaces/action/MoveToNamedPose``. Front-door for
"go to pose X" requests across the whole system (chat layer, grasp pipeline,
multi-view scan). It looks up the target named pose's joint values in the
``observation_poses.yaml`` registry, performs a MoveIt
``/check_state_validity`` pre-check to refuse poses that would collide with
the planning scene, then delegates execution to the replay node by publishing
on ``/ur3e_fast_observation_replay/replay``.

The split is deliberate: the **replay** node owns the controller action
client and the actual trajectory streaming; this **server** owns the
high-level "is the target safe?" check and the cancellation semantics. That
way the heavy controller plumbing is not duplicated for every consumer that
needs a named-pose move.

Cancellation:

* ``CancelResponse.ACCEPT`` for in-flight goals — the server forwards a
  cancel signal on ``/ur3e_fast_observation_replay/cancel`` so the replay
  node stops the trajectory mid-move.
* ``ReentrantCallbackGroup`` so a cancel can be processed while a goal is
  still executing (otherwise the goal would hold the spin and block).

Velocity scaling: the goal's ``velocity_scaling`` field is mapped to one of
three pre-tuned move durations (``slow_duration_s``, ``fast_duration_s``,
``very_fast_duration_s``) rather than passed through as a continuous scale —
this is more predictable on the real robot than letting callers pick
arbitrary speeds.

The ``poses_file`` default is ``$HOME/morty_ws/observation_poses.yaml``
(not the old machine path) — override via the parameter when running under a
different layout.
"""
from __future__ import annotations

import json
import math
import os
import faulthandler
import sys
import threading
import time
from typing import List, Optional, Sequence, Tuple

import rclpy
import yaml
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.wait_for_message import wait_for_message
from rclpy.node import Node

from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from morty_interfaces.action import MoveToNamedPose

# ── blended multi-pose trajectory construction ───────────────────────────
#
# Pure, ROS-free helpers so the geometry/timing can be unit-tested without
# a robot, a controller, or even an rclpy context.
#
# WHY (2026-08-28): the old flow sent the controller a SINGLE waypoint per
# leg, so lift->home->drop was two goals, each with its own start-from-rest
# / stop-at-rest profile AND its own ~2 s fixed overhead (action round
# trip + collision pre-check + controller goal acceptance). Measured:
# 3.7 s + 3.9 s = 7.6 s with a hard stop at home. Blending pays the
# overhead once and never lets the arm reach zero speed at the via.
#
# HOW: the whole path gets ONE normalised progress coordinate phi in
# [0, 1], measured as ARC LENGTH in joint space (see the note in
# build_blended_trajectory about why length, not clock share). Real time
# is then warped through the S-curve: phi = S(tau), tau = t / T. Sampling
# uniformly in phi (not in tau) is deliberate — it puts a trajectory point
# EXACTLY on every via pose, and the closed-form inverse S-curve gives
# that point its time. The velocity profile is therefore
# |dq/dt| = (path length / T) * S'(tau): zero at both ends, peaking at
# 1.5x the path average in the middle, and comfortably non-zero wherever
# the vias happen to fall. That is the blend.
#
# Because phi is arc length, the peak is the SAME 1.5x-of-average that a
# single quintic leg would produce, just taken over the whole path instead
# of per leg — the arm is not pushed harder than one of today's legs, it
# simply never stops between them.

BLEND_SAMPLES_PER_SEGMENT = 40   # 2026-09-23: 20 gave the controller a point every 0.2 s on an 8 s path
"""Trajectory points sampled per path segment (excluding the shared
start point). 20 keeps the controller interpolating between points that
are milliseconds apart instead of inventing its own start-stop ramp."""


def smoothstep(tau: float) -> float:
    """Classic S-curve S(t) = 3t^2 - 2t^3 on [0, 1]. S'(0) = S'(1) = 0."""
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return 1.0
    return tau * tau * (3.0 - 2.0 * tau)


def smoothstep_inverse(phi: float) -> float:
    """Exact inverse of :func:`smoothstep`.

    Solve 3t^2 - 2t^3 = phi on [0, 1]; the trigonometric root is

        tau = 1/2 - sin(asin(1 - 2*phi) / 3)

    Having the closed form (rather than sampling uniformly in time and
    accepting whatever positions fall out) is what lets us pin trajectory
    points to the via poses exactly while still S-curving the timing.
    """
    if phi <= 0.0:
        return 0.0
    if phi >= 1.0:
        return 1.0
    # asin argument is in [-1, 1] by construction; clamp against fp drift.
    arg = max(-1.0, min(1.0, 1.0 - 2.0 * phi))
    return 0.5 - math.sin(math.asin(arg) / 3.0)


def build_blended_trajectory(
    start_joints: Sequence[float],
    targets: Sequence[Sequence[float]],
    segment_durations: Sequence[float],
    samples_per_segment: int = BLEND_SAMPLES_PER_SEGMENT,
    min_dt_s: float = 1e-4,
) -> Tuple[List[List[float]], List[float]]:
    """Build ONE dense, S-curve-timed joint path through several poses.

    Parameters
    ----------
    start_joints:
        Current joint configuration; becomes trajectory point 0 at t=0
        (same convention MoveIt uses, so the controller has a consistent
        anchor instead of jumping to an unrelated first point).
    targets:
        The ordered poses to pass through; ``targets[-1]`` is the final
        destination and everything before it is a via.
    segment_durations:
        One duration per segment (``len(targets)`` of them). Only their
        SUM is binding: it is the total move time. How that total is
        spread over the path is decided by arc length, not by which
        segment contributed which duration. No new speeds are invented
        here — the caller passes the existing per-tier durations.
    samples_per_segment:
        Trajectory points generated per segment, excluding the shared
        start point. Via poses land exactly on a sample boundary.

    Returns
    -------
    (positions, times)
        ``positions[i]`` is a joint vector, ``times[i]`` its
        ``time_from_start`` in seconds. ``times`` is guaranteed strictly
        increasing, ``positions[0] == start_joints``,
        ``positions[-1] == targets[-1]``, and every via pose appears
        verbatim at a segment boundary.
    """
    n_seg = len(targets)
    if n_seg < 1:
        raise ValueError("build_blended_trajectory needs at least one target")
    if len(segment_durations) != n_seg:
        raise ValueError(
            f"segment_durations has {len(segment_durations)} entries for "
            f"{n_seg} segment(s)"
        )
    n_joints = len(start_joints)
    for i, tgt in enumerate(targets):
        if len(tgt) != n_joints:
            raise ValueError(
                f"target {i} has {len(tgt)} joints, expected {n_joints}"
            )
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be >= 1")
    durations = [float(d) for d in segment_durations]
    if any(d <= 0.0 for d in durations):
        raise ValueError(f"segment durations must be positive: {durations}")

    total_s = sum(durations)

    nodes: List[List[float]] = [[float(v) for v in start_joints]]
    nodes.extend([float(v) for v in t] for t in targets)

    # phi boundary of each segment. phi is ARC LENGTH along the joint-space
    # path, normalised to [0, 1] — NOT each segment's share of the clock.
    #
    # This matters on the real rig. lift->home is a ~1.5 rad move, but
    # home->drop_right_side swings the elbow 3.7 rad. Splitting the time
    # evenly (one tier duration each) would run the long segment ~2.5x
    # faster in joint space: at the 0.50 tier the elbow peaks at
    # 3.48 rad/s, i.e. 111% of the UR3e's 180 deg/s elbow limit, which is
    # exactly the overspeed that protective-stops the arm. Allocating phi
    # by arc length instead makes |dq/dt| identical everywhere on the
    # path, so the whole move runs at ONE speed set by (total arc length
    # / total duration) and the same 0.50 tier peaks at 76% of the limit
    # (measured on the taught home / drop_right_side values).
    #
    # The contract the caller cares about is unchanged: the TOTAL duration
    # is still exactly the sum of the per-segment tier durations, and no
    # new speed tiers are invented.
    seg_len = [
        math.sqrt(
            sum(
                (b - a) ** 2
                for a, b in zip(nodes[i], nodes[i + 1])
            )
        )
        for i in range(n_seg)
    ]
    path_len = sum(seg_len)
    bounds: List[float] = [0.0]
    acc = 0.0
    if path_len > 0.0:
        for L in seg_len:
            acc += L
            bounds.append(acc / path_len)
    else:
        # Degenerate: every node identical. Nothing to arc-length by, so
        # fall back to the clock split; the arm just dwells in place.
        for d in durations:
            acc += d
            bounds.append(acc / total_s)
    bounds[-1] = 1.0

    positions: List[List[float]] = [list(nodes[0])]
    times: List[float] = [0.0]

    # GEOMETRY (2026-09-23, Majd: "not smooth, it hits all its joints at
    # the via"): with vias the path used to be straight joint-space lines
    # meeting at a corner, and the S-curve keeps full speed there — so the
    # shoulder and elbow reversed direction instantly at the waypoint
    # (93 deg turn measured for home -> vla_via -> vla_drop_right). Now
    # the path is a Catmull-Rom spline THROUGH the nodes (end tangents
    # clamped): every via is still hit exactly, the direction turns
    # smoothly, and the curve bulges outward from the corner, i.e. away
    # from the chord the via was taught to avoid. One segment (no via)
    # stays the straight line it always was. phi is arc length along the
    # actual geometry, warped by the S-curve as before.
    if n_seg == 1:
        geom = [[list(nodes[0]), list(nodes[1])]]
    else:
        ext = [nodes[0]] + nodes + [nodes[-1]]

        def _cr(p0, p1, p2, p3, u):
            u2, u3 = u * u, u * u * u
            return [0.5 * ((2 * b) + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u2
                           + (-a + 3 * b - 3 * c + d) * u3)
                    for a, b, c, d in zip(p0, p1, p2, p3)]
        fine = 200
        geom = [[_cr(ext[seg], ext[seg + 1], ext[seg + 2], ext[seg + 3], k / fine) for k in range(fine + 1)]
                for seg in range(n_seg)]
    seg_len = [sum(math.sqrt(sum((b - a) ** 2 for a, b in zip(pts[i], pts[i + 1])))
                   for i in range(len(pts) - 1)) for pts in geom]
    path_len = sum(seg_len)
    bounds = [0.0]
    acc = 0.0
    for L in seg_len:
        acc += L
        bounds.append(acc / path_len if path_len > 0.0 else 0.0)
    if path_len <= 0.0:
        bounds = [i / float(n_seg) for i in range(n_seg + 1)]
    bounds[-1] = 1.0

    for seg in range(n_seg):
        pts = geom[seg]
        cum = [0.0]                       # arc length inside this segment
        for i in range(len(pts) - 1):
            cum.append(cum[-1] + math.sqrt(sum((b - a) ** 2 for a, b in zip(pts[i], pts[i + 1]))))
        seg_total = cum[-1] if cum[-1] > 0.0 else 1.0
        phi_a = bounds[seg]
        phi_b = bounds[seg + 1]
        for k in range(1, samples_per_segment + 1):
            frac = k / float(samples_per_segment)
            if k == samples_per_segment:
                q = list(nodes[seg + 1])                    # the via / target itself
            else:
                want = frac * seg_total
                j = 1
                while j < len(cum) - 1 and cum[j] < want:
                    j += 1
                span = cum[j] - cum[j - 1]
                w = (want - cum[j - 1]) / span if span > 0.0 else 1.0
                q = [(1.0 - w) * a + w * b for a, b in zip(pts[j - 1], pts[j])]
            phi = phi_a + (phi_b - phi_a) * frac
            t = total_s * smoothstep_inverse(phi)
            # fp guard: consecutive phi are far apart at these sample
            # counts, but never emit a non-increasing stamp — the
            # JointTrajectoryController rejects the whole goal for it.
            if t <= times[-1]:
                t = times[-1] + min_dt_s
            positions.append(q)
            times.append(t)

    # Pin the endpoints exactly (the interpolation above already lands on
    # them to fp precision; this removes any doubt for the controller's
    # goal-tolerance check).
    positions[-1] = list(nodes[-1])
    times[-1] = max(total_s, times[-2] + min_dt_s)

    for i in range(1, len(times)):
        assert times[i] > times[i - 1], (
            f"time_from_start not strictly increasing at point {i}: "
            f"{times[i - 1]} -> {times[i]}"
        )
    return positions, times


class MoveToNamedPoseServer(Node):
    """Action server that drives ur3e_fast_observation_replay via topics.

    The replay node subscribes to a String topic for the target pose name and
    publishes a String status topic with values like "starting:<name>",
    "done:<name>", and "failed:<name>". This server wraps that pattern in a
    proper Action with cancellation, timeout handling, and an optional
    collision pre-check.

    Collision pre-check
    -------------------
    Before publishing the replay request, we (optionally) interpolate the
    joint trajectory from the current state to the target pose into N
    waypoints and ask MoveIt's /check_state_validity service whether each
    waypoint is in collision against the published planning scene (table,
    walls, mount, etc.). If any waypoint collides, we refuse the goal with
    a clear error so the arm doesn't crash through known obstacles.

    Toggle with `collision_check:=false`.
    """

    def __init__(self) -> None:
        super().__init__("move_to_named_pose_server")

        self.declare_parameter("server_action_name", "/move_to_named_pose")
        self.declare_parameter("replay_topic", "/ur3e_fast_observation_replay/replay")
        self.declare_parameter("status_topic", "/ur3e_fast_observation_replay/status")
        self.declare_parameter("default_timeout_s", 30.0)
        # Collision pre-check params
        self.declare_parameter("collision_check", True)
        self.declare_parameter("collision_waypoints", 10)
        self.declare_parameter(
            "validity_service_name", "/check_state_validity"
        )
        self.declare_parameter("planning_group", "ur3e_manipulator")
        # Path to the same YAML the replay node uses (so we know target joints).
        # $HOME-relative so it is not tied to a specific machine/username.
        # Honour $WS_DIR (same convention as grasp_pipeline/selection/params.py)
        # and fall back to ~/morty_ws.
        self.declare_parameter(
            "poses_file",
            os.path.join(
                os.environ.get(
                    "WS_DIR",
                    os.path.expanduser("~/morty_ws"),
                ),
                "observation_poses.yaml",
            ),
        )
        self.declare_parameter("validity_call_timeout_s", 1.0)
        # Slow / fast / very_fast move durations (seconds). The action's
        # velocity_scaling field is mapped to one of these via the rule:
        #   <= 0.0       -> slow_duration_s   (default)
        #    0.0..0.5    -> slow
        #    0.5..0.8    -> fast
        #    > 0.8       -> very_fast
        # Speech: "go home"            -> slow      (default per user)
        #         "go home fast"       -> fast      (~65% of slow)
        #         "go home very fast"  -> very_fast (~35% of slow)
        # Five speed tiers (durations in seconds). Capped at the
        # 50% velocity-scaling ceiling — going faster overspeeds the
        # UR3e controller and triggers protective stops. Lower
        # duration = faster move.
        self.declare_parameter("very_slow_duration_s", 12.0)  # ~5%
        self.declare_parameter("slow_duration_s",       4.0)  # ~20% — default
        self.declare_parameter("normal_duration_s",     2.7)  # ~30%
        self.declare_parameter("fast_duration_s",       2.0)  # ~40%
        self.declare_parameter("very_fast_duration_s",  1.6)  # ~50% (ceiling)
        # Blended multi-pose paths (goal.via_poses non-empty): trajectory
        # points sampled per segment. Dense enough that the controller
        # interpolates instead of inventing a start-stop profile of its
        # own; 20 x 2 segments + the start point = 41 points, ~2.5 kB of
        # JSON, which the String topic carries comfortably.
        self.declare_parameter(
            "blend_samples_per_segment", BLEND_SAMPLES_PER_SEGMENT
        )
        # SMOOTH SINGLE MOVES (2026-08-28). Yesterday only the BLENDED
        # path (goal.via_poses non-empty) got a dense trajectory; every
        # ordinary named move — "go home" after release, recovery moves,
        # look_left/look_right, the observation poses — still went out as
        # ONE JointTrajectoryPoint and the controller invented its own
        # start-stop profile for it. That hard start and hard stop is the
        # mechanical noise Majd still hears ("even when it goes from drop
        # to home back it is fast movement and a lot of sounds").
        #
        # With this true the single-pose path runs the SAME builder with
        # ONE segment: build_blended_trajectory(current, [target],
        # [duration]). Same total duration, same end pose, but the
        # controller is handed ~21 points a few tens of ms apart, so the
        # profile is the S-curve's (zero speed at both ends, 1.5x the
        # average in the middle) instead of the controller's own ramp.
        #
        # Set false to get the historical single-waypoint request back
        # verbatim — that is the whole point of the gate. Any failure of
        # the dense build also falls back to it, with a log line saying so.
        self.declare_parameter("smooth_single_moves", True)

        self._default_timeout = float(self.get_parameter("default_timeout_s").value)
        self._collision_check = bool(self.get_parameter("collision_check").value)
        # Blended transport (lift -> home -> drop): the pipeline has already
        # cleared the neighbour boxes and the arc runs between taught poses
        # over free space, so a coarser sampling suffices. MoveIt answers
        # the validity calls one by one (~0.15 s each under load): 22 checks
        # were ~3 s of standing still after "Drop right" (2026-09-16).
        self.declare_parameter("blend_collision_waypoints", 3)
        self._blend_collision_waypoints = max(
            2, int(self.get_parameter("blend_collision_waypoints").value))
        self._collision_waypoints = max(
            2, int(self.get_parameter("collision_waypoints").value)
        )
        self._validity_call_timeout_s = float(
            self.get_parameter("validity_call_timeout_s").value
        )
        self._planning_group = str(self.get_parameter("planning_group").value)
        self._very_slow_duration_s = float(
            self.get_parameter("very_slow_duration_s").value
        )
        self._slow_duration_s = float(self.get_parameter("slow_duration_s").value)
        self._normal_duration_s = float(
            self.get_parameter("normal_duration_s").value
        )
        self._fast_duration_s = float(self.get_parameter("fast_duration_s").value)
        self._very_fast_duration_s = float(
            self.get_parameter("very_fast_duration_s").value
        )
        self._blend_samples_per_segment = max(
            1, int(self.get_parameter("blend_samples_per_segment").value)
        )
        self._smooth_single_moves = bool(
            self.get_parameter("smooth_single_moves").value
        )

        self._cb = ReentrantCallbackGroup()
        # DEDICATED group for the momentary /joint_states sampler — and
        # this is load-bearing (2026-08-31 live hang: sharing self._cb
        # with the RUNNING action-execute callback wedged the goal inside
        # _current_joint_positions; 'drop_right_side' sat for minutes
        # with the operator asking "it doesnt want to move?"). replay
        # node's _capture_current_joints carries the same warning.
        self._js_cb = ReentrantCallbackGroup()
        # 2026-09-02 ROOT CAUSE of every "wedge" since #279, finally with
        # a traceback: `InvalidHandle: cannot use Destroyable because
        # destruction was requested` raised INSIDE executor.spin() at
        # _take_subscription — the MultiThreadedExecutor's other threads
        # were taking from the momentary /joint_states subscription at
        # the instant the goal thread destroyed it. spin() died, main
        # fell into its finally, the node tore down under the running
        # goal, and the process exited 1 (or lingered half-dead). A
        # 494 Hz topic makes that race near-certain within a few goals.
        # Cure: sample on a PRIVATE helper node that is never handed to
        # any executor, via rclpy.wait_for_message (its own wait set).
        # Nothing else ever spins that node, so create/destroy cannot
        # race anything.
        self._sampler_node = rclpy.create_node(
            "move_to_named_pose_joint_sampler")
        # The helper's FIRST /joint_states match pays DDS discovery
        # (~0.8 s measured) — longer than the 0.5 s sampling budget, so
        # the first goal after startup would silently skip its collision
        # pre-check. Warm the match once, off the goal path.
        self._warmup_timer = self.create_timer(
            1.5, self._warm_joint_sampler, callback_group=self._js_cb)
        self._lock = threading.Lock()
        self._active_goal: ServerGoalHandle | None = None

        self._replay_pub = self.create_publisher(
            String, str(self.get_parameter("replay_topic").value), 10
        )
        # 2026-09-18: a move this server gave up on must not run later.
        # Twice today a request sat 40 s inside a blocked replay node and
        # was executed after the operator had been told it failed.
        self._cancel_pub = self.create_publisher(
            String, "/ur3e_fast_observation_replay/cancel", 10
        )

        self._status_event = threading.Event()
        self._latest_status: str = ""
        self._expected_pose: str = ""
        self._replay_seen = False
        self.create_subscription(
            String,
            str(self.get_parameter("status_topic").value),
            self._on_status,
            10,
            callback_group=self._cb,
        )

        # Joint states are sampled ON DEMAND in _current_joint_positions()
        # (2026-08-31). The standing 494 Hz subscription that used to live
        # here kept this idle server at ~96% of a core — the same rclpy
        # per-message wait-set cost that was purged from chat_node and
        # intent_router on 08-27 — and that load was starving move_group's
        # action round trips machine-wide. This node runs a
        # MultiThreadedExecutor with a ReentrantCallbackGroup, so a
        # momentary subscription IS delivered while a goal thread blocks
        # (same pattern as replay_node._capture_current_joints).
        self._joint_state_lock = threading.Lock()

        # Load YAML so we know target joint values without re-querying replay.
        self._joint_names: List[str] = []
        self._poses: dict[str, List[float]] = {}
        self._load_poses(str(self.get_parameter("poses_file").value))

        # MoveIt validity check service client.
        self._validity_client = self.create_client(
            GetStateValidity,
            str(self.get_parameter("validity_service_name").value),
            callback_group=self._cb,
        )

        self._server = ActionServer(
            self,
            MoveToNamedPose,
            str(self.get_parameter("server_action_name").value),
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cb,
        )

        self.get_logger().info(
            f"MoveToNamedPose server ready on "
            f"{self.get_parameter('server_action_name').value}; "
            f"publishes to {self.get_parameter('replay_topic').value}, "
            f"awaits {self.get_parameter('status_topic').value}; "
            f"collision_check={self._collision_check} "
            f"(waypoints={self._collision_waypoints}, "
            f"poses_loaded={len(self._poses)}); "
            f"smooth_single_moves={self._smooth_single_moves} "
            f"(samples={self._blend_samples_per_segment})"
        )

    # ── YAML loading ────────────────────────────────────────────────────

    def _file_via_poses(self, target: str) -> list:
        """Detour for current -> target from the file: leaving a pose that
        has a via list means passing its vias in reverse; arriving at one
        means passing them forward. The pose we are AT is the nearest
        listed pose within 0.35 rad (L2 over the six joints)."""
        vias = getattr(self, "_pose_via", {}) or {}
        if not vias:
            return []
        out: list = []
        cur = self._current_joint_positions()
        if cur is not None and len(cur) == len(self._joint_names):
            best, best_d = "", 1e9
            for name, pos in self._poses.items():
                d = sum((a - b) ** 2 for a, b in zip(cur, pos)) ** 0.5
                if d < best_d:
                    best, best_d = name, d
            if best and best != target and best_d < 0.35:
                out.extend(reversed(vias.get(best, [])))
        out.extend(v for v in vias.get(target, []) if v not in out)
        return [v for v in out if v in self._poses and v != target]

    def _load_poses(self, path: str) -> None:
        if not os.path.exists(path):
            self.get_logger().warn(
                f"Poses file '{path}' not found. Collision check will skip "
                "interpolated waypoints (only target reachability checked at "
                "the controller level)."
            )
            return
        try:
            with open(path, "r") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as exc:
            self.get_logger().error(f"Could not parse '{path}': {exc}")
            return
        joint_names = data.get("joint_names") or []
        poses = data.get("poses") or {}
        if not isinstance(joint_names, list) or not joint_names:
            self.get_logger().warn(
                f"Poses file '{path}' has no joint_names. Collision check disabled."
            )
            return
        self._joint_names = [str(j) for j in joint_names]
        # 2026-09-23: a pose may name the poses to flow THROUGH on the way
        # ("via: [vla_via_right]"). Straight joint-space lines from home to
        # the new VLA drop poses swept the elbow through the table (199 deg
        # of elbow while the shoulder lifts 142 deg). The file owns the
        # detour, so every client (GUI, executor, router) gets it.
        self._pose_via = {}
        for label, entry in poses.items():
            if not isinstance(entry, dict):
                continue
            positions = entry.get("positions")
            if (
                isinstance(positions, list)
                and len(positions) == len(self._joint_names)
            ):
                self._poses[str(label)] = [float(v) for v in positions]
                via = entry.get("via") or []
                if isinstance(via, list) and via:
                    self._pose_via[str(label)] = [str(v) for v in via]
        self.get_logger().info(
            f"Loaded {len(self._poses)} pose(s) from {path} for collision pre-check."
        )

    # ── ROS callbacks ────────────────────────────────────────────────────

    def _goal_cb(self, _request) -> GoalResponse:
        with self._lock:
            if self._active_goal is not None:
                self.get_logger().warn(
                    "Rejecting MoveToNamedPose: another goal is in flight"
                )
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _on_status(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        self._replay_seen = True
        if not self._expected_pose:
            return
        # replay_node's grammar is phase:label[:reason] — a failure such as
        # "failed:drop_right_side:status_6" carries the controller's reason.
        # Splitting once glued the reason onto the label (2026-09-22): every
        # failure was ignored here and surfaced as a 30 s timeout instead.
        parts = text.split(":")
        if len(parts) < 2:
            return
        phase, name = parts[0], parts[1]
        if name.strip() != self._expected_pose:
            return
        if phase in ("done", "failed"):
            self._latest_status = text
            self._status_event.set()

    def _publish_phase(
        self, goal_handle: ServerGoalHandle, phase: str, progress: float
    ) -> None:
        """Phase marker — LOG ONLY as of 2026-08-31.

        This used to call ``goal_handle.publish_feedback()``. Three live
        wedges tonight (goals frozen forever between the arrival log and
        the pre-check, arm stranded mid-task while the operator repeated
        himself) bracketed EXACTLY to these feedback publishes — the
        try/except around it catches exceptions, not a DEADLOCK inside
        rclpy's action-server locking (known Humble failure mode when
        feedback is published from the execute thread while the server
        machinery holds its own lock). Nothing in the system ever
        subscribed to this feedback — the executor waits only on the
        result, the GUI and router don't read it — so the publish was
        pure decoration with a landmine inside. Do not re-add it on
        Humble; a phase topic (plain publisher) is the safe alternative
        if progress display is ever wanted.
        """
        _ = goal_handle  # kept for signature stability
        self.get_logger().debug(f"[phase] {phase} ({progress:.2f})")

    # ── collision pre-check ──────────────────────────────────────────────

    def _warm_joint_sampler(self) -> None:
        """One-shot: pay the helper node's discovery cost at startup."""
        try:
            self._warmup_timer.cancel()
            self.destroy_timer(self._warmup_timer)  # Humble: cancel alone leaves it hot
        except Exception:
            pass
        got = self._current_joint_positions(wait_timeout_s=3.0)
        self.get_logger().info(
            "joint sampler warmed: "
            + ("ok" if got else "no /joint_states yet (driver down?)"))

    def _current_joint_positions(
        self, wait_timeout_s: float = 0.5, poll_s: float = 0.02
    ) -> Optional[List[float]]:
        """Return current joint positions in the order of self._joint_names.

        On a busy system /joint_states messages occasionally arrive partial
        (e.g. only the gripper joint in one tick) or the subscription cache
        is briefly empty between callbacks. Rather than failing the move,
        poll for up to `wait_timeout_s` until we see a message containing
        all the arm joints. Returns None only after the wait expires.
        """
        if not self._joint_names:
            return None
        # Momentary subscription — mirrors replay_node's proven
        # _capture_current_joints byte for byte in the parts that
        # matter: a DEDICATED callback group (self._js_cb), a plain
        # member-poll loop, and a destroy guarded by try/except.
        # Accumulates across messages because the arm joints and the
        # gripper joint arrive in SEPARATE partial JointState messages
        # (UR driver vs gripper adapter).
        pos_by_name: dict[str, float] = {}
        wanted = list(self._joint_names)
        deadline = time.monotonic() + max(0.0, wait_timeout_s)
        # Momentary sampling on the private, never-spun helper node —
        # see the constructor comment for why it must NOT be `self`.
        # The arm joints arrive together in one UR-driver message; the
        # loop only bridges the separate gripper-joint messages.
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                ok, msg = wait_for_message(
                    JointState, self._sampler_node, "/joint_states",
                    time_to_wait=min(remaining, 0.2))
            except Exception as exc:
                self.get_logger().warn(f"joint sampling failed: {exc}")
                break
            if ok and msg is not None:
                for n, p in zip(msg.name, msg.position):
                    pos_by_name[n] = float(p)
                if all(n in pos_by_name for n in wanted):
                    return [pos_by_name[n] for n in wanted]
        if True:
            missing = [n for n in wanted if n not in pos_by_name]
            if not pos_by_name:
                self.get_logger().warn(
                    f"No /joint_states received within {wait_timeout_s:.2f}s; "
                    "cannot collision-check."
                )
            elif missing:
                self.get_logger().warn(
                    f"Joint '{missing[0]}' missing from /joint_states for "
                    f"{wait_timeout_s:.2f}s; cannot collision-check."
                )
            return None

    def _check_state_validity(
        self, joints: List[float]
    ) -> tuple[Optional[bool], str]:
        """Call /check_state_validity for a single joint configuration.
        Returns (valid, detail). valid is True/False/None (None = no answer).
        detail is a short string naming the colliding pair, or "" if valid /
        unavailable."""
        if not self._validity_client.service_is_ready():
            if not self._validity_client.wait_for_service(timeout_sec=0.3):
                return None, ""
        rs = RobotState()
        rs.joint_state.name = list(self._joint_names)
        rs.joint_state.position = list(joints)
        req = GetStateValidity.Request()
        req.robot_state = rs
        req.group_name = self._planning_group
        future = self._validity_client.call_async(req)
        deadline = time.monotonic() + self._validity_call_timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().warn(
                    "/check_state_validity timed out; skipping check for this waypoint."
                )
                return None, ""
            time.sleep(0.005)
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f"/check_state_validity errored: {exc}")
            return None, ""
        if resp.valid:
            return True, ""
        # Format the first contact pair for diagnostics: (body_1 vs body_2).
        detail = ""
        try:
            contacts = list(resp.contacts) if resp.contacts else []
            if contacts:
                c = contacts[0]
                detail = f"{c.contact_body_1} ↔ {c.contact_body_2}"
        except Exception:
            pass
        return False, detail

    def _path_collision_free(
        self, target_joints: List[float], start_joints: List[float]
    ) -> tuple[bool, str]:
        """Linearly interpolate N waypoints from start to target and check
        each for collision. Returns (clean, reason).

        Special-cases the start: if the FIRST waypoint (the current
        configuration) is reported in collision, we don't refuse the move —
        the robot is already there, so refusing wouldn't help, and a false
        positive at the start (e.g. tight allowed-collision matrix) would
        make every move impossible. We log it and check the rest of the path.
        """
        n = self._collision_waypoints
        first_invalid_warned = False
        # PIPELINED (2026-08-26): the answers take milliseconds each — the
        # old one-at-a-time loop paid a full messaging round trip PER
        # waypoint (~0.4 s x 11 = the ~5 s "going home" stall Majd asked
        # about). Fire ALL requests at once, then collect; same checks,
        # same refusals, ~one round trip total.
        if not self._validity_client.service_is_ready():
            if not self._validity_client.wait_for_service(timeout_sec=0.3):
                self.get_logger().warn(
                    "Validity service unavailable; proceeding without check."
                )
                return True, "validity service unavailable"
        futures = []
        for i in range(n + 1):
            alpha = i / n
            interp = [
                (1.0 - alpha) * s + alpha * t
                for s, t in zip(start_joints, target_joints)
            ]
            rs = RobotState()
            rs.joint_state.name = list(self._joint_names)
            rs.joint_state.position = list(interp)
            req = GetStateValidity.Request()
            req.robot_state = rs
            req.group_name = self._planning_group
            futures.append(self._validity_client.call_async(req))
        deadline = time.monotonic() + self._validity_call_timeout_s + 0.1 * n
        while any(not f.done() for f in futures):
            if time.monotonic() > deadline:
                break
            time.sleep(0.005)
        for i, future in enumerate(futures):
            alpha = i / n
            valid: Optional[bool]
            detail = ""
            if not future.done():
                valid = None
            else:
                try:
                    resp = future.result()
                    valid = bool(resp.valid)
                    if not valid:
                        contacts = list(resp.contacts) if resp.contacts else []
                        if contacts:
                            c = contacts[0]
                            detail = f"{c.contact_body_1} ↔ {c.contact_body_2}"
                except Exception as exc:
                    self.get_logger().warn(f"/check_state_validity errored: {exc}")
                    valid = None
            if valid is None:
                self.get_logger().warn(
                    f"Validity service gave no answer at step {i}/{n}; "
                    "proceeding without check."
                )
                return True, "validity service unavailable"
            if valid:
                continue
            # Step 0 is the current pose. If THAT is reported colliding,
            # don't block the move — log once and keep checking.
            if i == 0:
                self.get_logger().warn(
                    f"Start configuration reports collision ({detail or 'no contact info'}); "
                    "treating as already-resolved and continuing path check."
                )
                first_invalid_warned = True
                continue
            self.get_logger().error(
                f"Collision at waypoint {i}/{n} (alpha={alpha:.2f}): {detail or 'no contact info'}"
            )
            reason = f"waypoint {i}/{n} would collide"
            if detail:
                reason += f" ({detail})"
            return False, reason
        if first_invalid_warned:
            self.get_logger().info(
                "Path clean from waypoint 1 onward; proceeding despite start-state warning."
            )
        return True, ""

    def _path_collision_free_multi(
        self, nodes: List[List[float]]
    ) -> tuple[bool, str]:
        """Collision-check EVERY segment of a multi-pose path.

        Same interpolation density as :meth:`_path_collision_free`
        (``collision_waypoints`` steps per segment) and the same PIPELINED
        discipline: ALL requests for ALL segments are fired first, then
        collected, so a 2-segment path still costs about one messaging
        round trip instead of 22.

        ``nodes[0]`` is the current configuration; a collision reported
        there is a warn-and-continue, exactly as in the single-segment
        path (the robot is already standing in it — refusing cannot help).
        """
        n = self._blend_collision_waypoints
        n_seg = len(nodes) - 1
        if n_seg < 1:
            return True, ""
        if not self._validity_client.service_is_ready():
            if not self._validity_client.wait_for_service(timeout_sec=0.3):
                self.get_logger().warn(
                    "Validity service unavailable; proceeding without check."
                )
                return True, "validity service unavailable"
        # (segment index, step index) in the same order as `futures`.
        keys: List[tuple[int, int]] = []
        futures = []
        for seg in range(n_seg):
            a = nodes[seg]
            b = nodes[seg + 1]
            # Step 0 of segments 1..N-1 is the previous segment's step n —
            # already queued, so skip it and check each configuration once.
            first_step = 0 if seg == 0 else 1
            for i in range(first_step, n + 1):
                alpha = i / n
                interp = [
                    (1.0 - alpha) * s + alpha * t for s, t in zip(a, b)
                ]
                rs = RobotState()
                rs.joint_state.name = list(self._joint_names)
                rs.joint_state.position = list(interp)
                req = GetStateValidity.Request()
                req.robot_state = rs
                req.group_name = self._planning_group
                keys.append((seg, i))
                futures.append(self._validity_client.call_async(req))
        deadline = (
            time.monotonic()
            + self._validity_call_timeout_s
            + 0.1 * n * n_seg
        )
        while any(not f.done() for f in futures):
            if time.monotonic() > deadline:
                break
            time.sleep(0.005)
        first_invalid_warned = False
        for (seg, i), future in zip(keys, futures):
            alpha = i / n
            valid: Optional[bool]
            detail = ""
            if not future.done():
                valid = None
            else:
                try:
                    resp = future.result()
                    valid = bool(resp.valid)
                    if not valid:
                        contacts = list(resp.contacts) if resp.contacts else []
                        if contacts:
                            c = contacts[0]
                            detail = f"{c.contact_body_1} ↔ {c.contact_body_2}"
                except Exception as exc:
                    self.get_logger().warn(f"/check_state_validity errored: {exc}")
                    valid = None
            if valid is None:
                self.get_logger().warn(
                    f"Validity service gave no answer at segment "
                    f"{seg + 1}/{n_seg} step {i}/{n}; proceeding without check."
                )
                return True, "validity service unavailable"
            if valid:
                continue
            if seg == 0 and i == 0:
                self.get_logger().warn(
                    f"Start configuration reports collision "
                    f"({detail or 'no contact info'}); treating as "
                    "already-resolved and continuing path check."
                )
                first_invalid_warned = True
                continue
            self.get_logger().error(
                f"Collision on segment {seg + 1}/{n_seg} at waypoint "
                f"{i}/{n} (alpha={alpha:.2f}): {detail or 'no contact info'}"
            )
            reason = (
                f"segment {seg + 1}/{n_seg} waypoint {i}/{n} would collide"
            )
            if detail:
                reason += f" ({detail})"
            return False, reason
        if first_invalid_warned:
            self.get_logger().info(
                "Path clean from waypoint 1 onward; proceeding despite "
                "start-state warning."
            )
        return True, ""

    # ── main execute ─────────────────────────────────────────────────────

    def _execute(self, goal_handle: ServerGoalHandle) -> MoveToNamedPose.Result:
        with self._lock:
            self._active_goal = goal_handle
        try:
            goal: MoveToNamedPose.Goal = goal_handle.request
            pose_name = (goal.pose_name or "").strip()
            if not pose_name:
                return self._fail(goal_handle, "pose_name is empty")
            # Optional intermediate poses. Empty list (the overwhelmingly
            # common case) leaves every line below untouched.
            via_poses = [
                str(v).strip()
                for v in (getattr(goal, "via_poses", None) or [])
            ]
            via_poses = [v for v in via_poses if v]
            if not via_poses:
                via_poses = self._file_via_poses(pose_name)
                if via_poses:
                    self.get_logger().info(
                        f"via poses from observation_poses.yaml: {via_poses} (before {pose_name!r})")

            self._expected_pose = pose_name
            self._status_event.clear()
            self._latest_status = ""

            self.get_logger().info(f"MoveToNamedPose -> {pose_name!r}")
            # HANG WATCHDOG (2026-08-31): two live sessions wedged right
            # after this log line with nothing external able to ptrace
            # the process. If this goal takes >20 s before completing,
            # every thread's stack is dumped to stderr (-> the launch
            # log), naming the exact blocked line. Cancelled in the
            # finally below on every normal exit.
            try:
                faulthandler.dump_traceback_later(
                    20.0, repeat=True, file=sys.stderr, exit=False)
            except Exception:
                pass
            # WATCHDOG v2 (2026-09-02): the faulthandler dump above never
            # showed up in any captured stream across two wedges, so this
            # one routes through the ROS logger (which demonstrably lands
            # in the launch log). A daemon Timer fires after 20 s unless
            # the goal finishes first; it formats every thread's stack
            # via sys._current_frames — independent of the wedged thread.
            _wd_ticket = getattr(self, "_wd_ticket", 0) + 1
            self._wd_ticket = _wd_ticket
            self._wd_goal_done = False

            def _dump_all_threads():
                if getattr(self, "_wd_goal_done", True) or \
                        getattr(self, "_wd_ticket", 0) != _wd_ticket:
                    return
                import traceback
                names = {t.ident: t.name for t in threading.enumerate()}
                out = [f"WATCHDOG: goal {pose_name!r} still running after 20 s — all thread stacks:"]
                for tid, frame in sys._current_frames().items():
                    out.append(f"--- thread {names.get(tid, '?')} ({tid}) ---")
                    out.extend(l.rstrip() for l in traceback.format_stack(frame))
                self.get_logger().error("\n".join(out))

            _wd = threading.Timer(20.0, _dump_all_threads)
            _wd.daemon = True
            _wd.start()
            self._publish_phase(goal_handle, "planning", 0.0)
            self.get_logger().info("[trace] feedback published; sampling joints")

            # Resolve target + current joints ONCE. The collision
            # pre-check already needed them; the smooth single-pose build
            # further down needs exactly the same two vectors, and
            # _current_joint_positions() polls /joint_states, so fetching
            # them twice would just add latency to every move.
            target: Optional[List[float]] = None
            start: Optional[List[float]] = None
            if not via_poses and (
                self._collision_check or self._smooth_single_moves
            ):
                target = self._poses.get(pose_name)
                start = self._current_joint_positions()
                self.get_logger().info(
                    f"[trace] joints sampled: {'ok' if start else 'NONE'}")

            # Collision pre-check (fast, uses MoveIt planning scene).
            # Blended (via_poses) goals run their own multi-segment check
            # inside _execute_blended, so skip this one for them.
            if self._collision_check and not via_poses:
                if target is None:
                    self.get_logger().info(
                        f"Pose '{pose_name}' not in YAML cache; skipping "
                        "interpolated check (replay_node will handle it)."
                    )
                elif start is None:
                    self.get_logger().warn(
                        "No /joint_states yet; skipping collision pre-check."
                    )
                else:
                    self._publish_phase(goal_handle, "checking_collision", 0.05)
                    clean, reason = self._path_collision_free(target, start)
                    if not clean:
                        return self._fail(
                            goal_handle,
                            f"refusing: {reason}",
                        )
                    self.get_logger().info(
                        f"Collision pre-check passed for '{pose_name}'."
                    )

            # Map velocity_scaling -> duration across five speed buckets.
            # Capped at 50% — anything higher protective-stops the UR3e.
            # Bucket boundaries are mid-points between adjacent
            # speed_to_vs values in intent_router: 0.05, 0.20, 0.30,
            # 0.40, 0.50. Default (vs <= 0.0) is SLOW.
            vs = float(getattr(goal, "velocity_scaling", 0.0) or 0.0)
            # Safety clamp: never let an over-eager caller exceed the
            # 0.50 ceiling, even if the LLM/router schema is bypassed.
            if vs > 0.50:
                self.get_logger().warn(
                    f"velocity_scaling={vs:.2f} exceeds 0.50 safety "
                    "ceiling; clamping to 0.50."
                )
                vs = 0.50
            if vs <= 0.0:
                duration_s = self._slow_duration_s
                speed_label = "slow"
            elif vs <= 0.12:        # bucket for very_slow (~0.05)
                duration_s = self._very_slow_duration_s
                speed_label = "very_slow"
            elif vs <= 0.25:        # bucket for slow (~0.20)
                duration_s = self._slow_duration_s
                speed_label = "slow"
            elif vs <= 0.35:        # bucket for normal (~0.30)
                duration_s = self._normal_duration_s
                speed_label = "normal"
            elif vs <= 0.45:        # bucket for fast (~0.40)
                duration_s = self._fast_duration_s
                speed_label = "fast"
            else:                   # bucket for very_fast (~0.50, ceiling)
                duration_s = self._very_fast_duration_s
                speed_label = "very_fast"

            # ── blended multi-pose path ──────────────────────────────
            # One dense trajectory through the vias instead of N goals
            # with N start-stop profiles and N fixed overheads.
            if via_poses:
                return self._execute_blended(
                    goal_handle, pose_name, via_poses, duration_s, speed_label
                )

            # ── single-pose request ──────────────────────────────────
            # The classic payload is ONE waypoint and the controller
            # invents a start-stop profile around it. With
            # smooth_single_moves (default) we run the SAME builder the
            # blended path uses, with ONE segment, and ship the result
            # through the SAME multi-point payload: identical total
            # duration and end pose, eased at both ends instead.
            #
            # Every guard below falls back to the classic payload rather
            # than failing the goal — a move that worked yesterday must
            # not start failing because the smoothing could not be built.
            plain_payload = {"label": pose_name, "duration_s": duration_s}
            req_payload = plain_payload
            smooth_note = "smooth=off (single waypoint)"
            if self._smooth_single_moves:
                if target is None:
                    smooth_note = (
                        "smooth=skipped (pose not in the YAML cache; "
                        "replay_node resolves the label)"
                    )
                elif start is None:
                    smooth_note = (
                        "smooth=skipped (no /joint_states to anchor the "
                        "path to)"
                    )
                else:
                    try:
                        positions, times = build_blended_trajectory(
                            start,
                            [list(target)],
                            [duration_s],
                            samples_per_segment=self._blend_samples_per_segment,
                        )
                        # Same belt-and-braces re-check the blended path
                        # does: a non-increasing time_from_start is the #1
                        # way JointTrajectoryController rejects a goal.
                        for i in range(1, len(times)):
                            if times[i] <= times[i - 1]:
                                raise ValueError(
                                    "time_from_start not strictly increasing "
                                    f"at point {i} ({times[i - 1]:.6f} -> "
                                    f"{times[i]:.6f})"
                                )
                        req_payload = {
                            "label": pose_name,
                            # Kept so a replay node that does not
                            # understand "waypoints" still lands on target.
                            "positions": [round(v, 6) for v in target],
                            "duration_s": round(times[-1], 6),
                            "waypoints": [
                                [round(v, 6) for v in q] for q in positions
                            ],
                            "times": [round(t, 6) for t in times],
                        }
                        smooth_note = f"smooth={len(positions)} waypoints"
                    except Exception as exc:
                        req_payload = plain_payload
                        smooth_note = "smooth=FAILED (fell back)"
                        self.get_logger().warn(
                            f"Smooth single-pose build failed for "
                            f"'{pose_name}' ({exc}); falling back to the "
                            "classic single-waypoint request."
                        )
            req_payload["sent_at"] = time.time()   # replay_node drops stale requests
            req_msg = String()
            req_msg.data = json.dumps(req_payload, separators=(",", ":"))
            self._replay_pub.publish(req_msg)
            self.get_logger().info(
                f"Replay request: pose='{pose_name}' speed={speed_label} "
                f"duration_s={duration_s:.2f} {smooth_note} "
                f"(payload {len(req_msg.data)} B)"
            )

            self._publish_phase(goal_handle, "moving", 0.2)

            deadline = time.monotonic() + self._default_timeout

            while not self._status_event.is_set():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = MoveToNamedPose.Result()
                    result.success = False
                    result.error = "cancelled"
                    return result
                if time.monotonic() > deadline:
                    if not self._replay_seen:
                        return self._fail(
                            goal_handle, "the replay node is not running"
                        )
                    self._cancel_pub.publish(String(data=pose_name))
                    return self._fail(
                        goal_handle,
                        f"the robot didn't reach pose {pose_name!r} in time "
                        f"({self._default_timeout:.0f}s)",
                    )
                self._status_event.wait(timeout=0.2)

            self._publish_phase(goal_handle, "settling", 0.9)

            result = MoveToNamedPose.Result()
            if self._latest_status.startswith("done:"):
                result.success = True
                result.error = ""
                goal_handle.succeed()
            else:
                result.success = False
                result.error = f"replay reported: {self._latest_status}"
                goal_handle.abort()
            return result
        finally:
            self._wd_goal_done = True
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass
            self._expected_pose = ""
            with self._lock:
                self._active_goal = None

    # ── blended (multi-pose) execute ─────────────────────────────────────

    def _execute_blended(
        self,
        goal_handle: ServerGoalHandle,
        pose_name: str,
        via_poses: List[str],
        segment_duration_s: float,
        speed_label: str,
    ) -> MoveToNamedPose.Result:
        """Flow current -> via_poses[0] -> ... -> pose_name as ONE move.

        Everything the single-pose path does per leg (collision pre-check,
        replay request, wait for ``done:``) happens ONCE here, and the
        trajectory handed to the replay node is dense and S-curve timed so
        the arm never decelerates to zero at a via.
        """
        path_names = list(via_poses) + [pose_name]

        if not self._joint_names:
            return self._fail(
                goal_handle,
                "refusing: no joint_names loaded from the poses file, so a "
                "blended path cannot be built",
            )
        unknown = [nm for nm in path_names if nm not in self._poses]
        if unknown:
            return self._fail(
                goal_handle,
                f"refusing: pose(s) {unknown} are not in the poses file, so "
                "a blended path cannot be built",
            )
        start = self._current_joint_positions()
        if start is None:
            return self._fail(
                goal_handle,
                "refusing: no /joint_states, so a blended path cannot be "
                "anchored to the current configuration",
            )

        targets = [list(self._poses[nm]) for nm in path_names]
        nodes = [list(start)] + targets
        n_seg = len(targets)
        path_desc = " -> ".join(["<current>"] + path_names)

        # Collision-check the path the arm will REALLY follow (a spline
        # through the vias since 2026-09-23): build it first, then check
        # states sampled along it, all requests pipelined together.
        try:
            _pos, _ = build_blended_trajectory(
                start, targets, [segment_duration_s] * n_seg,
                samples_per_segment=self._blend_samples_per_segment)
        except Exception as exc:
            return self._fail(goal_handle, f"refusing: could not build the blended path ({exc})")
        if self._collision_check:
            self._publish_phase(goal_handle, "checking_collision", 0.05)
            t_chk = time.monotonic()
            stride = max(1, self._blend_samples_per_segment // self._blend_collision_waypoints)
            check_nodes = _pos[::stride] + ([_pos[-1]] if (len(_pos) - 1) % stride else [])
            clean, reason = self._path_collision_free_multi(check_nodes)
            if not clean:
                return self._fail(goal_handle, f"refusing: {reason}")
            self.get_logger().info(
                f"Collision pre-check passed for blended path {path_desc} "
                f"({len(check_nodes)} sampled states along the spline in {time.monotonic() - t_chk:.2f} s)."
            )

        # One tier duration per segment — no new speeds are invented, the
        # blend only removes the dead time BETWEEN the legs.
        # DURATION (2026-09-23, Majd: "smooth but slow"): the tier's time
        # used to be charged per segment (2 segments = 8 s for 'slow'),
        # although a via only makes the path somewhat longer than the
        # direct line. Now the tier time is scaled by the length ratio
        # path/direct (never below the tier time): home -> vla_via ->
        # vla_drop_right is 1.5x the direct line, so 'slow' gives ~6 s,
        # 'fast' ~3 s. Only the sum of the segment times is binding.
        _direct = math.sqrt(sum((b - a) ** 2 for a, b in zip(start, targets[-1])))
        _path = 0.0
        for _i in range(1, len(_pos)):
            _path += math.sqrt(sum((b - a) ** 2 for a, b in zip(_pos[_i - 1], _pos[_i])))
        _ratio = (_path / _direct) if _direct > 1e-6 else 1.0
        total_planned = segment_duration_s * max(1.0, _ratio)
        self.get_logger().info(
            f"blended duration: tier {segment_duration_s:.1f}s x path/direct {_ratio:.2f} = {total_planned:.1f}s")
        durations = [total_planned / n_seg] * n_seg
        try:
            positions, times = build_blended_trajectory(
                start,
                targets,
                durations,
                samples_per_segment=self._blend_samples_per_segment,
            )
        except Exception as exc:
            return self._fail(
                goal_handle, f"refusing: blended trajectory build failed: {exc}"
            )
        total_s = times[-1]

        # Belt and braces: the builder asserts this, re-assert before the
        # message leaves the process. A non-monotonic time_from_start is
        # the #1 way JointTrajectoryController rejects a goal outright.
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                return self._fail(
                    goal_handle,
                    f"refusing: time_from_start not strictly increasing at "
                    f"point {i} ({times[i - 1]:.6f} -> {times[i]:.6f})",
                )

        req_payload = {
            "label": pose_name,
            # Kept for the replay node's existing single-point path; a node
            # that does not understand "waypoints" still lands on target.
            "positions": [round(v, 6) for v in targets[-1]],
            "duration_s": round(total_s, 6),
            "waypoints": [[round(v, 6) for v in q] for q in positions],
            "times": [round(t, 6) for t in times],
        }
        req_msg = String()
        req_msg.data = json.dumps(req_payload, separators=(",", ":"))
        self._replay_pub.publish(req_msg)
        self.get_logger().info(
            f"Blended replay request: {path_desc} speed={speed_label} "
            f"segments={n_seg} waypoints={len(positions)} "
            f"total_duration_s={total_s:.2f} "
            f"(payload {len(req_msg.data)} B)"
        )

        self._publish_phase(goal_handle, "moving", 0.2)

        deadline = time.monotonic() + self._default_timeout
        while not self._status_event.is_set():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = MoveToNamedPose.Result()
                result.success = False
                result.error = "cancelled"
                result.waypoints = len(positions)
                return result
            if time.monotonic() > deadline:
                if not self._replay_seen:
                    return self._fail(
                        goal_handle, "the replay node is not running"
                    )
                return self._fail(
                    goal_handle,
                    f"the robot didn't finish the blended path to "
                    f"{pose_name!r} in time ({self._default_timeout:.0f}s)",
                )
            self._status_event.wait(timeout=0.2)

        self._publish_phase(goal_handle, "settling", 0.9)

        result = MoveToNamedPose.Result()
        result.waypoints = len(positions)
        if self._latest_status.startswith("done:"):
            result.success = True
            result.error = ""
            self.get_logger().info(
                f"Blended path {path_desc} done "
                f"({n_seg} segments, {len(positions)} waypoints, "
                f"{total_s:.2f}s commanded)."
            )
            goal_handle.succeed()
        else:
            result.success = False
            result.error = f"replay reported: {self._latest_status}"
            goal_handle.abort()
        return result

    def _fail(
        self, goal_handle: ServerGoalHandle, error: str
    ) -> MoveToNamedPose.Result:
        self.get_logger().error(error)
        result = MoveToNamedPose.Result()
        result.success = False
        result.error = error
        goal_handle.abort()
        return result


def main(args: list[str] | None = None) -> None:
    # 2026-09-02: if this process ever dies of a NATIVE fault (segfault /
    # abort inside rclpy or the DDS layer), print every Python thread's
    # stack on the way out — the 08-31/09-02 "wedges" may in fact be
    # crashes whose stderr never reached a captured stream.
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass
    rclpy.init(args=args)
    node = MoveToNamedPoseServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        try:
            node._sampler_node.destroy_node()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
