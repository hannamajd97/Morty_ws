#!/usr/bin/env python3
"""Replay taught (or ad-hoc) joint poses via FollowJointTrajectory.

Originally a "name -> SRDF pose" replayer, this node now also accepts:

- **Ad-hoc joint vectors** for non-cached targets (e.g. NBV interpolated
  poses) — the caller passes ``positions`` directly.
- **Per-request duration** so callers can request a fast snap (~2 s) or a
  slow sweep (e.g. 25 s for a frame-capture sweep that used to run as a
  ``vel=0.01`` MoveIt PTP).
- **Cancellation** via ``~/cancel`` — used when the perception side
  decides it has gathered enough frames mid-sweep.

Two request formats are accepted on ``~/replay`` (std_msgs/String):

- Plain string  -> replay the named pose with the default duration.
- JSON object   -> ``{"label": "...", "positions": [..6..], "duration_s": <float>}``
  ``positions`` is optional; if absent the cached YAML entry is used.
  ``label`` is required even for ad-hoc moves so the status messages can
  be correlated to the request.
- JSON object with a **dense multi-point path** (2026-08-28) ->
  ``{"label": "...", "waypoints": [[..6..], ...], "times": [<float>, ...]}``
  When ``waypoints`` is present the node forwards the whole thing as a
  multi-point ``JointTrajectory`` and the controller interpolates through
  it, so a lift->home->drop flows as ONE continuous motion instead of two
  goals with a full stop between them. ``times`` are ``time_from_start``
  seconds, one per waypoint, strictly increasing. Positions ONLY — no
  explicit velocities: an inconsistent position/velocity pair is a common
  reason for ``JointTrajectoryController`` to reject the goal, and the
  controller's own interpolation is what produces the blend.
  Requests that arrive WITHOUT ``waypoints`` are shaped here instead
  (2026-08-28): unless ``smooth_single_moves`` is false, the node builds
  the same one-segment dense path from the current joint state to the
  target, so a plain label or the GUI's ⚡ fast-replay JSON gets the same
  eased profile as the blended flow. Same duration, same end pose, no
  controller-invented start-stop ramp. Set ``smooth_single_moves:=false``
  for the historical single-point behaviour.

Status messages keep the same ``state:LABEL[:reason]`` format. Add a fifth
state: ``cancelled:LABEL`` for explicit aborts.

Safety
------
- Joint limits and velocity scaling stay enforced by the
  ``scaled_joint_trajectory_controller`` itself.
- For ad-hoc joint vectors: the caller is responsible for collision
  freedom (same trust model as the SRDF group_states themselves).
- This node never plans Cartesian motions and never commands the gripper.

Topics & actions
----------------
- Subscribes: ``/joint_states`` (sensor_msgs/JointState).
- Subscribes: ``~/replay``  (std_msgs/String, plain label or JSON).
- Subscribes: ``~/cancel``  (std_msgs/String, label of currently running goal).
- Publishes:  ``~/status``  (std_msgs/String).
- Action client: ``<controller_action_ns>/follow_joint_trajectory``.
"""

from __future__ import annotations

import faulthandler
import json
import os
import threading
import time
from typing import Dict, List, Optional

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# The dense, S-curve trajectory builder lives with the named-pose server
# (same package, same algorithm, one implementation). Guarded because it
# pulls moveit_msgs / morty_interfaces at import time: if either is
# missing this node keeps working exactly as it did, just unsmoothed.
try:
    from ur3e_fast_observation.named_pose_server import (
        build_blended_trajectory,
    )
except Exception:  # pragma: no cover - depends on the install layout
    build_blended_trajectory = None


class ReplayNode(Node):
    """Send taught joint poses straight to the trajectory controller."""

    def __init__(self) -> None:
        super().__init__('ur3e_fast_observation_replay')

        self.declare_parameter('poses_file', 'observation_poses.yaml')
        self.declare_parameter(
            'controller_action_ns', '/scaled_joint_trajectory_controller'
        )
        self.declare_parameter('move_duration_s', 2.0)
        self.declare_parameter('default_pose_name', '')
        # SMOOTH SINGLE MOVES (2026-08-28). Requests that arrive WITHOUT
        # a "waypoints" array (a plain label, or the GUI's ⚡ fast-replay
        # JSON, which publishes here directly and never passes through
        # move_to_named_pose_server) used to become ONE
        # JointTrajectoryPoint; the controller then invented its own
        # start-stop profile, which is the hard start / hard stop Majd
        # hears as noise. With this true we build the same dense,
        # S-curve-timed path the blended flow uses — one segment, same
        # total duration, same end pose — and send THAT.
        #
        # Requests that already carry "waypoints" are untouched: the
        # sender (the named-pose server) has already shaped them.
        self.declare_parameter('smooth_single_moves', True)
        self.declare_parameter('smooth_samples', 20)
        # How long to wait for a /joint_states sample to anchor the dense
        # path to. On timeout we fall back to the single waypoint and log
        # it — never fail a move that used to work.
        self.declare_parameter('smooth_joint_state_timeout_s', 0.5)

        self._poses_file: str = str(self.get_parameter('poses_file').value)
        self._controller_ns: str = str(
            self.get_parameter('controller_action_ns').value
        )
        self._move_duration_s: float = float(
            self.get_parameter('move_duration_s').value
        )
        self._default_pose: str = str(
            self.get_parameter('default_pose_name').value
        )
        self._smooth_single_moves: bool = bool(
            self.get_parameter('smooth_single_moves').value
        )
        self._smooth_samples: int = max(
            1, int(self.get_parameter('smooth_samples').value)
        )
        self._smooth_js_timeout_s: float = float(
            self.get_parameter('smooth_joint_state_timeout_s').value
        )
        if self._smooth_single_moves and build_blended_trajectory is None:
            self.get_logger().warn(
                'smooth_single_moves requested but the trajectory builder '
                'could not be imported; single-waypoint behaviour retained.'
            )
            self._smooth_single_moves = False

        self._joint_names: List[str] = []
        self._poses: Dict[str, List[float]] = {}
        self._latest_joint_state: Optional[JointState] = None
        # Group for the momentary /joint_states subscription used to
        # anchor a smoothed path (see _capture_current_joints).
        self._js_cb_group = ReentrantCallbackGroup()
        # 2026-09-18 watchdog: two moves today produced no controller goal
        # for 30-40 s and nothing was logged (stderr went to a terminal).
        # A request is 'pending' from arrival until the controller answers
        # the goal; if that takes longer than _stuck_after_s the node logs
        # it and dumps every thread's stack, once per request.
        self._max_request_age_s = 5.0
        self._stuck_after_s = 8.0
        self._pending: Optional[tuple] = None      # (label, monotonic arrival)
        self._pending_reported = False
        self.create_timer(1.0, self._watchdog_tick)
        self._load_poses()

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            f'{self._controller_ns.rstrip("/")}/follow_joint_trajectory',
        )
        # Only subscribe to the ~490 Hz joint stream when a default pose
        # actually needs it (2026-08-27: with default_pose_name empty this
        # subscription was pure wait-set churn — ~0.85 of a CPU core).
        if self._default_pose:
            self.create_subscription(
                JointState, '/joint_states', self._on_joint_state, 50
            )
        self.create_subscription(
            String, '~/replay', self._on_replay_request, 10
        )
        self.create_subscription(
            String, '~/cancel', self._on_cancel_request, 10
        )
        self._status_pub = self.create_publisher(String, '~/status', 10)
        # Mirror successful named-pose moves to /robot/current_pose so the
        # LLM (via chat_node) gets the right answer to "which position
        # are you at?". This is the SAME topic intent_router publishes on
        # after a MoveToNamedPose action; the executor's pick/drop cycle
        # bypasses intent_router and calls us directly, which used to
        # leave /robot/current_pose stuck at the LAST intent_router move.
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._current_pose_pub = self.create_publisher(
            String, '/robot/current_pose', latched_qos
        )

        # Track the currently active goal for cancellation.
        self._active_goal_lock = threading.Lock()
        self._active_goal_handle = None
        self._active_goal_label: str = ''

        self.get_logger().info(
            f"Replay node ready. poses_file='{self._poses_file}', "
            f"controller_ns='{self._controller_ns}', "
            f"loaded {len(self._poses)} pose(s)."
        )

        if self._default_pose:
            # Defer until we have at least one /joint_states.
            self.create_timer(0.5, self._maybe_run_default)
            self._default_done = False

    # -------------------------------------------------------------- Loading

    def _load_poses(self) -> None:
        if not os.path.exists(self._poses_file):
            self.get_logger().warn(
                f"Poses file '{self._poses_file}' does not exist yet. "
                'Replay requests will be rejected until it is populated by '
                'teach_node.'
            )
            return
        try:
            with open(self._poses_file, 'r') as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as exc:
            self.get_logger().error(
                f"Could not parse '{self._poses_file}': {exc}"
            )
            return
        joint_names = data.get('joint_names') or []
        poses = data.get('poses') or {}
        if not isinstance(joint_names, list) or not joint_names:
            self.get_logger().error(
                f"Poses file is missing a non-empty 'joint_names' list."
            )
            return
        self._joint_names = [str(j) for j in joint_names]
        self._poses = {}
        for label, entry in poses.items():
            if not isinstance(entry, dict):
                continue
            positions = entry.get('positions')
            if (
                isinstance(positions, list)
                and len(positions) == len(self._joint_names)
            ):
                self._poses[str(label)] = [float(v) for v in positions]

    # -------------------------------------------------------------- Helpers

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _capture_current_joints(self) -> Optional[List[float]]:
        """Current joint positions, ordered like ``self._joint_names``.

        Deliberately does NOT hold a standing /joint_states subscription:
        that stream runs at ~490 Hz and a permanent subscription cost
        ~0.85 of a CPU core here (2026-08-27), which is why it exists only
        when ``default_pose_name`` needs it. A named move happens a few
        times a minute, so we subscribe for the few milliseconds it takes
        to catch one message and drop the subscription again.

        Returns None on timeout or a partial message — the caller then
        sends the classic single waypoint, exactly as before.
        """
        if not self._joint_names:
            return None

        def _extract(js: Optional[JointState]) -> Optional[List[float]]:
            if js is None:
                return None
            by_name = {n: p for n, p in zip(js.name, js.position)}
            if any(n not in by_name for n in self._joint_names):
                # /joint_states occasionally arrives partial (e.g. only the
                # gripper joint in one tick); keep waiting for a full one.
                return None
            return [float(by_name[n]) for n in self._joint_names]

        # The standing subscription exists only in default-pose mode.
        if self._default_pose:
            return _extract(self._latest_joint_state)

        self._latest_joint_state = None
        # OWN callback group, and this is load-bearing. Every other
        # subscription here sits in the node's default group, which rclpy
        # makes MUTUALLY EXCLUSIVE — a /joint_states callback in that
        # group could never run while _on_replay_request (our caller) is
        # still on the stack, so this wait would always time out and
        # every move would silently pay 0.5 s and go out unsmoothed.
        # A separate group lets the MultiThreadedExecutor deliver it.
        sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10,
            callback_group=self._js_cb_group,
        )
        try:
            deadline = time.monotonic() + max(0.0, self._smooth_js_timeout_s)
            while True:
                joints = _extract(self._latest_joint_state)
                if joints is not None:
                    return joints
                if time.monotonic() >= deadline:
                    return None
                time.sleep(0.002)
        finally:
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
            self._latest_joint_state = None

    def _publish_status(self, text: str) -> None:
        self._status_pub.publish(String(data=text))
        self.get_logger().info(f'status: {text}')

    def _on_replay_request(self, msg: String) -> None:
        text = (msg.data or '').strip()
        if not text:
            return
        # JSON payload or plain label?
        if text.startswith('{'):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                self.get_logger().warn(
                    f'replay JSON parse failed ({exc}); ignoring request.'
                )
                return
            label = str(payload.get('label', '')).strip()
            if not label:
                self.get_logger().warn('replay JSON missing "label"; ignoring.')
                return
            sent_at = payload.get('sent_at')
            age = (time.time() - float(sent_at)) if sent_at is not None else 0.0
            self.get_logger().info(f"request '{label}' received (sent {age:.2f}s ago)")
            if age > self._max_request_age_s:
                self.get_logger().error(
                    f"request '{label}' is {age:.1f}s old — this node was blocked; "
                    "dropping it instead of moving the arm late.")
                self._publish_status(f'failed:{label}:stale_request')
                return
            self._pending = (label, time.monotonic()); self._pending_reported = False
            positions = payload.get('positions')
            if positions is not None:
                try:
                    positions = [float(v) for v in positions]
                except Exception:
                    self.get_logger().warn(
                        'replay JSON has invalid "positions"; ignoring.'
                    )
                    return
            duration_s = payload.get('duration_s')
            if duration_s is not None:
                try:
                    duration_s = float(duration_s)
                except Exception:
                    duration_s = None
            # Optional dense multi-point path. Absent => classic single
            # waypoint, byte for byte as before.
            waypoints = payload.get('waypoints')
            times = payload.get('times')
            if waypoints is not None:
                try:
                    waypoints = [[float(v) for v in wp] for wp in waypoints]
                    times = [float(t) for t in times]
                except Exception:
                    self.get_logger().warn(
                        'replay JSON has invalid "waypoints"/"times"; '
                        'ignoring request.'
                    )
                    self._publish_status(f'failed:{label}:invalid_waypoints')
                    return
            self.replay(
                label,
                positions=positions,
                duration_s=duration_s,
                waypoints=waypoints,
                times=times,
            )
        else:
            self.replay(text)

    def _on_cancel_request(self, msg: String) -> None:
        target_label = (msg.data or '').strip()
        with self._active_goal_lock:
            handle = self._active_goal_handle
            current_label = self._active_goal_label
        if handle is None:
            return
        if target_label and target_label != current_label:
            return
        self.get_logger().info(
            f"cancel requested for '{current_label}'."
        )
        try:
            handle.cancel_goal_async()
        except Exception as exc:
            self.get_logger().warn(f'cancel_goal_async failed: {exc}')

    def _watchdog_tick(self) -> None:
        p = self._pending
        if not p or self._pending_reported:
            return
        waited = time.monotonic() - p[1]
        if waited > self._stuck_after_s:
            self._pending_reported = True
            self.get_logger().error(
                f"request '{p[0]}' has produced no controller goal response for "
                f"{waited:.0f}s — all thread stacks follow:")
            try:
                faulthandler.dump_traceback(all_threads=True)
            except Exception:
                pass

    def _maybe_run_default(self) -> None:
        if getattr(self, '_default_done', False):
            return
        if self._latest_joint_state is None:
            return
        self._default_done = True
        self.replay(self._default_pose)

    # ----------------------------------------------------------- Replay API

    def replay(
        self,
        label: str,
        positions: Optional[List[float]] = None,
        duration_s: Optional[float] = None,
        waypoints: Optional[List[List[float]]] = None,
        times: Optional[List[float]] = None,
    ) -> bool:
        # Resolve target positions: ad-hoc vector or cached label.
        if positions is None:
            if label not in self._poses:
                self._publish_status(f'failed:{label}:unknown_label')
                return False
            target_positions = list(self._poses[label])
        else:
            if len(positions) != len(self._joint_names):
                self._publish_status(
                    f'failed:{label}:positions_size_mismatch'
                )
                return False
            target_positions = list(positions)

        # Optional dense multi-point path (blended multi-pose move). All
        # of these refusals happen BEFORE 'starting:' is published, so a
        # malformed payload never looks like a move that began.
        use_waypoints = False
        if waypoints is not None:
            if len(waypoints) < 2:
                self._publish_status(f'failed:{label}:too_few_waypoints')
                return False
            if times is None or len(times) != len(waypoints):
                self._publish_status(f'failed:{label}:waypoint_time_mismatch')
                return False
            if any(len(wp) != len(self._joint_names) for wp in waypoints):
                self._publish_status(f'failed:{label}:waypoint_size_mismatch')
                return False
            if times[0] < 0.0 or any(
                times[i] <= times[i - 1] for i in range(1, len(times))
            ):
                # JointTrajectoryController rejects the whole goal for a
                # non-increasing time_from_start; catch it here where the
                # status message can say why.
                self._publish_status(f'failed:{label}:times_not_monotonic')
                return False
            use_waypoints = True

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self._publish_status(f'failed:{label}:controller_unavailable')
            return False

        # Effective duration: per-request override or node default.
        effective_duration = (
            float(duration_s) if duration_s is not None else self._move_duration_s
        )
        effective_duration = max(0.05, effective_duration)

        # ── smooth an unshaped single-waypoint request ────────────────
        # Only when the sender did NOT already supply a dense path. Same
        # builder, one segment: identical total duration and end pose,
        # but an eased profile instead of the controller's own start-stop
        # ramp. Every failure path here falls back to the classic single
        # waypoint, so a move that worked before still works.
        if not use_waypoints and self._smooth_single_moves:
            start_joints = self._capture_current_joints()
            if start_joints is None:
                self.get_logger().warn(
                    f"no complete /joint_states within "
                    f"{self._smooth_js_timeout_s:.2f}s; sending '{label}' as "
                    "a single waypoint (unsmoothed)."
                )
            else:
                try:
                    positions, times = build_blended_trajectory(
                        start_joints,
                        [list(target_positions)],
                        [effective_duration],
                        samples_per_segment=self._smooth_samples,
                    )
                    if times[0] < 0.0 or any(
                        times[i] <= times[i - 1] for i in range(1, len(times))
                    ):
                        raise ValueError('times not strictly increasing')
                    waypoints = positions
                    use_waypoints = True
                    self.get_logger().info(
                        f"smoothed '{label}': {len(positions)} points over "
                        f'{times[-1]:.2f}s (S-curve, eased ends).'
                    )
                except Exception as exc:
                    self.get_logger().warn(
                        f"smoothing '{label}' failed ({exc}); sending a "
                        'single waypoint instead.'
                    )

        self._publish_status(f'starting:{label}')

        traj = JointTrajectory()
        traj.joint_names = list(self._joint_names)
        if use_waypoints:
            # POSITIONS ONLY, deliberately. The controller derives its own
            # consistent velocities by interpolating between points that
            # are tens of milliseconds apart; supplying explicit
            # velocities that disagree with the position/time spacing is a
            # classic goal-rejection cause. The dense sampling plus the
            # sender's S-curve timing is what makes the arm flow through
            # the via poses instead of stopping at them.
            pts = []
            for q, t in zip(waypoints, times):
                pt = JointTrajectoryPoint()
                pt.positions = list(q)
                secs = int(t)
                nsec = int(round((t - secs) * 1e9))
                if nsec >= 1000000000:
                    secs += 1
                    nsec -= 1000000000
                pt.time_from_start = DurationMsg(sec=secs, nanosec=nsec)
                pts.append(pt)
            traj.points = pts
            self.get_logger().info(
                f'multi-point trajectory: {len(pts)} points, '
                f'{times[-1]:.2f}s total (dense path).'
            )
        else:
            point = JointTrajectoryPoint()
            point.positions = target_positions
            point.velocities = [0.0] * len(target_positions)
            secs = int(effective_duration)
            nsec = int((effective_duration - secs) * 1e9)
            point.time_from_start = DurationMsg(sec=secs, nanosec=nsec)
            traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(f"sending goal '{label}' to the controller")
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda fut: self._on_goal_response(fut, label)
        )
        return True

    def _on_goal_response(self, future, label: str) -> None:
        if self._pending and self._pending[0] == label:
            self._pending = None
        try:
            handle = future.result()
        except Exception as exc:
            self._publish_status(f'failed:{label}:send_error:{exc}')
            return
        if not handle.accepted:
            self._publish_status(f'failed:{label}:rejected')
            return
        with self._active_goal_lock:
            self._active_goal_handle = handle
            self._active_goal_label = label
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda fut: self._on_goal_result(fut, label)
        )

    def _on_goal_result(self, future, label: str) -> None:
        with self._active_goal_lock:
            if self._active_goal_label == label:
                self._active_goal_handle = None
                self._active_goal_label = ''
        try:
            result = future.result()
        except Exception as exc:
            self._publish_status(f'failed:{label}:result_error:{exc}')
            return
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._publish_status(f'done:{label}')
            # Latched broadcast for "where is the arm" questions.
            try:
                self._current_pose_pub.publish(String(data=label))
            except Exception:
                pass
        elif status in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_CANCELING):
            self._publish_status(f'cancelled:{label}')
        else:
            self._publish_status(f'failed:{label}:status_{status}')


def main(args=None) -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    rclpy.init(args=args)
    node = ReplayNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
