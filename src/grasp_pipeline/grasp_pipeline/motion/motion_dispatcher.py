"""Dispatcher that owns every motion path used by the grasp pipeline.

Routes a high-level request like "go to pose 'home'" through the right
backend (cached fast-replay vs PTP commander subprocess), waits for
completion, and returns success/failure to the caller. Wraps
:class:`FastReplayHandle` so multi-view sweep code that expects a
``subprocess.Popen`` keeps working unchanged when the move is dispatched
via fast-replay.

Public methods consumed by ``YoloLiveSelectGrasp``:

    move_to_registration_target(name) -> bool
    move_to_joint_target(name, joints) -> bool
    move_to_registration_pose(name, pose, frame) -> bool
    move_to_registration_candidate(candidate) -> bool
    pose_delta(T_ref, T_cur) -> (dt, dr)
    dynamic_candidate_achieved_is_distinct(candidate, first_obs_tf, reg_cfg) -> bool
    cancel_fast_replay(label) -> None
    start_joint_target_process(...)
    start_cartesian_lookat_process(...)
    start_pose_target_process(...)
    wait_for_motion_process(...)
    reload_fast_replay_cache()

The dispatcher reads parameters and uses TF on demand from the owning
node; all parameter names are unchanged so launch files keep working.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import rclpy
from std_msgs.msg import String
from tf_transformations import quaternion_from_matrix

from ..nbv_candidates import generate_look_at_arc_waypoints
from ..registration import lookup_transform_matrix
from .fast_replay import FastReplayHandle

if TYPE_CHECKING:  # avoid circular import at runtime
    from rclpy.node import Node
    from ..registration import RegistrationConfig


class MotionDispatcher:
    """Owns fast-replay state + every motion entry point used by the node."""

    def __init__(self, node: "Node") -> None:
        self._node = node

        # ---- Fast-replay state (moved verbatim from grasp_selection_node).
        self._fast_replay_enabled = bool(
            node.get_parameter("registration_use_fast_observation_moves").value
        )
        self._fast_replay_poses_file = str(
            node.get_parameter("registration_fast_observation_poses_file").value
        )
        self._fast_replay_timeout_s = float(
            node.get_parameter("registration_fast_observation_timeout_s").value
        )
        self._fast_replay_topic = str(
            node.get_parameter("registration_fast_observation_replay_topic").value
        )
        self._fast_replay_status_topic = str(
            node.get_parameter("registration_fast_observation_status_topic").value
        )
        self._fast_replay_cache: set[str] = set()
        self._fast_replay_status_lock = threading.Lock()
        self._fast_replay_status_event = threading.Event()
        self._fast_replay_last_status: str = ""
        self._fast_replay_pub = None
        self._fast_replay_cancel_pub = None
        self._fast_replay_active_handles: Dict[str, FastReplayHandle] = {}

        if self._fast_replay_enabled:
            self.reload_fast_replay_cache()
            self._fast_replay_pub = node.create_publisher(
                String, self._fast_replay_topic, 10
            )
            cancel_topic = self._fast_replay_topic.rsplit("/", 1)[0] + "/cancel"
            self._fast_replay_cancel_pub = node.create_publisher(
                String, cancel_topic, 10
            )
            node.create_subscription(
                String,
                self._fast_replay_status_topic,
                self._on_fast_replay_status,
                10,
            )
            node.get_logger().info(
                "[fast-replay] enabled. cache=%s entries=%d topic=%s status=%s timeout=%.1fs"
                % (
                    self._fast_replay_poses_file,
                    len(self._fast_replay_cache),
                    self._fast_replay_topic,
                    self._fast_replay_status_topic,
                    self._fast_replay_timeout_s,
                )
            )
        else:
            node.get_logger().info(
                "[fast-replay] disabled; using MoveIt PTP for all observation moves."
            )

    # ------------------------------------------------------------------
    # Fast-replay primitives
    # ------------------------------------------------------------------

    def reload_fast_replay_cache(self) -> None:
        """Read the cache YAML and update the set of available pose labels."""
        path = self._fast_replay_poses_file
        if not path or not os.path.exists(path):
            self._fast_replay_cache = set()
            return
        try:
            import yaml as _yaml
            with open(path, "r") as fh:
                data = _yaml.safe_load(fh) or {}
            poses = data.get("poses") or {}
            if isinstance(poses, dict):
                self._fast_replay_cache = {str(k) for k in poses.keys()}
            else:
                self._fast_replay_cache = set()
        except Exception as exc:
            self._node.get_logger().warn(
                f"[fast-replay] could not read cache '{path}': {exc}"
            )
            self._fast_replay_cache = set()

    def _on_fast_replay_status(self, msg: String) -> None:
        text = msg.data or ""
        with self._fast_replay_status_lock:
            self._fast_replay_last_status = text
            parts = text.split(":", 2)
            if len(parts) >= 2:
                state = parts[0]
                label = parts[1]
                reason = parts[2] if len(parts) > 2 else ""
                if state in ("done", "failed", "cancelled"):
                    handle = self._fast_replay_active_handles.pop(label, None)
                    if handle is not None:
                        handle._finish(state, reason)
                elif state == "starting":
                    handle = self._fast_replay_active_handles.get(label)
                    if handle is not None:
                        handle.started = True
        self._fast_replay_status_event.set()

    def cancel_fast_replay(self, label: str) -> None:
        """Send a cancel request for the named fast-replay goal."""
        if self._fast_replay_cancel_pub is None:
            return
        try:
            self._fast_replay_cancel_pub.publish(String(data=label))
        except Exception as exc:
            self._node.get_logger().warn(
                f"[fast-replay] cancel publish failed for '{label}': {exc}"
            )

    def try_fast_replay(self, target_name: str) -> Optional[bool]:
        """Attempt a fast replay for the given named target.

        Returns:
            None  -> feature disabled or pose not in cache; caller should
                    fall back to the PTP path.
            True  -> replay completed successfully (`done:LABEL`).
            False -> replay was attempted but failed; caller should fall
                    back to the PTP path for safety.
        """
        if not self._fast_replay_enabled or self._fast_replay_pub is None:
            return None
        if not target_name:
            return None
        if target_name not in self._fast_replay_cache:
            self._node.get_logger().info(
                f"[fast-replay] '{target_name}' not in cache; using PTP fallback."
            )
            return None

        with self._fast_replay_status_lock:
            self._fast_replay_last_status = ""
        self._fast_replay_status_event.clear()
        self._fast_replay_pub.publish(String(data=target_name))
        self._node.get_logger().info(f"[fast-replay] requested '{target_name}'")

        deadline = time.time() + max(0.5, self._fast_replay_timeout_s)
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            self._fast_replay_status_event.wait(timeout=min(0.25, remaining))
            self._fast_replay_status_event.clear()
            with self._fast_replay_status_lock:
                status = self._fast_replay_last_status
            if not status:
                continue
            parts = status.split(":", 2)
            if len(parts) < 2 or parts[1] != target_name:
                continue
            state = parts[0]
            if state == "done":
                self._node.get_logger().info(f"[fast-replay] done: {target_name}")
                return True
            if state == "failed":
                reason = parts[2] if len(parts) > 2 else "unknown"
                self._node.get_logger().warn(
                    f"[fast-replay] failed for '{target_name}': {reason}; "
                    "falling back to PTP."
                )
                return False
            # state == "starting": keep waiting.

        self._node.get_logger().warn(
            f"[fast-replay] timeout waiting for '{target_name}' "
            f"(>{self._fast_replay_timeout_s:.1f}s); falling back to PTP."
        )
        return False

    # ------------------------------------------------------------------
    # High-level moves
    # ------------------------------------------------------------------

    def move_to_registration_target(self, target_name: str) -> bool:
        # Try the cached fast-replay path first; fall through to PTP on
        # miss/failure so taught-but-uncached or replay-unavailable poses
        # still reach the robot via the original code path.
        node = self._node
        fast_result = self.try_fast_replay(target_name)
        if fast_result is True:
            settle_s = max(
                0.0,
                float(node.get_parameter("registration_motion_settle_s").value),
            )
            if settle_s > 0.0:
                time.sleep(settle_s)
            return True

        exe = str(
            node.get_parameter("registration_observation_commander_executable").value
        ).strip()
        if not exe:
            node.get_logger().warn(
                "[registration] no observation commander executable configured; skipping auto-move."
            )
            return False
        if not os.path.exists(exe):
            node.get_logger().warn(
                f"[registration] observation commander executable not found: {exe}"
            )
            return False

        velocity_scaling = float(
            node.get_parameter("registration_motion_velocity_scaling").value
        )
        acceleration_scaling = float(
            node.get_parameter("registration_motion_acceleration_scaling").value
        )
        timeout_s = float(node.get_parameter("registration_motion_timeout_s").value)
        settle_s = max(
            0.0, float(node.get_parameter("registration_motion_settle_s").value)
        )

        cmd = [
            exe,
            "--ros-args",
            "-p", f"target_name:={target_name}",
            "-p", "execute:=true",
            "-p", "one_shot:=true",
            "-p", f"velocity_scaling:={velocity_scaling}",
            "-p", f"acceleration_scaling:={acceleration_scaling}",
        ]
        self._append_observation_commander_limits(cmd)
        node.get_logger().info(
            f"[registration] moving to observation target '{target_name}'"
        )
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            node.get_logger().warn(
                f"[registration] move to '{target_name}' timed out after {timeout_s:.1f}s"
            )
            return False
        except Exception as e:
            node.get_logger().warn(
                f"[registration] failed to start observation move '{target_name}': {e}"
            )
            return False

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr if stderr else stdout
            node.get_logger().warn(
                f"[registration] move to '{target_name}' failed with code "
                f"{completed.returncode}: {detail}"
            )
            return False

        if settle_s > 0.0:
            time.sleep(settle_s)
        return True

    def _append_observation_commander_limits(self, cmd: List[str]) -> None:
        node = self._node
        planning_time = max(
            0.1,
            float(
                node.get_parameter("registration_observation_planning_time_s").value
            ),
        )
        attempts = max(
            1,
            int(
                node.get_parameter(
                    "registration_observation_num_planning_attempts"
                ).value
            ),
        )
        allow_fallbacks = bool(
            node.get_parameter("registration_observation_allow_fallbacks").value
        )
        cmd.extend(
            [
                "-p", f"planning_time:={planning_time}",
                "-p", f"num_attempts:={attempts}",
                "-p",
                f"allow_lin_fallback:={'true' if allow_fallbacks else 'false'}",
                "-p",
                f"allow_ompl_fallback:={'true' if allow_fallbacks else 'false'}",
            ]
        )

    def move_to_joint_target(
        self, target_name: str, joint_values: np.ndarray
    ) -> bool:
        """Move the robot to a joint-space target via the observation commander."""
        proc_info = self.start_joint_target_process(target_name, joint_values)
        if proc_info is None:
            return False
        completed, timeout_s, settle_s = proc_info
        return self.wait_for_motion_process(
            target_name,
            completed,
            timeout_s,
            log_prefix="[dynamic-nbv]",
            settle_s=settle_s,
        )

    def start_fast_replay_process(
        self,
        target_name: str,
        joint_values: np.ndarray,
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
    ) -> Optional[tuple[FastReplayHandle, float, float]]:
        """Send a joint-target via the trajectory controller (fast path).

        Returns a Popen-compatible handle so existing callers that use
        ``proc.poll() / .terminate() / .wait() / .kill() / .stdout`` keep
        working unchanged.
        """
        node = self._node
        if not self._fast_replay_enabled or self._fast_replay_pub is None:
            return None

        # Map MoveIt-style velocity_scaling (0–1) to a replay duration.
        # Empirically a full SRDF-to-SRDF sweep covers ~3 rad on the
        # dominant joint and the controller is happy at ~0.6 rad/s under
        # vel_scaling=1.0. So duration ≈ 5 s / vel_scaling. We clamp.
        vel_scaling = (
            velocity_scaling_override
            if velocity_scaling_override is not None
            else float(
                node.get_parameter("registration_motion_velocity_scaling").value
            )
        )
        vel_scaling = max(0.005, min(1.0, float(vel_scaling)))
        # Default move time at full speed; a vel_scaling of 0.009 (the
        # "slow sweep" default) yields ~30 s, matching the legacy MoveIt
        # behavior the perception code was tuned around.
        base_duration_s = 0.27
        duration_s = base_duration_s / vel_scaling

        timeout_s = (
            timeout_override
            if timeout_override is not None
            else float(node.get_parameter("registration_motion_timeout_s").value)
        )
        timeout_s = max(timeout_s, duration_s + 5.0)
        settle_s = max(
            0.0, float(node.get_parameter("registration_motion_settle_s").value)
        )

        jv = np.asarray(joint_values, dtype=np.float64).ravel().tolist()
        if len(jv) != 6:
            return None

        jv_strs = [f"{v:.4f}" for v in jv]
        node.get_logger().info(
            f"[fast-replay-sweep] '{target_name}' duration={duration_s:.2f}s "
            f"vel_scaling={vel_scaling:.3f} joints=[{', '.join(jv_strs)}]"
        )

        handle = FastReplayHandle(
            owner=self,
            label=target_name,
            timeout_s=timeout_s,
        )

        payload = {
            "label": target_name,
            "positions": jv,
            "duration_s": duration_s,
        }
        # Registered BEFORE the request goes out, so the replay node's
        # "starting:" answer always finds its handle.
        with self._fast_replay_status_lock:
            self._fast_replay_active_handles[target_name] = handle
        try:
            self._fast_replay_pub.publish(String(data=json.dumps(payload)))
        except Exception as exc:
            node.get_logger().warn(
                f"[fast-replay-sweep] failed to publish '{target_name}': {exc}"
            )
            with self._fast_replay_status_lock:
                self._fast_replay_active_handles.pop(target_name, None)
            handle._finish("failed", "publish_error")
            return None

        return handle, timeout_s, settle_s

    def start_joint_target_process(
        self,
        target_name: str,
        joint_values: np.ndarray,
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
    ) -> Optional[tuple[Any, float, float]]:
        """Start a joint-target motion command asynchronously.

        If fast-replay is enabled, the move is dispatched through the
        ``ur3e_fast_observation`` replay node and wrapped in a Popen-like
        adapter. Otherwise the original PTP-commander subprocess is used.
        """
        node = self._node
        adapter = self.start_fast_replay_process(
            target_name,
            joint_values,
            velocity_scaling_override=velocity_scaling_override,
            timeout_override=timeout_override,
        )
        if adapter is not None:
            return adapter

        exe = str(
            node.get_parameter("registration_observation_commander_executable").value
        ).strip()
        if not exe or not os.path.exists(exe):
            node.get_logger().warn(
                "[registration] observation commander executable not found; skipping."
            )
            return None

        jv = np.asarray(joint_values, dtype=np.float64).ravel()
        velocity_scaling = (
            velocity_scaling_override
            if velocity_scaling_override is not None
            else float(
                node.get_parameter("registration_motion_velocity_scaling").value
            )
        )
        acceleration_scaling = float(
            node.get_parameter("registration_motion_acceleration_scaling").value
        )
        timeout_s = (
            timeout_override
            if timeout_override is not None
            else float(node.get_parameter("registration_motion_timeout_s").value)
        )
        settle_s = max(
            0.0, float(node.get_parameter("registration_motion_settle_s").value)
        )

        cmd = [
            "stdbuf", "-oL", "-eL",
            exe,
            "--ros-args",
            "-p", "use_joint_target:=true",
            "-p", "execute:=true",
            "-p", "one_shot:=true",
            "-p", f"velocity_scaling:={velocity_scaling}",
            "-p", f"acceleration_scaling:={acceleration_scaling}",
        ]
        self._append_observation_commander_limits(cmd)
        jv_strs = []
        for i, v in enumerate(jv):
            cmd.extend(["-p", f"joint_target_{i}:={float(v)}"])
            jv_strs.append(f"{float(v):.4f}")

        node.get_logger().info(
            f"[dynamic-nbv] moving to joint target '{target_name}' "
            f"joints=[{', '.join(jv_strs)}]"
        )
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
        except Exception as e:
            node.get_logger().warn(
                f"[dynamic-nbv] failed to start joint move '{target_name}': {e}"
            )
            return None
        return proc, timeout_s, settle_s

    def start_cartesian_lookat_process(
        self,
        target_name: str,
        camera_pose: np.ndarray,
        object_centroid: np.ndarray,
        direction: str,
        reg_cfg: "RegistrationConfig",
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
        n_waypoints: int = 8,
    ) -> Optional[tuple[subprocess.Popen, float, float]]:
        """Launch the commander in Cartesian look-at arc mode.

        Generates *n_waypoints* EE poses along a great-circle arc around the
        object centroid (all looking at it), then executes as a Cartesian path.
        """
        node = self._node
        exe = str(
            node.get_parameter("registration_observation_commander_executable").value
        ).strip()
        if not exe or not os.path.exists(exe):
            node.get_logger().warn(
                "[registration] commander executable not found; skipping."
            )
            return None

        ee_waypoints = self._generate_lookat_ee_waypoints(
            camera_pose,
            object_centroid,
            direction,
            reg_cfg,
            n_waypoints=n_waypoints,
        )
        if not ee_waypoints:
            return None

        velocity_scaling = (
            velocity_scaling_override
            if velocity_scaling_override is not None
            else float(
                node.get_parameter("registration_motion_velocity_scaling").value
            )
        )
        acceleration_scaling = float(
            node.get_parameter("registration_motion_acceleration_scaling").value
        )
        timeout_s = (
            timeout_override
            if timeout_override is not None
            else float(node.get_parameter("registration_motion_timeout_s").value)
        )
        settle_s = max(
            0.0, float(node.get_parameter("registration_motion_settle_s").value)
        )
        reference_frame = reg_cfg.target_frame or "base_link"
        ee_link = str(
            node.get_parameter("registration_observation_ee_link").value
        ).strip()

        cmd = [
            "stdbuf", "-oL", "-eL",
            exe,
            "--ros-args",
            "-p", "use_cartesian_path:=true",
            "-p", f"cartesian_num_waypoints:={len(ee_waypoints)}",
            "-p", "cartesian_eef_step:=0.005",
            "-p", "cartesian_jump_threshold:=0.0",
            "-p", f"reference_frame:={reference_frame}",
            "-p", f"end_effector_link:={ee_link}",
            "-p", "execute:=true",
            "-p", "one_shot:=true",
            "-p", f"velocity_scaling:={velocity_scaling}",
            "-p", f"acceleration_scaling:={acceleration_scaling}",
        ]
        self._append_observation_commander_limits(cmd)

        for i, T_ee in enumerate(ee_waypoints):
            qx, qy, qz, qw = quaternion_from_matrix(T_ee)
            prefix = f"wp_{i}_"
            cmd.extend([
                "-p", f"{prefix}x:={float(T_ee[0, 3])}",
                "-p", f"{prefix}y:={float(T_ee[1, 3])}",
                "-p", f"{prefix}z:={float(T_ee[2, 3])}",
                "-p", f"{prefix}qx:={float(qx)}",
                "-p", f"{prefix}qy:={float(qy)}",
                "-p", f"{prefix}qz:={float(qz)}",
                "-p", f"{prefix}qw:={float(qw)}",
            ])

        node.get_logger().info(
            f"[cartesian-lookat] launching Cartesian arc '{target_name}' "
            f"direction='{direction}' waypoints={len(ee_waypoints)} "
            f"vel={velocity_scaling:.3f}"
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
        except Exception as e:
            node.get_logger().warn(f"[cartesian-lookat] failed to start: {e}")
            return None
        return proc, timeout_s, settle_s

    def _generate_lookat_ee_waypoints(
        self,
        camera_pose: np.ndarray,
        object_centroid: np.ndarray,
        direction: str,
        reg_cfg: "RegistrationConfig",
        *,
        n_waypoints: int = 8,
    ) -> Optional[List[np.ndarray]]:
        node = self._node
        lateral_m = float(
            node.get_parameter("registration_dynamic_nbv_lateral_offset_m").value
        )
        vertical_m = float(
            node.get_parameter("registration_dynamic_nbv_vertical_offset_m").value
        )
        upward_bias = float(
            node.get_parameter("registration_dynamic_nbv_upward_bias_m").value
        )
        cam_waypoints = generate_look_at_arc_waypoints(
            camera_pose, object_centroid, direction,
            n_waypoints=n_waypoints,
            lateral_offset_m=lateral_m,
            vertical_offset_m=vertical_m,
            upward_bias_m=upward_bias,
        )
        if not cam_waypoints:
            node.get_logger().warn(
                f"[cartesian-lookat] no waypoints for direction '{direction}'"
            )
            return None

        camera_frame = ""
        try:
            ee_link = str(
                node.get_parameter("registration_observation_ee_link").value
            ).strip()
            for candidate_frame in (
                "camera_color_optical_frame",
                "camera_depth_optical_frame",
            ):
                try:
                    T_cam_ee = lookup_transform_matrix(
                        node.tf_buffer,
                        candidate_frame,
                        ee_link,
                        rclpy.time.Time().to_msg(),
                        reg_cfg.tf_timeout_s,
                        logger=node.get_logger(),
                    )
                    camera_frame = candidate_frame
                    break
                except Exception:
                    continue
            if not camera_frame:
                node.get_logger().warn(
                    "[cartesian-lookat] could not find camera→EE transform"
                )
                return None
        except Exception as e:
            node.get_logger().warn(f"[cartesian-lookat] TF lookup failed: {e}")
            return None

        T_cam_from_ee = T_cam_ee  # lookup gives T_cam_from_ee
        T_ee_from_cam = np.linalg.inv(T_cam_from_ee)

        ee_waypoints = []
        for T_base_cam_wp in cam_waypoints:
            T_base_cam_wp = np.asarray(T_base_cam_wp, dtype=np.float64).reshape(4, 4)
            T_base_ee = T_base_cam_wp @ T_ee_from_cam
            ee_waypoints.append(T_base_ee)
        return ee_waypoints

    def start_pose_target_process(
        self,
        target_name: str,
        pose_matrix: np.ndarray,
        reference_frame: str,
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
        planner_id_override: Optional[str] = None,
    ) -> Optional[tuple[subprocess.Popen, float, float]]:
        node = self._node
        exe = str(
            node.get_parameter("registration_observation_commander_executable").value
        ).strip()
        if not exe or not os.path.exists(exe):
            node.get_logger().warn(
                "[registration] commander executable not found; skipping."
            )
            return None
        T = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
        qx, qy, qz, qw = quaternion_from_matrix(T)
        velocity_scaling = (
            velocity_scaling_override
            if velocity_scaling_override is not None
            else float(
                node.get_parameter("registration_motion_velocity_scaling").value
            )
        )
        acceleration_scaling = float(
            node.get_parameter("registration_motion_acceleration_scaling").value
        )
        timeout_s = (
            timeout_override
            if timeout_override is not None
            else float(node.get_parameter("registration_motion_timeout_s").value)
        )
        settle_s = max(
            0.0, float(node.get_parameter("registration_motion_settle_s").value)
        )
        ee_link = str(
            node.get_parameter("registration_observation_ee_link").value
        ).strip()
        planner_id = planner_id_override or "LIN"

        cmd = [
            "stdbuf", "-oL", "-eL",
            exe,
            "--ros-args",
            "-p", "use_pose_target:=true",
            "-p", f"reference_frame:={reference_frame}",
            "-p", f"end_effector_link:={ee_link}",
            "-p", f"pose_x:={float(T[0, 3])}",
            "-p", f"pose_y:={float(T[1, 3])}",
            "-p", f"pose_z:={float(T[2, 3])}",
            "-p", f"pose_qx:={float(qx)}",
            "-p", f"pose_qy:={float(qy)}",
            "-p", f"pose_qz:={float(qz)}",
            "-p", f"pose_qw:={float(qw)}",
            "-p", "execute:=true",
            "-p", "one_shot:=true",
            "-p", f"velocity_scaling:={velocity_scaling}",
            "-p", f"acceleration_scaling:={acceleration_scaling}",
            "-p", "planning_pipeline:=pilz_industrial_motion_planner",
            "-p", f"planner_id:={planner_id}",
            "-p", "allow_lin_fallback:=false",
        ]
        self._append_observation_commander_limits(cmd)
        node.get_logger().info(
            f"[lookat-lin] launching pose-target '{target_name}' planner='{planner_id}' "
            f"pos=({float(T[0, 3]):.4f}, {float(T[1, 3]):.4f}, {float(T[2, 3]):.4f})"
        )
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
        except Exception as e:
            node.get_logger().warn(f"[lookat-lin] failed to start: {e}")
            return None
        return proc, timeout_s, settle_s

    def wait_for_motion_process(
        self,
        target_name: str,
        proc: subprocess.Popen,
        timeout_s: float,
        *,
        log_prefix: str,
        settle_s: float = 0.0,
    ) -> bool:
        node = self._node
        try:
            stdout_out, stderr_out = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_out, stderr_out = proc.communicate()
            node.get_logger().warn(
                f"{log_prefix} move to '{target_name}' timed out after {timeout_s:.1f}s"
            )
            return False
        except Exception as e:
            node.get_logger().warn(
                f"{log_prefix} failed while waiting on move '{target_name}': {e}"
            )
            return False

        stderr_out = (stderr_out or "").strip()
        stdout_out = (stdout_out or "").strip()
        if proc.returncode != 0:
            detail = stderr_out if stderr_out else stdout_out
            node.get_logger().warn(
                f"{log_prefix} move to '{target_name}' failed with code "
                f"{proc.returncode}: {detail}"
            )
            return False

        moveit_detail = stderr_out if stderr_out else stdout_out
        if moveit_detail:
            for line in moveit_detail.splitlines()[-5:]:
                node.get_logger().info(f"{log_prefix} commander: {line.strip()}")

        if settle_s > 0.0:
            time.sleep(settle_s)
        return True

    def move_to_registration_pose(
        self, target_name: str, pose_matrix: np.ndarray, reference_frame: str
    ) -> bool:
        node = self._node
        exe = str(
            node.get_parameter("registration_observation_commander_executable").value
        ).strip()
        if not exe:
            node.get_logger().warn(
                "[registration] no observation commander executable configured; skipping auto-move."
            )
            return False
        if not os.path.exists(exe):
            node.get_logger().warn(
                f"[registration] observation commander executable not found: {exe}"
            )
            return False

        T = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
        qx, qy, qz, qw = quaternion_from_matrix(T)
        velocity_scaling = float(
            node.get_parameter("registration_motion_velocity_scaling").value
        )
        acceleration_scaling = float(
            node.get_parameter("registration_motion_acceleration_scaling").value
        )
        timeout_s = float(
            node.get_parameter("registration_motion_timeout_s").value
        )
        settle_s = max(
            0.0, float(node.get_parameter("registration_motion_settle_s").value)
        )
        ee_link = str(
            node.get_parameter("registration_observation_ee_link").value
        ).strip()

        cmd = [
            exe,
            "--ros-args",
            "-p", "use_pose_target:=true",
            "-p", f"reference_frame:={reference_frame}",
            "-p", f"end_effector_link:={ee_link}",
            "-p", f"pose_x:={float(T[0, 3])}",
            "-p", f"pose_y:={float(T[1, 3])}",
            "-p", f"pose_z:={float(T[2, 3])}",
            "-p", f"pose_qx:={float(qx)}",
            "-p", f"pose_qy:={float(qy)}",
            "-p", f"pose_qz:={float(qz)}",
            "-p", f"pose_qw:={float(qw)}",
            "-p", "execute:=true",
            "-p", "one_shot:=true",
            "-p", f"velocity_scaling:={velocity_scaling}",
            "-p", f"acceleration_scaling:={acceleration_scaling}",
        ]
        self._append_observation_commander_limits(cmd)
        node.get_logger().info(
            f"[registration] moving to dynamic NBV target '{target_name}' "
            f"ee_link={ee_link} frame={reference_frame} "
            f"pos=({float(T[0, 3]):.4f}, {float(T[1, 3]):.4f}, {float(T[2, 3]):.4f}) "
            f"quat=({float(qx):.4f}, {float(qy):.4f}, {float(qz):.4f}, {float(qw):.4f})"
        )
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            node.get_logger().warn(
                f"[registration] move to dynamic target '{target_name}' timed out after {timeout_s:.1f}s"
            )
            return False
        except Exception as e:
            node.get_logger().warn(
                f"[registration] failed to start dynamic move '{target_name}': {e}"
            )
            return False

        stderr_out = (completed.stderr or "").strip()
        stdout_out = (completed.stdout or "").strip()
        if completed.returncode != 0:
            detail = stderr_out if stderr_out else stdout_out
            node.get_logger().warn(
                f"[registration] move to dynamic target '{target_name}' failed with code "
                f"{completed.returncode}: {detail}"
            )
            return False

        moveit_detail = stderr_out if stderr_out else stdout_out
        if moveit_detail:
            for line in moveit_detail.splitlines()[-5:]:
                node.get_logger().info(f"[dynamic-nbv] commander: {line.strip()}")

        if settle_s > 0.0:
            time.sleep(settle_s)
        return True

    @staticmethod
    def pose_delta(T_ref: np.ndarray, T_cur: np.ndarray) -> tuple[float, float]:
        T0 = np.asarray(T_ref, dtype=np.float64).reshape(4, 4)
        T1 = np.asarray(T_cur, dtype=np.float64).reshape(4, 4)
        dt = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        dR = T0[:3, :3].T @ T1[:3, :3]
        cos_theta = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
        dr = math.degrees(math.acos(cos_theta))
        return dt, dr

    def move_to_registration_candidate(self, candidate: Dict[str, Any]) -> bool:
        move_type = str(candidate.get("move_type", "named"))
        target_name = str(candidate.get("target_name", "candidate"))
        display_name = str(candidate.get("display_name", target_name))
        if move_type == "joint":
            return self.move_to_joint_target(
                display_name,
                np.asarray(candidate["joint_values"], dtype=np.float64),
            )
        if move_type == "pose":
            return self.move_to_registration_pose(
                display_name,
                np.asarray(candidate["pose_matrix"], dtype=np.float32),
                str(candidate.get("reference_frame", "base_link")),
            )
        return self.move_to_registration_target(target_name)

    def dynamic_candidate_achieved_is_distinct(
        self,
        candidate: Dict[str, Any],
        first_obs_tf: Dict[str, Any],
        reg_cfg: "RegistrationConfig",
    ) -> bool:
        node = self._node
        candidate_name = str(
            candidate.get("display_name", candidate.get("target_name", "candidate"))
        )
        source_frame = str(first_obs_tf.get("source_frame", "")).strip()
        if not source_frame:
            node.get_logger().warn(
                f"[dynamic-nbv] candidate '{candidate_name}' has no source camera frame; "
                "accepting achieved motion without validation."
            )
            return True

        try:
            T_seed = np.asarray(
                first_obs_tf["capture_transform"], dtype=np.float64
            ).reshape(4, 4)
            T_cur = lookup_transform_matrix(
                node.tf_buffer,
                reg_cfg.target_frame,
                source_frame,
                rclpy.time.Time().to_msg(),
                reg_cfg.tf_timeout_s,
                logger=node.get_logger(),
            )
        except Exception as e:
            node.get_logger().warn(
                f"[dynamic-nbv] failed to validate achieved pose for candidate "
                f"'{candidate_name}': {e}. Accepting move."
            )
            return True

        dt, dr = self.pose_delta(T_seed, T_cur)
        node.get_logger().info(
            f"[dynamic-nbv] achieved candidate='{candidate_name}' "
            f"camera_delta: dt={dt:.3f} m, dr={dr:.1f} deg"
        )
        min_dt = float(
            node.get_parameter("registration_dynamic_nbv_min_translation_m").value
        )
        min_dr = float(
            node.get_parameter("registration_dynamic_nbv_min_rotation_deg").value
        )
        if dt < min_dt and dr < min_dr:
            node.get_logger().warn(
                f"[dynamic-nbv] rejecting candidate '{candidate_name}' after motion: "
                f"achieved camera motion is too small (dt={dt:.3f} m, dr={dr:.1f} deg, "
                f"thresholds=({min_dt:.3f} m, {min_dr:.1f} deg))."
            )
            return False
        return True
