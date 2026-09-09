"""Multi-view registration orchestrator.

Pulled out of grasp_selection_node so the orchestrator stops carrying
the ~700-line ``_maybe_register_multi_view_cloud`` master loop plus all
the NBV candidate generation / continuous sweep / dynamic candidate
helpers. Method names match the legacy node 1:1 — wrappers on the
orchestrator delegate to ``self._registrar.<method>(...)`` so every
internal call site keeps working.

State stays on the owning ``YoloLiveSelectGrasp`` (TF buffer, frame
buffer, cloud / motion / grasping / debug subsystems are reached via
``self._node``). Parameter names are unchanged.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import rclpy
from std_msgs.msg import Header

from ..geometry_utils import _aabb_min_max
from ..nbv_candidates import (
    NBVCandidatePose,
    NBVJointCandidate,
    generate_dynamic_nbv_candidates,
    generate_joint_interpolated_candidates,
    generate_look_at_arc_waypoints,
    make_look_at_camera_pose,
    parse_srdf_named_targets,
)
from ..registration import (
    RegistrationConfig,
    RegistrationView,
    compute_observation_gain,
    lookup_transform_matrix,
    register_and_fuse_views,
    transform_points,
    voxel_downsample,
)

if TYPE_CHECKING:
    from rclpy.node import Node


class MultiViewRegistrar:
    """Orchestrates the multi-view observation -> register -> fuse loop."""

    def __init__(self, node: "Node") -> None:
        self._node = node

    def _build_registration_config(self) -> RegistrationConfig:
        return RegistrationConfig(
            target_frame=str(self._node.get_parameter("registration_target_frame").value),
            voxel_size_m=float(self._node.get_parameter("registration_voxel_size_m").value),
            use_icp=bool(self._node.get_parameter("registration_use_icp").value),
            icp_use_point_to_plane=bool(self._node.get_parameter("registration_icp_point_to_plane").value),
            icp_max_correspondence_m=float(self._node.get_parameter("registration_icp_max_correspondence_m").value),
            icp_max_iterations=int(self._node.get_parameter("registration_icp_max_iterations").value),
            icp_min_fitness=float(self._node.get_parameter("registration_icp_min_fitness").value),
            icp_max_rmse_m=float(self._node.get_parameter("registration_icp_max_rmse_m").value),
            icp_max_translation_correction_m=float(
                self._node.get_parameter("registration_icp_max_translation_correction_m").value
            ),
            icp_max_rotation_correction_deg=float(
                self._node.get_parameter("registration_icp_max_rotation_correction_deg").value
            ),
            icp_local_map_radius_m=float(self._node.get_parameter("registration_icp_local_map_radius_m").value),
            icp_normal_radius_m=float(self._node.get_parameter("registration_icp_normal_radius_m").value),
        )


    def _transform_observation_to_target_frame(
        self,
        obs: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        *,
        note_prefix: str,
    ) -> Dict[str, Any]:
        src_frame = str(obs["header"].frame_id)
        if src_frame == reg_cfg.target_frame:
            out = dict(obs)
            out.setdefault("source_frame", src_frame)
            return out

        try:
            T_target_source = lookup_transform_matrix(
                self._node.tf_buffer,
                reg_cfg.target_frame,
                src_frame,
                obs["header"].stamp,
                reg_cfg.tf_timeout_s,
                logger=self._node.get_logger(),
            )
            pts_tf = transform_points(obs["points"], T_target_source)
        except Exception as e:
            self._node.get_logger().warn(
                f"[registration] {note_prefix}: failed to transform observation to "
                f"'{reg_cfg.target_frame}' at capture time: {e}"
            )
            return obs

        header_tf = copy.deepcopy(obs["header"])
        header_tf.frame_id = reg_cfg.target_frame
        return {
            **obs,
            "points": pts_tf,
            "header": header_tf,
            "capture_transform": T_target_source.astype(np.float32),
            "source_frame": src_frame,
        }

    # ------------------------------------------------------------------
    # Motion delegation (legacy method names kept as thin wrappers so
    # internal call sites in registration / sweep / utility code keep
    # working unchanged. Real implementation lives in
    # grasp_pipeline.motion.MotionDispatcher.)
    # ------------------------------------------------------------------


    def _generate_dynamic_registration_candidates(
        self,
        first_obs_tf: Dict[str, Any],
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        requested_names: List[str],
    ) -> List[Dict[str, Any]]:
        use_joint_interp = bool(
            self._node.get_parameter("registration_dynamic_nbv_use_joint_interpolation").value
        )
        if use_joint_interp:
            return self._generate_joint_interpolated_candidates(requested_names, reg_cfg)
        return self._generate_camera_sphere_candidates(first_obs_tf, tracked_roi, reg_cfg, requested_names)


    def _partition_dynamic_joint_candidates(
        self,
        candidate_specs: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        probe_alpha = float(self._node.get_parameter("registration_dynamic_nbv_probe_alpha").value)
        refine_top_k = max(0, int(self._node.get_parameter("registration_dynamic_nbv_refine_top_k").value))

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for spec in candidate_specs:
            base_name = str(spec.get("base_name", spec.get("target_name", "")))
            grouped.setdefault(base_name, []).append(spec)

        coarse_specs: List[Dict[str, Any]] = []
        refine_specs: List[Dict[str, Any]] = []
        for base_name, group in grouped.items():
            ordered = sorted(
                group,
                key=lambda spec: (
                    abs(float(spec.get("alpha", 1.0)) - probe_alpha),
                    -float(spec.get("alpha", 1.0)),
                ),
            )
            coarse_specs.append(ordered[0])
            refine_specs.extend(ordered[1:])
            self._node.get_logger().info(
                f"[dynamic-nbv] coarse probe for '{base_name}' uses alpha={float(ordered[0].get('alpha', 1.0)):.2f}"
            )

        coarse_specs.sort(key=lambda spec: str(spec.get("base_name", spec.get("target_name", ""))))
        refine_specs.sort(
            key=lambda spec: (
                str(spec.get("base_name", spec.get("target_name", ""))),
                -float(spec.get("alpha", 1.0)),
            )
        )
        if refine_top_k <= 0:
            refine_specs = []
        return coarse_specs, refine_specs


    def _generate_joint_interpolated_candidates(
        self,
        requested_names: List[str],
        reg_cfg: RegistrationConfig,
    ) -> List[Dict[str, Any]]:
        """Generate candidates by interpolating joint-space between SRDF named targets."""
        srdf_path = str(self._node.get_parameter("registration_dynamic_nbv_srdf_path").value).strip()
        group_name = str(self._node.get_parameter("registration_dynamic_nbv_planning_group").value).strip()
        seed_name = str(self._node.get_parameter("registration_dynamic_nbv_seed_target").value).strip()
        alpha = float(self._node.get_parameter("registration_dynamic_nbv_joint_alpha").value)
        alpha_values_raw = str(self._node.get_parameter("registration_dynamic_nbv_joint_alpha_values").value).strip()

        if not srdf_path:
            self._node.get_logger().warn(
                "[dynamic-nbv] registration_dynamic_nbv_srdf_path is empty; "
                "cannot generate joint-interpolated candidates."
            )
            return []

        try:
            all_targets = parse_srdf_named_targets(srdf_path, group_name)
        except Exception as e:
            self._node.get_logger().warn(f"[dynamic-nbv] failed to parse SRDF '{srdf_path}': {e}")
            return []

        if seed_name not in all_targets:
            self._node.get_logger().warn(
                f"[dynamic-nbv] seed target '{seed_name}' not found in SRDF. "
                f"Available: {list(all_targets.keys())}"
            )
            return []

        seed_joints = all_targets[seed_name]
        target_configs = {n: all_targets[n] for n in requested_names if n in all_targets}
        if not target_configs:
            self._node.get_logger().warn(
                f"[dynamic-nbv] none of requested targets {requested_names} found in SRDF."
            )
            return []

        alpha_values: List[float] = []
        if alpha_values_raw:
            for tok in alpha_values_raw.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    alpha_values.append(float(np.clip(float(tok), 0.0, 1.0)))
                except Exception:
                    continue
        if not alpha_values:
            alpha_values = [float(np.clip(alpha, 0.0, 1.0))]
        # Larger moves first, then finer moves.
        alpha_values = sorted(set(alpha_values), reverse=True)

        candidates = []
        for a in alpha_values:
            candidates.extend(
                generate_joint_interpolated_candidates(
                    seed_joints, target_configs, requested_names, alpha=a,
                )
            )

        specs: List[Dict[str, Any]] = []
        for cand in candidates:
            jv_str = ", ".join(f"{v:.4f}" for v in cand.joint_values)
            self._node.get_logger().info(
                f"[dynamic-nbv] joint candidate='{cand.name}' alpha={cand.alpha:.2f} "
                f"joints=[{jv_str}]"
            )
            specs.append(
                {
                    "target_name": f"dyn_{cand.name}",
                    "display_name": f"dynamic/{cand.name}@{cand.alpha:.2f}",
                    "base_name": cand.name,
                    "move_type": "joint",
                    "joint_values": cand.joint_values.copy(),
                    "alpha": float(cand.alpha),
                }
            )

        if not specs:
            self._node.get_logger().warn(
                "[dynamic-nbv] no joint-interpolated candidates generated; falling back to named targets."
            )
        return specs


    def _generate_single_joint_candidate_spec(
        self,
        base_name: str,
        alpha: float,
    ) -> Optional[Dict[str, Any]]:
        srdf_path = str(self._node.get_parameter("registration_dynamic_nbv_srdf_path").value).strip()
        group_name = str(self._node.get_parameter("registration_dynamic_nbv_planning_group").value).strip()
        seed_name = str(self._node.get_parameter("registration_dynamic_nbv_seed_target").value).strip()
        if not srdf_path:
            return None
        try:
            all_targets = parse_srdf_named_targets(srdf_path, group_name)
        except Exception as e:
            self._node.get_logger().warn(f"[continuous] failed to parse SRDF '{srdf_path}': {e}")
            return None
        if seed_name not in all_targets or base_name not in all_targets:
            return None
        generated = generate_joint_interpolated_candidates(
            all_targets[seed_name],
            {base_name: all_targets[base_name]},
            [base_name],
            alpha=float(np.clip(alpha, 0.0, 1.0)),
        )
        if not generated:
            return None
        cand = generated[0]
        return {
            "target_name": f"dyn_{cand.name}",
            "display_name": f"dynamic/{cand.name}@{cand.alpha:.2f}",
            "base_name": cand.name,
            "move_type": "joint",
            "joint_values": cand.joint_values.copy(),
            "alpha": float(cand.alpha),
        }


    def _collect_continuous_joint_sweep_observations(
        self,
        best_cand: Dict[str, Any],
        candidate_observations: List[Dict[str, Any]],
        sel: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        tracked_roi: Dict[str, Any],
        refresh: bool,
        use_roi_fallback: bool,
        min_pts: int,
        last_meta: Dict[str, Any],
        preview_seed_views: Optional[List[RegistrationView]] = None,
        preview_seed_obs_meta: Optional[Dict[str, Any]] = None,
        skip_return_to_seed: bool = False,
        max_frames_override: Optional[int] = None,
        prior_sweep_observations: Optional[List[Dict[str, Any]]] = None,
        utility_prior_obs_meta: Optional[List[Dict[str, Any]]] = None,
        utility_stop_callback: Optional[Callable[[str, List[RegistrationView], List[Dict[str, Any]]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Capture frames during a stepwise slow move to a target.

        When chaining multiple sweep legs (sweep-all mode), set
        ``skip_return_to_seed=True`` so the robot continues from its
        current position instead of returning to initgrasp first.
        ``prior_sweep_observations`` should contain observations from
        earlier legs so that incremental TSDF publishes include them.
        """
        # Safety reset: a previous sweep that died mid-motion must not leave
        # the in-motion flag stuck (it would silently disable multiframe
        # fusion for every later stationary capture).
        self._node._sweep_in_motion = False
        base_name = str(best_cand.get("base_name", best_cand["target_name"]))
        best_alpha = float(best_cand["candidate"].get("alpha", 1.0))
        slow_vel = max(
            0.01, float(self._node.get_parameter("registration_continuous_observation_velocity_scaling").value)
        )
        max_frames = max(
            1,
            max_frames_override
            if max_frames_override is not None
            else int(self._node.get_parameter("registration_continuous_observation_max_frames").value),
        )
        step_alpha = float(self._node.get_parameter("registration_continuous_observation_step_alpha").value)
        step_alpha = float(np.clip(step_alpha, 0.05, 1.0))
        max_consecutive_rejections = max(
            1,
            int(self._node.get_parameter("registration_continuous_observation_max_consecutive_rejections").value),
        )
        preview_seed_views = [] if preview_seed_views is None else list(preview_seed_views)
        prior_obs = [] if prior_sweep_observations is None else list(prior_sweep_observations)
        utility_prior_obs = [] if utility_prior_obs_meta is None else list(utility_prior_obs_meta)
        cont_min_dt = max(
            0.0, float(self._node.get_parameter("registration_continuous_observation_min_camera_translation_m").value)
        )
        cont_min_dr = max(
            0.0, float(self._node.get_parameter("registration_continuous_observation_min_camera_rotation_deg").value)
        )

        # Build target spec for the full winning direction
        target_spec = self._generate_single_joint_candidate_spec(base_name, best_alpha)
        if target_spec is None:
            self._node.get_logger().warn(
                f"[continuous] could not generate target for '{base_name}' alpha={best_alpha:.2f}"
            )
            return []

        # ── Step 1: move back to seed (initgrasp) at normal speed ──
        if not skip_return_to_seed:
            seed_name = str(self._node.get_parameter("registration_dynamic_nbv_seed_target").value).strip()
            self._node.get_logger().info(
                f"[continuous] returning to seed '{seed_name}' before slow sweep"
            )
            self._node._move_to_registration_target(seed_name)

        # ── Step 2: launch ONE slow motion and capture frames while moving ──
        slow_timeout = max(60.0, float(self._node.get_parameter("registration_motion_timeout_s").value) * 3)
        accepted: List[Dict[str, Any]] = []
        last_accepted_meta = last_meta
        tracked_roi_local = {
            k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
            for k, v in tracked_roi.items()
        }

        self._node.get_logger().info(
            f"[continuous] launching slow motion to '{base_name}' alpha={best_alpha:.2f} "
            f"vel={slow_vel:.3f} — capturing frames during motion"
        )

        # Launch the motion command (non-blocking)
        proc_info = self._node._start_joint_target_process(
            target_spec["display_name"],
            np.asarray(target_spec["joint_values"], dtype=np.float64),
            velocity_scaling_override=slow_vel,
            timeout_override=slow_timeout,
        )
        if proc_info is None:
            self._node.get_logger().warn(f"[continuous] failed to start motion to '{base_name}'")
            return []

        proc, timeout_s, settle_s = proc_info
        # Arm is moving from here until the motion process exits: in-motion
        # captures must stay single-frame (multiframe median would smear).
        self._node._sweep_in_motion = True
        motion_start_time = time.monotonic()
        consecutive_rejections = 0
        abort_motion = False
        abort_reason = ""

        # Get initial camera TF for motion detection
        camera_frame = "camera_color_optical_frame"
        try:
            T_last_cam = lookup_transform_matrix(
                self._node.tf_buffer, reg_cfg.target_frame, camera_frame,
                rclpy.time.Time(), reg_cfg.tf_timeout_s,
                logger=self._node.get_logger(),
            )
        except Exception:
            T_last_cam = np.eye(4, dtype=np.float64)

        motion_detected = False
        capture_poll_interval = 0.2  # seconds between TF checks
        preview_interval_s = 0.25
        last_preview_t = 0.0

        # ── Poll loop: capture frames while robot moves ──
        while len(accepted) < max_frames:
            elapsed = time.monotonic() - motion_start_time
            if elapsed > timeout_s:
                self._node.get_logger().warn(
                    f"[continuous] sweep timeout ({timeout_s:.0f}s) reached."
                )
                break

            # Check if motion process has finished
            proc_done = (proc.poll() is not None)

            # Read current camera TF
            try:
                T_cam_now = lookup_transform_matrix(
                    self._node.tf_buffer, reg_cfg.target_frame, camera_frame,
                    rclpy.time.Time(), reg_cfg.tf_timeout_s,
                    logger=None,
                )
            except Exception:
                if proc_done:
                    break
                time.sleep(capture_poll_interval)
                continue

            # Check motion since last capture
            dt_cam = float(np.linalg.norm(T_cam_now[:3, 3] - T_last_cam[:3, 3]))
            dR_cam = T_last_cam[:3, :3].T @ T_cam_now[:3, :3]
            cos_cam = np.clip((np.trace(dR_cam) - 1.0) * 0.5, -1.0, 1.0)
            dr_cam = math.degrees(math.acos(cos_cam))

            if not motion_detected:
                if dt_cam > 0.003 or dr_cam > 0.3:
                    motion_detected = True
                    self._node.get_logger().info(
                        f"[continuous] detected motion (dt={dt_cam:.3f}m, "
                        f"dr={dr_cam:.1f}deg) — beginning frame capture"
                    )
                else:
                    if proc_done:
                        break
                    time.sleep(capture_poll_interval)
                    continue

            now_preview_t = time.monotonic()
            if now_preview_t - last_preview_t >= preview_interval_s:
                try:
                    self._node._publish_followup_tracking_preview(sel, tracked_roi_local, reg_cfg)
                except Exception as e:
                    self._node.get_logger().warn(f"[tracking] motion preview publish failed: {e}")
                last_preview_t = now_preview_t

            # Enough motion since last accepted frame to warrant capture?
            T_prev_accepted = np.asarray(last_accepted_meta["capture_transform"], dtype=np.float64)
            dt_from_last = float(np.linalg.norm(T_cam_now[:3, 3] - T_prev_accepted[:3, 3]))
            dR_from_last = T_prev_accepted[:3, :3].T @ T_cam_now[:3, :3]
            cos_from_last = np.clip((np.trace(dR_from_last) - 1.0) * 0.5, -1.0, 1.0)
            dr_from_last = math.degrees(math.acos(cos_from_last))

            if dt_from_last < cont_min_dt and dr_from_last < cont_min_dr:
                if proc_done:
                    break
                time.sleep(capture_poll_interval)
                continue

            # ── Capture a frame ──
            frame_idx = len(accepted) + 1
            obs_tf = self._capture_registration_observation(
                sel,
                reg_cfg,
                tracked_roi_local,
                refresh=refresh,
                use_roi_fallback=use_roi_fallback,
                attempt_label=f"continuous sweep '{base_name}' frame {frame_idx}",
            )
            if obs_tf is None or obs_tf["points"].shape[0] < min_pts:
                consecutive_rejections += 1
                self._node.get_logger().warn(
                    f"[continuous] frame {frame_idx} no usable observation "
                    f"(consecutive_bad={consecutive_rejections}/{max_consecutive_rejections})."
                )
                if consecutive_rejections >= max_consecutive_rejections:
                    self._node.get_logger().warn(
                        f"[continuous] continuing sweep '{base_name}' to the endpoint despite repeated "
                        "poor follow-up observations."
                    )
                if proc_done:
                    break
                continue

            framing = self._node._evaluate_continuous_bbox_framing(
                [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])],
                tracked_roi_local,
            )
            if framing["reject"]:
                consecutive_rejections += 1
                self._node.get_logger().warn(
                    f"[continuous] frame {frame_idx} rejected for border clipping "
                    f"(area_ratio={framing['area_ratio']:.2f}, height_ratio={framing['height_ratio']:.2f}, "
                    f"edges=L{int(framing['touches_left'])}/T{int(framing['touches_top'])}/"
                    f"R{int(framing['touches_right'])}/B{int(framing['touches_bottom'])})."
                )
                if framing["stop_now"] or consecutive_rejections >= max_consecutive_rejections:
                    self._node.get_logger().warn(
                        f"[continuous] continuing sweep '{base_name}' to the endpoint despite edge clipping."
                    )
                if proc_done:
                    break
                continue

            # Compute motion metrics vs last accepted frame
            T_cur = np.asarray(obs_tf["capture_transform"], dtype=np.float64)
            T_prev = np.asarray(last_accepted_meta["capture_transform"], dtype=np.float64)
            dt = float(np.linalg.norm(T_cur[:3, 3] - T_prev[:3, 3]))
            dR = T_prev[:3, :3].T @ T_cur[:3, :3]
            cos_theta = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
            dr = math.degrees(math.acos(cos_theta))

            if bool(self._node.get_parameter("registration_tsdf_endpoint_only").value):
                obs_tf["tsdf_eligible"] = False
                obs_tf["tsdf_skip_reason"] = "moving sweep frame"
            obs_tf["sweep_phase"] = "moving"
            accepted.append(obs_tf)
            last_accepted_meta = obs_tf
            consecutive_rejections = 0
            self._node.get_logger().info(
                f"[continuous] accepted frame {frame_idx}/{max_frames} "
                f"(dt={dt:.3f}m, dr={dr:.1f}deg, pts={obs_tf['points'].shape[0]})"
            )
            self._node._debug_status_text = (
                f"SWEEP  dir={best_cand.get('base_name','?')}  frame {frame_idx}/{max_frames}\n"
                f"pts={obs_tf['points'].shape[0]}  dt={dt:.3f}m  dr={dr:.1f}deg"
            )

            # Update tracked ROI from accepted observation
            pts_acc = np.asarray(obs_tf["points"], dtype=np.float32)
            aabb_min = np.min(pts_acc, axis=0)
            aabb_max = np.max(pts_acc, axis=0)
            margin = float(self._node.get_parameter("registration_roi_margin_m").value)
            min_half = float(self._node.get_parameter("registration_roi_min_half_extent_m").value)
            tracked_roi_local["center"] = (0.5 * (aabb_min + aabb_max)).astype(np.float32)
            tracked_roi_local["half_extent"] = np.maximum(
                0.5 * (aabb_max - aabb_min) + margin,
                min_half,
            ).astype(np.float32)
            bbox_obs = [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])]
            tracked_roi_local["reference_bbox_area_px"] = int(
                max(1, (bbox_obs[2] - bbox_obs[0]) * (bbox_obs[3] - bbox_obs[1]))
            )
            tracked_roi_local["reference_bbox_xyxy"] = bbox_obs
            # Update reference points for tighter ROI projection next frame
            ref_voxel = float(self._node.get_parameter("registration_roi_fallback_reference_voxel_m").value)
            ref_points, _ = voxel_downsample(pts_acc, None, ref_voxel)
            tracked_roi_local["reference_points"] = np.asarray(ref_points, dtype=np.float32)

            # Update camera TF baseline for next motion check
            T_last_cam = T_cam_now.copy()

            # Publish preview + TSDF
            if preview_seed_views:
                preview_views = preview_seed_views + [
                    RegistrationView(
                        points=np.asarray(v["points"], dtype=np.float32),
                        colors=None if v["colors"] is None else np.asarray(v["colors"]),
                        frame_id=str(v["header"].frame_id),
                        stamp=v["header"].stamp,
                        label=str(sel["label"]),
                    )
                    for v in accepted
                ]
                self._node._publish_registration_preview_cloud(preview_views, reg_cfg)
            self._node._publish_incremental_tsdf_cloud(
                preview_seed_obs_meta, prior_obs + accepted, reg_cfg
            )

            if utility_stop_callback is not None:
                eval_views = list(preview_seed_views) + [
                    RegistrationView(
                        points=np.asarray(v["points"], dtype=np.float32),
                        colors=None if v["colors"] is None else np.asarray(v["colors"]),
                        frame_id=str(v["header"].frame_id),
                        stamp=v["header"].stamp,
                        label=str(sel["label"]),
                    )
                    for v in accepted
                ]
                eval_obs_meta = list(utility_prior_obs) + list(accepted)
                context = f"view_{len(eval_views)}"
                if utility_stop_callback(context, eval_views, eval_obs_meta):
                    self._node.get_logger().info(
                        f"[continuous] stopping sweep '{base_name}' because grasp utility stopped improving."
                    )
                    abort_motion = True
                    abort_reason = "grasp utility stopped improving"
                    break

            if proc_done:
                break

        # ── Stop or wait for motion process ───────────────────────────────
        if abort_motion and proc.poll() is None:
            self._node.get_logger().warn(
                f"[continuous] aborting active observation motion to '{base_name}' "
                f"because {abort_reason}."
            )
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                self._node.get_logger().warn("[continuous] motion process killed after perception abort")
        elif proc.poll() is None:
            try:
                proc.wait(timeout=max(10.0, timeout_s - (time.monotonic() - motion_start_time)))
            except subprocess.TimeoutExpired:
                proc.kill()
                self._node.get_logger().warn(f"[continuous] motion process killed after timeout")
        # Log commander output
        self._node._sweep_in_motion = False
        stdout_text = proc.stdout.read() if proc.stdout else ""
        stderr_text = proc.stderr.read() if proc.stderr else ""
        for line in (stdout_text + stderr_text).strip().splitlines():
            if line.strip():
                self._node.get_logger().info(f"[continuous] commander: {line.strip()}")

        # ── Capture final frame at destination if robot stopped ──
        if (not abort_motion) and len(accepted) < max_frames and proc.returncode == 0:
            time.sleep(max(0.3, settle_s))
            obs_tf = self._capture_registration_observation(
                sel,
                reg_cfg,
                tracked_roi_local,
                refresh=refresh,
                use_roi_fallback=use_roi_fallback,
                attempt_label=f"continuous sweep '{base_name}' final",
            )
            if obs_tf is not None and obs_tf["points"].shape[0] >= min_pts:
                framing = self._node._evaluate_continuous_bbox_framing(
                    [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])],
                    tracked_roi_local,
                )
                if framing["reject"]:
                    self._node.get_logger().warn(
                        f"[continuous] final frame rejected for border clipping "
                        f"(area_ratio={framing['area_ratio']:.2f}, height_ratio={framing['height_ratio']:.2f}, "
                        f"edges=L{int(framing['touches_left'])}/T{int(framing['touches_top'])}/"
                        f"R{int(framing['touches_right'])}/B{int(framing['touches_bottom'])})."
                    )
                    obs_tf = None
            if obs_tf is not None and obs_tf["points"].shape[0] >= min_pts:
                T_cur = np.asarray(obs_tf["capture_transform"], dtype=np.float64)
                T_prev = np.asarray(last_accepted_meta["capture_transform"], dtype=np.float64)
                dt = float(np.linalg.norm(T_cur[:3, 3] - T_prev[:3, 3]))
                dR = T_prev[:3, :3].T @ T_cur[:3, :3]
                cos_theta = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
                dr = math.degrees(math.acos(cos_theta))
                if dt >= cont_min_dt or dr >= cont_min_dr:
                    obs_tf["tsdf_eligible"] = True
                    obs_tf["sweep_phase"] = "endpoint"
                    accepted.append(obs_tf)
                    self._node.get_logger().info(
                        f"[continuous] accepted final frame "
                        f"(dt={dt:.3f}m, dr={dr:.1f}deg, pts={obs_tf['points'].shape[0]})"
                    )
                    if preview_seed_views:
                        preview_views = preview_seed_views + [
                            RegistrationView(
                                points=np.asarray(v["points"], dtype=np.float32),
                                colors=None if v["colors"] is None else np.asarray(v["colors"]),
                                frame_id=str(v["header"].frame_id),
                                stamp=v["header"].stamp,
                                label=str(sel["label"]),
                            )
                            for v in accepted
                        ]
                        self._node._publish_registration_preview_cloud(preview_views, reg_cfg)
                    self._node._publish_incremental_tsdf_cloud(
                        preview_seed_obs_meta, prior_obs + accepted, reg_cfg
                    )

        self._node.get_logger().info(
            f"[continuous] sweep leg '{base_name}' complete: "
            f"captured {len(accepted)} frames (prior={len(prior_obs)})"
        )
        self._node._debug_status_text = (
            f"SWEEP DONE  dir={base_name}\n"
            f"captured={len(accepted)} frames  prior={len(prior_obs)}\n"
            f"running GraspGen..."
        )
        return accepted


    def _choose_continuous_direction_without_probe(
        self,
        first_obs_tf: Dict[str, Any],
        dynamic_coarse_specs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not dynamic_coarse_specs:
            return None
        by_base: Dict[str, Dict[str, Any]] = {}
        for spec in dynamic_coarse_specs:
            bname = str(spec.get("base_name", spec.get("target_name", "")))
            if bname not in by_base:
                by_base[bname] = spec

        forced_direction = str(
            self._node.get_parameter("registration_continuous_observation_force_direction").value
        ).strip()
        if forced_direction:
            if forced_direction in by_base:
                self._node.get_logger().warn(
                    f"[continuous] force-direction override selected '{forced_direction}'."
                )
                return by_base[forced_direction]
            self._node.get_logger().warn(
                f"[continuous] force-direction override '{forced_direction}' is unavailable; "
                f"available={sorted(by_base.keys())}. Falling back to scoring."
            )

        pts = np.asarray(first_obs_tf.get("points", np.empty((0, 3))), dtype=np.float32)
        bbox = [int(v) for v in first_obs_tf.get("bbox", [0, 0, 0, 0])]
        cx_img = float(self._node.fb.cx) if self._node.fb.cx is not None else 0.0
        cy_img = float(self._node.fb.cy) if self._node.fb.cy is not None else 0.0
        score: Dict[str, float] = {k: 0.0 for k in by_base.keys()}
        score_terms: Dict[str, Dict[str, float]] = {
            k: {
                "shape": 0.0,
                "height": 0.0,
                "depth_slope": 0.0,
                "center_bias": 0.0,
                "framing": 0.0,
                "tie_break": 0.0,
            }
            for k in by_base.keys()
        }
        ex = ey = ez = 0.0
        x_span_cam = y_span_cam = z_span_cam = 0.0
        height_ratio = 0.0
        lateral_score = 0.0
        higher_bonus = 0.0
        depth_slope = 0.0
        depth_slope_bonus = 0.0
        center_bias_bonus = 0.0
        higher_low_penalty = 0.0
        higher_edge_penalty = 0.0
        x_var = 0.0
        img_centered = True
        depth_slope_deadband_for_tiebreak = max(
            0.0,
            float(
                self._node.get_parameter(
                    "registration_continuous_observation_depth_slope_deadband"
                ).value
            ),
        )

        if pts.shape[0] >= 10:
            aabb_min = np.min(pts, axis=0)
            aabb_max = np.max(pts, axis=0)
            ex, ey, ez = [float(v) for v in (aabb_max - aabb_min)]
            T_base_cam = np.asarray(
                first_obs_tf.get("capture_transform", np.eye(4, dtype=np.float32)),
                dtype=np.float64,
            ).reshape(4, 4)
            R_base_cam = T_base_cam[:3, :3]
            t_base_cam = T_base_cam[:3, 3]
            pts_cam = (pts.astype(np.float64) - t_base_cam[None, :]) @ R_base_cam
            cam_min = np.min(pts_cam, axis=0)
            cam_max = np.max(pts_cam, axis=0)
            x_span_cam, y_span_cam, z_span_cam = [float(v) for v in (cam_max - cam_min)]

            # Higher view only matters if the object is genuinely tall in the image/camera frame.
            height_ratio = y_span_cam / max(x_span_cam, 1e-6)
            higher_bonus = min(0.25, max(0.0, height_ratio - 1.8) * 0.15)
            bbox_bottom = float(bbox[3]) if len(bbox) >= 4 else 0.0
            _, img_h = self._node._get_active_image_size()
            low_penalty_cap = max(
                0.0,
                float(self._node.get_parameter("registration_continuous_observation_higher_low_penalty").value),
            )
            bottom_edge_penalty = max(
                0.0,
                float(self._node.get_parameter("registration_continuous_observation_higher_bottom_edge_penalty").value),
            )
            bottom_offset = max(0.0, bbox_bottom - (cy_img + 20.0))
            low_norm = bottom_offset / max(float(img_h) - (cy_img + 20.0), 1.0)
            higher_low_penalty = min(low_penalty_cap, max(0.0, low_norm) * low_penalty_cap)
            framing = self._node._evaluate_continuous_bbox_framing(
                bbox,
                {"reference_bbox_xyxy": bbox, "reference_bbox_area_px": 1},
            )
            if framing["touches_bottom"]:
                higher_edge_penalty = bottom_edge_penalty
            higher_bonus = max(0.0, higher_bonus - higher_low_penalty - higher_edge_penalty)
            if "detect_higher" in score:
                score["detect_higher"] += higher_bonus
                score_terms["detect_higher"]["height"] += higher_bonus
                score_terms["detect_higher"]["framing"] -= (higher_low_penalty + higher_edge_penalty)

            # Use depth-vs-horizontal-position slope as a real 3D cue for which
            # side is receding — but only when the object is roughly centered in
            # the image (|img_offset| < 60 px).  When the object is far off-center
            # the image-position signal is much stronger and depth_slope often
            # reflects camera-angle bias rather than true occlusion geometry.
            x_vals = pts_cam[:, 0]
            z_vals = pts_cam[:, 2]
            x_var = float(np.var(x_vals))
            if x_var > 1e-8:
                depth_slope = float(np.cov(x_vals, z_vals, bias=True)[0, 1] / x_var)
                depth_weight = max(
                    0.0,
                    float(
                        self._node.get_parameter(
                            "registration_continuous_observation_depth_slope_weight"
                        ).value
                    ),
                )
                depth_deadband = depth_slope_deadband_for_tiebreak
                depth_slope_effective = max(0.0, abs(depth_slope) - depth_deadband)
                depth_slope_bonus = min(0.10, depth_slope_effective * depth_weight)

        x1, _, x2, _ = bbox
        bbox_cx = 0.5 * (x1 + x2)

        # ── Fix A — occlusion-aware bbox center via mask centroid ────────
        # If a segmentation mask of the target is available, use its
        # centroid in image space as the "target center" instead of the
        # raw bbox center. The bbox can include occluder pixels, but the
        # mask is target-only. This makes img_offset robust to other
        # objects that happen to share the bbox region.
        mask_used = False
        if bool(
            self._node.get_parameter(
                "registration_continuous_observation_use_mask_centroid"
            ).value
        ):
            sel_mask = first_obs_tf.get("selected_mask")
            if sel_mask is not None:
                try:
                    mask_np = np.asarray(sel_mask, dtype=bool)
                    if mask_np.any():
                        ys, xs = np.where(mask_np)
                        bbox_cx = float(xs.mean())
                        mask_used = True
                except Exception:
                    pass

        img_offset = bbox_cx - cx_img

        img_bias_scale = 1.0 / 200.0
        img_bias_cap = 0.40
        img_edge_penalty = 0.20
        img_centered = abs(img_offset) < 60.0

        if img_offset < -15.0:
            # Object left of centre → reward detect_left (same-side: move
            # toward the object to reveal its currently-unseen left face).
            bonus = min(img_bias_cap, max(0.0, abs(img_offset) - 15.0) * img_bias_scale)
            center_bias_bonus = bonus
            if "detect_left" in score:
                score["detect_left"] += bonus
                score_terms["detect_left"]["center_bias"] += bonus
            if not img_centered and "detect_right" in score:
                score["detect_right"] -= img_edge_penalty
                score_terms["detect_right"]["framing"] -= img_edge_penalty
        elif img_offset > 15.0:
            # Object right of centre → reward detect_right (same-side).
            bonus = min(img_bias_cap, max(0.0, abs(img_offset) - 15.0) * img_bias_scale)
            center_bias_bonus = bonus
            if "detect_right" in score:
                score["detect_right"] += bonus
                score_terms["detect_right"]["center_bias"] += bonus
            if not img_centered and "detect_left" in score:
                score["detect_left"] -= img_edge_penalty
                score_terms["detect_left"]["framing"] -= img_edge_penalty

        # depth_slope only contributes when object is near image center
        if img_centered and x_var > 1e-8:
            if depth_slope > depth_slope_deadband_for_tiebreak and "detect_right" in score:
                score["detect_right"] += depth_slope_bonus
                score_terms["detect_right"]["depth_slope"] += depth_slope_bonus
            elif depth_slope < -depth_slope_deadband_for_tiebreak and "detect_left" in score:
                score["detect_left"] += depth_slope_bonus
                score_terms["detect_left"]["depth_slope"] += depth_slope_bonus

        # ── Fix B — occlusion-aware bias ────────────────────────────────
        # Look for other YOLO detections whose bbox overlaps the target.
        # If one is found, push the camera AWAY from the occluder side so
        # the sweep can look around it. This complements Fix A: A cleans
        # up the target signal, B adds a positive cue based on occluder
        # geometry.
        occluder_offset_px = 0.0
        occluder_bonus_left = 0.0
        occluder_bonus_right = 0.0
        try:
            iou_thresh = float(
                self._node.get_parameter(
                    "registration_continuous_observation_occluder_iou_threshold"
                ).value
            )
            bonus_max = float(
                self._node.get_parameter(
                    "registration_continuous_observation_occluder_bonus_max"
                ).value
            )
            min_offset_px = float(
                self._node.get_parameter(
                    "registration_continuous_observation_occluder_min_offset_px"
                ).value
            )
        except Exception:
            iou_thresh = 0.10
            bonus_max = 0.30
            min_offset_px = 30.0

        def _bbox_iou(b1: List[float], b2: List[float]) -> float:
            ax1, ay1, ax2, ay2 = b1
            bx1, by1, bx2, by2 = b2
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = inter_w * inter_h
            if inter <= 0.0:
                return 0.0
            a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = a_area + b_area - inter
            return inter / union if union > 0.0 else 0.0

        occluder_det = None
        try:
            target_bbox_f = [float(v) for v in bbox]
            best_iou = 0.0
            for det in (self._node.detections_cache or []):
                d_bbox = det.get("bbox")
                if not d_bbox or len(d_bbox) < 4:
                    continue
                # Skip the target itself by exact bbox match (close enough)
                d_bf = [float(v) for v in d_bbox[:4]]
                if (abs(d_bf[0] - target_bbox_f[0]) < 1.0 and
                        abs(d_bf[1] - target_bbox_f[1]) < 1.0 and
                        abs(d_bf[2] - target_bbox_f[2]) < 1.0 and
                        abs(d_bf[3] - target_bbox_f[3]) < 1.0):
                    continue
                iou = _bbox_iou(d_bf, target_bbox_f)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    occluder_det = det
        except Exception:
            occluder_det = None

        if occluder_det is not None:
            try:
                ob = [float(v) for v in occluder_det.get("bbox", [0, 0, 0, 0])[:4]]
                occ_cx = 0.5 * (ob[0] + ob[2])
                # Vector from target's mask-centroid to the occluder centroid
                # in image space. Positive = occluder to the RIGHT of target →
                # bias sweep LEFT (look around it from the left).
                occluder_offset_px = occ_cx - bbox_cx
                if abs(occluder_offset_px) >= min_offset_px:
                    bonus = min(
                        bonus_max,
                        (abs(occluder_offset_px) - min_offset_px) / 200.0 * bonus_max,
                    )
                    if occluder_offset_px > 0.0 and "detect_left" in score:
                        score["detect_left"] += bonus
                        score_terms["detect_left"]["framing"] += bonus
                        occluder_bonus_left = bonus
                    elif occluder_offset_px < 0.0 and "detect_right" in score:
                        score["detect_right"] += bonus
                        score_terms["detect_right"]["framing"] += bonus
                        occluder_bonus_right = bonus
                self._node.get_logger().info(
                    f"[continuous] occluder detected: label='{occluder_det.get('label', '?')}' "
                    f"iou={best_iou:.2f} dx_from_target_px={occluder_offset_px:+.1f} "
                    f"bonus_left={occluder_bonus_left:.3f} bonus_right={occluder_bonus_right:.3f}"
                )
            except Exception as e:
                self._node.get_logger().warn(f"[continuous] occluder bias failed: {e}")

        # Tie-break order: depth_slope first (when centered), then img_offset.
        pref_order = {"detect_left": 0, "detect_right": 0, "detect_higher": 0}
        if img_centered and abs(depth_slope) > depth_slope_deadband_for_tiebreak:
            if depth_slope > 0.0:
                pref_order["detect_right"] = 2
                pref_order["detect_left"] = 1
            else:
                pref_order["detect_left"] = 2
                pref_order["detect_right"] = 1
        elif img_offset < -15.0:
            pref_order["detect_left"] = 1
        elif img_offset > 15.0:
            pref_order["detect_right"] = 1

        max_score = max(score.values())
        near_tied = [name for name, value in score.items() if abs(value - max_score) < 1e-6]
        tie_break_used = len(near_tied) > 1
        chosen_base = max(score.keys(), key=lambda k: (score[k], pref_order.get(k, -1)))
        if tie_break_used:
            score_terms[chosen_base]["tie_break"] = 1.0
        chosen_spec = by_base[chosen_base]
        self._node.get_logger().info(
            "[continuous] direction score details: "
            f"extents_m=(x={ex:.3f}, y={ey:.3f}, z={ez:.3f}), "
            f"camera_spans_m=(x={x_span_cam:.3f}, y={y_span_cam:.3f}, z={z_span_cam:.3f}), "
            f"height_ratio={height_ratio:.3f}, "
            f"depth_slope={depth_slope:.3f}, depth_slope_bonus={depth_slope_bonus:.3f}, "
            f"img_centered={img_centered}, "
            f"target_center_x={bbox_cx:.1f} (mask_used={mask_used}), image_center_x={cx_img:.1f}, "
            f"img_offset_px={img_offset:.1f}, center_bias_bonus={center_bias_bonus:.3f}, "
            f"occluder_offset_px={occluder_offset_px:+.1f}, "
            f"higher_low_penalty={higher_low_penalty:.3f}, higher_edge_penalty={higher_edge_penalty:.3f}, "
            f"tie_break_used={tie_break_used}"
        )
        for base_name in sorted(score.keys()):
            parts = score_terms[base_name]
            self._node.get_logger().info(
                f"[continuous] direction '{base_name}': total={score[base_name]:.3f} "
                f"(shape={parts['shape']:.3f}, height={parts['height']:.3f}, "
                f"depth_slope={parts['depth_slope']:.3f}, "
                f"center_bias={parts['center_bias']:.3f}, framing={parts['framing']:.3f}, "
                f"tie_break={parts['tie_break']:.0f})"
            )
        self._node.get_logger().info(
            f"[continuous] one-direction score selected '{chosen_base}' "
            f"(left={score.get('detect_left', float('-inf')):.3f}, "
            f"right={score.get('detect_right', float('-inf')):.3f}, "
            f"higher={score.get('detect_higher', float('-inf')):.3f})"
        )
        return chosen_spec


    def _generate_camera_sphere_candidates(
        self,
        first_obs_tf: Dict[str, Any],
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        requested_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate candidates on a camera viewing sphere (legacy approach)."""
        if "capture_transform" not in first_obs_tf:
            return []
        camera_frame = str(first_obs_tf.get("source_frame", "")).strip()
        if not camera_frame:
            self._node.get_logger().warn(
                "[registration] dynamic NBV needs the source camera frame; falling back."
            )
            return []

        ee_link = str(self._node.get_parameter("registration_observation_ee_link").value).strip()
        stamp = first_obs_tf["header"].stamp
        try:
            T_base_cam = np.asarray(first_obs_tf["capture_transform"], dtype=np.float64).reshape(4, 4)
            T_base_ee = lookup_transform_matrix(
                self._node.tf_buffer, reg_cfg.target_frame, ee_link,
                stamp, reg_cfg.tf_timeout_s, logger=self._node.get_logger(),
            )
        except Exception as e:
            self._node.get_logger().warn(f"[registration] dynamic NBV TF failed: {e}. Falling back.")
            return []

        T_ee_cam = np.linalg.inv(T_base_ee) @ T_base_cam
        try:
            cam_candidates = generate_dynamic_nbv_candidates(
                requested_names, T_base_cam,
                np.asarray(tracked_roi["center"], dtype=np.float32),
                float(self._node.get_parameter("registration_dynamic_nbv_lateral_offset_m").value),
                float(self._node.get_parameter("registration_dynamic_nbv_vertical_offset_m").value),
                float(self._node.get_parameter("registration_dynamic_nbv_upward_bias_m").value),
            )
        except Exception as e:
            self._node.get_logger().warn(f"[registration] dynamic NBV generation failed: {e}. Falling back.")
            return []

        specs: List[Dict[str, Any]] = []
        for cand in cam_candidates:
            T_base_cam_des = np.asarray(cand.camera_pose, dtype=np.float64).reshape(4, 4)
            T_base_ee_des = T_base_cam_des @ np.linalg.inv(T_ee_cam)
            specs.append(
                {
                    "target_name": f"dyn_{cand.name}",
                    "display_name": f"dynamic/{cand.name}",
                    "base_name": cand.name,
                    "move_type": "pose",
                    "reference_frame": reg_cfg.target_frame,
                    "pose_matrix": T_base_ee_des.astype(np.float32),
                    "camera_pose": T_base_cam_des.astype(np.float32),
                }
            )
        if not specs:
            self._node.get_logger().warn("[dynamic-nbv] no camera-sphere candidates; falling back.")
        return specs

    # Motion delegation helpers (real impl in motion.MotionDispatcher).


    def _capture_registration_observation(
        self,
        sel: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        tracked_roi: Dict[str, Any],
        *,
        refresh: bool,
        use_roi_fallback: bool,
        attempt_label: str,
    ) -> Optional[Dict[str, Any]]:
        sel_now = sel
        if refresh:
            obs = None
            if bool(self._node.get_parameter("registration_followup_use_tracked_segmentation").value):
                obs = self._node._extract_followup_object_cloud_tracked(sel, tracked_roi, reg_cfg)
                if obs is not None:
                    self._node.get_logger().info(
                        f"[tracking] {attempt_label}: used tracked segmentation for '{sel['label']}'."
                    )
            if obs is None:
                maybe_sel = self._node._detect_current_selection(sel["label"])
                if maybe_sel is None:
                    if use_roi_fallback:
                        self._node.get_logger().warn(
                            f"[registration] {attempt_label}: could not redetect '{sel['label']}', trying 3D ROI fallback."
                        )
                        obs = self._node._extract_followup_object_cloud_roi(tracked_roi, reg_cfg)
                        if obs is None:
                            self._node.get_logger().warn(
                                f"[registration] {attempt_label}: ROI fallback failed, skipping."
                            )
                            return None
                    else:
                        self._node.get_logger().warn(
                            f"[registration] {attempt_label}: could not redetect '{sel['label']}', skipping."
                        )
                        return None
                else:
                    sel_now = maybe_sel
                    obs = self._node._extract_filtered_object_cloud(sel_now, publish_outputs=False)
        else:
            obs = self._node._extract_filtered_object_cloud(sel_now, publish_outputs=False)

        if obs is None:
            return None

        obs_tf = self._transform_observation_to_target_frame(
            obs,
            reg_cfg,
            note_prefix=attempt_label,
        )
        if "capture_transform" not in obs_tf:
            obs_tf["capture_transform"] = np.eye(4, dtype=np.float32)
        obs_tf["used_roi_fallback"] = bool(obs_tf.get("used_roi_fallback", False))
        if obs_tf["used_roi_fallback"] and not self._node._roi_fallback_is_plausible(obs_tf, tracked_roi):
            return None
        if bool(obs_tf.get("used_tracked_segmentation", False)) and not self._node._tracked_followup_is_plausible(obs_tf, tracked_roi):
            return None
        return obs_tf


    def _finalize_registered_views(
        self,
        first_obs: Dict[str, Any],
        views: List[RegistrationView],
        accepted_obs_meta: List[Dict[str, Any]],
        reg_cfg: RegistrationConfig,
        min_pts: int,
    ) -> Dict[str, Any]:
        def _tsdf_final_cloud() -> Optional[Dict[str, Any]]:
            tsdf_surface = self._node._publish_tsdf_cloud(accepted_obs_meta, reg_cfg)
            if tsdf_surface is None:
                return None
            tsdf_pts, tsdf_cols = tsdf_surface
            if tsdf_pts is None or len(tsdf_pts) < min_pts:
                return None
            self._node.get_logger().info(
                f"[registration] using TSDF surface cloud for final grasp input "
                f"({len(tsdf_pts)} points, raw_views={len(accepted_obs_meta)})."
            )
            return {
                "points": tsdf_pts,
                "colors": tsdf_cols,
                "frame_id": reg_cfg.target_frame,
                "registration_frame_is_target": True,
            }

        if len(views) == 1:
            self._node.get_logger().warn("[registration] no extra valid observations collected; using first cloud only.")
            tsdf_cloud = _tsdf_final_cloud()
            if tsdf_cloud is not None:
                return tsdf_cloud
            return {
                "points": first_obs["points"],
                "colors": first_obs["colors"],
                "frame_id": first_obs["header"].frame_id,
                "registration_frame_is_target": False,
            }

        result = register_and_fuse_views(views, self._node.tf_buffer, reg_cfg, logger=self._node.get_logger())
        if result.fused_points is None or len(result.fused_points) < min_pts:
            self._node.get_logger().warn(
                "[registration] fused cloud too small after registration; using first cloud only."
            )
            tsdf_cloud = _tsdf_final_cloud()
            if tsdf_cloud is not None:
                return tsdf_cloud
            return {
                "points": first_obs["points"],
                "colors": first_obs["colors"],
                "frame_id": first_obs["header"].frame_id,
                "registration_frame_is_target": False,
            }

        try:
            out_header = Header()
            out_header.stamp = self._node.get_clock().now().to_msg()
            out_header.frame_id = reg_cfg.target_frame
            fused_msg = self._node._create_cloud_xyzrgb(out_header, result.fused_points, result.fused_colors)
            self._node.pub_fused_cloud.publish(fused_msg)
        except Exception as e:
            self._node.get_logger().warn(f"Failed to publish fused registration cloud: {e}")

        tsdf_cloud = _tsdf_final_cloud()
        if tsdf_cloud is not None:
            return tsdf_cloud

        return {
            "points": result.fused_points,
            "colors": result.fused_colors,
            "frame_id": reg_cfg.target_frame,
            "registration_frame_is_target": True,
        }

    # ------------------------------------------------------------------
    # Grasping delegation (real impl in grasping/{graspgen_client,
    # grasp_scorer, filter_pipeline}). Wrappers preserve method names so
    # all internal call sites in registration / sweep code keep working.
    # ------------------------------------------------------------------


    def _maybe_register_multi_view_cloud(
        self,
        sel: Dict[str, Any],
        first_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        enabled = bool(self._node.get_parameter("registration_enabled").value)
        n_obs = max(1, int(self._node.get_parameter("registration_num_observations").value))
        if not enabled or n_obs <= 1:
            return {
                "points": first_obs["points"],
                "colors": first_obs["colors"],
                "frame_id": first_obs["header"].frame_id,
                "registration_frame_is_target": False,
            }

        reg_cfg = self._build_registration_config()
        min_pts = max(10, int(self._node.get_parameter("registration_min_points").value))
        interval_s = max(0.0, float(self._node.get_parameter("registration_observation_interval_s").value))
        refresh = bool(self._node.get_parameter("registration_refresh_detection").value)
        max_extra_attempts = max(0, int(self._node.get_parameter("registration_max_extra_attempts").value))
        min_dt = max(0.0, float(self._node.get_parameter("registration_min_camera_translation_m").value))
        min_dr = max(0.0, float(self._node.get_parameter("registration_min_camera_rotation_deg").value))
        max_same_iou = float(self._node.get_parameter("registration_max_bbox_iou_same_view").value)
        use_roi_fallback = bool(self._node.get_parameter("registration_followup_use_roi_fallback").value)
        nbv_enabled = bool(self._node.get_parameter("registration_nbv_enabled").value)
        dynamic_nbv_enabled = bool(self._node.get_parameter("registration_dynamic_nbv_enabled").value)
        nbv_gain_voxel = float(self._node.get_parameter("registration_nbv_gain_voxel_size_m").value)
        nbv_min_gain_ratio = float(self._node.get_parameter("registration_nbv_min_gain_ratio").value)
        nbv_min_new_voxels = int(self._node.get_parameter("registration_nbv_min_new_voxels").value)
        continuous_enabled = bool(self._node.get_parameter("registration_continuous_observation_enabled").value)
        named_targets = [
            s.strip()
            for s in str(self._node.get_parameter("registration_observation_named_targets").value).split(",")
            if s.strip()
        ]
        first_obs_tf = self._transform_observation_to_target_frame(first_obs, reg_cfg, note_prefix="view 0")
        if "capture_transform" not in first_obs_tf:
            first_obs_tf["capture_transform"] = np.eye(4, dtype=np.float32)

        views: List[RegistrationView] = [
            RegistrationView(
                points=np.asarray(first_obs_tf["points"], dtype=np.float32),
                colors=None if first_obs_tf["colors"] is None else np.asarray(first_obs_tf["colors"]),
                frame_id=str(first_obs_tf["header"].frame_id),
                stamp=first_obs_tf["header"].stamp,
                label=str(sel["label"]),
            )
        ]
        accepted_obs_meta: List[Dict[str, Any]] = [first_obs_tf]
        tracked_roi = self._node._make_tracking_roi(first_obs_tf, reg_cfg)
        nbv_fused_points = np.asarray(first_obs_tf["points"], dtype=np.float32)
        self._node._publish_registration_preview_cloud(views, reg_cfg)
        last_logged_utility = self._node._evaluate_registration_grasp_utility(
            sel,
            first_obs,
            views,
            accepted_obs_meta,
            reg_cfg,
            min_pts,
            context="view_1_seed",
        )
        utility_stop_streak = 0
        utility_stop_requested = False

        def maybe_log_registration_utility(
            context: str,
            observation_gain: Optional[Any] = None,
            *,
            views_override: Optional[List[RegistrationView]] = None,
            accepted_obs_override: Optional[List[Dict[str, Any]]] = None,
        ) -> bool:
            nonlocal last_logged_utility, utility_stop_streak
            if observation_gain is not None:
                self._node.get_logger().info(
                    f"[grasp-utility] {context}: voxel_gain "
                    f"(new_voxels={int(observation_gain.new_voxels)}, "
                    f"gain_ratio={float(observation_gain.gain_ratio):.3f}, "
                    f"total_voxels={int(observation_gain.total_voxels)})"
                )
            eval_views = views if views_override is None else views_override
            eval_obs = accepted_obs_meta if accepted_obs_override is None else accepted_obs_override
            utility_now = self._node._evaluate_registration_grasp_utility(
                sel,
                first_obs,
                eval_views,
                eval_obs,
                reg_cfg,
                min_pts,
                context=context,
            )
            if utility_now is None:
                return False
            self._node._debug_status_text = (
                f"UTILITY  {context}\n"
                f"U={utility_now['total']:.3f}  count={int(utility_now['surviving_count'])}\n"
                f"score={utility_now['mean_score']:.3f}  div={utility_now['diversity_deg']:.1f}deg"
            )
            stop_now = False
            if last_logged_utility is not None:
                delta_u = float(utility_now["total"] - last_logged_utility["total"])
                self._node.get_logger().info(
                    f"[grasp-utility] {context}: delta_U={delta_u:.3f} "
                    f"(prev={last_logged_utility['total']:.3f} -> now={utility_now['total']:.3f})"
                )
                stop_now, utility_stop_streak, _ = self._node._update_registration_utility_stop_state(
                    context=context,
                    view_count=len(eval_views),
                    prev_utility=last_logged_utility,
                    utility_now=utility_now,
                    small_gain_streak=utility_stop_streak,
                )
            last_logged_utility = utility_now
            return stop_now

        def build_predictive_utility_stop_callback():
            predictive_last_utility = last_logged_utility
            predictive_streak = utility_stop_streak

            def _callback(
                context: str,
                eval_views: List[RegistrationView],
                eval_obs_meta: List[Dict[str, Any]],
            ) -> bool:
                nonlocal predictive_last_utility, predictive_streak
                utility_now = self._node._evaluate_registration_grasp_utility(
                    sel,
                    first_obs,
                    eval_views,
                    eval_obs_meta,
                    reg_cfg,
                    min_pts,
                    context=context,
                    log_details=False,
                )
                if utility_now is None:
                    return False
                stop_now = False
                if predictive_last_utility is not None:
                    stop_now, predictive_streak, _ = self._node._update_registration_utility_stop_state(
                        context=context,
                        view_count=len(eval_views),
                        prev_utility=predictive_last_utility,
                        utility_now=utility_now,
                        small_gain_streak=predictive_streak,
                    )
                predictive_last_utility = utility_now
                return stop_now

            return _callback

        max_attempts = n_obs + max_extra_attempts
        attempt_idx = 1

        if nbv_enabled and named_targets:
            candidate_specs: List[Dict[str, Any]] = []
            using_dynamic_specs = False
            if dynamic_nbv_enabled:
                candidate_specs = self._generate_dynamic_registration_candidates(
                    first_obs_tf,
                    tracked_roi,
                    reg_cfg,
                    named_targets,
                )
                using_dynamic_specs = bool(candidate_specs)
            if not candidate_specs:
                candidate_specs = [{"target_name": name, "move_type": "named"} for name in named_targets]

            dynamic_coarse_specs: List[Dict[str, Any]] = []
            dynamic_refine_specs: List[Dict[str, Any]] = []
            if using_dynamic_specs and bool(self._node.get_parameter("registration_dynamic_nbv_use_joint_interpolation").value):
                dynamic_coarse_specs, dynamic_refine_specs = self._partition_dynamic_joint_candidates(candidate_specs)
            else:
                dynamic_coarse_specs = candidate_specs

            candidate_observations: List[Dict[str, Any]] = []
            prev_meta = accepted_obs_meta[-1]

            def probe_candidates(
                specs_to_probe: List[Dict[str, Any]],
                start_attempt_idx: int,
                attempt_prefix: str,
            ) -> int:
                nonlocal prev_meta, candidate_observations
                local_attempt_idx = start_attempt_idx
                for cand_spec in specs_to_probe:
                    if local_attempt_idx >= max_attempts:
                        break
                    target_name = str(cand_spec.get("display_name", cand_spec["target_name"]))
                    moved = self._node._move_to_registration_candidate(cand_spec)
                    if not moved:
                        local_attempt_idx += 1
                        continue
                    if str(cand_spec.get("move_type", "named")) == "pose":
                        if not self._node._dynamic_candidate_achieved_is_distinct(cand_spec, first_obs_tf, reg_cfg):
                            local_attempt_idx += 1
                            continue
                    if interval_s > 0.0:
                        time.sleep(interval_s)

                    obs_tf = self._capture_registration_observation(
                        sel,
                        reg_cfg,
                        tracked_roi,
                        refresh=refresh,
                        use_roi_fallback=use_roi_fallback,
                        attempt_label=f"{attempt_prefix} {local_attempt_idx}",
                    )
                    if obs_tf is None:
                        local_attempt_idx += 1
                        continue
                    if obs_tf["points"].shape[0] < min_pts:
                        n_pts = int(obs_tf["points"].shape[0])
                        self._node.get_logger().warn(
                            f"[registration] {attempt_prefix} {local_attempt_idx}: insufficient points ({n_pts} < {min_pts}), skipping."
                        )
                        local_attempt_idx += 1
                        continue

                    T_prev = np.asarray(prev_meta["capture_transform"], dtype=np.float64)
                    T_cur = np.asarray(obs_tf["capture_transform"], dtype=np.float64)
                    dt = float(np.linalg.norm(T_cur[:3, 3] - T_prev[:3, 3]))
                    dR = T_prev[:3, :3].T @ T_cur[:3, :3]
                    cos_theta = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
                    dr = math.degrees(math.acos(cos_theta))
                    bbox_iou = self._node._bbox_iou_xyxy(prev_meta["bbox"], obs_tf["bbox"])
                    if dt < min_dt and dr < min_dr and bbox_iou >= max_same_iou:
                        self._node.get_logger().warn(
                            f"[registration] {attempt_prefix} {local_attempt_idx}: view too similar to previous "
                            f"(dt={dt:.3f} m, dr={dr:.1f} deg, bbox_iou={bbox_iou:.2f}); move more and retry."
                        )
                        local_attempt_idx += 1
                        continue

                    gain = compute_observation_gain(
                        nbv_fused_points,
                        np.asarray(obs_tf["points"], dtype=np.float32),
                        nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                    )
                    self._node.get_logger().info(
                        f"[nbv] probed target='{target_name}' total_voxels={gain.total_voxels} "
                        f"new_voxels={gain.new_voxels} gain_ratio={gain.gain_ratio:.3f}"
                    )
                    candidate_observations.append(
                        {
                            "target_name": target_name,
                            "base_name": str(cand_spec.get("base_name", target_name)),
                            "obs_tf": obs_tf,
                            "candidate": cand_spec,
                            "gain_total_voxels": int(gain.total_voxels),
                            "gain_new_voxels": int(gain.new_voxels),
                            "gain_ratio": float(gain.gain_ratio),
                        }
                    )
                    local_attempt_idx += 1
                return local_attempt_idx

            if continuous_enabled and using_dynamic_specs and dynamic_coarse_specs:
                chosen_spec = self._choose_continuous_direction_without_probe(first_obs_tf, dynamic_coarse_specs)
                if chosen_spec is None:
                    by_base: Dict[str, Dict[str, Any]] = {}
                    for spec in dynamic_coarse_specs:
                        bname = str(spec.get("base_name", spec.get("target_name", "")))
                        if bname not in by_base:
                            by_base[bname] = spec
                    chosen_spec = next(iter(by_base.values()))
                chosen_base = str(chosen_spec.get("base_name", chosen_spec.get("target_name", "")))
                chosen_alpha = float(chosen_spec.get("alpha", 1.0))

                sweep_specs = [chosen_spec]
                if bool(self._node.get_parameter("registration_continuous_observation_sweep_all_directions").value):
                    sweep_all_alpha = float(
                        self._node.get_parameter("registration_continuous_observation_sweep_all_alpha").value
                    )
                    by_base: Dict[str, Dict[str, Any]] = {}
                    for spec in dynamic_coarse_specs:
                        bname = str(spec.get("base_name", spec.get("target_name", "")))
                        if not bname:
                            continue
                        if bname not in by_base:
                            by_base[bname] = spec
                        elif sweep_all_alpha > 0.0:
                            prev_alpha = float(by_base[bname].get("alpha", 1.0))
                            cur_alpha = float(spec.get("alpha", 1.0))
                            if abs(cur_alpha - sweep_all_alpha) < abs(prev_alpha - sweep_all_alpha):
                                by_base[bname] = spec
                    direction_order = [chosen_base]
                    for name in named_targets:
                        if name in by_base and name not in direction_order:
                            direction_order.append(name)
                    for spec in dynamic_coarse_specs:
                        bname = str(spec.get("base_name", spec.get("target_name", "")))
                        if bname and bname not in direction_order:
                            direction_order.append(bname)
                    sweep_specs = [by_base[name] for name in direction_order if name in by_base]
                    self._node.get_logger().info(
                        "[continuous] sweep-all enabled: "
                        + ", ".join(
                            f"{str(spec.get('base_name', spec.get('target_name', '')))}@"
                            f"{float(spec.get('alpha', 1.0)):.2f}"
                            for spec in sweep_specs
                        )
                    )

                prior_sweep_observations: List[Dict[str, Any]] = []
                return_to_seed_between_legs = bool(
                    self._node.get_parameter("registration_continuous_observation_sweep_all_return_to_seed").value
                )
                for leg_idx, sweep_spec in enumerate(sweep_specs):
                    leg_base = str(sweep_spec.get("base_name", sweep_spec.get("target_name", "")))
                    synthetic_cand = {
                        "target_name": sweep_spec.get("display_name", leg_base),
                        "base_name": leg_base,
                        "candidate": sweep_spec,
                    }
                    skip_return_to_seed = (
                        self._node._registration_init_move_failed_this_cycle
                        or (
                            leg_idx == 0
                            and self._node._registration_prepositioned_for_selection
                        )
                        or (
                            leg_idx > 0
                            and not return_to_seed_between_legs
                        )
                    )
                    if skip_return_to_seed:
                        self._node.get_logger().info(
                            "[continuous] skipping return-to-seed before sweep "
                            "(already prepositioned, seed move failed, or continuing sweep-all)."
                        )
                    sweep_obs = self._collect_continuous_joint_sweep_observations(
                        synthetic_cand,
                        [],
                        sel,
                        reg_cfg,
                        tracked_roi,
                        refresh,
                        use_roi_fallback,
                        min_pts,
                        accepted_obs_meta[-1],
                        preview_seed_views=views,
                        preview_seed_obs_meta=accepted_obs_meta[0],
                        skip_return_to_seed=skip_return_to_seed,
                        prior_sweep_observations=prior_sweep_observations,
                        utility_prior_obs_meta=accepted_obs_meta,
                        utility_stop_callback=(
                            build_predictive_utility_stop_callback()
                            if bool(self._node.get_parameter("registration_grasp_utility_stop_during_motion").value)
                            else None
                        ),
                    )
                    prior_sweep_observations.extend(sweep_obs)
                    for obs_tf in sweep_obs:
                        gain_now = compute_observation_gain(
                            nbv_fused_points,
                            np.asarray(obs_tf["points"], dtype=np.float32),
                            nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                        )
                        views.append(
                            RegistrationView(
                                points=np.asarray(obs_tf["points"], dtype=np.float32),
                                colors=None if obs_tf["colors"] is None else np.asarray(obs_tf["colors"]),
                                frame_id=str(obs_tf["header"].frame_id),
                                stamp=obs_tf["header"].stamp,
                                label=str(sel["label"]),
                            )
                        )
                        accepted_obs_meta.append(obs_tf)
                        pts_acc = np.asarray(obs_tf["points"], dtype=np.float32)
                        aabb_min = np.min(pts_acc, axis=0)
                        aabb_max = np.max(pts_acc, axis=0)
                        tracked_roi["center"] = (0.5 * (aabb_min + aabb_max)).astype(np.float32)
                        margin = float(self._node.get_parameter("registration_roi_margin_m").value)
                        min_half = float(self._node.get_parameter("registration_roi_min_half_extent_m").value)
                        tracked_roi["half_extent"] = np.maximum(
                            0.5 * (aabb_max - aabb_min) + margin,
                            min_half,
                        ).astype(np.float32)
                        if not bool(obs_tf.get("used_roi_fallback", False)):
                            tracked_roi["reference_half_extent"] = tracked_roi["half_extent"].copy()
                            tracked_roi["reference_point_count"] = int(len(pts_acc))
                            bbox_obs = [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])]
                            tracked_roi["reference_bbox_xyxy"] = bbox_obs
                            tracked_roi["reference_bbox_area_px"] = int(
                                max(1, (bbox_obs[2] - bbox_obs[0]) * (bbox_obs[3] - bbox_obs[1]))
                            )
                            ref_voxel = float(
                                self._node.get_parameter("registration_roi_fallback_reference_voxel_m").value
                            )
                            ref_points, _ = voxel_downsample(pts_acc, None, ref_voxel)
                            tracked_roi["reference_points"] = np.asarray(ref_points, dtype=np.float32)
                        nbv_fused_points, _ = voxel_downsample(
                            np.vstack([nbv_fused_points, np.asarray(obs_tf["points"], dtype=np.float32)]),
                            None,
                            nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                        )
                        self._node._publish_registration_preview_cloud(views, reg_cfg)
                        utility_stop_requested = maybe_log_registration_utility(
                            f"view_{len(views)}",
                            observation_gain=gain_now,
                        )
                        if utility_stop_requested:
                            break
                    if utility_stop_requested:
                        break

                if utility_stop_requested:
                    return self._finalize_registered_views(first_obs, views, accepted_obs_meta, reg_cfg, min_pts)

                if len(views) < n_obs:
                    self._node.get_logger().warn(
                        f"[registration] collected {len(views)}/{n_obs} sufficiently distinct observations."
                    )
                return self._finalize_registered_views(first_obs, views, accepted_obs_meta, reg_cfg, min_pts)

            elif using_dynamic_specs and dynamic_coarse_specs:
                attempt_idx = probe_candidates(dynamic_coarse_specs, attempt_idx, "coarse observation attempt")

                refine_top_k = max(0, int(self._node.get_parameter("registration_dynamic_nbv_refine_top_k").value))
                if dynamic_refine_specs and refine_top_k > 0 and candidate_observations:
                    coarse_scored = []
                    for cand in candidate_observations:
                        gain = compute_observation_gain(
                            nbv_fused_points,
                            np.asarray(cand["obs_tf"]["points"], dtype=np.float32),
                            nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                        )
                        coarse_scored.append((cand, gain))
                    coarse_scored.sort(key=lambda item: (item[1].new_voxels, item[1].gain_ratio), reverse=True)
                    winning_base_names = {
                        str(item[0].get("base_name", item[0]["target_name"]))
                        for item in coarse_scored[:refine_top_k]
                    }
                    refine_probe_specs = [
                        spec for spec in dynamic_refine_specs
                        if str(spec.get("base_name", spec.get("target_name", ""))) in winning_base_names
                    ]
                    if refine_probe_specs:
                        self._node.get_logger().info(
                            f"[dynamic-nbv] refining alpha for winning directions: {sorted(winning_base_names)}"
                        )
                        if bool(self._node.get_parameter("registration_dynamic_nbv_drop_nonwinning_coarse").value):
                            candidate_observations = [
                                cand for cand in candidate_observations
                                if str(cand.get("base_name", cand["target_name"])) in winning_base_names
                            ]
                        attempt_idx = probe_candidates(
                            refine_probe_specs,
                            attempt_idx,
                            "refine observation attempt",
                        )
            else:
                attempt_idx = probe_candidates(candidate_specs, attempt_idx, "observation attempt")

            if using_dynamic_specs and not candidate_observations:
                self._node.get_logger().warn(
                    "[dynamic-nbv] no distinct generated candidates achieved useful camera motion; "
                    "falling back to named-target NBV candidates."
                )
                candidate_specs = [{"target_name": name, "move_type": "named"} for name in named_targets]
                fallback_idx = 1
                prev_meta = accepted_obs_meta[-1]
                while fallback_idx < max_attempts and (fallback_idx - 1) < len(candidate_specs):
                    cand_spec = candidate_specs[fallback_idx - 1]
                    target_name = str(cand_spec.get("display_name", cand_spec["target_name"]))
                    moved = self._node._move_to_registration_candidate(cand_spec)
                    if not moved:
                        fallback_idx += 1
                        continue
                    if interval_s > 0.0:
                        time.sleep(interval_s)

                    obs_tf = self._capture_registration_observation(
                        sel,
                        reg_cfg,
                        tracked_roi,
                        refresh=refresh,
                        use_roi_fallback=use_roi_fallback,
                        attempt_label=f"fallback observation attempt {fallback_idx}",
                    )
                    if obs_tf is None:
                        fallback_idx += 1
                        continue
                    if obs_tf["points"].shape[0] < min_pts:
                        n_pts = int(obs_tf["points"].shape[0])
                        self._node.get_logger().warn(
                            f"[registration] fallback observation attempt {fallback_idx}: insufficient points "
                            f"({n_pts} < {min_pts}), skipping."
                        )
                        fallback_idx += 1
                        continue

                    T_prev = np.asarray(prev_meta["capture_transform"], dtype=np.float64)
                    T_cur = np.asarray(obs_tf["capture_transform"], dtype=np.float64)
                    dt = float(np.linalg.norm(T_cur[:3, 3] - T_prev[:3, 3]))
                    dR = T_prev[:3, :3].T @ T_cur[:3, :3]
                    cos_theta = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
                    dr = math.degrees(math.acos(cos_theta))
                    bbox_iou = self._node._bbox_iou_xyxy(prev_meta["bbox"], obs_tf["bbox"])
                    if dt < min_dt and dr < min_dr and bbox_iou >= max_same_iou:
                        self._node.get_logger().warn(
                            f"[registration] fallback observation attempt {fallback_idx}: view too similar to previous "
                            f"(dt={dt:.3f} m, dr={dr:.1f} deg, bbox_iou={bbox_iou:.2f}); skipping."
                        )
                        fallback_idx += 1
                        continue

                    gain = compute_observation_gain(
                        nbv_fused_points,
                        np.asarray(obs_tf["points"], dtype=np.float32),
                        nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                    )
                    self._node.get_logger().info(
                        f"[nbv] fallback target='{target_name}' total_voxels={gain.total_voxels} "
                        f"new_voxels={gain.new_voxels} gain_ratio={gain.gain_ratio:.3f}"
                    )
                    candidate_observations.append(
                        {
                            "target_name": target_name,
                            "base_name": str(cand_spec.get("base_name", target_name)),
                            "obs_tf": obs_tf,
                            "candidate": cand_spec,
                        }
                    )
                    fallback_idx += 1

            while len(views) < n_obs and candidate_observations:
                scored = []
                for cand in candidate_observations:
                    gain = compute_observation_gain(
                        nbv_fused_points,
                        np.asarray(cand["obs_tf"]["points"], dtype=np.float32),
                        nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                    )
                    scored.append((cand, gain))
                scored.sort(key=lambda item: (item[1].new_voxels, item[1].gain_ratio), reverse=True)
                best_cand, best_gain = scored[0]
                if best_gain.total_voxels <= 0 or best_gain.new_voxels < nbv_min_new_voxels or best_gain.gain_ratio < nbv_min_gain_ratio:
                    self._node.get_logger().warn(
                        f"[nbv] no remaining candidate passes gain thresholds "
                        f"(best target='{best_cand['target_name']}', new_voxels={best_gain.new_voxels}, "
                        f"gain_ratio={best_gain.gain_ratio:.3f})."
                    )
                    break

                self._node.get_logger().info(
                    f"[nbv] selecting target='{best_cand['target_name']}' "
                    f"(new_voxels={best_gain.new_voxels}, gain_ratio={best_gain.gain_ratio:.3f})"
                )
                selected_obs: List[Dict[str, Any]] = [best_cand["obs_tf"]]
                if (
                    continuous_enabled
                    and using_dynamic_specs
                    and str(best_cand["candidate"].get("move_type", "")) == "joint"
                ):
                    skip_return_to_seed = (
                        self._node._registration_prepositioned_for_selection
                        or self._node._registration_init_move_failed_this_cycle
                    )
                    if skip_return_to_seed:
                        self._node.get_logger().info(
                            "[continuous] skipping return-to-seed before sweep "
                            "(already prepositioned or seed move failed this cycle)."
                        )
                    sweep_obs = self._collect_continuous_joint_sweep_observations(
                        best_cand,
                        candidate_observations,
                        sel,
                        reg_cfg,
                        tracked_roi,
                        refresh,
                        use_roi_fallback,
                        min_pts,
                        accepted_obs_meta[-1],
                        preview_seed_views=views,
                        preview_seed_obs_meta=accepted_obs_meta[0],
                        skip_return_to_seed=skip_return_to_seed,
                        utility_prior_obs_meta=accepted_obs_meta,
                        utility_stop_callback=(
                            build_predictive_utility_stop_callback()
                            if bool(self._node.get_parameter("registration_grasp_utility_stop_during_motion").value)
                            else None
                        ),
                    )
                    if sweep_obs:
                        selected_obs = sweep_obs

                for obs_tf in selected_obs:
                    gain_now = compute_observation_gain(
                        nbv_fused_points,
                        np.asarray(obs_tf["points"], dtype=np.float32),
                        nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                    )
                    views.append(
                        RegistrationView(
                            points=np.asarray(obs_tf["points"], dtype=np.float32),
                            colors=None if obs_tf["colors"] is None else np.asarray(obs_tf["colors"]),
                            frame_id=str(obs_tf["header"].frame_id),
                            stamp=obs_tf["header"].stamp,
                            label=str(sel["label"]),
                        )
                    )
                    accepted_obs_meta.append(obs_tf)
                    pts_acc = np.asarray(obs_tf["points"], dtype=np.float32)
                    aabb_min = np.min(pts_acc, axis=0)
                    aabb_max = np.max(pts_acc, axis=0)
                    tracked_roi["center"] = (0.5 * (aabb_min + aabb_max)).astype(np.float32)
                    margin = float(self._node.get_parameter("registration_roi_margin_m").value)
                    min_half = float(self._node.get_parameter("registration_roi_min_half_extent_m").value)
                    tracked_roi["half_extent"] = np.maximum(
                        0.5 * (aabb_max - aabb_min) + margin,
                        min_half,
                    ).astype(np.float32)
                    if not bool(obs_tf.get("used_roi_fallback", False)):
                        tracked_roi["reference_half_extent"] = tracked_roi["half_extent"].copy()
                        tracked_roi["reference_point_count"] = int(len(pts_acc))
                        bbox_obs = [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])]
                        tracked_roi["reference_bbox_xyxy"] = bbox_obs
                        tracked_roi["reference_bbox_area_px"] = int(
                            max(1, (bbox_obs[2] - bbox_obs[0]) * (bbox_obs[3] - bbox_obs[1]))
                        )
                        ref_voxel = float(self._node.get_parameter("registration_roi_fallback_reference_voxel_m").value)
                        ref_points, _ = voxel_downsample(pts_acc, None, ref_voxel)
                        tracked_roi["reference_points"] = np.asarray(ref_points, dtype=np.float32)
                    nbv_fused_points, _ = voxel_downsample(
                        np.vstack([nbv_fused_points, np.asarray(obs_tf["points"], dtype=np.float32)]),
                        None,
                        nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                    )
                    self._node._publish_registration_preview_cloud(views, reg_cfg)
                    utility_stop_requested = maybe_log_registration_utility(
                        f"view_{len(views)}",
                        observation_gain=gain_now,
                    )
                    if utility_stop_requested:
                        break

                candidate_observations = [
                    cand for cand in candidate_observations
                    if str(cand.get("base_name", cand["target_name"])) != str(
                        best_cand.get("base_name", best_cand["target_name"])
                    )
                ]
                if continuous_enabled and using_dynamic_specs:
                    break
                if utility_stop_requested:
                    break

            if utility_stop_requested:
                return self._finalize_registered_views(first_obs, views, accepted_obs_meta, reg_cfg, min_pts)
            if len(views) < n_obs:
                self._node.get_logger().warn(
                    f"[registration] collected {len(views)}/{n_obs} sufficiently distinct observations."
                )
            return self._finalize_registered_views(first_obs, views, accepted_obs_meta, reg_cfg, min_pts)

        while len(views) < n_obs and attempt_idx < max_attempts:
            target_idx = min(attempt_idx - 1, len(named_targets) - 1) if named_targets else -1
            target_name = named_targets[target_idx] if target_idx < len(named_targets) else ""
            if target_idx < len(named_targets):
                moved = self._node._move_to_registration_target(target_name)
                if not moved:
                    attempt_idx += 1
                    continue
            if interval_s > 0.0:
                time.sleep(interval_s)
            obs_tf = self._capture_registration_observation(
                sel,
                reg_cfg,
                tracked_roi,
                refresh=refresh,
                use_roi_fallback=use_roi_fallback,
                attempt_label=f"observation attempt {attempt_idx}",
            )
            if obs_tf is None:
                attempt_idx += 1
                continue
            if obs_tf["points"].shape[0] < min_pts:
                n_pts = int(obs_tf["points"].shape[0])
                self._node.get_logger().warn(
                    f"[registration] observation attempt {attempt_idx}: insufficient points ({n_pts} < {min_pts}), skipping."
                )
                attempt_idx += 1
                continue

            prev_meta = accepted_obs_meta[-1]
            T_prev = np.asarray(prev_meta["capture_transform"], dtype=np.float64)
            T_cur = np.asarray(obs_tf["capture_transform"], dtype=np.float64)
            dt = float(np.linalg.norm(T_cur[:3, 3] - T_prev[:3, 3]))
            dR = T_prev[:3, :3].T @ T_cur[:3, :3]
            cos_theta = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
            dr = math.degrees(math.acos(cos_theta))
            bbox_iou = self._node._bbox_iou_xyxy(prev_meta["bbox"], obs_tf["bbox"])
            if dt < min_dt and dr < min_dr and bbox_iou >= max_same_iou:
                self._node.get_logger().warn(
                    f"[registration] observation attempt {attempt_idx}: view too similar to previous "
                    f"(dt={dt:.3f} m, dr={dr:.1f} deg, bbox_iou={bbox_iou:.2f}); move more and retry."
                )
                attempt_idx += 1
                continue

            if nbv_enabled:
                gain = compute_observation_gain(
                    nbv_fused_points,
                    np.asarray(obs_tf["points"], dtype=np.float32),
                    nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
                )
                self._node.get_logger().info(
                    f"[nbv] target='{target_name or 'manual'}' total_voxels={gain.total_voxels} "
                    f"new_voxels={gain.new_voxels} gain_ratio={gain.gain_ratio:.3f}"
                )
                if gain.total_voxels > 0 and (
                    gain.new_voxels < nbv_min_new_voxels or gain.gain_ratio < nbv_min_gain_ratio
                ):
                    self._node.get_logger().warn(
                        f"[nbv] rejecting target '{target_name or 'manual'}' due to low information gain "
                        f"(new_voxels={gain.new_voxels}, gain_ratio={gain.gain_ratio:.3f})."
                    )
                    attempt_idx += 1
                    continue

            views.append(
                RegistrationView(
                    points=np.asarray(obs_tf["points"], dtype=np.float32),
                    colors=None if obs_tf["colors"] is None else np.asarray(obs_tf["colors"]),
                    frame_id=str(obs_tf["header"].frame_id),
                    stamp=obs_tf["header"].stamp,
                    label=str(sel["label"]),
                )
            )
            accepted_obs_meta.append(obs_tf)
            pts_acc = np.asarray(obs_tf["points"], dtype=np.float32)
            aabb_min = np.min(pts_acc, axis=0)
            aabb_max = np.max(pts_acc, axis=0)
            tracked_roi["center"] = (0.5 * (aabb_min + aabb_max)).astype(np.float32)
            margin = float(self._node.get_parameter("registration_roi_margin_m").value)
            min_half = float(self._node.get_parameter("registration_roi_min_half_extent_m").value)
            tracked_roi["half_extent"] = np.maximum(
                0.5 * (aabb_max - aabb_min) + margin,
                min_half,
            ).astype(np.float32)
            if not bool(obs_tf.get("used_roi_fallback", False)):
                tracked_roi["reference_half_extent"] = tracked_roi["half_extent"].copy()
                tracked_roi["reference_point_count"] = int(len(pts_acc))
                bbox_obs = [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])]
                tracked_roi["reference_bbox_xyxy"] = bbox_obs
                tracked_roi["reference_bbox_area_px"] = int(
                    max(1, (bbox_obs[2] - bbox_obs[0]) * (bbox_obs[3] - bbox_obs[1]))
                )
                ref_voxel = float(self._node.get_parameter("registration_roi_fallback_reference_voxel_m").value)
                ref_points, _ = voxel_downsample(pts_acc, None, ref_voxel)
                tracked_roi["reference_points"] = np.asarray(ref_points, dtype=np.float32)
            nbv_fused_points, _ = voxel_downsample(
                np.vstack([nbv_fused_points, np.asarray(obs_tf["points"], dtype=np.float32)]),
                None,
                nbv_gain_voxel if nbv_gain_voxel > 0.0 else reg_cfg.voxel_size_m,
            )
            self._node._publish_registration_preview_cloud(views, reg_cfg)
            utility_stop_requested = maybe_log_registration_utility(
                f"view_{len(views)}",
                observation_gain=gain if nbv_enabled else None,
            )
            if utility_stop_requested:
                break
            attempt_idx += 1

        if len(views) < n_obs:
            self._node.get_logger().warn(
                f"[registration] collected {len(views)}/{n_obs} sufficiently distinct observations."
            )
        return self._finalize_registered_views(first_obs, views, accepted_obs_meta, reg_cfg, min_pts)
