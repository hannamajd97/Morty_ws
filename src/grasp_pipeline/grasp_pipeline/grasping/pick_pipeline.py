"""End-to-end pick pipeline for one user selection.

Decomposes the legacy 580-line ``_publish_selected`` orchestrator method
into a single class with one entry point — :meth:`PickPipeline.run` —
that runs the full sequence:

    1. (optional) move arm to the registration initial seed pose
    2. extract the first object cloud for the chosen detection
    3. multi-view register / fuse if registration is enabled
    4. downsample the cloud and call GraspGen
    5. run the geometric filter chain (no-under, penetration, reject
       from behind, upright, reject-side, view-consistent, side-center)
    6. centroid + EE-orientation reweight, publish centroid
    7. final orientation-consistency gate + utility logging
    8. publish top-K grasp candidates as a PoseArray
    9. pick the single best grasp via the heuristic selector
   10. publish best pose in source and target frames; remember it for
       continuous re-publishing on the timer

Reads / writes node state via ``self._node.<attr>``. The method body is
otherwise byte-identical to the legacy implementation so behaviour is
preserved exactly.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, TYPE_CHECKING, Optional

import numpy as np
import time as _time
import rclpy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, Header
from tf_transformations import quaternion_from_matrix

from ..geometry_utils import (
    _aabb_min_max,
    _denorm_T_from_unit_cube,
    _inside_aabb,
    _make_T_grasp_to_tcp,
    _publish_marker,
)
from ..grasp_client import send_points_to_graspgen
from ..grasp_filters import (
    _filter_finger_penetration,
    _filter_no_under_grasp,
    _filter_reject_from_behind,
    _filter_reject_side_grasps,
    _filter_roll_upright_by_plane,
    _filter_side_center_window,
    _filter_view_consistent,
    _reweight_scores_by_centroid,
)
from ..utils import pick_best_grasp

if TYPE_CHECKING:
    from rclpy.node import Node


class PickPipeline:
    """Run the full pick pipeline for one user selection."""

    def __init__(self, node: "Node") -> None:
        self._node = node
        # True while _compute runs for the precompute worker. Foreground and
        # background never overlap (GraspPrecomputer.foreground() holds the
        # model lock around run()), so one flag is enough.
        self._background = False

    def _aborted(self, where: str) -> bool:
        """Operator stop check at phase boundaries (trial 1001).

        A background job is not the operator's pick: a stale stop flag must
        not kill it, and the operator's SELECTION must — the moment someone
        chooses an object the worker's job is worthless and only delays the
        real pick, so it ends quietly at the next phase boundary."""
        if self._background:
            return bool(getattr(self._node, "selection_made", False))
        if getattr(self._node, "_pick_abort", False):
            self._node.get_logger().warn(
                f"[abort] operator stop honoured during {where} — "
                "ending this pick cycle, publishing nothing."
            )
            return True
        return False

    def run(self, sel: Dict[str, Any]) -> bool:
        """Run one selection and return whether a grasp command was published.

        Cache first (2026-09-11): if the precompute worker already holds
        valid grasps for this object at this arm pose, replay its captured
        outputs, rebuild the neighbour boxes from the live frame, and publish
        the stored candidates; otherwise compute as always. The worker's
        model lock is held for the whole pick so no background job can run
        in between."""
        pre = getattr(self._node, "_precompute", None)
        if pre is None:
            return self._run_locked(sel, None)
        with pre.foreground():
            return self._run_locked(sel, pre)

    def _run_locked(self, sel: Dict[str, Any], pre) -> bool:
        if pre is not None and pre.enabled and not sel.get("force_fresh", False):
            entry, why = pre.lookup(sel)
            if entry is not None:
                t0 = _time.monotonic()
                try:
                    if pre.replay_capture(entry, sel):
                        ok = self._publish(entry.result, sel)
                        self._node.get_logger().info(
                            f"[precompute] HIT '{entry.label}' (age {_time.monotonic() - entry.t_done:.0f} s) "
                            f"-> published in {_time.monotonic() - t0:.2f} s")
                        return ok
                except Exception as exc:
                    self._node.get_logger().warn(f"[precompute] replay failed ({exc}) — computing fresh.")
            else:
                self._node.get_logger().info(f"[precompute] MISS: {why} — fresh path")
        res = self._compute(sel, background=False)
        if res is None:
            return False
        return self._publish(res, sel)

    def _compute(self, sel: Dict[str, Any], background: bool = False) -> Optional[Dict[str, Any]]:
        """Phases 0-7: capture the object, run GraspGen, filter. Returns the
        result dict `_publish` consumes, or None when the cycle must end.

        ``background=True`` (2026-09-11, the precompute cache): the same
        computation for an object nobody has selected — nothing is
        published, the arm is never moved (no registration pre-positioning),
        and no node state that the overlay/voice/GUI read is written."""
        grasps: List[Dict[str, Any]] = []
        raw_grasps: List[Dict[str, Any]] = []
        self._background = bool(background)
        centroid_msg = None          # kept in the result so a cache replay
        centroid_base_msg = None     # publishes exactly what a fresh pick does
        if not background:
            self._node.last_candidates_base = None
            self._node.last_candidate_scores = None

        # ── Phase 0: the robot's transforms must exist ────────────────
        # Every later phase needs camera -> base_link (obstacles, centroid,
        # candidates). Without the driver there is no robot_state_publisher
        # and no such chain; the pick then failed late, silently, and left
        # the cycle waiting for an executor that never got candidates
        # (2026-09-09). Say it once, at the start, and end the cycle.
        try:
            import rclpy.time as _rt
            import rclpy.duration as _rd
            cam_frame = str(getattr(sel.get("header"), "frame_id", "") or "camera_color_optical_frame")
            if not self._node.tf_buffer.can_transform(
                    "base_link", cam_frame, _rt.Time(), timeout=_rd.Duration(seconds=0.5)):
                self._node.get_logger().error(
                    f"No transform base_link <- {cam_frame}: the robot driver is not running "
                    "(no robot_state_publisher). Start the driver, then select again.")
                return False
        except Exception as exc:
            self._node.get_logger().warn(f"[tf-precheck] skipped: {exc}")

        # ── Phase 1: pre-position to the registration seed pose ───────
        if not background and bool(self._node.get_parameter("registration_enabled").value) and bool(
            self._node.get_parameter("registration_move_to_initial_target").value
        ):
            init_target = str(self._node.get_parameter("registration_initial_named_target").value).strip()
            if init_target:
                if self._node._registration_prepositioned_for_selection and not self._node._registration_init_move_failed_this_cycle:
                    self._node.get_logger().info(
                        f"[registration] already at '{init_target}' from preselection; "
                        "capturing first view without another seed move."
                    )
                elif self._node._registration_init_move_failed_this_cycle:
                    self._node.get_logger().warn(
                        f"[registration] skipping repeated move to '{init_target}' because it "
                        "already failed this cycle; capturing first view at current pose."
                    )
                else:
                    moved = self._node._move_to_registration_target(init_target)
                    if moved:
                        self._node._registration_prepositioned_for_selection = True
                    else:
                        self._node._registration_init_move_failed_this_cycle = True
                        self._node.get_logger().warn(
                            f"[registration] initial move to '{init_target}' failed; capturing first view at current pose."
                        )

        # ── Phase 2: capture first cloud + multi-view registration ────
        import time as _time
        _t_phase = _time.monotonic()
        sel_capture = self._node._refresh_selection_for_capture(sel)
        self._node.get_logger().info(
            f"[timing] refresh-detection {_time.monotonic() - _t_phase:.2f}s")
        _t_phase = _time.monotonic()
        obs0 = self._node._extract_filtered_object_cloud(sel_capture, publish_outputs=not background, background=background)
        self._node.get_logger().info(
            f"[timing] capture+segment+cloud {_time.monotonic() - _t_phase:.2f}s")
        _t_phase = _time.monotonic()
        if obs0 is None:
            self._node.get_logger().warn(
                "Selected crop produced no valid depth points; cannot run GraspGen for this selection."
            )
            return False
        pts_np = obs0["points"]
        cols_np = obs0["colors"]
        plane_n = obs0["plane_n"]
        plane_p = obs0["plane_p"]

        reg = self._node._maybe_register_multi_view_cloud(sel_capture, obs0)
        pts_np = np.asarray(reg["points"], dtype=np.float32)
        cols_np = reg["colors"]
        registration_frame_is_target = bool(reg["registration_frame_is_target"])
        cloud_frame_id = str(reg.get("frame_id", "") or "")

        if registration_frame_is_target:
            plane_n = None
            plane_p = None
            self._node.get_logger().info(
                f"[registration] using fused cloud in fixed frame '{reg['frame_id']}' "
                f"with {pts_np.shape[0]} points — skipping post-fusion filters "
                f"(trusting per-view mask + filtering)."
            )
            plane_n = None
            plane_p = None

        if self._aborted("capture/registration"):
            return False

        # ── Phase 3: sample / downsample for GraspGen ─────────────────
        min_pts = int(self._node.get_parameter("min_points_for_graspgen").value)
        if pts_np is None or pts_np.shape[0] < min_pts:
            self._node.get_logger().warn(f"Not enough points after pre-GraspGen filtering (<{min_pts}).")
            return False

        cloud_msg = PointCloud2()
        cloud_msg.header = Header()
        cloud_msg.header.stamp = sel["header"].stamp
        cloud_msg.header.frame_id = str(reg["frame_id"])

        max_graspgen_points = int(self._node.get_parameter("max_graspgen_points").value)
        if max_graspgen_points > 0 and pts_np.shape[0] > max_graspgen_points:
            idx = np.random.choice(pts_np.shape[0], max_graspgen_points, replace=False)
            pts_send = pts_np[idx]
            cols_send = cols_np[idx] if cols_np is not None else None
            self._node.get_logger().info(
                f"Downsampled cloud for GraspGen: {pts_np.shape[0]} -> {max_graspgen_points} points."
            )
        else:
            pts_send = pts_np
            cols_send = cols_np

        if self._aborted("pre-GraspGen filtering"):
            return False

        # ── Phase 4: publish input cloud + call GraspGen ──────────────
        self._node.get_logger().info(
            f"[timing] registration+downsample {_time.monotonic() - _t_phase:.2f}s")
        _t_phase = _time.monotonic()
        if not background and bool(self._node.get_parameter("publish_graspgen_input_cloud").value):
            try:
                out_header = Header()
                out_header.stamp = self._node.get_clock().now().to_msg()
                out_header.frame_id = str(reg["frame_id"])
                cloud_out = self._node._create_cloud_xyzrgb(out_header, pts_send, cols_send)
                self._node.pub_graspgen_cloud.publish(cloud_out)
            except Exception as e:
                self._node.get_logger().warn(f"Failed to publish GraspGen input cloud: {e}")
        # 2026-08-27: if the startup preload is still building the model,
        # this call blocks inside grasp_client until that build finishes and
        # then REUSES the same instance (module singleton). Say so up front —
        # the grasp_gen "Loading ... checkpoint" lines that can appear right
        # after the user's first selection are the preload's own load, not a
        # second one.
        import os as _os
        if _os.environ.get("GRASPGEN_MODE", "inprocess").strip().lower() != "zmq":
            from ..grasp_client import sampler_is_warm
            if not sampler_is_warm():
                self._node.get_logger().info(
                    "[graspgen] model not warm yet — waiting for the in-flight "
                    "preload and reusing its instance (single load, no reload)."
                )
        try:
            resp = send_points_to_graspgen(
                pts_send,
                endpoint=self._node.get_parameter("graspgen_endpoint").value,
                threshold=float(self._node.get_parameter("grasp_threshold").value),
                colors=cols_send,
                num_grasps=int(self._node.get_parameter("graspgen_num_grasps").value),
                topk_num_grasps=-1,  # server topk costs 5-7 s; cut is client-side
                min_grasps=40,
                max_tries=2,
                remove_outliers=False,
                fx=self._node.fb.fx,
                fy=self._node.fb.fy,
                cx=self._node.fb.cx,
                cy=self._node.fb.cy,
                depth_scale=self._node.fb.depth_scale,
                # A background job must never hold the model lock for long:
                # a selection made meanwhile waits behind it (2026-09-11).
                timeout_ms=(int(float(self._node.get_parameter("precompute_request_timeout_s").value) * 1000)
                            if background else 60000),
            )
        except Exception as e:
            self._node.get_logger().error(
                f"Failed contacting GraspGen at {self._node.get_parameter('graspgen_endpoint').value}: {e}"
            )
            return False
        if not resp or not resp.get("ok", False):
            self._node.get_logger().error(f"GraspGen error: {None if resp is None else resp.get('error')}")
            return False
        grasps_obj = resp.get("grasps", [])
        if grasps_obj is None:
            grasps = []
        elif isinstance(grasps_obj, list):
            grasps = grasps_obj
        else:
            try:
                grasps = list(grasps_obj)
            except Exception:
                grasps = []
        raw_grasps = list(grasps)
        if not grasps:
            self._node.get_logger().warn("GraspGen returned no grasps.")
            return False
        # Client-side top-K by score BEFORE the filter chain. Doing this in
        # the GraspGen server (topk_num_grasps) costs 5-7 s; a sort here is
        # free, and the chain's cost scales with grasp count (200→3 s,
        # 399→9.6 s measured 2026-08-12).
        topk = int(self._node.get_parameter("graspgen_topk_num_grasps").value)
        if topk > 0 and len(grasps) > topk:
            grasps = sorted(
                grasps, key=lambda g: float(g.get("score", 0.0)), reverse=True
            )[:topk]
            self._node.get_logger().info(
                f"[topk] kept best {topk}/{len(raw_grasps)} grasps by score "
                "(client-side, ~0 ms)."
            )
        if self._aborted("GraspGen inference"):
            return False

        # ── Phase 5: geometric filter chain ───────────────────────────
        filter_input_count = len(grasps)
        self._node.get_logger().info(
            f"[filter-chain] START: {filter_input_count} score-ranked candidates "
            f"of {len(raw_grasps)} generated"
        )
        # True-RAW orientation census, before ANY filter touches the set.
        from ..grasp_filters import apply_gravity_view_filter, orientation_census
        try:
            import rclpy.time as _rt
            _tr = self._node.tf_buffer.lookup_transform(
                cloud_frame_id, "base_link", _rt.Time())
            _q = _tr.transform.rotation
            _upq = np.array([
                2*(_q.x*_q.z+_q.y*_q.w), 2*(_q.y*_q.z-_q.x*_q.w),
                1-2*(_q.x*_q.x+_q.y*_q.y)])
            _upq = _upq/np.linalg.norm(_upq)
            orientation_census(
                grasps, _upq,
                lambda m: self._node.get_logger().info(f"[filter-chain] {m}"),
                "RAW from GraspGen (post topk)")
        except Exception:
            pass
        # NOTE 2026-08-24 (thesis teardown): the old _filter_no_under_grasp
        # call was retired here — its from-below test used the +z WRIST
        # direction (inverted: it killed descending grasps and PASSED
        # ascending ones, hence "the grasp goes up through the object")
        # and compared unit-cube translations against the metric plane.
        # Thesis stage 2 (support-plane) now runs correctly inside
        # apply_gravity_view_filter below.
        # Finger penetration filter: rejects grasps whose fingers would
        # crush the object. Runs early to prune the set before the more
        # expensive orientation reweights downstream.
        try:
            if bool(self._node.get_parameter("apply_finger_penetration_filter").value):
                aabb_min = np.min(pts_np, axis=0)
                aabb_max = np.max(pts_np, axis=0)
                grasps_pen = _filter_finger_penetration(
                    grasps,
                    pts_np,
                    matrices_normalized=bool(self._node.get_parameter("graspgen_matrix_normalized").value),
                    column_major=False,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    finger_length=float(self._node.get_parameter("finger_penetration_finger_length_m").value),
                    finger_thickness=float(self._node.get_parameter("finger_penetration_finger_thickness_m").value),
                    finger_height=float(self._node.get_parameter("finger_penetration_finger_height_m").value),
                    gripper_opening=float(self._node.get_parameter("finger_penetration_gripper_opening_m").value),
                    finger_z_offset=float(self._node.get_parameter("finger_penetration_finger_z_offset_m").value),
                    max_points_per_finger=int(self._node.get_parameter("finger_penetration_max_points_per_finger").value),
                    min_points_in_closing_volume=int(self._node.get_parameter("finger_penetration_min_points_in_closing").value),
                )
                if grasps_pen:
                    grasps = grasps_pen
                else:
                    self._node.get_logger().warn(
                        "All grasps failed the finger-penetration filter; keeping previous candidates."
                    )
                self._node.get_logger().info(f"[filter-chain] after finger-penetration: {len(grasps)}")
        except Exception as e:
            self._node.get_logger().warn(f"Finger-penetration filter error: {e}")
        try:
            if bool(self._node.get_parameter("apply_reject_from_behind_filter").value):
                # Observation point = the REAL gripper TCP (Majd,
                # 2026-08-24: "the center should be the Gripper TCP, not
                # the camera"). "Behind" now means "facing away from
                # where the HAND approaches". Fallback: the old synthetic
                # point above the centroid when TF is unavailable.
                obs_pt = None
                try:
                    import rclpy.time as _rt
                    _trh = self._node.tf_buffer.lookup_transform(
                        cloud_frame_id, "gripper_tcp", _rt.Time())
                    _t = _trh.transform.translation
                    obs_pt = np.array([_t.x, _t.y, _t.z], dtype=np.float32)
                except Exception:
                    pass
                if obs_pt is None:
                    cz = float(np.mean(pts_np[:, 2])) \
                        + float(self._node.get_parameter("reject_from_behind_observation_height_m").value)
                    centroid_xy = np.mean(pts_np[:, :2], axis=0)
                    obs_pt = np.array([centroid_xy[0], centroid_xy[1], cz], dtype=np.float32)
                aabb_min_b = np.min(pts_np, axis=0)
                aabb_max_b = np.max(pts_np, axis=0)
                grasps_back = _filter_reject_from_behind(
                    grasps,
                    pts_np,
                    observation_point=obs_pt,
                    matrices_normalized=bool(self._node.get_parameter("graspgen_matrix_normalized").value),
                    column_major=False,
                    aabb_min=aabb_min_b,
                    aabb_max=aabb_max_b,
                    min_forward_dot=float(self._node.get_parameter("reject_from_behind_min_forward_dot").value),
                )
                if grasps_back:
                    grasps = grasps_back
                else:
                    self._node.get_logger().warn(
                        "All grasps failed the reject-from-behind filter; keeping previous candidates."
                    )
                self._node.get_logger().info(f"[filter-chain] after reject-from-behind: {len(grasps)}")
        except Exception as e:
            self._node.get_logger().warn(f"Reject-from-behind filter error: {e}")
        try:
            if bool(self._node.get_parameter("reject_flipped_grasps").value) and plane_n is not None:
                up_idx = int(self._node.get_parameter("up_axis_index").value)
                min_up = float(self._node.get_parameter("min_up_dot").value)
                grasps_upright = _filter_roll_upright_by_plane(
                    grasps,
                    plane_n=plane_n,
                    plane_p=plane_p,
                    up_axis=up_idx,
                    min_up_dot=min_up,
                )
                if grasps_upright:
                    grasps = grasps_upright
                else:
                    self._node.get_logger().warn("Upright-only filter removed all; keeping previous candidates.")
            self._node.get_logger().info(f"[filter-chain] after upright: {len(grasps)}")
        except Exception as e:
            self._node.get_logger().warn(f"Upright-only filter error: {e}")
        try:
            if bool(self._node.get_parameter("reject_side_grasps").value) and plane_n is not None:
                min_vert = float(self._node.get_parameter("min_vertical_dot").value)
                grasps_vertical = _filter_reject_side_grasps(
                    grasps,
                    plane_n=plane_n,
                    min_vertical_dot=min_vert,
                    approach_axis=2,
                    matrices_normalized=bool(self._node.get_parameter("graspgen_matrix_normalized").value),
                    column_major=bool(self._node.get_parameter("graspgen_column_major").value),
                )
                if grasps_vertical:
                    grasps = grasps_vertical
                else:
                    self._node.get_logger().warn("Vertical-only filter removed all; keeping previous candidates.")
            self._node.get_logger().info(f"[filter-chain] after reject-side: {len(grasps)}")
        except Exception as e:
            self._node.get_logger().warn(f"Vertical-only filter error: {e}")

        # THE thesis §4.6.14 cascade — SINGLE shared implementation
        # (grasp_filters.apply_gravity_view_filter); never duplicate it
        # again (2026-08-21).
        grasps = apply_gravity_view_filter(
            self._node, grasps, pts_np,
            plane_n=plane_n, plane_p=plane_p,
            frame_id=cloud_frame_id,
            registration_frame_is_target=registration_frame_is_target,
            log_info=lambda m: self._node.get_logger().info(f"[filter-chain] {m}"),
            log_warn=lambda m: self._node.get_logger().warn(f"[filter-chain] {m}"),
        )

        try:
            if bool(self._node.get_parameter("forbid_top_grasps").value) and plane_n is not None and plane_p is not None:
                side_center = _filter_side_center_window(
                    grasps,
                    pts_np,
                    plane_n=plane_n,
                    plane_p=plane_p,
                    approach_axis=2,
                    max_vertical_dot=float(self._node.get_parameter("max_vertical_dot").value),
                    centroid_down_only=True,
                    max_down=float(self._node.get_parameter("max_down_from_centroid").value),
                    max_up=float(self._node.get_parameter("max_up_from_centroid").value),
                    matrices_normalized=bool(self._node.get_parameter("graspgen_matrix_normalized").value),
                    column_major=bool(self._node.get_parameter("graspgen_column_major").value),
                )
                if side_center:
                    grasps = side_center
                else:
                    self._node.get_logger().warn("Side/center filter removed all; keeping previous candidates.")
            self._node.get_logger().info(f"[filter-chain] after side-center: {len(grasps)}")
        except Exception as e:
            self._node.get_logger().warn(f"Side/center filter error: {e}")

        if not grasps:
            self._node.get_logger().warn(
                "No grasp candidates available (crop/filters produced no valid cloud or GraspGen did not return candidates)."
            )
            return False

        try:
        # ── Phase 6: centroid + EE-orientation reweight ───────────────
            cloud_for_stats = pts_send if 'pts_send' in locals() else pts_np
            if cloud_for_stats is None or getattr(cloud_for_stats, "size", 0) == 0:
                raise ValueError("No valid cloud points available for centroid reweight.")
            aabb_min, aabb_max = _aabb_min_max(cloud_for_stats)
            centroid = cloud_for_stats.mean(axis=0).astype(np.float32)
            w_center = float(self._node.get_parameter("center_distance_weight").value)
            if w_center > 0.0:
                reweighted = _reweight_scores_by_centroid(
                    grasps,
                    centroid_metric=centroid,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    matrices_normalized=bool(self._node.get_parameter("graspgen_matrix_normalized").value),
                    column_major=bool(self._node.get_parameter("graspgen_column_major").value),
                    weight=w_center,
                )
                if reweighted:
                    grasps = reweighted
                else:
                    self._node.get_logger().warn("Centroid reweight returned no candidates; keeping previous grasp list.")
            self._node.get_logger().info(f"[filter-chain] after centroid-reweight: {len(grasps)}")
            frame_for_pick = cloud_msg.header.frame_id
            grasps = self._node._reweight_scores_by_ee_orientation(
                grasps,
                frame_id=frame_for_pick,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
            )
            self._node.get_logger().info(f"[filter-chain] after ee-orientation-reweight: {len(grasps)}")
            try:
                ps = PoseStamped()
                ps.header = cloud_msg.header
                ps.pose.position.x = float(centroid[0])
                ps.pose.position.y = float(centroid[1])
                ps.pose.position.z = float(centroid[2])
                ps.pose.orientation.w = 1.0
                centroid_msg = ps
                if not background:
                    self._node.pub_centroid.publish(ps)
                    self._node.last_centroid = ps
                target_frame = self._node.get_parameter("target_frame").value
                try:
                    if ps.header.frame_id == target_frame:
                        transformed_centroid = PoseStamped()
                        transformed_centroid.header.frame_id = target_frame
                        transformed_centroid.header.stamp = self._node.get_clock().now().to_msg()
                        transformed_centroid.pose = ps.pose
                    else:
                        # Latest transform (stamp 0): the arm stands still while
                        # a pick is prepared, and a stamped lookup waited its
                        # full 0.5 s timeout on every pick (2026-09-09).
                        ps.header.stamp = rclpy.time.Time().to_msg()
                        transformed_centroid = self._node.tf_buffer.transform(
                            ps, target_frame, timeout=rclpy.duration.Duration(seconds=0.2)
                        )
                        transformed_centroid.header.frame_id = target_frame
                    try:
                        tcp_tf = self._node.tf_buffer.lookup_transform(
                            target_frame, "gripper_tcp",
                            rclpy.time.Time(),
                            timeout=rclpy.duration.Duration(seconds=0.3),
                        )
                        transformed_centroid.pose.orientation = tcp_tf.transform.rotation
                        self._node.get_logger().info(
                            f"Centroid orientation set from gripper_tcp: "
                            f"({tcp_tf.transform.rotation.x:.3f}, "
                            f"{tcp_tf.transform.rotation.y:.3f}, "
                            f"{tcp_tf.transform.rotation.z:.3f}, "
                            f"{tcp_tf.transform.rotation.w:.3f})"
                        )
                    except Exception as etf:
                        self._node.get_logger().warn(
                            f"Could not look up gripper_tcp orientation: {etf}; using identity"
                        )
                        transformed_centroid.pose.orientation.x = 0.0
                        transformed_centroid.pose.orientation.y = 0.0
                        transformed_centroid.pose.orientation.z = 0.0
                        transformed_centroid.pose.orientation.w = 1.0
                    centroid_base_msg = transformed_centroid
                    if not background:
                        self._node.pub_centroid_base.publish(transformed_centroid)
                        self._node.last_centroid_base = transformed_centroid
                except Exception as e:
                    self._node.get_logger().warn(f"Centroid transform to {target_frame} failed: {e}")
            except Exception as e:
                self._node.get_logger().warn(f"Centroid publish failed: {e}")
        except Exception as e:
            self._node.get_logger().warn(f"Centroid reweight failed: {e}")

        # ── Phase 7: final orientation gate + utility logging ─────────
        if (
            not grasps
            and raw_grasps
            and not bool(self._node.get_parameter("require_horizontal_approach").value)
        ):
            self._node.get_logger().warn("All grasps were removed by filters/reweight; falling back to raw GraspGen grasps.")
            grasps = raw_grasps
        if not grasps:
            self._node.get_logger().warn("No grasp candidates available after filtering/reweight; skipping.")
            return False

        cloud_for_pick = pts_send if 'pts_send' in locals() else pts_np
        if cloud_for_pick is None or getattr(cloud_for_pick, "size", 0) == 0:
            self._node.get_logger().warn("No valid cloud points available for orientation consistency filter.")
            return False
        aabb_min_pick, aabb_max_pick = _aabb_min_max(cloud_for_pick)
        frame_for_pick = cloud_msg.header.frame_id
        grasps = self._node._filter_by_ee_orientation_consistency(
            grasps,
            frame_id=frame_for_pick,
            aabb_min=aabb_min_pick,
            aabb_max=aabb_max_pick,
        )
        self._node.get_logger().info(f"[filter-chain] after orientation-consistency: {len(grasps)}")
        if not grasps:
            self._node.get_logger().warn("No grasp candidates available after orientation consistency filter; skipping.")
            return False
        self._node.get_logger().info(
            f"[filter-chain] SURVIVED: {len(grasps)}/{len(raw_grasps)} generated "
            f"({filter_input_count} entered geometric filters)"
        )
        if bool(self._node.get_parameter("registration_grasp_utility_logging_enabled").value):
            utility_now = self._node._compute_grasp_utility(
                grasps,
                aabb_min_pick,
                aabb_max_pick,
                instability_penalty=0.0,
            )
            self._node.get_logger().info(
                f"[grasp-utility] final_cloud: U={utility_now['total']:.3f} "
                f"(count={int(utility_now['surviving_count'])}, diversity_deg={utility_now['diversity_deg']:.2f}, "
                f"mean_score={utility_now['mean_score']:.3f}, instability={utility_now['instability_penalty']:.3f})"
            )

        # ── Grasp visualisation: kept vs rejected, camera frame (2026-08-06)
        # Feeds grasp_viz_viewer: green = survived the whole filter chain,
        # red = produced by GraspGen but filtered out. Raw grasp frames (no
        # TCP offset) in `frame_for_pick`, same frame as the selected cloud.
        return {
            "sel_capture": sel_capture, "obs0": obs0,
            "grasps": grasps, "raw_grasps": raw_grasps,
            "pts_np": pts_np, "cols_np": cols_np, "pts_send": pts_send,
            "cloud_msg": cloud_msg, "frame_for_pick": frame_for_pick,
            "registration_frame_is_target": registration_frame_is_target,
            "aabb_min_pick": aabb_min_pick, "aabb_max_pick": aabb_max_pick,
            "filter_input_count": filter_input_count,
            # replay-only extras (precompute cache)
            "cols_send": cols_send, "input_cloud_frame": str(reg["frame_id"]),
            "centroid_msg": centroid_msg, "centroid_base_msg": centroid_base_msg,
        }

    def _publish(self, res: Dict[str, Any], sel: Dict[str, Any]) -> bool:
        """Phases 8-10 on a compute result: candidates to the executor first,
        then the viewer poses, then the single best pose."""
        grasps = res["grasps"]; raw_grasps = res["raw_grasps"]
        pts_np = res["pts_np"]; pts_send = res["pts_send"]; cloud_msg = res["cloud_msg"]
        frame_for_pick = res["frame_for_pick"]
        registration_frame_is_target = res["registration_frame_is_target"]
        aabb_min_pick = res["aabb_min_pick"]; aabb_max_pick = res["aabb_max_pick"]
        # ── Phase 8: publish top-K candidates as PoseArray ────────────
        if bool(self._node.get_parameter("publish_grasp_candidates").value):
            try:
                target_frame = str(self._node.get_parameter("target_frame").value)
                top_k = max(int(self._node.get_parameter("grasp_candidates_top_k").value), 1)
                use_norm = bool(self._node.get_parameter("graspgen_matrix_normalized").value)
                col_major = bool(self._node.get_parameter("graspgen_column_major").value)
                fixed_rot = str(self._node.get_parameter("grasp2tcp_rot").value)
                ordered = sorted(grasps, key=lambda g: float(g.get("score", 0.0)), reverse=True)[:top_k]
                pose_array = PoseArray()
                pose_array.header.frame_id = target_frame
                pose_array.header.stamp = self._node.get_clock().now().to_msg()
                scores_msg = Float32MultiArray()

                tf_cam_to_target = None
                if frame_for_pick != target_frame:
                    tf_cam_to_target = self._node.tf_buffer.lookup_transform(
                        target_frame,
                        frame_for_pick,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0),
                    )
                for g in ordered:
                    M_g = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
                    if use_norm:
                        T_g = _denorm_T_from_unit_cube(M_g, aabb_min_pick, aabb_max_pick, column_major=col_major)
                    else:
                        T_g = M_g.T if col_major else M_g
                    try:
                        T_tcp_g = T_g @ _make_T_grasp_to_tcp(fixed_rot)
                    except Exception:
                        T_tcp_g = T_g
                    qx_g, qy_g, qz_g, qw_g = quaternion_from_matrix(T_tcp_g)
                    ps_src = PoseStamped()
                    ps_src.header.frame_id = frame_for_pick
                    ps_src.header.stamp = pose_array.header.stamp
                    ps_src.pose.position.x = float(T_tcp_g[0, 3])
                    ps_src.pose.position.y = float(T_tcp_g[1, 3])
                    ps_src.pose.position.z = float(T_tcp_g[2, 3])
                    ps_src.pose.orientation.x = float(qx_g)
                    ps_src.pose.orientation.y = float(qy_g)
                    ps_src.pose.orientation.z = float(qz_g)
                    ps_src.pose.orientation.w = float(qw_g)
                    if tf_cam_to_target is not None:
                        pose_target = tf2_geometry_msgs.do_transform_pose(ps_src.pose, tf_cam_to_target)
                    else:
                        pose_target = ps_src.pose
                    pose_array.poses.append(pose_target)
                    scores_msg.data.append(float(g.get("score", 0.0)))
                # Table-clearance floor (Majd's "it takes forever", 2026-08-11):
                # every candidate below the table top + clearance is doomed —
                # the executor burns ~8 s of LIN planning per rejection. Drop
                # them here so only physically plannable grasps are sent.
                min_wz = float(
                    self._node.get_parameter("candidate_min_world_z").value
                )
                if min_wz > 0.0 and pose_array.poses:
                    try:
                        tf_t_w = None
                        if target_frame != "world":
                            tf_t_w = self._node.tf_buffer.lookup_transform(
                                "world", target_frame, rclpy.time.Time(),
                                timeout=rclpy.duration.Duration(seconds=0.5),
                            )
                        keep_p, keep_s = [], []
                        for pose_t, sc in zip(pose_array.poses, scores_msg.data):
                            if tf_t_w is not None:
                                z_w = tf2_geometry_msgs.do_transform_pose(
                                    pose_t, tf_t_w
                                ).position.z
                            else:
                                z_w = pose_t.position.z
                            if z_w >= min_wz:
                                keep_p.append(pose_t)
                                keep_s.append(sc)
                        if keep_p:
                            n_drop = len(pose_array.poses) - len(keep_p)
                            if n_drop:
                                self._node.get_logger().info(
                                    f"[cand-floor] dropped {n_drop}/{len(pose_array.poses)} "
                                    f"candidates below world z={min_wz:.3f} (table clearance)"
                                )
                            pose_array.poses = keep_p
                            scores_msg.data = keep_s
                        else:
                            self._node.get_logger().warn(
                                f"[cand-floor] ALL candidates below world z={min_wz:.3f} — "
                                "refusing execution to protect the table."
                            )
                            return False
                    except Exception as e:
                        self._node.get_logger().warn(f"[cand-floor] skipped ({e})")
                if self._aborted("candidate publishing"):
                    return False
                self._node.pub_candidates_base.publish(pose_array)
                self._node.pub_candidate_scores.publish(scores_msg)
                self._node.last_candidates_base = copy.deepcopy(pose_array)
                self._node.last_candidate_scores = copy.deepcopy(scores_msg)
                # Tag the keep-alive cache with the cycle these belong to —
                # the republish timer only re-sends them while this is still
                # the ACTIVE cycle (ghost-pick hardening, 2026-08-12).
                self._node._candidates_cycle_id = int(
                    getattr(self._node, "_waiting_cycle_id", -1)
                )
                self._node.get_logger().info(
                    f"Published {len(pose_array.poses)} grasp candidates on "
                    f"'{self._node.get_parameter('grasp_candidates_topic').value}'."
                )
            except Exception as e:
                # Nothing reached the executor, so nothing will ever answer
                # with cycle_done: end the cycle here (2026-09-09, a pick
                # attempted before the driver was up sat in that wait until
                # a manual abort).
                self._node.get_logger().error(
                    f"Failed to publish grasp candidates: {e} — ending this cycle.")
                return False

        # ── Phase 9: pick the single best grasp ───────────────────────
        try:
            use_norm_v = bool(self._node.get_parameter("graspgen_matrix_normalized").value)
            col_major_v = bool(self._node.get_parameter("graspgen_column_major").value)

            def _grasp_to_pose_v(g):
                M_v = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
                if use_norm_v:
                    T_v = _denorm_T_from_unit_cube(
                        M_v, aabb_min_pick, aabb_max_pick, column_major=col_major_v
                    )
                else:
                    T_v = M_v.T if col_major_v else M_v
                qx_v, qy_v, qz_v, qw_v = quaternion_from_matrix(T_v)
                p_v = Pose()
                p_v.position.x = float(T_v[0, 3])
                p_v.position.y = float(T_v[1, 3])
                p_v.position.z = float(T_v[2, 3])
                p_v.orientation.x = float(qx_v)
                p_v.orientation.y = float(qy_v)
                p_v.orientation.z = float(qz_v)
                p_v.orientation.w = float(qw_v)
                return p_v

            _viz_cap = 300
            stamp_v = self._node.get_clock().now().to_msg()

            def _mk_pose_array_v(items):
                pa_v = PoseArray()
                pa_v.header.frame_id = frame_for_pick
                pa_v.header.stamp = stamp_v
                step_v = max(1, len(items) // _viz_cap)
                for g_v in items[::step_v][:_viz_cap]:
                    pa_v.poses.append(_grasp_to_pose_v(g_v))
                return pa_v

            kept_keys_v = {
                tuple(np.asarray(g["matrix"], dtype=np.float32).round(6).ravel())
                for g in grasps
            }
            rejected_v = [
                g for g in raw_grasps
                if tuple(np.asarray(g["matrix"], dtype=np.float32).round(6).ravel())
                not in kept_keys_v
            ]
            # Final cloud for the result window (2026-08-07): the cloud these
            # grasps were computed on. frame_for_pick tells us which one that
            # was — base_link means the fused/TSDF path ran, camera frame
            # means the single-view crop.
            try:
                final_msg = (
                    self._node.last_tsdf_cloud_msg
                    if str(frame_for_pick) == "base_link"
                    and self._node.last_tsdf_cloud_msg is not None
                    else self._node.last_selected_cloud_rgb_msg
                )
                if final_msg is not None:
                    self._node.pub_final_cloud.publish(final_msg)
            except Exception as e:
                self._node.get_logger().warn(f"final_cloud publish failed: {e}")

            # Remember the viz frame so the executor's selected pose can be
            # bridged into the same scene (2026-08-07, ORANGE = robot's pick).
            self._node.last_viz_frame = str(frame_for_pick)

            # Publish the camera's pose in the viz frame so the viewer can
            # orient its initial view exactly as the camera saw the object.
            try:
                cam_ps = PoseStamped()
                cam_ps.header.frame_id = str(frame_for_pick)
                cam_ps.header.stamp = stamp_v
                if str(frame_for_pick) == "camera_color_optical_frame":
                    cam_ps.pose.orientation.w = 1.0  # camera at origin
                else:
                    tf_cam = self._node.tf_buffer.lookup_transform(
                        str(frame_for_pick), "camera_color_optical_frame",
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5),
                    )
                    cam_ps.pose.position.x = tf_cam.transform.translation.x
                    cam_ps.pose.position.y = tf_cam.transform.translation.y
                    cam_ps.pose.position.z = tf_cam.transform.translation.z
                    cam_ps.pose.orientation = tf_cam.transform.rotation
                self._node.pub_viz_campose.publish(cam_ps)
            except Exception as e:
                self._node.get_logger().warn(f"camera-pose viz publish failed: {e}")

            self._node.pub_viz_kept.publish(_mk_pose_array_v(list(grasps)))
            self._node.pub_viz_rejected.publish(_mk_pose_array_v(rejected_v))
            # The best is simply the top-scoring survivor, in the same raw
            # frame — so the blue glyph orients exactly like its green peers.
            best_v = max(grasps, key=lambda g: float(g.get("score", 0.0)))
            self._node.pub_viz_best.publish(_mk_pose_array_v([best_v]))
            # Ground truth for "is the grasp from behind?" (Majd, 2026-08-06):
            # in the camera optical frame the view axis is +z, so a physically
            # front-side approach must have approach·z > 0. Logged every cycle
            # so a from-behind best can never hide behind rendering conventions.
            M_b = np.asarray(best_v["matrix"], dtype=np.float32).reshape(4, 4)
            if use_norm_v:
                T_b = _denorm_T_from_unit_cube(
                    M_b, aabb_min_pick, aabb_max_pick, column_major=col_major_v
                )
            else:
                T_b = M_b.T if col_major_v else M_b
            # CONVENTION (established 2026-08-06 from measurement): GraspGen's
            # raw grasp frame has +z pointing fingers->wrist, so the physical
            # approach direction is -z. That is exactly why the grasp->tcp
            # conversion applies flipY (making TCP z = +approach for the UR
            # tool). A raw z·view of -0.99 therefore means a dead-on FRONT
            # approach, not a from-behind grasp.
            az_v = float(-T_b[2, 2])  # physical approach (-z raw) · camera view z
            self._node.get_logger().info(
                f"[viz] best grasp approach·view = {az_v:+.2f} -> "
                + ("FRONT approach (correct)" if az_v > 0.0 else "FROM BEHIND — INVESTIGATE!")
            )
            self._node.get_logger().info(
                f"[viz] published grasp poses for viewer: {min(len(grasps), _viz_cap)} kept, "
                f"{min(len(rejected_v), _viz_cap)} rejected (frame={frame_for_pick})"
            )
        except Exception as e:
            self._node.get_logger().warn(f"Grasp viz publish failed: {e}")
        best = pick_best_grasp(
            grasps,
            pts_np,
            table_margin=0.01,
            min_open=0.05,
            max_open=0.09,
            prefer_side=bool(self._node.get_parameter("prefer_side_grasps").value),
            ignore_behind=(False if registration_frame_is_target else bool(self._node.get_parameter("ignore_behind_grasps").value)),
            behind_threshold=float(self._node.get_parameter("behind_grasp_threshold").value),
            ignore_above=(False if registration_frame_is_target else bool(self._node.get_parameter("ignore_above_grasps").value)),
            above_threshold=float(self._node.get_parameter("above_grasp_threshold").value),
        )
        if best is None:
            self._node.get_logger().warn("No grasp chosen after filtering.")
            return False

        # ── Phase 10: publish best pose in source + target frames ─────
        M = np.asarray(best["matrix"], dtype=np.float32).reshape(4, 4)
        frame_id = cloud_msg.header.frame_id
        stamp = sel["header"].stamp
        aabb_min, aabb_max = _aabb_min_max(pts_send if pts_send is not None else pts_np)
        use_norm = bool(self._node.get_parameter("graspgen_matrix_normalized").value)
        col_major = bool(self._node.get_parameter("graspgen_column_major").value)
        if use_norm:
            T = _denorm_T_from_unit_cube(M, aabb_min, aabb_max, column_major=col_major)
        else:
            T = M.T if col_major else M
        t = T[:3, 3].copy()
        if not _inside_aabb(t, aabb_min, aabb_max, margin=5e-3):
            self._node.get_logger().warn(
                f"[guard] t={t.round(3)} outside AABB "
                f"x{aabb_min[0]:.3f}..{aabb_max[0]:.3f} "
                f"y{aabb_min[1]:.3f}..{aabb_max[1]:.3f} "
                f"z{aabb_min[2]:.3f}..{aabb_max[2]:.3f}; snapping to centroid."
            )
            c = (pts_send if pts_send is not None else pts_np).mean(axis=0)
            T[:3, 3] = c

        fixed_rot = self._node.get_parameter("grasp2tcp_rot").value
        try:
            T_tcp = T @ _make_T_grasp_to_tcp(fixed_rot)
        except Exception as e:
            self._node.get_logger().warn(f"Failed to apply grasp2tcp_rot='{fixed_rot}': {e}; using identity.")
            T_tcp = T
        self._node.get_logger().info(f"[fixed] grasp→tcp rotation: {fixed_rot}")
        _publish_marker(self._node, T, frame_id, topic="/graspgen/raw_pose")
        _publish_marker(self._node, T_tcp, frame_id, topic="/graspgen/tcp_pose")
        qx, qy, qz, qw = quaternion_from_matrix(T_tcp)
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp if (stamp.sec or stamp.nanosec) else self._node.get_clock().now().to_msg()
        pose.pose.position.x = float(T_tcp[0, 3])
        pose.pose.position.y = float(T_tcp[1, 3])
        pose.pose.position.z = float(T_tcp[2, 3])
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        self._node.last_published_pose = pose
        self._node.pub_best.publish(pose)

        target_frame = self._node.get_parameter("target_frame").value
        try:
            current_time_msg = self._node.get_clock().now().to_msg()
            if frame_id != target_frame:
                transform = self._node.tf_buffer.lookup_transform(
                    target_frame,
                    frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                pose_for_transform = PoseStamped()
                pose_for_transform.header.frame_id = frame_id
                pose_for_transform.header.stamp = current_time_msg
                pose_for_transform.pose = pose.pose
                transformed_pose_msg = tf2_geometry_msgs.do_transform_pose(pose_for_transform.pose, transform)
            else:
                transformed_pose_msg = pose.pose

            transformed_pose = PoseStamped()
            transformed_pose.header.frame_id = target_frame
            transformed_pose.header.stamp = current_time_msg
            transformed_pose.pose = transformed_pose_msg
            self._node.last_published_pose_base = transformed_pose
            self._node.pub_best_base.publish(transformed_pose)

            self._node.get_logger().info(f"Published TCP grasp pose transformed to {target_frame}")
            self._node.get_logger().info(
                f"Source frame pose: t=({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},{pose.pose.position.z:.3f})"
            )
            self._node.get_logger().info(
                f"{target_frame} pose: t=({transformed_pose.pose.position.x:.3f},{transformed_pose.pose.position.y:.3f},{transformed_pose.pose.position.z:.3f})"
            )
        except (TypeError, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self._node.get_logger().error(f"Failed to transform pose from {frame_id} to {target_frame}: {e}")
            self._node.get_logger().warn("Only source-frame pose will be available")

        self._node.get_logger().info("Published TCP-aligned grasp pose (will republish continuously)")
        self._node.get_logger().info(
            f"Best grasp score={best.get('score', 0.0):.3f} @ t=({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},{pose.pose.position.z:.3f}) "
            f"frame={pose.header.frame_id}"
        )
        centroid_metric = pts_np.mean(axis=0)
        self._node.get_logger().info(
            f"[debug] Cloud centroid=({centroid_metric[0]:.3f},{centroid_metric[1]:.3f},{centroid_metric[2]:.3f}), "
            f"Best grasp=({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},{pose.pose.position.z:.3f})"
        )
        return True
