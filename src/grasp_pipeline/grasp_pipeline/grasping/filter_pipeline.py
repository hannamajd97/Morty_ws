"""Geometric filter chain that runs on every GraspGen candidate set.

Reproduces the legacy ``_filter_grasps_for_utility`` flow exactly: each
filter is applied if its enabling parameter is true, and an empty result
set is treated as "skip this filter rather than nuke the pipeline" — so
we keep candidates flowing even when one filter is too aggressive for a
given object.

Plays alongside :class:`OrientationFilter` (passed in) for the
TF-dependent re-weight + consistency passes that close out the chain.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..geometry_utils import _aabb_min_max
from ..grasp_filters import (
    _filter_no_under_grasp,
    _filter_finger_penetration,
    _filter_reject_from_behind,
    _filter_roll_upright_by_plane,
    _filter_reject_side_grasps,
    _filter_view_consistent,
    _filter_side_center_window,
    _reweight_scores_by_centroid,
)

if TYPE_CHECKING:
    from rclpy.node import Node
    from .orientation import OrientationFilter


class GraspFilterPipeline:
    """Apply the legacy filter / re-weight chain in the same order."""

    def __init__(self, node: "Node", orientation: "OrientationFilter") -> None:
        self._node = node
        self._orientation = orientation

    def filter_for_utility(
        self,
        grasps: List[Dict[str, Any]],
        pts_filter: np.ndarray,
        pts_metric: np.ndarray,
        plane_n: Optional[np.ndarray],
        plane_p: Optional[np.ndarray],
        registration_frame_is_target: bool,
        frame_id: str,
        *,
        log_prefix: Optional[str],
    ) -> List[Dict[str, Any]]:
        node = self._node

        def log_info(msg: str) -> None:
            if log_prefix:
                node.get_logger().info(f"{log_prefix} {msg}")

        def log_warn(msg: str) -> None:
            if log_prefix:
                node.get_logger().warn(f"{log_prefix} {msg}")

        raw_grasps = list(grasps)
        log_info(f"START: {len(grasps)} grasps from GraspGen")
        # NOTE 2026-08-24 (thesis teardown): _filter_no_under_grasp retired
        # (inverted from-below test on the +z wrist axis + no unit-cube
        # denormalisation). Thesis stage 2 (support-plane) now runs inside
        # apply_gravity_view_filter below — same cascade the pick uses, so
        # the utility U(v) counts what is actually executable.
        try:
            if bool(node.get_parameter("apply_finger_penetration_filter").value):
                aabb_min_f = np.min(pts_filter, axis=0)
                aabb_max_f = np.max(pts_filter, axis=0)
                grasps_pen = _filter_finger_penetration(
                    grasps,
                    pts_filter,
                    matrices_normalized=bool(
                        node.get_parameter("graspgen_matrix_normalized").value
                    ),
                    column_major=False,
                    aabb_min=aabb_min_f,
                    aabb_max=aabb_max_f,
                    finger_length=float(
                        node.get_parameter(
                            "finger_penetration_finger_length_m"
                        ).value
                    ),
                    finger_thickness=float(
                        node.get_parameter(
                            "finger_penetration_finger_thickness_m"
                        ).value
                    ),
                    finger_height=float(
                        node.get_parameter(
                            "finger_penetration_finger_height_m"
                        ).value
                    ),
                    gripper_opening=float(
                        node.get_parameter(
                            "finger_penetration_gripper_opening_m"
                        ).value
                    ),
                    finger_z_offset=float(
                        node.get_parameter(
                            "finger_penetration_finger_z_offset_m"
                        ).value
                    ),
                    max_points_per_finger=int(
                        node.get_parameter(
                            "finger_penetration_max_points_per_finger"
                        ).value
                    ),
                    min_points_in_closing_volume=int(
                        node.get_parameter(
                            "finger_penetration_min_points_in_closing"
                        ).value
                    ),
                )
                if grasps_pen:
                    grasps = grasps_pen
                log_info(f"after finger-penetration: {len(grasps)}")
        except Exception as e:
            log_warn(f"finger-penetration filter error: {e}")
        try:
            if bool(node.get_parameter("apply_reject_from_behind_filter").value):
                cz = float(
                    np.mean(np.asarray(pts_filter, dtype=np.float32)[:, 2])
                ) + float(
                    node.get_parameter(
                        "reject_from_behind_observation_height_m"
                    ).value
                )
                centroid_xy = np.mean(
                    np.asarray(pts_filter, dtype=np.float32)[:, :2], axis=0
                )
                obs_pt = np.array(
                    [centroid_xy[0], centroid_xy[1], cz], dtype=np.float32
                )
                aabb_min_b = np.min(pts_filter, axis=0)
                aabb_max_b = np.max(pts_filter, axis=0)
                grasps_back = _filter_reject_from_behind(
                    grasps,
                    pts_filter,
                    observation_point=obs_pt,
                    matrices_normalized=bool(
                        node.get_parameter("graspgen_matrix_normalized").value
                    ),
                    column_major=False,
                    aabb_min=aabb_min_b,
                    aabb_max=aabb_max_b,
                    min_forward_dot=float(
                        node.get_parameter(
                            "reject_from_behind_min_forward_dot"
                        ).value
                    ),
                )
                if grasps_back:
                    grasps = grasps_back
                log_info(f"after reject-from-behind: {len(grasps)}")
        except Exception as e:
            log_warn(f"reject-from-behind filter error: {e}")
        try:
            if (
                bool(node.get_parameter("reject_flipped_grasps").value)
                and plane_n is not None
            ):
                grasps_upright = _filter_roll_upright_by_plane(
                    grasps,
                    plane_n=plane_n,
                    plane_p=plane_p,
                    up_axis=int(node.get_parameter("up_axis_index").value),
                    min_up_dot=float(node.get_parameter("min_up_dot").value),
                )
                if grasps_upright:
                    grasps = grasps_upright
            log_info(f"after upright: {len(grasps)}")
        except Exception as e:
            log_warn(f"upright-only filter error: {e}")
        try:
            if (
                bool(node.get_parameter("reject_side_grasps").value)
                and plane_n is not None
            ):
                grasps_vertical = _filter_reject_side_grasps(
                    grasps,
                    plane_n=plane_n,
                    min_vertical_dot=float(
                        node.get_parameter("min_vertical_dot").value
                    ),
                    approach_axis=2,
                    matrices_normalized=bool(
                        node.get_parameter("graspgen_matrix_normalized").value
                    ),
                    column_major=bool(
                        node.get_parameter("graspgen_column_major").value
                    ),
                )
                if grasps_vertical:
                    grasps = grasps_vertical
            log_info(f"after reject-side: {len(grasps)}")
        except Exception as e:
            log_warn(f"vertical-only filter error: {e}")

        # Gravity-aware view stage — SINGLE shared implementation
        # (grasp_filters.apply_gravity_view_filter); never duplicate it
        # again (2026-08-21).
        from ..grasp_filters import apply_gravity_view_filter
        grasps = apply_gravity_view_filter(
            node, grasps, pts_filter,
            plane_n=plane_n, plane_p=plane_p,
            frame_id=str(frame_id),
            registration_frame_is_target=registration_frame_is_target,
            log_info=log_info, log_warn=log_warn,
        )

        try:
            if (
                bool(node.get_parameter("forbid_top_grasps").value)
                and plane_n is not None
                and plane_p is not None
            ):
                side_center = _filter_side_center_window(
                    grasps,
                    pts_filter,
                    plane_n=plane_n,
                    plane_p=plane_p,
                    approach_axis=2,
                    max_vertical_dot=float(
                        node.get_parameter("max_vertical_dot").value
                    ),
                    centroid_down_only=True,
                    max_down=float(
                        node.get_parameter("max_down_from_centroid").value
                    ),
                    max_up=float(node.get_parameter("max_up_from_centroid").value),
                    matrices_normalized=bool(
                        node.get_parameter("graspgen_matrix_normalized").value
                    ),
                    column_major=bool(
                        node.get_parameter("graspgen_column_major").value
                    ),
                )
                if side_center:
                    grasps = side_center
            log_info(f"after side-center: {len(grasps)}")
        except Exception as e:
            log_warn(f"side/center filter error: {e}")

        aabb_min_metric, aabb_max_metric = _aabb_min_max(pts_metric)
        try:
            w_center = float(node.get_parameter("center_distance_weight").value)
            if w_center > 0.0:
                reweighted = _reweight_scores_by_centroid(
                    grasps,
                    centroid_metric=np.asarray(
                        pts_metric.mean(axis=0), dtype=np.float32
                    ),
                    aabb_min=aabb_min_metric,
                    aabb_max=aabb_max_metric,
                    matrices_normalized=bool(
                        node.get_parameter("graspgen_matrix_normalized").value
                    ),
                    column_major=bool(
                        node.get_parameter("graspgen_column_major").value
                    ),
                    weight=w_center,
                )
                if reweighted:
                    grasps = reweighted
            log_info(f"after centroid-reweight: {len(grasps)}")
            grasps = self._orientation.reweight_scores_by_ee_orientation(
                grasps,
                frame_id=frame_id,
                aabb_min=aabb_min_metric,
                aabb_max=aabb_max_metric,
            )
            log_info(f"after ee-orientation-reweight: {len(grasps)}")
        except Exception as e:
            log_warn(f"centroid/orientation reweight failed: {e}")

        if (
            not grasps
            and raw_grasps
            and not bool(node.get_parameter("require_horizontal_approach").value)
        ):
            log_warn(
                "all grasps were removed by filters/reweight; falling back to raw grasps."
            )
            grasps = raw_grasps

        if grasps:
            grasps = self._orientation.filter_by_ee_orientation_consistency(
                grasps,
                frame_id=frame_id,
                aabb_min=aabb_min_metric,
                aabb_max=aabb_max_metric,
            )
        log_info(f"after orientation-consistency: {len(grasps)}")
        return grasps
