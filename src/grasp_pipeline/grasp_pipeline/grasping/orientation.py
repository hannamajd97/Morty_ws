"""TF-aware EE-orientation consistency filter and score re-weighter.

Both methods need to know:
  - the current TCP pose (looked up against ``target_frame``);
  - the cloud's frame (so a grasp expressed in the cloud frame can be
    re-expressed in the target frame).

If the lookup fails (TF stale, frame missing) we degrade to a no-op
rather than nuking the candidate set — same behaviour as the legacy
inline implementation.

Also exposes the small static quaternion helpers that used to live on
the orchestrator.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import rclpy
from tf_transformations import quaternion_from_matrix

from ..geometry_utils import _denorm_T_from_unit_cube, _make_T_grasp_to_tcp

if TYPE_CHECKING:
    from rclpy.node import Node


class OrientationFilter:
    """EE-orientation aware filter and re-weighter."""

    def __init__(self, node: "Node") -> None:
        self._node = node

    # ------------------------------------------------------------------
    # Quaternion helpers (kept here so other code can import them too)
    # ------------------------------------------------------------------

    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(q))
        if n <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return (q / n).astype(np.float64)

    @staticmethod
    def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = [float(v) for v in q1]
        x2, y2, z2, w2 = [float(v) for v in q2]
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
        q1n = OrientationFilter.quat_normalize(q1)
        q2n = OrientationFilter.quat_normalize(q2)
        dot = float(np.clip(np.abs(np.dot(q1n, q2n)), 0.0, 1.0))
        return math.degrees(2.0 * math.acos(dot))

    # ------------------------------------------------------------------
    # Filter / re-weight
    # ------------------------------------------------------------------

    def filter_by_ee_orientation_consistency(
        self,
        grasps: List[Dict[str, Any]],
        frame_id: str,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> List[Dict[str, Any]]:
        node = self._node
        if not bool(node.get_parameter("orientation_consistency_filter").value):
            return grasps
        if not grasps:
            return grasps

        target_frame = str(node.get_parameter("target_frame").value)
        ee_link = str(node.get_parameter("ee_link_for_orientation_filter").value)
        max_delta_deg = float(
            node.get_parameter("max_orientation_delta_deg").value
        )
        use_norm = bool(node.get_parameter("graspgen_matrix_normalized").value)
        col_major = bool(node.get_parameter("graspgen_column_major").value)
        fixed_rot = str(node.get_parameter("grasp2tcp_rot").value)

        try:
            tf_ee = node.tf_buffer.lookup_transform(
                target_frame,
                ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3),
            )
            q_ee = self.quat_normalize(
                np.array(
                    [
                        tf_ee.transform.rotation.x,
                        tf_ee.transform.rotation.y,
                        tf_ee.transform.rotation.z,
                        tf_ee.transform.rotation.w,
                    ],
                    dtype=np.float64,
                )
            )
            if frame_id == target_frame:
                q_cloud_to_target = np.array(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float64
                )
            else:
                tf_cloud = node.tf_buffer.lookup_transform(
                    target_frame,
                    frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.3),
                )
                q_cloud_to_target = self.quat_normalize(
                    np.array(
                        [
                            tf_cloud.transform.rotation.x,
                            tf_cloud.transform.rotation.y,
                            tf_cloud.transform.rotation.z,
                            tf_cloud.transform.rotation.w,
                        ],
                        dtype=np.float64,
                    )
                )
        except Exception as e:
            node.get_logger().warn(
                f"Orientation consistency filter disabled for this cycle (TF unavailable): {e}"
            )
            return grasps

        kept: List[Dict[str, Any]] = []
        for g in grasps:
            try:
                M = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
                if use_norm:
                    T = _denorm_T_from_unit_cube(
                        M, aabb_min, aabb_max, column_major=col_major
                    )
                else:
                    T = M.T if col_major else M
                try:
                    T_tcp = T @ _make_T_grasp_to_tcp(fixed_rot)
                except Exception:
                    T_tcp = T
                q_cloud = self.quat_normalize(
                    np.asarray(quaternion_from_matrix(T_tcp), dtype=np.float64)
                )
                q_target = self.quat_multiply(q_cloud_to_target, q_cloud)
                ang = self.quat_angle_deg(q_target, q_ee)
                if ang <= max_delta_deg:
                    kept.append(g)
            except Exception:
                continue

        if not kept:
            node.get_logger().warn(
                f"Orientation consistency filter removed all candidates "
                f"(max_orientation_delta_deg={max_delta_deg:.1f})."
            )
            return []
        node.get_logger().info(
            f"Orientation consistency kept {len(kept)}/{len(grasps)} candidates "
            f"(<= {max_delta_deg:.1f} deg to {ee_link})."
        )
        return kept

    def reweight_scores_by_ee_orientation(
        self,
        grasps: List[Dict[str, Any]],
        frame_id: str,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> List[Dict[str, Any]]:
        node = self._node
        if not bool(node.get_parameter("orientation_score_reweight").value):
            return grasps
        if not grasps:
            return grasps

        w_orient = float(node.get_parameter("orientation_score_weight").value)
        if w_orient <= 0.0:
            return grasps

        ref_deg = float(node.get_parameter("orientation_score_ref_deg").value)
        if ref_deg <= 1e-6:
            ref_deg = 45.0

        target_frame = str(node.get_parameter("target_frame").value)
        ee_link = str(node.get_parameter("ee_link_for_orientation_filter").value)
        use_norm = bool(node.get_parameter("graspgen_matrix_normalized").value)
        col_major = bool(node.get_parameter("graspgen_column_major").value)
        fixed_rot = str(node.get_parameter("grasp2tcp_rot").value)

        try:
            tf_ee = node.tf_buffer.lookup_transform(
                target_frame,
                ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3),
            )
            q_ee = self.quat_normalize(
                np.array(
                    [
                        tf_ee.transform.rotation.x,
                        tf_ee.transform.rotation.y,
                        tf_ee.transform.rotation.z,
                        tf_ee.transform.rotation.w,
                    ],
                    dtype=np.float64,
                )
            )
            if frame_id == target_frame:
                q_cloud_to_target = np.array(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float64
                )
            else:
                tf_cloud = node.tf_buffer.lookup_transform(
                    target_frame,
                    frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.3),
                )
                q_cloud_to_target = self.quat_normalize(
                    np.array(
                        [
                            tf_cloud.transform.rotation.x,
                            tf_cloud.transform.rotation.y,
                            tf_cloud.transform.rotation.z,
                            tf_cloud.transform.rotation.w,
                        ],
                        dtype=np.float64,
                    )
                )
        except Exception as e:
            node.get_logger().warn(
                f"Orientation score reweight disabled for this cycle (TF unavailable): {e}"
            )
            return grasps

        reweighted: List[Dict[str, Any]] = []
        for g in grasps:
            try:
                M = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
                if use_norm:
                    T = _denorm_T_from_unit_cube(
                        M, aabb_min, aabb_max, column_major=col_major
                    )
                else:
                    T = M.T if col_major else M
                try:
                    T_tcp = T @ _make_T_grasp_to_tcp(fixed_rot)
                except Exception:
                    T_tcp = T
                q_cloud = self.quat_normalize(
                    np.asarray(quaternion_from_matrix(T_tcp), dtype=np.float64)
                )
                q_target = self.quat_multiply(q_cloud_to_target, q_cloud)
                ang = self.quat_angle_deg(q_target, q_ee)
                orient_bonus = math.exp(-w_orient * (ang / ref_deg))
                g2 = dict(g)
                # Clamp at 0 first: upstream stages subtract penalties and
                # can push a score negative, and multiplying a NEGATIVE
                # score by a small bonus RAISES it — which promoted the
                # most flipped gripper orientation (bug, 2026-08-21).
                base_score = max(0.0, float(g.get("score", 0.0)))
                g2["score"] = base_score * float(orient_bonus)
                reweighted.append(g2)
            except Exception:
                reweighted.append(g)

        node.get_logger().info(
            f"Orientation score reweight applied to {len(reweighted)} candidates "
            f"(weight={w_orient:.2f}, ref_deg={ref_deg:.1f})."
        )
        return reweighted
