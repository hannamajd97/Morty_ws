"""Utility scoring + early-stop policy for the multi-view grasp sweep.

Stateless apart from a back-reference to the node (used to read
parameters). The orchestrator ticks the multi-view loop and consults
this scorer after every accepted observation.

Public surface:

    grasp_matrix_to_pose(grasp, aabb_min, aabb_max) -> np.ndarray
    angular_diversity_deg(grasps, aabb_min, aabb_max) -> float
    instability_penalty(accepted_obs_meta) -> float
    compute_utility(grasps, aabb_min, aabb_max, instability_penalty)
        -> dict[str, float]
    update_stop_state(...) -> tuple[bool, int, float | None]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..geometry_utils import _denorm_T_from_unit_cube

if TYPE_CHECKING:
    from rclpy.node import Node


class GraspScorer:
    """Stateless utility math (parameters read from the node)."""

    def __init__(self, node: "Node") -> None:
        self._node = node

    def grasp_matrix_to_pose(
        self,
        grasp: Dict[str, Any],
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> np.ndarray:
        node = self._node
        M = np.asarray(grasp["matrix"], dtype=np.float32).reshape(4, 4)
        use_norm = bool(node.get_parameter("graspgen_matrix_normalized").value)
        col_major = bool(node.get_parameter("graspgen_column_major").value)
        if use_norm:
            return _denorm_T_from_unit_cube(
                M, aabb_min, aabb_max, column_major=col_major
            )
        return M.T if col_major else M

    def angular_diversity_deg(
        self,
        grasps: List[Dict[str, Any]],
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> float:
        if len(grasps) < 2:
            return 0.0
        dirs: List[np.ndarray] = []
        for grasp in grasps:
            try:
                T = self.grasp_matrix_to_pose(grasp, aabb_min, aabb_max)
                axis = np.asarray(T[:3, 2], dtype=np.float64)
                nrm = float(np.linalg.norm(axis))
                if nrm > 1e-6:
                    dirs.append(axis / nrm)
            except Exception:
                continue
        if len(dirs) < 2:
            return 0.0
        dirs_np = np.asarray(dirs, dtype=np.float64)
        mean_dir = np.mean(dirs_np, axis=0)
        mean_nrm = float(np.linalg.norm(mean_dir))
        if mean_nrm <= 1e-6:
            mean_dir = dirs_np[0]
            mean_nrm = float(np.linalg.norm(mean_dir))
            if mean_nrm <= 1e-6:
                return 0.0
        mean_dir = mean_dir / mean_nrm
        cos_theta = np.clip(dirs_np @ mean_dir, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_theta))
        return float(np.std(angles_deg))

    @staticmethod
    def instability_penalty(accepted_obs_meta: List[Dict[str, Any]]) -> float:
        penalties = [
            float(obs.get("tracking_instability_score", 0.0))
            for obs in accepted_obs_meta[1:]
            if float(obs.get("tracking_instability_score", 0.0)) > 0.0
        ]
        return float(np.mean(penalties)) if penalties else 0.0

    def compute_utility(
        self,
        grasps: List[Dict[str, Any]],
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        instability_penalty: float = 0.0,
    ) -> Dict[str, float]:
        node = self._node
        surviving_count = int(len(grasps))
        diversity_deg = self.angular_diversity_deg(grasps, aabb_min, aabb_max)
        mean_score = (
            float(np.mean([float(g.get("score", 0.0)) for g in grasps]))
            if grasps
            else 0.0
        )
        count_term = float(math.log1p(max(0, surviving_count)))
        diversity_term = float(max(0.0, diversity_deg) / 45.0)
        score_term = float(max(0.0, mean_score))
        instability_term = float(max(0.0, instability_penalty))
        w_count = float(
            node.get_parameter("registration_grasp_utility_weight_count").value
        )
        w_div = float(
            node.get_parameter("registration_grasp_utility_weight_diversity").value
        )
        w_score = float(
            node.get_parameter("registration_grasp_utility_weight_score").value
        )
        w_inst = float(
            node.get_parameter("registration_grasp_utility_weight_instability").value
        )
        total = (
            w_count * count_term
            + w_div * diversity_term
            + w_score * score_term
            - w_inst * instability_term
        )
        return {
            "total": float(total),
            "surviving_count": float(surviving_count),
            "diversity_deg": float(diversity_deg),
            "mean_score": float(mean_score),
            "instability_penalty": float(instability_term),
            "count_term": float(count_term),
            "diversity_term": float(diversity_term),
            "score_term": float(score_term),
            "weights_count": float(w_count),
            "weights_diversity": float(w_div),
            "weights_score": float(w_score),
            "weights_instability": float(w_inst),
        }

    def update_stop_state(
        self,
        *,
        context: str,
        view_count: int,
        prev_utility: Optional[Dict[str, float]],
        utility_now: Optional[Dict[str, float]],
        small_gain_streak: int,
    ) -> tuple[bool, int, Optional[float]]:
        node = self._node
        if not bool(
            node.get_parameter("registration_grasp_utility_stop_enabled").value
        ):
            return False, 0, None
        if prev_utility is None or utility_now is None:
            return False, 0, None

        delta_u = float(utility_now["total"] - prev_utility["total"])
        min_views_after_seed = max(
            0,
            int(
                node.get_parameter(
                    "registration_grasp_utility_stop_min_views_after_seed"
                ).value
            ),
        )
        min_delta_u = float(
            node.get_parameter("registration_grasp_utility_stop_min_delta_u").value
        )
        patience = max(
            1,
            int(
                node.get_parameter("registration_grasp_utility_stop_patience").value
            ),
        )
        large_drop_delta_u = float(
            node.get_parameter(
                "registration_grasp_utility_stop_large_drop_delta_u"
            ).value
        )
        extra_views = max(0, int(view_count) - 1)
        if extra_views < min_views_after_seed:
            return False, 0, delta_u

        if delta_u < large_drop_delta_u:
            node.get_logger().info(
                f"[grasp-utility] {context}: utility-stop triggered "
                f"(delta_U={delta_u:.3f} < {large_drop_delta_u:.3f})."
            )
            return True, 0, delta_u

        if delta_u <= min_delta_u:
            next_streak = small_gain_streak + 1
            if next_streak >= patience:
                node.get_logger().info(
                    f"[grasp-utility] {context}: utility-stop triggered "
                    f"(delta_U={delta_u:.3f} <= {min_delta_u:.3f}, "
                    f"streak={next_streak}/{patience})."
                )
                return True, next_streak, delta_u
            return False, next_streak, delta_u

        return False, 0, delta_u
