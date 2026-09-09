"""ZMQ-RPC wrapper around the GraspGen service.

Hides the parameter plumbing and intrinsics lookup so the orchestrator
just calls ``request_candidates(points, colors)``. Logging is opt-in
via ``log_prefix`` so the multi-view sweep can label its calls without
spamming the steady-state pick path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..grasp_client import send_points_to_graspgen

if TYPE_CHECKING:
    from rclpy.node import Node


class GraspGenClient:
    """Issue requests to the GraspGen service."""

    def __init__(self, node: "Node") -> None:
        self._node = node

    def request_candidates(
        self,
        pts_send: np.ndarray,
        cols_send: Optional[np.ndarray],
        *,
        log_prefix: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        node = self._node
        # 2026-08-27: same note as in pick_pipeline — a not-yet-warm model
        # means this call waits for the in-flight preload build and shares
        # its instance; it never loads a second copy.
        import os as _os
        if _os.environ.get("GRASPGEN_MODE", "inprocess").strip().lower() != "zmq":
            from ..grasp_client import sampler_is_warm
            if not sampler_is_warm():
                node.get_logger().info(
                    "[graspgen] model not warm yet — waiting for the in-flight "
                    "preload and reusing its instance (single load, no reload)."
                )
        try:
            resp = send_points_to_graspgen(
                pts_send,
                endpoint=node.get_parameter("graspgen_endpoint").value,
                threshold=float(node.get_parameter("grasp_threshold").value),
                colors=cols_send,
                num_grasps=int(node.get_parameter("graspgen_num_grasps").value),
                topk_num_grasps=-1,  # server topk costs 5-7 s; cut is client-side
                min_grasps=40,
                max_tries=2,
                remove_outliers=False,
                fx=node.fb.fx,
                fy=node.fb.fy,
                cx=node.fb.cx,
                cy=node.fb.cy,
                depth_scale=node.fb.depth_scale,
            )
        except Exception as e:
            if log_prefix:
                node.get_logger().warn(f"{log_prefix} failed contacting GraspGen: {e}")
            return None
        if not resp or not resp.get("ok", False):
            if log_prefix:
                node.get_logger().warn(
                    f"{log_prefix} GraspGen error: "
                    f"{None if resp is None else resp.get('error')}"
                )
            return None
        grasps_obj = resp.get("grasps", [])
        if grasps_obj is None:
            return []
        if isinstance(grasps_obj, list):
            return list(grasps_obj)
        try:
            return list(grasps_obj)
        except Exception:
            return []
