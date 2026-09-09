"""Annotated debug-image rendering.

Wraps :class:`~grasp_pipeline.debug_image_publisher.DebugImagePublisher`
and pulls the per-tick render context out of the orchestrator. Reads
state directly off the node (frame buffer, detection cache, selection
state, hand landmarks, status text) — keeps that state on the node so
other components see the same thing the renderer does.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rclpy.node import Node


class DebugRenderer:
    """Publish the annotated debug image at a fixed cadence."""

    def __init__(self, node: "Node") -> None:
        self._node = node

    def _annotate_distances(self, node, dets) -> None:
        """Attach a compact base distance such as ``D:70cm``:
        median depth of a 13x13 window at the bbox centre -> 3D point ->
        distance from base_link."""
        if not dets:
            return
        rgb, depth, (fx, fy, cx, cy), depth_scale, header = node.fb.latest()
        if depth is None or header is None or not fx:
            return
        from ..registration import lookup_transform_matrix
        from ..perception.reach_envelope import reach_info
        cam = str(header.frame_id)
        t_base = lookup_transform_matrix(
            node.tf_buffer, "base_link", cam, header.stamp, 0.05)
        h, w = depth.shape[:2]
        for d in dets:
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            u = int((bbox[0] + bbox[2]) * 0.5)
            v = int((bbox[1] + bbox[3]) * 0.5)
            u0, u1 = max(0, u - 6), min(w, u + 7)
            v0, v1 = max(0, v - 6), min(h, v + 7)
            patch = np.asarray(depth[v0:v1, u0:u1], dtype=np.float64)
            patch = patch[patch > 0]
            if patch.size < 5:
                continue
            z = float(np.median(patch)) / float(depth_scale)
            p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0])
            p_base = (t_base @ p_cam)[:3]
            d_base = float(np.linalg.norm(p_base))
            # Reach warning (2026-09-09, Majd): distance over the front-grasp
            # limit at this angle from the base axis; see reach_envelope.py.
            info = reach_info(p_base)
            d["reach_pct"] = round(float(info["pct"]), 1)
            d["reach_angle_deg"] = round(float(info["angle_deg"]), 1)
            d["reach_state"] = str(info["state"])
            d["reach_closer_cm"] = round(float(info["closer_cm"]), 1)
            d["dist_note"] = f"D:{d_base * 100.0:.0f}cm" + (f"  {info['note']}" if info["note"] else "")

    def publish(self) -> None:
        """Publish debug image at a fixed rate independent of detector throughput."""
        node = self._node
        if not bool(node.get_parameter("publish_debug_image").value):
            return
        # Always use the live RGB frame so the debug window tracks robot motion.
        rgb = node.fb.rgb_frame
        header = node.fb.rgb_header
        if rgb is None or header is None:
            return

        dets = list(node.detections_cache) if node.detections_cache else []
        sel_idx = node.selected_idx if node.selection_made else None
        selected_mask = None

        if node.selection_made and node.last_selected_debug_det is not None:
            # Show the selected object bbox on the LIVE frame.
            dets = [dict(node.last_selected_debug_det)]
            sel_idx = 0
            # Overlay the segmentation mask only when it is fresh (<2 s old).
            # Stale masks drawn on a live/moved frame are misleading — skip them.
            mask_age_s = max(
                0.0,
                time.monotonic() - float(node.last_selected_debug_stamp_mono or 0.0),
            )
            if node.last_selected_mask_np is not None and mask_age_s < 2.0:
                selected_mask = np.asarray(node.last_selected_mask_np)

        # One useful distance only: compact base-to-object range in cm.
        try:
            self._annotate_distances(node, dets)
        except Exception:
            pass

        # Build status: pipeline state first, then mask age if relevant.
        status_lines = []
        if node._debug_status_text:
            status_lines.append(node._debug_status_text)
        if node.selection_made and node.last_selected_debug_det is not None:
            mask_age_s = max(
                0.0,
                time.monotonic() - float(node.last_selected_debug_stamp_mono or 0.0),
            )
            label = (
                node.last_selected_debug_det.get("display")
                or node.last_selected_debug_det.get("label", "?")
            )
            status_lines.append(f"obj={label}  mask_age={mask_age_s:.1f}s")
        status_text = "\n".join(status_lines) if status_lines else None

        try:
            node.debug_pub.publish(
                header,
                rgb,
                dets,
                selected_idx=sel_idx,
                pointer_px=node._pointer_px,
                hand_landmarks_px=node._hand_landmarks_px,
                hand_connections=node._hand_connections,
                selected_mask=selected_mask,
                status_text=status_text,
            )
        except Exception as e:
            node.get_logger().warn(f"Debug image publish failed: {e}")
