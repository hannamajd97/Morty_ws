"""Single-view cloud extraction + per-frame filtering + multi-view tracking.

Pulled out of grasp_selection_node into one focused class so the
orchestrator stops carrying ~1,000 lines of NumPy point-cloud surgery.

Method names match the legacy implementation 1:1 — wrappers on the
orchestrator simply delegate to ``self._cloud.<method>(...)`` so every
existing call site (registration sweep, dynamic NBV, single-view pick)
keeps working unchanged.

State lives on the owning ``YoloLiveSelectGrasp`` node (frame buffer,
selection state, debug publishers, TF). The extractor reaches into the
node for everything it needs; this keeps state ownership centralised
and lets the orchestrator stay the single source of truth for pipeline
state.
"""

from __future__ import annotations

import copy
import time
import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np
import rclpy
from std_msgs.msg import Header

from ..crop_utils import crop_depth_to_cloud_and_numpy
from .neighbour_obstacles import publish_neighbour_obstacles
from ..plane_filter import estimate_plane_svd, filter_points_above_plane
from ..registration import (
    RegistrationConfig,
    lookup_transform_matrix,
    transform_points,
    voxel_downsample,
)
from ..utils import (
    _clean_cropped_cloud,
    keep_largest_cluster,
    radius_outlier_filter,
)

if TYPE_CHECKING:
    from rclpy.node import Node


class CloudExtractor:
    """Extract and filter the per-object point cloud for a YOLO selection."""

    def _capture_depth_multiframe(self) -> "np.ndarray":
        """Return the depth image for a stationary single-view capture.

        Multi-frame temporal fusion (2026-08-07, Majd's item 4): one D435
        frame carries mm-level ripple and edge dropouts. With the arm still,
        we grab ``single_view_frames`` consecutive depth frames (~0.2 s for 5
        at 30 Hz) and take the per-pixel MEDIAN over valid (non-zero) samples
        — the same temporal denoising the TSDF's running average provides,
        without the volume overhead, so the capture stays fast. Dropout
        pixels get filled by any frame that saw them.

        Active whenever ``single_view_multiframe`` is true and the arm is
        not mid-sweep (2026-08-07, Majd: multiview too). Stationary multiview
        captures — the initial view, NBV stops, sweep endpoints — fuse just
        like single view; frames grabbed while the sweep motion process runs
        stay single-frame (averaging in-motion frames would smear). Falls
        back to the live frame on any problem.
        """
        node = self._node
        try:
            if not node.has_parameter("single_view_multiframe"):
                node.declare_parameter("single_view_multiframe", True)
                node.declare_parameter("single_view_frames", 5)
            enabled = bool(node.get_parameter("single_view_multiframe").value)
            n = max(2, int(node.get_parameter("single_view_frames").value))
        except Exception:
            return node.fb.depth_frame
        if not enabled:
            return node.fb.depth_frame
        if getattr(node, "_sweep_in_motion", False):
            return node.fb.depth_frame
        try:
            t_start = time.monotonic()
            frames = [node.fb.depth_frame.copy()]
            last_stamp = getattr(node.fb.depth_header, "stamp", None)
            deadline = time.monotonic() + 1.0
            while len(frames) < n and time.monotonic() < deadline:
                time.sleep(0.01)
                stamp = getattr(node.fb.depth_header, "stamp", None)
                if stamp is not None and stamp != last_stamp:
                    frames.append(node.fb.depth_frame.copy())
                    last_stamp = stamp
            if len(frames) < 2:
                return node.fb.depth_frame
            stack = np.stack([f.astype(np.float32) for f in frames])
            stack[stack <= 0] = np.nan
            # Pixels with zero depth in ALL frames are all-NaN slices; the
            # median is legitimately NaN there (mapped to 0 below), so numpy's
            # RuntimeWarning about it is expected noise.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
            med = np.nan_to_num(med, nan=0.0)
            out = med.astype(frames[0].dtype)
            node.get_logger().info(
                f"[multiframe] fused {len(frames)} depth frames (median) in "
                f"{time.monotonic() - t_start:.2f}s for single-view capture."
            )
            return out
        except Exception as e:
            node.get_logger().warn(f"[multiframe] fusion failed ({e}); using live frame.")
            return node.fb.depth_frame

    def __init__(self, node: "Node") -> None:
        self._node = node

    def _extract_filtered_object_cloud(
        self,
        sel: Dict[str, Any],
        *,
        publish_outputs: bool,
    ) -> Optional[Dict[str, Any]]:
        if self._node.fb.depth_frame is None or self._node.fb.rgb_frame is None:
            return None

        capture_header = copy.deepcopy(self._node.fb.depth_header if self._node.fb.depth_header is not None else self._node.fb.rgb_header)
        if capture_header is None:
            capture_header = copy.deepcopy(sel["header"])
        capture_sel = dict(sel)
        capture_sel["header"] = capture_header

        if publish_outputs:
            self._node._publish_detection_message(capture_sel)
            x1, y1, x2, y2 = capture_sel["bbox"]
            self._node.get_logger().info(
                f"Published selected '{capture_sel['label']}' bbox=({x1},{y1},{x2},{y2}) | depth=yes"
            )

        selected_mask = self._node._compute_selected_mask(capture_sel, publish_outputs=publish_outputs)
        if selected_mask is not None:
            try:
                x1, y1, x2, y2 = [int(v) for v in capture_sel["bbox"]]
                H, W = self._node.fb.depth_frame.shape[:2]
                x1 = max(0, min(W - 1, x1))
                x2 = max(0, min(W, x2))
                y1 = max(0, min(H - 1, y1))
                y2 = max(0, min(H, y2))
                if x2 > x1 and y2 > y1:
                    z_roi = self._node.fb.depth_frame[y1:y2, x1:x2]
                    valid_depth_roi = z_roi > 0
                    mask_roi = selected_mask[y1:y2, x1:x2]
                    overlap = int(np.count_nonzero(valid_depth_roi & mask_roi))
                    valid_bbox = int(np.count_nonzero(valid_depth_roi))
                    mask_px = int(np.count_nonzero(mask_roi))
                    self._node.get_logger().info(
                        f"[seg-depth] bbox_valid_depth_px={valid_bbox}, mask_px={mask_px}, overlap_px={overlap}"
                    )
            except Exception as e:
                self._node.get_logger().warn(f"Seg-depth diagnostics failed: {e}")

        # Stationary capture: optionally fuse several depth frames (item 4).
        depth_for_capture = self._capture_depth_multiframe()
        # Padded collision boxes for the OTHER objects on the table
        # (2026-08-27) — the target itself is deliberately left out.
        # Gated on publish_outputs: the real pick capture passes True
        # (pick_pipeline), the multi-view sweep passes False — so the
        # boxes are rebuilt once per PICK, not once per view.
        if publish_outputs:
            publish_neighbour_obstacles(
                self._node, capture_sel, depth_for_capture)
        cloud_msg, pts_np, cols_np = crop_depth_to_cloud_and_numpy(
            depth_for_capture,
            self._node.fb.rgb_frame,
            capture_sel["bbox"],
            capture_sel["header"],
            self._node.fb.fx,
            self._node.fb.fy,
            self._node.fb.cx,
            self._node.fb.cy,
            self._node.fb.depth_scale,
            mask_full=selected_mask,
        )
        if (cloud_msg is None or pts_np is None or pts_np.size == 0) and selected_mask is not None:
            self._node.get_logger().warn("Segmented crop has no valid depth points; retrying with bbox-only crop.")
            cloud_msg, pts_np, cols_np = crop_depth_to_cloud_and_numpy(
                depth_for_capture,
                self._node.fb.rgb_frame,
                capture_sel["bbox"],
                capture_sel["header"],
                self._node.fb.fx,
                self._node.fb.fy,
                self._node.fb.cx,
                self._node.fb.cy,
                self._node.fb.depth_scale,
                mask_full=None,
            )
        if cloud_msg is None or pts_np is None or pts_np.size == 0:
            return None

        cloud_msg.header = capture_sel["header"]
        if publish_outputs:
            self._node.pub_cloud.publish(cloud_msg)
            self._node.last_selected_cloud_msg = copy.deepcopy(cloud_msg)
            self._node.get_logger().info(f"Published cropped PointCloud2 ({pts_np.shape} pts).")
            # RGB twin for the viewers (2026-08-07): cols_np is already
            # computed above — publish a coloured copy, and in single-view
            # mode also feed the frames window (fused_cloud) so it shows the
            # one captured frame; multiview overwrites it with the real
            # accumulated preview during the sweep.
            try:
                if cols_np is not None and len(cols_np) == len(pts_np):
                    # cols_np comes from the OpenCV frame buffer = BGR order;
                    # swap to RGB or yellow renders as cyan (2026-08-07).
                    rgb_msg = self._node._cloud_pub.create_cloud_xyzrgb(
                        cloud_msg.header, pts_np, np.ascontiguousarray(cols_np[:, ::-1])
                    )
                    self._node.pub_cloud_rgb.publish(rgb_msg)
                    self._node.last_selected_cloud_rgb_msg = rgb_msg
            except Exception as e:
                self._node.get_logger().warn(f"RGB cloud publish failed: {e}")
            self._node.get_logger().info(
                f"[diag] selected_cloud frame={cloud_msg.header.frame_id}, "
                f"K=({self._node.fb.fx:.1f},{self._node.fb.fy:.1f},{self._node.fb.cx:.1f},{self._node.fb.cy:.1f}), "
                f"depth_scale={self._node.fb.depth_scale}, bbox={capture_sel['bbox']}, "
                f"z_range=({float(pts_np[:,2].min()):.3f},{float(pts_np[:,2].max()):.3f}) m"
            )

        filtered = self._filter_object_cloud(pts_np, cols_np)
        if filtered is None:
            return None
        pts_np, cols_np, plane_n, plane_p = filtered
        # ── TABLE plane from the ring AROUND the object (2026-08-24) ──
        # The thesis (§4.6.7) fits the support plane where the crop still
        # contains table pixels; a tight seg-mask leaves none, so the old
        # plane fit either never ran (apply_plane_above_filter=False) or
        # fitted the OBJECT's own face. Measure the actual tabletop from
        # the non-object pixels around/below the bbox instead. This plane
        # defines "up" for the whole filter cascade — camera-tilt-proof
        # and TF-free (Majd: measure "up" from the table in the picture).
        # COLOUR first (2026-08-27): the yellow tabletop itself is the
        # PRIMARY plane source; the ring below stays the fallback with
        # its own memory/TF ladder untouched. fb.rgb_frame is the same
        # capture this call site already pairs with depth_for_capture.
        ring = None
        if bool(self._node.get_parameter("table_from_colour").value):
            try:
                ring = self._estimate_table_plane_from_colour(
                    self._node.fb.rgb_frame, depth_for_capture,
                    self._collect_all_object_masks(
                        selected_mask, capture_sel["bbox"]),
                    obj_pts=pts_np)
            except Exception as e:
                self._node.get_logger().warn(f"[table] colour plane failed: {e}")
        try:
            if ring is None:
                ring = self._estimate_table_plane_from_ring(
                    depth_for_capture, capture_sel["bbox"], selected_mask,
                    obj_pts=pts_np)
            if ring is not None:
                plane_n, plane_p = ring
        except Exception as e:
            self._node.get_logger().warn(f"[table] ring plane failed: {e}")
        h, w = self._node.fb.depth_frame.shape[:2]
        return {
            "cloud_msg": cloud_msg,
            "points": pts_np,
            "colors": cols_np,
            "header": cloud_msg.header,
            "bbox": [int(v) for v in capture_sel["bbox"]],
            "plane_n": plane_n,
            "plane_p": plane_p,
            # The TSDF integrates obs["depth_frame"] — store the (possibly
            # median-fused) capture depth, not the raw live frame.
            "depth_frame": np.asarray(depth_for_capture).copy(),
            "rgb_frame": self._node.fb.rgb_frame.copy(),
            "selected_mask": None if selected_mask is None else np.asarray(selected_mask, dtype=bool).copy(),
            "intrinsics": {
                "fx": float(self._node.fb.fx),
                "fy": float(self._node.fb.fy),
                "cx": float(self._node.fb.cx),
                "cy": float(self._node.fb.cy),
                "width": int(w),
                "height": int(h),
            },
            "depth_scale": float(self._node.fb.depth_scale),
            "tsdf_eligible": True,
        }

    def _filter_object_cloud(
        self,
        pts_np: Optional[np.ndarray],
        cols_np: Optional[np.ndarray],
        *,
        apply_depth_clamp: bool = True,
        lightweight: bool = False,
        skip_cluster_filter: bool = False,
        skip_radius_outlier: bool = False,
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
        """Apply pregrasp filters to a point cloud.

        ``lightweight=True`` skips the expensive cluster + radius outlier
        filters and the cloud-cleaning pass; only depth clamp and the
        cheap IQR / median distance trim run. Use this for per-frame
        evaluation during a multi-view sweep, where the full filter would
        otherwise dominate runtime as the fused cloud grows.
        """
        if pts_np is None or pts_np.size == 0:
            return None

        n0 = int(pts_np.shape[0])
        if apply_depth_clamp and bool(self._node.get_parameter("apply_depth_clamp").value):
            z_min = float(self._node.get_parameter("min_depth_m").value)
            z_max = float(self._node.get_parameter("max_depth_m").value)
            depth_mask = (pts_np[:, 2] >= z_min) & (pts_np[:, 2] <= z_max)
            if not depth_mask.any():
                self._node.get_logger().warn(
                    f"All points filtered by depth mask z[{z_min:.2f},{z_max:.2f}] m; skipping grasp generation."
                )
                return None
            pts_np = pts_np[depth_mask]
            if cols_np is not None:
                cols_np = cols_np[depth_mask]
        n1 = int(pts_np.shape[0])

        # Optional voxel pre-downsample before heavy filters (full-mode
        # only — the lightweight path doesn't need it because it doesn't
        # hit the expensive O(N*k) filters).
        if not lightweight:
            try:
                voxel_pre = float(
                    self._node.get_parameter("pregrasp_filter_voxel_pre_downsample_m").value
                )
            except Exception:
                voxel_pre = 0.0
            # The final GraspGen request is capped at 1000 points, so running
            # DBSCAN/radius filters on 6k-14k nearly duplicate depth pixels is
            # wasted work (2-5 s in the 2026-08-21 timing logs). Preserve a
            # generous geometric margin, but voxelise before those filters
            # once the crop exceeds 5000 points.
            if voxel_pre > 0.0 and pts_np is not None and pts_np.shape[0] > 5000:
                try:
                    pts_np, cols_np = voxel_downsample(pts_np, cols_np, voxel_pre)
                except Exception as exc:
                    self._node.get_logger().warn(
                        f"[pregrasp] voxel pre-downsample failed ({exc}); continuing on full cloud."
                    )

        # Skip cluster filter when explicitly requested (e.g. on the
        # post-fusion path where multi-view clouds may form near-but-not-
        # touching clusters that DBSCAN would erroneously prune).
        if (not lightweight
                and not skip_cluster_filter
                and bool(self._node.get_parameter("apply_largest_cluster_filter").value)):
            pts_np, cols_np = keep_largest_cluster(
                pts_np,
                cols_np,
                eps=float(self._node.get_parameter("cluster_eps_m").value),
                min_samples=int(self._node.get_parameter("cluster_min_samples").value),
            )
        n_cluster = int(pts_np.shape[0]) if pts_np is not None else 0

        # Skip radius outlier filter when explicitly requested (e.g.
        # post-fusion clouds that have already been voxel-downsampled —
        # radius outlier is calibrated for full-density single-view
        # clouds and would over-prune sparser fused/downsampled clouds).
        if (not lightweight
                and not skip_radius_outlier
                and bool(self._node.get_parameter("apply_radius_outlier_filter").value)):
            pts_np, cols_np = radius_outlier_filter(
                pts_np,
                cols_np,
                radius_m=float(self._node.get_parameter("radius_filter_m").value),
                min_neighbors=int(self._node.get_parameter("radius_min_neighbors").value),
            )
        n_radius = int(pts_np.shape[0]) if pts_np is not None else 0

        if not lightweight and bool(self._node.get_parameter("apply_cloud_cleaning").value):
            pts_np, cols_np = _clean_cropped_cloud(pts_np, cols_np)
        n2 = int(pts_np.shape[0]) if pts_np is not None else 0

        max_cd = float(self._node.get_parameter("max_grasp_centroid_dist_m").value)
        if max_cd > 0.0 and pts_np is not None and len(pts_np) > 0:
            n_before = len(pts_np)
            keep = np.ones(len(pts_np), dtype=bool)
            for ax in range(3):
                q1, q3 = np.percentile(pts_np[:, ax], [25, 75])
                iqr = q3 - q1
                margin = 1.5 * iqr
                keep &= (pts_np[:, ax] >= q1 - margin) & (pts_np[:, ax] <= q3 + margin)
            median = np.median(pts_np, axis=0)
            dists = np.linalg.norm(pts_np - median, axis=1)
            p95 = np.percentile(dists, 95)
            cutoff = max(max_cd, p95)
            keep &= dists <= cutoff
            if keep.any():
                pts_np = pts_np[keep]
                if cols_np is not None:
                    cols_np = cols_np[keep]
                self._node.get_logger().info(
                    f"[pregrasp] spatial-outlier filter: kept {int(keep.sum())}/{n_before} "
                    f"(cutoff={cutoff:.3f}m, p95={p95:.3f}m)"
                )
            else:
                self._node.get_logger().warn("Spatial outlier filter would remove all points; keeping all.")

        plane_n = None
        plane_p = None
        if bool(self._node.get_parameter("apply_plane_above_filter").value) and pts_np is not None and pts_np.shape[0] >= 3:
            try:
                plane_n, plane_p = estimate_plane_svd(pts_np)
                mask_above = filter_points_above_plane(
                    pts_np,
                    plane_n,
                    plane_p,
                    clearance=float(self._node.get_parameter("plane_clearance_m").value),
                )
                pts_np = pts_np[mask_above]
                if cols_np is not None:
                    cols_np = cols_np[mask_above]
            except Exception as e:
                self._node.get_logger().warn(f"Plane estimation/filtering failed: {e}")
                plane_n = None
                plane_p = None
        n3 = int(pts_np.shape[0]) if pts_np is not None else 0
        if bool(self._node.get_parameter("log_pregrasp_filter_counts").value):
            self._node.get_logger().info(
                f"[pregrasp] counts raw={n0} depth={n1} cluster={n_cluster} radius={n_radius} clean={n2} plane={n3}"
            )

        if pts_np is None or pts_np.shape[0] == 0:
            return None
        return pts_np, cols_np, plane_n, plane_p

    def _estimate_table_plane_from_ring(
        self,
        depth_frame,
        bbox,
        selected_mask,
        *,
        obj_pts,
    ):
        """Measure the TABLE plane from the pixels around/below the object
        (2026-08-24). Returns (plane_n, plane_p) in the camera/cloud frame
        with plane_n oriented toward the object (= up), or None.

        The ring: the object's bbox grown sideways and downward, minus the
        object's own mask/bbox — i.e. the visible tabletop surrounding the
        object. Robust-fit a plane there; sanity-check that the object
        actually sits on it (bottom of the object cloud within a few cm of
        the plane). "Up" for the whole cascade comes from this plane —
        measured from the real table in the picture, no TF involved.
        """
        import numpy as _np
        if depth_frame is None or obj_pts is None or len(obj_pts) < 10:
            return None
        H, W = depth_frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        # grow: 80% sideways, 60% downward, 10% upward
        gx1 = max(0, x1 - int(0.8 * bw)); gx2 = min(W, x2 + int(0.8 * bw))
        gy1 = max(0, y1 - int(0.1 * bh)); gy2 = min(H, y2 + int(0.6 * bh))
        ring = _np.zeros((H, W), dtype=bool)
        ring[gy1:gy2, gx1:gx2] = True
        if selected_mask is not None:
            ring &= ~_np.asarray(selected_mask, dtype=bool)
        else:
            ring[y1:y2, x1:x2] = False
        raw = _np.asarray(depth_frame, dtype=_np.float32)
        # depth_scale semantics differ by source: frame_buffer stores a
        # DIVISOR (1000.0 for uint16 mm), RealSense metadata a MULTIPLIER
        # (0.001 m/unit). Handle both like crop_utils does — the first
        # version multiplied blindly and got "ring has only 0 px".
        ds = float(self._node.fb.depth_scale or 0.0)
        if ds > 1.0:
            d = raw / ds
        elif ds > 0.0:
            d = raw * ds
        else:
            d = raw / 1000.0
        ok = ring & (d > 0.15) & (d < 1.2)
        vs, us = _np.nonzero(ok)
        n_ring_px = int(len(vs))
        if n_ring_px < 400:
            self._node.get_logger().warn(
                f"[table] ring has only {n_ring_px} px — no plane this cycle")
            return self._remembered_plane("ring too thin")
        step = max(1, len(vs) // 4000)
        vs, us = vs[::step], us[::step]
        dd = d[vs, us]
        fx, fy = self._node.fb.fx, self._node.fb.fy
        cx, cy = self._node.fb.cx, self._node.fb.cy
        pts = _np.column_stack([
            (us - cx) / fx * dd, (vs - cy) / fy * dd, dd
        ]).astype(_np.float32)
        plane, why = self._fit_table_plane_and_update_memory(
            pts, obj_pts, source="ring", px_info=f"ring px {n_ring_px}")
        if plane is not None:
            return plane
        if why in ("fit failed", "sanity check failed"):
            return self._remembered_plane(why)
        return None

    def _estimate_table_plane_from_colour(
        self,
        rgb_frame,
        depth_frame,
        all_masks,
        *,
        obj_pts,
    ):
        """Measure the TABLE plane from the tabletop's own COLOUR
        (2026-08-27, Majd's design). The tabletop is uniform saturated
        YELLOW, so an HSV threshold finds it anywhere in the image —
        this is the PRIMARY table measurement; the bbox-ring stays the
        FALLBACK (the caller runs the ring whenever this returns None).

        WALL 1: subtract EVERY detected object (all_masks — the
        selected seg-mask plus bbox rectangles for all other cached
        detections) so no object pixel leaks into the fit.
        WALL 2: keep only the LARGEST connected yellow component —
        kills small yellow lookalikes (labels, caps, prints).
        Returns (plane_n, plane_p) in the camera/cloud frame with
        plane_n oriented toward the object (= up), or None.
        """
        import numpy as _np
        if rgb_frame is None or depth_frame is None or obj_pts is None or len(obj_pts) < 10:
            return None
        if rgb_frame.shape[:2] != depth_frame.shape[:2]:
            return None
        h_lo = int(self._node.get_parameter("table_colour_h_lo").value)
        h_hi = int(self._node.get_parameter("table_colour_h_hi").value)
        s_min = int(self._node.get_parameter("table_colour_s_min").value)
        v_min = int(self._node.get_parameter("table_colour_v_min").value)
        # fb frames are BGR (frame_buffer converts to bgr8) — OpenCV HSV,
        # H in 0..179.
        hsv = cv2.cvtColor(_np.asarray(rgb_frame), cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(
            hsv, (h_lo, s_min, v_min), (h_hi, 255, 255)) > 0
        # WALL 1: subtract ALL detected objects, not just the selected
        # one — any object standing on the table must not join the fit.
        for m in (all_masks or []):
            if m is not None and _np.asarray(m).shape == yellow.shape:
                yellow &= ~_np.asarray(m, dtype=bool)
        n_yellow = int(_np.count_nonzero(yellow))
        if n_yellow == 0:
            self._node.get_logger().warn(
                "[table] colour plane: no yellow found — falling back to"
                " ring")
            return None
        # WALL 2: largest connected component only (cv2 is this module's
        # resident image library; scipy.ndimage is not imported here).
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
            yellow.astype(_np.uint8), connectivity=8)
        if n_lbl <= 1:
            self._node.get_logger().warn(
                "[table] colour plane: no yellow found — falling back to"
                " ring")
            return None
        largest = 1 + int(_np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[largest, cv2.CC_STAT_AREA])
        if area < 800:
            self._node.get_logger().warn(
                f"[table] colour plane: largest region too small (<800 px):"
                f" {area}px — falling back to ring")
            return None
        region = lbl == largest
        raw = _np.asarray(depth_frame, dtype=_np.float32)
        # depth_scale semantics differ by source: frame_buffer stores a
        # DIVISOR (1000.0 for uint16 mm), RealSense metadata a MULTIPLIER
        # (0.001 m/unit). Handle both like crop_utils does — the first
        # ring version multiplied blindly and got "ring has only 0 px".
        ds = float(self._node.fb.depth_scale or 0.0)
        if ds > 1.0:
            d = raw / ds
        elif ds > 0.0:
            d = raw * ds
        else:
            d = raw / 1000.0
        ok = region & (d > 0.25) & (d < 1.5)
        n_ok = int(_np.count_nonzero(ok))
        if n_ok < 400:
            self._node.get_logger().warn(
                f"[table] colour plane: depth-invalid ({n_ok} of {area}"
                " yellow px carry usable depth) — falling back to ring")
            return None
        # EVEN sampling: uniform pixel grid, stride from region size —
        # NOT random. <=500 points is plenty for the robust SVD fit.
        stride = max(1, int(_np.ceil(_np.sqrt(n_ok / 500.0))))
        grid = _np.zeros_like(ok)
        grid[::stride, ::stride] = True
        vs, us = _np.nonzero(ok & grid)
        if len(vs) > 500:
            keep = _np.linspace(0, len(vs) - 1, 500).astype(int)
            vs, us = vs[keep], us[keep]
        if len(vs) < 3:
            self._node.get_logger().warn(
                f"[table] colour plane: depth-invalid ({len(vs)} sampled"
                " px) — falling back to ring")
            return None
        dd = d[vs, us]
        fx, fy = self._node.fb.fx, self._node.fb.fy
        cx, cy = self._node.fb.cx, self._node.fb.cy
        pts = _np.column_stack([
            (us - cx) / fx * dd, (vs - cy) / fy * dd, dd
        ]).astype(_np.float32)
        plane, why = self._fit_table_plane_and_update_memory(
            pts, obj_pts, source="COLOUR",
            px_info=f"yellow px {n_yellow} sampled {int(len(pts))}")
        if plane is None:
            self._node.get_logger().warn(
                f"[table] colour plane: {why} — falling back to ring")
            return None
        return plane

    def _collect_all_object_masks(self, selected_mask, bbox):
        """WALL 1 input for the colour table measurement (2026-08-27):
        the full per-object mask list. The detector returns bbox-only
        detections (yolo_detector.predict — there are NO per-detection
        pixel masks anywhere on the node); the ONLY true pixel mask is
        the SELECTED object's segmentation mask. So the list is: the
        selected mask (or its bbox rectangle when segmentation failed)
        plus a bbox-rectangle mask for every entry in the node's
        detections_cache. Rectangles over-subtract slightly — exactly
        right here: better to lose a strip of yellow around an object
        than to fit that object's pixels into the table plane.
        """
        import numpy as _np
        frame = self._node.fb.depth_frame if self._node.fb.depth_frame is not None else self._node.fb.rgb_frame
        if frame is None:
            return []
        H, W = frame.shape[:2]
        masks = []
        if selected_mask is not None:
            masks.append(_np.asarray(selected_mask, dtype=bool))
        elif bbox is not None:
            m = _np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
            masks.append(m)
        dets = list(getattr(self._node, "detections_cache", []) or [])
        if not dets:
            self._node.get_logger().debug(
                "[table] colour plane: detections cache empty — subtracting"
                " only the selected mask")
            return masks
        for det in dets:
            try:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            except Exception:
                continue
            x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
            if x2 > x1 and y2 > y1:
                m = _np.zeros((H, W), dtype=bool)
                m[y1:y2, x1:x2] = True
                masks.append(m)
        return masks

    def _fit_table_plane_and_update_memory(
        self,
        pts,
        obj_pts,
        *,
        source,
        px_info,
    ):
        """Shared tail of the table-plane measurement (2026-08-27):
        robust fit + toward-object orientation + object-on-plane sanity
        + the plane-memory/up-prior block, extracted VERBATIM from
        _estimate_table_plane_from_ring so the ring and the colour
        measurement share ONE copy of the logic.

        Returns ``((plane_n, plane_p), None)`` on success (fresh, or
        the remembered plane under edge/wall protection) or
        ``(None, why)`` on failure — the CALLER picks the fallback
        (ring -> _remembered_plane, colour -> the ring itself).
        """
        import numpy as _np
        try:
            n, p0 = estimate_plane_svd(pts)
        except Exception as e:
            self._node.get_logger().warn(f"[table] plane fit failed: {e}")
            return None, "fit failed"
        n = _np.asarray(n, dtype=_np.float32)
        n /= (float(_np.linalg.norm(n)) + 1e-9)
        p0 = _np.asarray(p0, dtype=_np.float32)
        centroid = _np.asarray(obj_pts, dtype=_np.float32).mean(axis=0)
        # orient toward the object: object sits ABOVE the table
        if float(_np.dot(centroid - p0, n)) < 0.0:
            n = -n
        heights = (_np.asarray(obj_pts, dtype=_np.float32) - p0) @ n
        h_min, h_max = float(heights.min()), float(heights.max())
        # sanity: the object must rest on this plane (bottom within 6 cm)
        # and stick out of it (top at least 1.5 cm above)
        if h_min < -0.08 or h_min > 0.06 or h_max < 0.015:
            self._node.get_logger().warn(
                f"[table] {source} plane REJECTED (object bottom {h_min*100:.1f}cm,"
                f" top {h_max*100:.1f}cm relative to plane) — not a support"
                " plane")
            return None, "sanity check failed"
        # ── PLANE MEMORY (2026-08-26, Majd's edge-object fix) ────────
        # The table does not move between cycles. A THIN ring whose
        # normal disagrees with the remembered plane is exactly the
        # table-edge failure (half the ring falls off the table) — use
        # the remembered plane instead. A RICH ring is trusted and
        # refreshes the memory (covers a genuinely different capture
        # pose).
        import time as _time
        cache = getattr(self._node, "_table_plane_cache", None)
        # UP-PRIOR (2026-08-26, hardening after the cardboard poisoning:
        # edge objects gave RICH rings full of WALL — normals [-0.91..]
        # sideways — and 'rich' wrongly overrode everything): in the
        # optical capture frame, a genuine tabletop normal points mostly
        # along -y (measured [-0.03,-0.99,-0.16] on every good cycle).
        # A candidate farther than ~50 deg from that is a wall, not a
        # table — never let it seed or update the memory.
        plausible_up = float(-n[1]) >= 0.64
        if cache is not None:
            cn, cp, ct = cache
            dev = float(_np.degrees(_np.arccos(
                min(1.0, abs(float(_np.dot(n, cn)))))))
            if dev > 20.0 or not plausible_up:
                self._node.get_logger().warn(
                    f"[table] fresh plane rejected (dev {dev:.0f}deg from "
                    f"memory, plausible_up={plausible_up}, {px_info}) — "
                    "using the REMEMBERED plane (edge/wall protection).")
                return (cn, cp), None
        elif not plausible_up:
            self._node.get_logger().warn(
                f"[table] plane normal [{n[0]:+.2f} {n[1]:+.2f} {n[2]:+.2f}]"
                f" is not a plausible tabletop ({px_info}) and no "
                "memory exists — falling back (likely a wall at the edge).")
            return None, "not plausible up"
        self._node._table_plane_cache = (n.copy(), p0.copy(), _time.time())
        self._node.get_logger().info(
            f"[table] plane from {source}: n=[{n[0]:+.2f} {n[1]:+.2f} {n[2]:+.2f}]"
            f" (cloud frame), object height {max(0.0, h_max)*100:.1f}cm,"
            f" bottom gap {h_min*100:.1f}cm, {px_info}"
            " [memory updated]")
        return (n, p0), None

    def _remembered_plane(self, why: str):
        '''Cached table plane when the fresh measurement is unusable
        (age-capped 30 min), else None.'''
        import time as _time
        cache = getattr(self._node, "_table_plane_cache", None)
        if cache is None:
            return None
        cn, cp, ct = cache
        if _time.time() - ct > 1800.0:
            self._node.get_logger().warn(
                f"[table] {why} and the remembered plane is older than "
                "30 min — not using it.")
            return None
        self._node.get_logger().warn(
            f"[table] {why} — using the REMEMBERED plane "
            f"(age {_time.time() - ct:.0f}s).")
        return cn, cp

    def _make_tracking_roi(
        self,
        obs_tf: Dict[str, Any],
        reg_cfg: RegistrationConfig,
    ) -> Dict[str, Any]:
        pts = np.asarray(obs_tf["points"], dtype=np.float32)
        aabb_min = np.min(pts, axis=0)
        aabb_max = np.max(pts, axis=0)
        center = 0.5 * (aabb_min + aabb_max)
        half_extent = 0.5 * (aabb_max - aabb_min)
        margin = float(self._node.get_parameter("registration_roi_margin_m").value)
        min_half = float(self._node.get_parameter("registration_roi_min_half_extent_m").value)
        ref_voxel = float(self._node.get_parameter("registration_roi_fallback_reference_voxel_m").value)
        ref_points, _ = voxel_downsample(pts, None, ref_voxel)
        half_extent = np.maximum(half_extent + margin, min_half).astype(np.float32)
        return {
            "target_frame": reg_cfg.target_frame,
            "center": center.astype(np.float32),
            "half_extent": half_extent,
            "reference_half_extent": half_extent.copy(),
            "reference_point_count": int(len(pts)),
            "reference_bbox_xyxy": [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])],
            "reference_bbox_area_px": int(
                max(1, (int(obs_tf["bbox"][2]) - int(obs_tf["bbox"][0])) * (int(obs_tf["bbox"][3]) - int(obs_tf["bbox"][1])))
            ),
            "reference_points": np.asarray(ref_points, dtype=np.float32),
        }

    @staticmethod
    def _mask_points_near_reference(
        points: np.ndarray,
        reference_points: np.ndarray,
        max_distance_m: float,
        *,
        chunk_size: int = 1024,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        ref = np.asarray(reference_points, dtype=np.float32)
        if len(pts) == 0 or len(ref) == 0:
            return np.zeros((len(pts),), dtype=bool)
        max_d2 = float(max_distance_m) ** 2
        keep = np.zeros((len(pts),), dtype=bool)
        for start in range(0, len(pts), chunk_size):
            chunk = pts[start:start + chunk_size]
            diff = chunk[:, None, :] - ref[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            keep[start:start + chunk_size] = np.min(d2, axis=1) <= max_d2
        return keep

    def _roi_fallback_is_plausible(
        self,
        obs_tf: Dict[str, Any],
        tracked_roi: Dict[str, Any],
    ) -> bool:
        pts = np.asarray(obs_tf["points"], dtype=np.float32)
        if len(pts) == 0:
            return False
        aabb_min = np.min(pts, axis=0)
        aabb_max = np.max(pts, axis=0)
        obs_half_extent = 0.5 * (aabb_max - aabb_min)
        ref_half_extent = np.asarray(
            tracked_roi.get("reference_half_extent", tracked_roi["half_extent"]),
            dtype=np.float32,
        )
        ref_point_count = max(1, int(tracked_roi.get("reference_point_count", len(pts))))
        point_ratio = float(len(pts)) / float(ref_point_count)
        extent_ratio = float(np.max(obs_half_extent / np.maximum(ref_half_extent, 1e-6)))
        max_point_ratio = float(self._node.get_parameter("registration_roi_fallback_max_point_ratio").value)
        max_extent_ratio = float(self._node.get_parameter("registration_roi_fallback_max_extent_ratio").value)
        if point_ratio > max_point_ratio or extent_ratio > max_extent_ratio:
            self._node.get_logger().warn(
                f"[registration] ROI fallback rejected as contaminated "
                f"(point_ratio={point_ratio:.2f}, extent_ratio={extent_ratio:.2f}, "
                f"max_point_ratio={max_point_ratio:.2f}, max_extent_ratio={max_extent_ratio:.2f})."
            )
            return False
        return True

    def _tracked_followup_is_plausible(
        self,
        obs_tf: Dict[str, Any],
        tracked_roi: Dict[str, Any],
    ) -> bool:
        pts = np.asarray(obs_tf["points"], dtype=np.float32)
        if len(pts) == 0:
            return False
        aabb_min = np.min(pts, axis=0)
        aabb_max = np.max(pts, axis=0)
        obs_half_extent = 0.5 * (aabb_max - aabb_min)
        ref_half_extent = np.asarray(
            tracked_roi.get("reference_half_extent", tracked_roi["half_extent"]),
            dtype=np.float32,
        )
        ref_point_count = max(1, int(tracked_roi.get("reference_point_count", len(pts))))
        point_ratio = float(len(pts)) / float(ref_point_count)
        extent_ratio = float(np.max(obs_half_extent / np.maximum(ref_half_extent, 1e-6)))
        centroid = np.mean(pts, axis=0)
        tracked_center = np.asarray(tracked_roi["center"], dtype=np.float32)
        centroid_offset = float(np.linalg.norm(centroid - tracked_center))
        bbox = [int(v) for v in obs_tf.get("bbox", [0, 0, 0, 0])]
        bbox_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
        ref_bbox_area = max(1, int(tracked_roi.get("reference_bbox_area_px", bbox_area if bbox_area > 0 else 1)))
        bbox_area_ratio = float(bbox_area) / float(ref_bbox_area)

        max_point_ratio = float(self._node.get_parameter("registration_tracked_followup_max_point_ratio").value)
        max_extent_ratio = float(self._node.get_parameter("registration_tracked_followup_max_extent_ratio").value)
        max_centroid_offset = float(self._node.get_parameter("registration_tracked_followup_max_centroid_offset_m").value)
        max_bbox_area_ratio = float(self._node.get_parameter("registration_tracked_followup_max_bbox_area_ratio").value)

        # 3D metrics are authoritative; 2D bbox inflates naturally when camera
        # viewpoint changes significantly (perspective projection artifact).
        three_d_fail = (
            point_ratio > max_point_ratio
            or extent_ratio > max_extent_ratio
            or centroid_offset > max_centroid_offset
        )
        # Only use bbox_area_ratio as a hard reject when it is extreme (>10×)
        # even if 3D metrics are fine, or when 3D metrics also fail.
        # Large viewpoint changes (e.g. 90° rotation) can inflate the projected
        # 2D bbox 10-15× without any real contamination.
        bbox_extreme = bbox_area_ratio > max_bbox_area_ratio * 10.0
        bbox_elevated = bbox_area_ratio > max_bbox_area_ratio

        point_instability = abs(point_ratio - 1.0)
        extent_instability = abs(extent_ratio - 1.0)
        centroid_instability = centroid_offset / max(max_centroid_offset, 1e-6)
        bbox_instability = max(0.0, (bbox_area_ratio / max(max_bbox_area_ratio, 1e-6)) - 1.0)
        instability_score = float(np.mean([
            point_instability,
            extent_instability,
            centroid_instability,
            bbox_instability,
        ]))
        obs_tf["tracking_instability_score"] = instability_score
        obs_tf["tracking_instability_metrics"] = {
            "point_ratio": float(point_ratio),
            "extent_ratio": float(extent_ratio),
            "centroid_offset_m": float(centroid_offset),
            "bbox_area_ratio": float(bbox_area_ratio),
            "instability_score": float(instability_score),
        }

        rejected = three_d_fail or bbox_extreme or (bbox_elevated and (
            point_ratio > max_point_ratio * 0.8
            or extent_ratio > max_extent_ratio * 0.8
            or centroid_offset > max_centroid_offset * 0.8
        ))

        self._node.get_logger().info(
            f"[tracking] plausibility check: point_ratio={point_ratio:.2f}, "
            f"extent_ratio={extent_ratio:.2f}, centroid_offset={centroid_offset:.3f} m, "
            f"bbox_area_ratio={bbox_area_ratio:.2f}, instability={instability_score:.3f} "
            f"(limits={max_point_ratio:.2f}/{max_extent_ratio:.2f}/"
            f"{max_centroid_offset:.3f}/{max_bbox_area_ratio:.2f})"
        )
        if rejected:
            self._node.get_logger().warn(
                f"[tracking] tracked follow-up rejected as contaminated "
                f"(3d_fail={three_d_fail}, bbox_extreme={bbox_extreme}, "
                f"bbox_elevated={bbox_elevated})."
            )
            return False
        return True

    def _extract_followup_object_cloud_roi(
        self,
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
    ) -> Optional[Dict[str, Any]]:
        if self._node.fb.depth_frame is None or self._node.fb.rgb_frame is None:
            return None

        H, W = self._node.fb.depth_frame.shape[:2]
        capture_header = copy.deepcopy(self._node.fb.depth_header if self._node.fb.depth_header is not None else self._node.fb.rgb_header)
        if capture_header is None:
            return None

        full_cloud_msg, pts_cam, cols_np = crop_depth_to_cloud_and_numpy(
            self._node.fb.depth_frame,
            self._node.fb.rgb_frame,
            [0, 0, W, H],
            capture_header,
            self._node.fb.fx,
            self._node.fb.fy,
            self._node.fb.cx,
            self._node.fb.cy,
            self._node.fb.depth_scale,
            mask_full=None,
        )
        if full_cloud_msg is None or pts_cam is None or pts_cam.size == 0:
            return None

        T_target_source = lookup_transform_matrix(
            self._node.tf_buffer,
            reg_cfg.target_frame,
            full_cloud_msg.header.frame_id,
            capture_header.stamp,
            reg_cfg.tf_timeout_s,
            logger=self._node.get_logger(),
        )
        pts_target = transform_points(pts_cam, T_target_source)

        center = np.asarray(tracked_roi["center"], dtype=np.float32)
        half_extent = np.asarray(tracked_roi["half_extent"], dtype=np.float32)
        keep = np.all(np.abs(pts_target - center.reshape(1, 3)) <= half_extent.reshape(1, 3), axis=1)
        if not np.any(keep):
            self._node.get_logger().warn("[registration] ROI fallback crop kept no points in target frame.")
            return None

        pts_target = pts_target[keep]
        cols_keep = None if cols_np is None else np.asarray(cols_np)[keep]
        pts_cam_keep = np.asarray(pts_cam, dtype=np.float32)[keep]

        ref_points = np.asarray(tracked_roi.get("reference_points", np.empty((0, 3), dtype=np.float32)), dtype=np.float32)
        max_surface_distance = float(self._node.get_parameter("registration_roi_fallback_max_surface_distance_m").value)
        if len(ref_points) > 0 and max_surface_distance > 0.0:
            near_mask = self._mask_points_near_reference(pts_target, ref_points, max_surface_distance)
            if not np.any(near_mask):
                self._node.get_logger().warn(
                    "[registration] ROI fallback shape filter removed all points; rejecting fallback crop."
                )
                return None
            pts_target = pts_target[near_mask]
            if cols_keep is not None:
                cols_keep = cols_keep[near_mask]
            pts_cam_keep = pts_cam_keep[near_mask]

        valid_z = pts_cam_keep[:, 2] > 1e-6
        if not np.any(valid_z):
            return None
        proj = pts_cam_keep[valid_z]
        us = (self._node.fb.fx * (proj[:, 0] / proj[:, 2]) + self._node.fb.cx).astype(np.float32)
        vs = (self._node.fb.fy * (proj[:, 1] / proj[:, 2]) + self._node.fb.cy).astype(np.float32)
        bbox = [
            int(np.clip(np.floor(np.min(us)), 0, W - 1)),
            int(np.clip(np.floor(np.min(vs)), 0, H - 1)),
            int(np.clip(np.ceil(np.max(us)), 0, W)),
            int(np.clip(np.ceil(np.max(vs)), 0, H)),
        ]

        filtered = self._filter_object_cloud(pts_target, cols_keep, apply_depth_clamp=False)
        if filtered is None:
            return None
        pts_target, cols_keep, plane_n, plane_p = filtered

        header_tf = copy.deepcopy(capture_header)
        header_tf.frame_id = reg_cfg.target_frame
        return {
            "cloud_msg": None,
            "points": pts_target,
            "colors": cols_keep,
            "header": header_tf,
            "bbox": bbox,
            "plane_n": plane_n,
            "plane_p": plane_p,
            "capture_transform": T_target_source.astype(np.float32),
            "source_frame": str(full_cloud_msg.header.frame_id),
            "used_roi_fallback": True,
            "depth_frame": self._node.fb.depth_frame.copy(),
            "rgb_frame": self._node.fb.rgb_frame.copy(),
            "selected_mask": None,
            "intrinsics": {
                "fx": float(self._node.fb.fx),
                "fy": float(self._node.fb.fy),
                "cx": float(self._node.fb.cx),
                "cy": float(self._node.fb.cy),
                "width": int(W),
                "height": int(H),
            },
            "depth_scale": float(self._node.fb.depth_scale),
            "tsdf_eligible": False,
        }

    def _project_tracking_roi_to_bbox(
        self,
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        *,
        use_latest_tf: bool = False,
    ) -> Optional[List[int]]:
        if self._node.fb.depth_frame is None:
            return None
        capture_header = copy.deepcopy(self._node.fb.depth_header if self._node.fb.depth_header is not None else self._node.fb.rgb_header)
        if capture_header is None:
            return None
        source_frame = str(capture_header.frame_id)
        if not source_frame:
            return None

        # Prefer projecting actual reference points (tight bbox) over box
        # corners (loose bbox).  Reference points are the voxel-downsampled
        # cloud from the last accepted observation — projecting them gives a
        # 2D footprint that matches the real object shape, not a blown-up
        # bounding box.
        ref_pts = tracked_roi.get("reference_points")
        if ref_pts is not None and len(ref_pts) >= 4:
            pts_target = np.asarray(ref_pts, dtype=np.float32)
        else:
            center = np.asarray(tracked_roi["center"], dtype=np.float32)
            half_extent = np.asarray(tracked_roi["half_extent"], dtype=np.float32)
            corners = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corners.append(center + np.array([sx, sy, sz], dtype=np.float32) * half_extent)
            pts_target = np.asarray(corners, dtype=np.float32)

        # Use latest available TF when called from preview path (avoids
        # extrapolation errors during fast motion).
        tf_time = rclpy.time.Time() if use_latest_tf else capture_header.stamp

        try:
            T_source_target = lookup_transform_matrix(
                self._node.tf_buffer,
                source_frame,
                reg_cfg.target_frame,
                tf_time,
                reg_cfg.tf_timeout_s,
                logger=None,
            )
        except Exception as e:
            self._node.get_logger().warn(f"[tracking] failed to project ROI into '{source_frame}': {e}")
            return None

        pts_cam = transform_points(pts_target, T_source_target)
        valid = pts_cam[:, 2] > 1e-6
        if not np.any(valid):
            return None
        pts_cam = pts_cam[valid]

        H, W = self._node.fb.depth_frame.shape[:2]
        us = self._node.fb.fx * (pts_cam[:, 0] / pts_cam[:, 2]) + self._node.fb.cx
        vs = self._node.fb.fy * (pts_cam[:, 1] / pts_cam[:, 2]) + self._node.fb.cy
        # Tight margin: just enough for mask dilation + a small buffer
        margin_px = int(max(8.0, float(self._node.get_parameter("segmentation_mask_dilate_px").value) * 2.0))
        x1 = int(np.clip(np.floor(np.min(us)) - margin_px, 0, W - 1))
        y1 = int(np.clip(np.floor(np.min(vs)) - margin_px, 0, H - 1))
        x2 = int(np.clip(np.ceil(np.max(us)) + margin_px, 0, W))
        y2 = int(np.clip(np.ceil(np.max(vs)) + margin_px, 0, H))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _project_reference_points_to_mask(
        self,
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
    ) -> Optional[tuple[List[int], np.ndarray]]:
        if self._node.fb.depth_frame is None:
            return None
        capture_header = copy.deepcopy(self._node.fb.depth_header if self._node.fb.depth_header is not None else self._node.fb.rgb_header)
        if capture_header is None:
            return None
        source_frame = str(capture_header.frame_id)
        if not source_frame:
            return None

        ref_pts = np.asarray(
            tracked_roi.get("reference_points", np.empty((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        if len(ref_pts) < 4:
            return None

        try:
            T_source_target = lookup_transform_matrix(
                self._node.tf_buffer,
                source_frame,
                reg_cfg.target_frame,
                capture_header.stamp,
                reg_cfg.tf_timeout_s,
                logger=self._node.get_logger(),
            )
        except Exception as e:
            self._node.get_logger().warn(f"[tracking] failed to project reference mask into '{source_frame}': {e}")
            return None

        pts_cam = transform_points(ref_pts, T_source_target)
        valid = pts_cam[:, 2] > 1e-6
        if not np.any(valid):
            return None
        pts_cam = pts_cam[valid]

        H, W = self._node.fb.depth_frame.shape[:2]
        us = self._node.fb.fx * (pts_cam[:, 0] / pts_cam[:, 2]) + self._node.fb.cx
        vs = self._node.fb.fy * (pts_cam[:, 1] / pts_cam[:, 2]) + self._node.fb.cy
        valid_uv = (
            np.isfinite(us) & np.isfinite(vs)
            & (us >= 0.0) & (us < float(W))
            & (vs >= 0.0) & (vs < float(H))
        )
        if not np.any(valid_uv):
            return None
        us = us[valid_uv]
        vs = vs[valid_uv]

        mask = np.zeros((H, W), dtype=np.uint8)
        try:
            import cv2

            pts_uv = np.stack([us, vs], axis=1).astype(np.int32)
            if len(pts_uv) >= 3:
                hull = cv2.convexHull(pts_uv.reshape(-1, 1, 2))
                cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 255)
            for u, v in pts_uv:
                cv2.circle(mask, (int(u), int(v)), 2, 255, thickness=-1)

            dilate_px = int(self._node.get_parameter("registration_projected_mask_dilate_px").value)
            if dilate_px > 0:
                k = max(1, 2 * dilate_px + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = cv2.dilate(mask, kernel, iterations=1)
        except Exception as e:
            self._node.get_logger().warn(f"[tracking] projected reference mask rasterization failed: {e}")
            return None

        mask_bool = mask > 0
        if not np.any(mask_bool):
            return None
        ys, xs = np.nonzero(mask_bool)
        bbox = [
            int(np.min(xs)),
            int(np.min(ys)),
            int(np.max(xs) + 1),
            int(np.max(ys) + 1),
        ]
        return bbox, mask_bool

    def _extract_object_cloud_with_mask(
        self,
        capture_sel: Dict[str, Any],
        selected_mask: Optional[np.ndarray],
        *,
        publish_outputs: bool,
        mask_label: str,
    ) -> Optional[Dict[str, Any]]:
        if publish_outputs:
            self._node._publish_detection_message(capture_sel)
            x1, y1, x2, y2 = capture_sel["bbox"]
            self._node.get_logger().info(
                f"Published selected '{capture_sel['label']}' bbox=({x1},{y1},{x2},{y2}) | depth=yes"
            )
        if selected_mask is not None:
            self._node._publish_selected_mask_outputs(
                capture_sel,
                selected_mask,
                publish_outputs=publish_outputs,
                mask_label=mask_label,
            )
            try:
                x1, y1, x2, y2 = [int(v) for v in capture_sel["bbox"]]
                H, W = self._node.fb.depth_frame.shape[:2]
                x1 = max(0, min(W - 1, x1))
                x2 = max(0, min(W, x2))
                y1 = max(0, min(H - 1, y1))
                y2 = max(0, min(H, y2))
                if x2 > x1 and y2 > y1:
                    z_roi = self._node.fb.depth_frame[y1:y2, x1:x2]
                    valid_depth_roi = z_roi > 0
                    mask_roi = selected_mask[y1:y2, x1:x2]
                    overlap = int(np.count_nonzero(valid_depth_roi & mask_roi))
                    valid_bbox = int(np.count_nonzero(valid_depth_roi))
                    mask_px = int(np.count_nonzero(mask_roi))
                    self._node.get_logger().info(
                        f"[seg-depth/{mask_label}] bbox_valid_depth_px={valid_bbox}, "
                        f"mask_px={mask_px}, overlap_px={overlap}"
                    )
            except Exception as e:
                self._node.get_logger().warn(f"Seg-depth diagnostics failed for {mask_label}: {e}")

        # Stationary capture: optionally fuse several depth frames (item 4).
        depth_for_capture = self._capture_depth_multiframe()
        # Padded collision boxes for the OTHER objects on the table
        # (2026-08-27) — the target itself is deliberately left out.
        # Gated on publish_outputs: the real pick capture passes True
        # (pick_pipeline), the multi-view sweep passes False — so the
        # boxes are rebuilt once per PICK, not once per view.
        if publish_outputs:
            publish_neighbour_obstacles(
                self._node, capture_sel, depth_for_capture)
        cloud_msg, pts_np, cols_np = crop_depth_to_cloud_and_numpy(
            depth_for_capture,
            self._node.fb.rgb_frame,
            capture_sel["bbox"],
            capture_sel["header"],
            self._node.fb.fx,
            self._node.fb.fy,
            self._node.fb.cx,
            self._node.fb.cy,
            self._node.fb.depth_scale,
            mask_full=selected_mask,
        )
        if (cloud_msg is None or pts_np is None or pts_np.size == 0) and selected_mask is not None:
            self._node.get_logger().warn(
                f"{mask_label} crop has no valid depth points; retrying with bbox-only crop."
            )
            cloud_msg, pts_np, cols_np = crop_depth_to_cloud_and_numpy(
                depth_for_capture,
                self._node.fb.rgb_frame,
                capture_sel["bbox"],
                capture_sel["header"],
                self._node.fb.fx,
                self._node.fb.fy,
                self._node.fb.cx,
                self._node.fb.cy,
                self._node.fb.depth_scale,
                mask_full=None,
            )
        if cloud_msg is None or pts_np is None or pts_np.size == 0:
            return None

        cloud_msg.header = capture_sel["header"]
        if publish_outputs:
            self._node.pub_cloud.publish(cloud_msg)
            self._node.last_selected_cloud_msg = copy.deepcopy(cloud_msg)
            self._node.get_logger().info(f"Published cropped PointCloud2 ({pts_np.shape} pts).")
            # RGB twin for the viewers (2026-08-07): cols_np is already
            # computed above — publish a coloured copy, and in single-view
            # mode also feed the frames window (fused_cloud) so it shows the
            # one captured frame; multiview overwrites it with the real
            # accumulated preview during the sweep.
            try:
                if cols_np is not None and len(cols_np) == len(pts_np):
                    # cols_np comes from the OpenCV frame buffer = BGR order;
                    # swap to RGB or yellow renders as cyan (2026-08-07).
                    rgb_msg = self._node._cloud_pub.create_cloud_xyzrgb(
                        cloud_msg.header, pts_np, np.ascontiguousarray(cols_np[:, ::-1])
                    )
                    self._node.pub_cloud_rgb.publish(rgb_msg)
                    self._node.last_selected_cloud_rgb_msg = rgb_msg
            except Exception as e:
                self._node.get_logger().warn(f"RGB cloud publish failed: {e}")
            self._node.get_logger().info(
                f"[diag] selected_cloud frame={cloud_msg.header.frame_id}, "
                f"K=({self._node.fb.fx:.1f},{self._node.fb.fy:.1f},{self._node.fb.cx:.1f},{self._node.fb.cy:.1f}), "
                f"depth_scale={self._node.fb.depth_scale}, bbox={capture_sel['bbox']}, "
                f"z_range=({float(pts_np[:,2].min()):.3f},{float(pts_np[:,2].max()):.3f}) m"
            )

        filtered = self._filter_object_cloud(pts_np, cols_np)
        if filtered is None:
            return None
        pts_np, cols_np, plane_n, plane_p = filtered
        # ── TABLE plane from the ring AROUND the object (2026-08-24) ──
        # The thesis (§4.6.7) fits the support plane where the crop still
        # contains table pixels; a tight seg-mask leaves none, so the old
        # plane fit either never ran (apply_plane_above_filter=False) or
        # fitted the OBJECT's own face. Measure the actual tabletop from
        # the non-object pixels around/below the bbox instead. This plane
        # defines "up" for the whole filter cascade — camera-tilt-proof
        # and TF-free (Majd: measure "up" from the table in the picture).
        # COLOUR first (2026-08-27): the yellow tabletop itself is the
        # PRIMARY plane source; the ring below stays the fallback with
        # its own memory/TF ladder untouched. fb.rgb_frame is the same
        # capture this call site already pairs with depth_for_capture.
        ring = None
        if bool(self._node.get_parameter("table_from_colour").value):
            try:
                ring = self._estimate_table_plane_from_colour(
                    self._node.fb.rgb_frame, depth_for_capture,
                    self._collect_all_object_masks(
                        selected_mask, capture_sel["bbox"]),
                    obj_pts=pts_np)
            except Exception as e:
                self._node.get_logger().warn(f"[table] colour plane failed: {e}")
        try:
            if ring is None:
                ring = self._estimate_table_plane_from_ring(
                    depth_for_capture, capture_sel["bbox"], selected_mask,
                    obj_pts=pts_np)
            if ring is not None:
                plane_n, plane_p = ring
        except Exception as e:
            self._node.get_logger().warn(f"[table] ring plane failed: {e}")
        h, w = self._node.fb.depth_frame.shape[:2]
        return {
            "cloud_msg": cloud_msg,
            "points": pts_np,
            "colors": cols_np,
            "header": cloud_msg.header,
            "bbox": [int(v) for v in capture_sel["bbox"]],
            "plane_n": plane_n,
            "plane_p": plane_p,
            # The TSDF integrates obs["depth_frame"] — store the (possibly
            # median-fused) capture depth, not the raw live frame.
            "depth_frame": np.asarray(depth_for_capture).copy(),
            "rgb_frame": self._node.fb.rgb_frame.copy(),
            "selected_mask": None if selected_mask is None else np.asarray(selected_mask, dtype=bool).copy(),
            "intrinsics": {
                "fx": float(self._node.fb.fx),
                "fy": float(self._node.fb.fy),
                "cx": float(self._node.fb.cx),
                "cy": float(self._node.fb.cy),
                "width": int(w),
                "height": int(h),
            },
            "depth_scale": float(self._node.fb.depth_scale),
            "tsdf_eligible": True,
        }

    def _build_followup_tracked_selection_and_mask(
        self,
        sel: Dict[str, Any],
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        *,
        log_details: bool,
    ) -> Optional[tuple[Dict[str, Any], Optional[np.ndarray], str, bool]]:
        bbox = self._project_tracking_roi_to_bbox(tracked_roi, reg_cfg)
        if bbox is None:
            return None
        capture_header = copy.deepcopy(self._node.fb.depth_header if self._node.fb.depth_header is not None else self._node.fb.rgb_header)
        if capture_header is None:
            capture_header = copy.deepcopy(sel["header"])
        tracked_sel = {
            "label": sel["label"],
            "score": float(sel.get("score", 1.0)),
            "bbox": bbox,
            "header": capture_header,
        }
        projected_prior = None
        use_projected_prior = bool(self._node.get_parameter("registration_followup_use_projected_mask_prior").value)
        if use_projected_prior:
            projected_prior = self._project_reference_points_to_mask(tracked_roi, reg_cfg)

        selected_mask = None
        mask_label = "tracked-seg"
        segmentation_mask = self._node._compute_selected_mask(tracked_sel, publish_outputs=False)
        if projected_prior is not None:
            prior_bbox, prior_mask = projected_prior
            tracked_sel["bbox"] = [int(v) for v in prior_bbox]
            selected_mask = prior_mask
            mask_label = "projected-prior"
            if log_details:
                self._node.get_logger().info(
                    f"[tracking] projected mask prior ready: bbox={prior_bbox}, area_px={int(np.count_nonzero(prior_mask))}"
                )
            if segmentation_mask is not None:
                overlap = int(np.count_nonzero(prior_mask & segmentation_mask))
                seg_area = max(1, int(np.count_nonzero(segmentation_mask)))
                overlap_ratio = float(overlap) / float(seg_area)
                min_overlap_ratio = float(self._node.get_parameter("registration_projected_mask_min_overlap_ratio").value)
                if overlap_ratio >= min_overlap_ratio:
                    selected_mask = np.asarray(prior_mask & segmentation_mask, dtype=bool)
                    mask_label = "projected+seg"
                    if log_details:
                        self._node.get_logger().info(
                            f"[tracking] combined projected prior with fresh segmentation "
                            f"(overlap_ratio={overlap_ratio:.2f})."
                        )
                elif log_details:
                    self._node.get_logger().warn(
                        f"[tracking] fresh segmentation disagrees with projected prior "
                        f"(overlap_ratio={overlap_ratio:.2f} < {min_overlap_ratio:.2f}); using projected prior."
                    )
        elif segmentation_mask is not None:
            selected_mask = segmentation_mask
            mask_label = "tracked-seg"

        return tracked_sel, selected_mask, mask_label, bool(projected_prior is not None)

    def _publish_followup_tracking_preview(
        self,
        sel: Dict[str, Any],
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
    ) -> bool:
        # ── Fast bbox-only preview: project 3D ROI to current camera frame ──
        bbox = self._project_tracking_roi_to_bbox(tracked_roi, reg_cfg, use_latest_tf=True)
        if bbox is None:
            self._node.get_logger().debug("[tracking-preview] ROI projection returned None — skipping")
            return False

        capture_header = copy.deepcopy(
            self._node.fb.depth_header if self._node.fb.depth_header is not None else self._node.fb.rgb_header
        )
        if capture_header is None:
            return False

        tracked_sel = {
            "label": sel["label"],
            "score": float(sel.get("score", 1.0)),
            "bbox": bbox,
            "header": capture_header,
        }

        # Update the debug det so the timer-driven debug frame picks it up
        self._node._publish_detection_message(tracked_sel)

        # Attempt segmentation mask — but don't let it block the preview
        selected_mask = None
        try:
            selected_mask = self._node._compute_selected_mask(tracked_sel, publish_outputs=False)
        except Exception:
            pass

        if selected_mask is not None:
            self._node._publish_selected_mask_outputs(tracked_sel, selected_mask, publish_outputs=True)
        else:
            # No mask available; still repaint the local debug window from the
            # motion loop because timer callbacks are blocked during the sweep.
            if bool(self._node.get_parameter("publish_debug_image").value):
                self._node.debug_pub.publish(
                    tracked_sel["header"],
                    self._node.fb.rgb_frame,
                    [{"label": tracked_sel["label"], "score": float(tracked_sel["score"]), "bbox": tracked_sel["bbox"]}],
                    selected_idx=0,
                    pointer_px=self._node._pointer_px,
                    hand_landmarks_px=self._node._hand_landmarks_px,
                    hand_connections=self._node._hand_connections,
                    selected_mask=None,
                    status_text="debug: projected tracking bbox only\nno mask for this frame",
                    render_window=False,
                )
        return True

    def _extract_followup_object_cloud_tracked(
        self,
        sel: Dict[str, Any],
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
    ) -> Optional[Dict[str, Any]]:
        built = self._build_followup_tracked_selection_and_mask(
            sel,
            tracked_roi,
            reg_cfg,
            log_details=True,
        )
        if built is None:
            return None
        tracked_sel, selected_mask, mask_label, used_projected_prior = built

        obs = self._extract_object_cloud_with_mask(
            tracked_sel,
            selected_mask,
            publish_outputs=True,
            mask_label=mask_label,
        )
        if obs is None:
            return None
        obs["used_tracked_segmentation"] = True
        obs["used_projected_mask_prior"] = bool(used_projected_prior)
        return obs


    @staticmethod
    def _bbox_iou_xyxy(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = area_a + area_b - inter
        if denom <= 1e-9:
            return 0.0
        return inter / denom

    def _get_active_image_size(self) -> tuple[int, int]:
        if self._node.fb.rgb_frame is not None:
            h, w = self._node.fb.rgb_frame.shape[:2]
            return int(w), int(h)
        if self._node.fb.depth_frame is not None:
            h, w = self._node.fb.depth_frame.shape[:2]
            return int(w), int(h)
        w = int(round(float(self._node.fb.cx or 0.0) * 2.0))
        h = int(round(float(self._node.fb.cy or 0.0) * 2.0))
        return max(1, w), max(1, h)

    def _evaluate_continuous_bbox_framing(
        self,
        bbox: List[int],
        tracked_roi: Dict[str, Any],
    ) -> Dict[str, Any]:
        img_w, img_h = self._get_active_image_size()
        margin_px = max(
            0,
            int(self._node.get_parameter("registration_continuous_observation_bbox_border_margin_px").value),
        )
        min_area_ratio = max(
            0.0,
            float(self._node.get_parameter("registration_continuous_observation_min_edge_bbox_area_ratio").value),
        )
        min_height_ratio = max(
            0.0,
            float(self._node.get_parameter("registration_continuous_observation_min_edge_bbox_height_ratio").value),
        )
        reject_bottom_edge = bool(
            self._node.get_parameter("registration_continuous_observation_reject_bottom_edge").value
        )

        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        bbox_area = max(1, bbox_w * bbox_h)
        ref_bbox = [int(v) for v in tracked_roi.get("reference_bbox_xyxy", bbox)]
        ref_area = max(
            1,
            int(tracked_roi.get("reference_bbox_area_px", max(1, (ref_bbox[2] - ref_bbox[0]) * (ref_bbox[3] - ref_bbox[1])))),
        )
        ref_height = max(1, ref_bbox[3] - ref_bbox[1])

        area_ratio = float(bbox_area) / float(ref_area)
        height_ratio = float(bbox_h) / float(ref_height)
        touches_left = x1 <= margin_px
        touches_top = y1 <= margin_px
        touches_right = x2 >= (img_w - margin_px)
        touches_bottom = y2 >= (img_h - margin_px)
        touches_any = touches_left or touches_top or touches_right or touches_bottom
        vertical_edge = touches_top or touches_bottom
        reject = touches_any and (area_ratio < min_area_ratio or height_ratio < min_height_ratio)
        reject = reject or (reject_bottom_edge and touches_bottom)
        stop_now = vertical_edge and (
            area_ratio < (0.5 * min_area_ratio)
            or height_ratio < (0.5 * min_height_ratio)
        )
        return {
            "image_size": (img_w, img_h),
            "margin_px": int(margin_px),
            "area_ratio": float(area_ratio),
            "height_ratio": float(height_ratio),
            "touches_left": bool(touches_left),
            "touches_top": bool(touches_top),
            "touches_right": bool(touches_right),
            "touches_bottom": bool(touches_bottom),
            "touches_any": bool(touches_any),
            "reject": bool(reject),
            "stop_now": bool(stop_now),
        }
