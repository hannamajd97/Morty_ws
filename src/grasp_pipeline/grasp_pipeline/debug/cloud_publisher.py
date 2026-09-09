"""Point-cloud packing + RViz preview/TSDF publishing.

Pulled out of ``YoloLiveSelectGrasp`` so the orchestrator stops carrying
PointField bit-twiddling and TSDF integration plumbing. The publishers
themselves (``/graspgen/fused_cloud``, ``/graspgen/tsdf_cloud``) are
created by the orchestrator and handed to this class — that keeps
topic-name parameters in one place (the node) and keeps this module free
of param plumbing where it isn't needed.

Public surface used by the orchestrator:

    create_cloud_xyzrgb(header, points, colors) -> PointCloud2
    colors_to_rgb_float(colors) -> np.ndarray
    build_tsdf_config() -> TSDFConfig
    publish_registration_preview(views, reg_cfg) -> None
    publish_tsdf_cloud(accepted_obs_meta, reg_cfg) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]
    publish_incremental_tsdf_cloud(seed_obs_meta, accepted_obs_meta, reg_cfg) -> None
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from std_msgs.msg import String

from ..registration import voxel_downsample
from ..tsdf_fusion import TSDFConfig, integrate_tsdf_surface_cloud

if TYPE_CHECKING:
    from rclpy.node import Node
    from rclpy.publisher import Publisher
    from ..registration import RegistrationConfig, RegistrationView


class CloudPublisher:
    """Produce PointCloud2 messages for the registration preview and TSDF."""

    def __init__(
        self,
        node: "Node",
        fused_pub: "Publisher",
        tsdf_pub: "Publisher",
    ) -> None:
        self._node = node
        self._fused_pub = fused_pub
        self._tsdf_pub = tsdf_pub
        self._status_pub = node.create_publisher(String, "/graspgen/cloud_status", 10)
        self._last_object_label = ""

    def _publish_cloud_status(
        self,
        *,
        topic: str,
        stage: str,
        object_label: str,
        point_count: int,
        view_count: int,
    ) -> None:
        label = str(object_label or self._last_object_label or "object")
        self._last_object_label = label
        try:
            self._status_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "topic": topic,
                            "stage": stage,
                            "object": label,
                            "points": int(point_count),
                            "views": int(view_count),
                        },
                        separators=(",", ":"),
                    )
                )
            )
        except Exception as exc:
            self._node.get_logger().warn(f"Failed to publish cloud status: {exc}")

    # ------------------------------------------------------------------
    # Static-ish encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def colors_to_rgb_float(colors: np.ndarray) -> np.ndarray:
        cols = np.asarray(colors)
        if cols.size == 0:
            return np.empty((0,), dtype=np.float32)
        if cols.dtype != np.uint8:
            cols = cols.astype(np.float32)
            if cols.max(initial=0.0) <= 1.0:
                cols = np.clip(np.round(cols * 255.0), 0.0, 255.0).astype(np.uint8)
            else:
                cols = np.clip(np.round(cols), 0.0, 255.0).astype(np.uint8)
        rgb_uint32 = (
            (cols[:, 0].astype(np.uint32) << 16)
            | (cols[:, 1].astype(np.uint32) << 8)
            | cols[:, 2].astype(np.uint32)
        )
        return rgb_uint32.view(np.float32)

    def create_cloud_xyzrgb(
        self,
        header: Header,
        points: np.ndarray,
        colors: Optional[np.ndarray],
    ) -> PointCloud2:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return pc2.create_cloud_xyz32(header, [])
        cols = None if colors is None else np.asarray(colors)
        if cols is None or len(cols) != len(pts):
            return pc2.create_cloud_xyz32(header, pts.tolist())
        rgb = self.colors_to_rgb_float(cols)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        xyzrgb = np.column_stack([pts, rgb])
        return pc2.create_cloud(header, fields, xyzrgb.tolist())

    # ------------------------------------------------------------------
    # Registration preview
    # ------------------------------------------------------------------

    def publish_registration_preview(
        self,
        views: "List[RegistrationView]",
        reg_cfg: "RegistrationConfig",
    ) -> None:
        if not views:
            return
        try:
            pts = np.vstack(
                [
                    np.asarray(v.points, dtype=np.float32)
                    for v in views
                    if v.points is not None and len(v.points) > 0
                ]
            )
        except ValueError:
            return
        if pts.size == 0:
            return
        cols_list = []
        for v in views:
            if (
                v.points is None
                or len(v.points) == 0
                or v.colors is None
                or len(v.colors) != len(v.points)
            ):
                cols_list = []
                break
            cols_list.append(np.asarray(v.colors))
        cols = np.vstack(cols_list) if cols_list else None
        try:
            pts_ds, cols_ds = voxel_downsample(
                pts, cols, float(reg_cfg.voxel_size_m)
            )
            out_header = Header()
            out_header.stamp = self._node.get_clock().now().to_msg()
            out_header.frame_id = reg_cfg.target_frame
            fused_msg = self.create_cloud_xyzrgb(out_header, pts_ds, cols_ds)
            self._fused_pub.publish(fused_msg)
            label = next((str(v.label) for v in reversed(views) if getattr(v, "label", "")), "")
            self._publish_cloud_status(
                topic=str(self._node.get_parameter("registration_fused_cloud_topic").value),
                stage="SEED ONLY" if len(views) <= 1 else "FUSED",
                object_label=label,
                point_count=len(pts_ds),
                view_count=len(views),
            )
        except Exception as e:
            self._node.get_logger().warn(
                f"Failed to publish registration preview cloud: {e}"
            )

    # ------------------------------------------------------------------
    # TSDF
    # ------------------------------------------------------------------

    def build_tsdf_config(self) -> TSDFConfig:
        node = self._node
        return TSDFConfig(
            voxel_length_m=float(
                node.get_parameter("registration_tsdf_voxel_length_m").value
            ),
            sdf_trunc_m=float(
                node.get_parameter("registration_tsdf_sdf_trunc_m").value
            ),
            depth_trunc_m=float(
                node.get_parameter("registration_tsdf_depth_trunc_m").value
            ),
            point_voxel_size_m=float(
                node.get_parameter("registration_tsdf_point_voxel_size_m").value
            ),
            mask_erode_px=int(
                node.get_parameter("registration_tsdf_mask_erode_px").value
            ),
            depth_band_m=float(
                node.get_parameter("registration_tsdf_depth_band_m").value
            ),
        )

    def publish_tsdf_cloud(
        self,
        accepted_obs_meta: List[Dict[str, Any]],
        reg_cfg: "RegistrationConfig",
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        node = self._node
        if not bool(node.get_parameter("registration_tsdf_enabled").value):
            return None
        try:
            tsdf_cfg = self.build_tsdf_config()
            n_total = len(accepted_obs_meta)
            n_eligible = sum(
                1 for obs in accepted_obs_meta if bool(obs.get("tsdf_eligible", False))
            )
            node.get_logger().info(
                f"[tsdf] attempting publish from {n_total} accepted observations "
                f"({n_eligible} eligible)."
            )
            pts, cols = integrate_tsdf_surface_cloud(
                accepted_obs_meta, tsdf_cfg, logger=node.get_logger()
            )
            if pts is None or len(pts) == 0:
                node.get_logger().warn(
                    "[tsdf] surface extraction produced no points; nothing will be published."
                )
                return None
            out_header = Header()
            out_header.stamp = node.get_clock().now().to_msg()
            out_header.frame_id = reg_cfg.target_frame
            tsdf_msg = self.create_cloud_xyzrgb(out_header, pts, cols)
            self._tsdf_pub.publish(tsdf_msg)
            # Remembered so the end-of-cycle final publish (Morty Final
            # Result window) can re-emit exactly this cloud. (2026-08-07)
            node.last_tsdf_cloud_msg = tsdf_msg
            self._publish_cloud_status(
                topic=str(node.get_parameter("registration_tsdf_cloud_topic").value),
                stage="TSDF",
                object_label=self._last_object_label,
                point_count=len(pts),
                view_count=n_total,
            )
            node.get_logger().info(
                f"[tsdf] published {len(pts)} points on "
                f"'{node.get_parameter('registration_tsdf_cloud_topic').value}'."
            )
            return pts, cols
        except Exception as e:
            node.get_logger().warn(f"Failed to publish TSDF cloud: {e}")
            return None

    def publish_incremental_tsdf_cloud(
        self,
        seed_obs_meta: Optional[Dict[str, Any]],
        accepted_obs_meta: List[Dict[str, Any]],
        reg_cfg: "RegistrationConfig",
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        node = self._node
        if not bool(node.get_parameter("registration_tsdf_enabled").value):
            return None
        obs_all: List[Dict[str, Any]] = []
        if seed_obs_meta is not None:
            obs_all.append(seed_obs_meta)
        obs_all.extend(accepted_obs_meta)
        n_eligible = sum(1 for obs in obs_all if bool(obs.get("tsdf_eligible", False)))
        if n_eligible < 2:
            return None
        return self.publish_tsdf_cloud(obs_all, reg_cfg)
