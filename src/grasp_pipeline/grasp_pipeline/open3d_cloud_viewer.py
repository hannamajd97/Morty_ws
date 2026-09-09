"""
open3d_cloud_viewer.py
======================

Live Open3D point-cloud viewer node. Subscribes to a PointCloud2 topic (e.g.
the fused/registered object cloud) and shows it in an interactive Open3D window,
updating as new clouds arrive, with a status string overlay. This is a debugging
/ demonstration tool — it does not affect grasping.

Threading note: Open3D's visualizer must own the main thread, so the ROS spin
runs in a background thread while the window's event loop runs in the foreground.

WSL caveat: the interactive window needs an OpenGL context via WSLg, which can be
unreliable. If the window won't open, prefer the offscreen render path in
``grasp_debug_viewer.py`` instead. (Launch with the X11 backend forced:
``env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 ros2 run grasp_pipeline open3d_cloud_viewer``.)
"""
from __future__ import annotations

import os
import json
import threading
import time
from typing import Any, Optional

import numpy as np
import open3d as o3d
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


def _extract_float32_field(points_flat: np.ndarray, offset: int) -> np.ndarray:
    field_bytes = points_flat[:, offset:offset + 4].copy()
    return np.frombuffer(field_bytes.tobytes(), dtype=np.float32)


def _rgb_float_to_colors(rgb_float: np.ndarray) -> np.ndarray:
    rgb_uint32 = rgb_float.view(np.uint32)
    r = ((rgb_uint32 >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_uint32 >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (rgb_uint32 & 0xFF).astype(np.float32) / 255.0
    return np.stack((r, g, b), axis=1)


def _to_o3d_cloud(points: np.ndarray, colors: Optional[np.ndarray]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def _source_label_for_topic(topic: str) -> str:
    topic_l = str(topic or "").lower()
    if "fused_cloud" in topic_l:
        return "FUSED multi-view cloud"
    if "tsdf_cloud" in topic_l:
        return "TSDF reconstructed cloud"
    if "selected_cloud" in topic_l:
        return "SELECTED raw crop"
    if "depth" in topic_l or "points" in topic_l:
        return "LIVE camera point cloud"
    return "Point cloud"


def _is_fused_topic(topic: str) -> bool:
    return "fused_cloud" in str(topic or "").lower()


def _downsample_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size_m: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size_m <= 0.0 or len(points) == 0:
        return points, colors
    cloud = _to_o3d_cloud(points, colors)
    cloud = cloud.voxel_down_sample(float(voxel_size_m))
    pts = np.asarray(cloud.points, dtype=np.float32)
    cols = None
    if cloud.has_colors():
        cols = np.asarray(cloud.colors, dtype=np.float32)
    return pts, cols


class Open3DCloudViewer(Node):
    def __init__(self) -> None:
        super().__init__("open3d_cloud_viewer")

        self.declare_parameter("topic", "/camera/camera/depth/color/points_downsampled")
        self.declare_parameter("window_name", "Open3D Cloud Viewer")
        self.declare_parameter("point_size", 2.0)
        self.declare_parameter("stride", 1)
        self.declare_parameter("voxel_size_m", 0.0)
        self.declare_parameter("max_points", 120000)
        self.declare_parameter("update_hz", 30.0)
        self.declare_parameter("use_rgb", True)
        self.declare_parameter("show_axes", True)
        self.declare_parameter("axes_size", 0.05)
        self.declare_parameter("width", 800)
        self.declare_parameter("height", 520)
        self.declare_parameter("fusion_ready_min_messages", 2)
        self.declare_parameter("status_log_period_s", 2.0)
        self.declare_parameter("status_topic", "/graspgen/cloud_status")
        self.declare_parameter("show_status_text", False)
        self.declare_parameter("status_text_scale", 0.0005)
        self.declare_parameter("status_text_flip", True)
        # Default False on this machine (2026-08-07): the status panel is a cv2
        # HighGUI window, and pip-opencv's bundled Qt cannot display here
        # (SYSTEM_STATE_LOG #34) — drawing it hangs the render loop, which then
        # freezes the Open3D window itself ("not responding" dialog, #25).
        # Status is logged to the terminal anyway.
        self.declare_parameter("show_status_panel", False)
        self.declare_parameter("status_panel_width", 520)
        self.declare_parameter("status_panel_height", 260)

        self.topic = str(self.get_parameter("topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.source_label = _source_label_for_topic(self.topic)
        if self.window_name == "Open3D Cloud Viewer":
            self.window_name = f"Open3D - {self.source_label} ({self.topic})"
        self.point_size = float(self.get_parameter("point_size").value)
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        self.max_points = max(1000, int(self.get_parameter("max_points").value))
        self.update_hz = max(1.0, float(self.get_parameter("update_hz").value))
        self.use_rgb = bool(self.get_parameter("use_rgb").value)
        self.show_axes = bool(self.get_parameter("show_axes").value)
        self.axes_size = float(self.get_parameter("axes_size").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fusion_ready_min_messages = max(
            1, int(self.get_parameter("fusion_ready_min_messages").value)
        )
        self.status_log_period_s = max(
            0.5, float(self.get_parameter("status_log_period_s").value)
        )
        self.show_status_text = bool(self.get_parameter("show_status_text").value)
        self.status_text_scale = max(0.0001, float(self.get_parameter("status_text_scale").value))
        self.status_text_flip = bool(self.get_parameter("status_text_flip").value)
        self.show_status_panel = bool(self.get_parameter("show_status_panel").value)
        self.status_panel_width = max(420, int(self.get_parameter("status_panel_width").value))
        self.status_panel_height = max(220, int(self.get_parameter("status_panel_height").value))
        self._is_fused_topic = _is_fused_topic(self.topic)

        self._lock = threading.Lock()
        self._latest_points: Optional[np.ndarray] = None
        self._latest_colors: Optional[np.ndarray] = None
        self._latest_frame_id = ""
        self._latest_stamp_ns = 0
        self._received_count = 0
        self._status_by_topic: dict[str, dict[str, Any]] = {}
        self._status_revision = 0

        self.create_subscription(
            PointCloud2,
            self.topic,
            self._cloud_callback,
            qos_profile_sensor_data,
        )
        if self.status_topic:
            self.create_subscription(
                String,
                self.status_topic,
                self._status_callback,
                10,
            )

        self.get_logger().info(
            f"Open3D viewer showing: {self.source_label}\n"
            f"  topic: {self.topic}\n"
            f"  window: {self.width}x{self.height}\n"
            f"  stride: {self.stride}\n"
            f"  voxel_size_m: {self.voxel_size_m}\n"
            f"  max_points: {self.max_points}\n"
            f"  use_rgb: {self.use_rgb}\n"
            f"  status_topic: {self.status_topic}\n"
            f"  status: WAITING for first cloud"
        )

    def _status_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data or "{}")
        except Exception:
            return
        topic = str(data.get("topic", "")).strip()
        if not topic:
            return
        with self._lock:
            self._status_by_topic[topic] = data
            self._status_revision += 1

    def _cloud_callback(self, msg: PointCloud2) -> None:
        num_points = int(msg.width) * int(msg.height)
        if num_points <= 0 or msg.point_step <= 0:
            return

        try:
            cloud_array = np.frombuffer(msg.data, dtype=np.uint8)
            points_flat = cloud_array.reshape(-1, msg.point_step)

            if self.stride > 1:
                points_flat = points_flat[::self.stride]

            field_offsets = {field.name: field.offset for field in msg.fields}
            if not {"x", "y", "z"}.issubset(field_offsets):
                self.get_logger().warn(
                    "Incoming cloud is missing XYZ fields",
                    throttle_duration_sec=5.0,
                )
                return

            points = np.column_stack([
                _extract_float32_field(points_flat, field_offsets["x"]),
                _extract_float32_field(points_flat, field_offsets["y"]),
                _extract_float32_field(points_flat, field_offsets["z"]),
            ])

            colors = None
            if self.use_rgb:
                color_field = "rgb" if "rgb" in field_offsets else ("rgba" if "rgba" in field_offsets else None)
                if color_field is not None:
                    rgb_float = _extract_float32_field(points_flat, field_offsets[color_field])
                    colors = _rgb_float_to_colors(rgb_float)

            finite_mask = np.isfinite(points).all(axis=1)
            points = points[finite_mask]
            if colors is not None:
                colors = colors[finite_mask]

            if len(points) == 0:
                return

            if len(points) > self.max_points:
                idx = np.linspace(0, len(points) - 1, self.max_points, dtype=np.int32)
                points = points[idx]
                if colors is not None:
                    colors = colors[idx]

            points, colors = _downsample_cloud(points, colors, self.voxel_size_m)

            with self._lock:
                self._latest_points = np.asarray(points, dtype=np.float32)
                self._latest_colors = None if colors is None else np.asarray(colors, dtype=np.float32)
                self._latest_frame_id = str(msg.header.frame_id)
                self._latest_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
                self._received_count += 1
        except Exception as exc:
            self.get_logger().error(
                f"Failed to parse point cloud: {exc}",
                throttle_duration_sec=5.0,
            )

    def consume_latest(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, int]:
        with self._lock:
            return self._latest_points, self._latest_colors, self._latest_frame_id, self._latest_stamp_ns

    def status_revision(self) -> int:
        with self._lock:
            return self._status_revision

    def metadata_for_topic(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status_by_topic.get(self.topic, {}))

    def cloud_status(self, points: Optional[np.ndarray]) -> str:
        meta = self.metadata_for_topic()
        stage = str(meta.get("stage", "")).strip()
        if stage:
            if stage == "SEED ONLY":
                return "NO FUSE YET: seed view only"
            if stage == "FUSED":
                return "FUSED: multi-view cloud"
            if stage == "TSDF":
                return "TSDF: reconstructed cloud"
            return stage
        if points is None:
            return "WAITING for first cloud"
        if self._is_fused_topic and self._received_count < self.fusion_ready_min_messages:
            return (
                "SEED ONLY: first registration preview received; "
                "waiting for fused follow-up views"
            )
        if self._is_fused_topic:
            return "FUSED/UPDATED: multi-view cloud has been published"
        return "LIVE: cloud message received"

    def status_text(self, points: Optional[np.ndarray]) -> str:
        meta = self.metadata_for_topic()
        obj = str(meta.get("object", "")).strip() or "object"
        stage = str(meta.get("stage", "")).strip()
        if stage == "SEED ONLY":
            stage_text = "NO FUSE YET"
        elif stage:
            stage_text = stage
        elif points is None:
            stage_text = "WAITING"
        else:
            stage_text = "LIVE"
        views = meta.get("views", self._received_count if points is not None else 0)
        pts = meta.get("points", len(points) if points is not None else 0)
        return f"{obj}\n{stage_text}\nviews={views} pts={pts}"

    def status_panel_data(self, points: Optional[np.ndarray]) -> dict[str, Any]:
        meta = self.metadata_for_topic()
        obj = str(meta.get("object", "")).strip() or "object"
        stage = str(meta.get("stage", "")).strip()
        if stage == "SEED ONLY":
            stage_text = "NO FUSE YET"
        elif stage:
            stage_text = stage
        elif points is None:
            stage_text = "WAITING"
        else:
            stage_text = "LIVE"
        return {
            "object": obj,
            "stage": stage_text,
            "views": meta.get("views", self._received_count if points is not None else 0),
            "points": meta.get("points", len(points) if points is not None else 0),
            "topic": self.topic,
        }


def _stage_color(text: str) -> np.ndarray:
    text_u = text.upper()
    if "NO FUSE" in text_u or "SEED" in text_u or "WAITING" in text_u:
        return np.asarray([1.0, 0.72, 0.16], dtype=np.float64)
    if "TSDF" in text_u:
        return np.asarray([0.25, 0.70, 1.0], dtype=np.float64)
    if "FUSED" in text_u:
        return np.asarray([0.28, 0.95, 0.44], dtype=np.float64)
    return np.asarray([0.95, 0.95, 0.95], dtype=np.float64)


def _make_status_text_mesh(
    text: str,
    points: Optional[np.ndarray],
    scale: float,
    flip_text: bool,
) -> Optional[o3d.geometry.TriangleMesh]:
    if points is None or len(points) == 0:
        return None
    try:
        clean_text = "".join(ch if 32 <= ord(ch) <= 126 or ch == "\n" else "?" for ch in text)
        color = _stage_color(clean_text)
        mesh_front = o3d.t.geometry.TriangleMesh.create_text(clean_text, depth=0.3).to_legacy()
        mesh_front.paint_uniform_color(color)
        if flip_text:
            mesh_front.transform(
                np.asarray(
                    [
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            )

        # Open3D's legacy visualizer shows both sides of a mesh, but the
        # backside of generated text is mirrored. Add a second front-facing
        # copy for the reverse side so the letters themselves stay readable.
        mesh_back = o3d.t.geometry.TriangleMesh.create_text(clean_text, depth=0.3).to_legacy()
        mesh_back.paint_uniform_color(color)
        if flip_text:
            mesh_back.transform(
                np.asarray(
                    [
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            )
        mesh_back.rotate(
            np.asarray(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=np.float64,
            ),
            center=(0.0, 0.0, 0.0),
        )
        mesh_back.translate((0.0, 0.0, -0.6))

        mesh = mesh_front + mesh_back
        mesh.scale(float(scale), center=(0.0, 0.0, 0.0))
        # Put the text in a vertical X/Z plane and face it toward the
        # default Open3D camera side. Otherwise the viewer sees the
        # backside of the text mesh and the label appears mirrored.
        text_basis = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        mesh.rotate(text_basis, center=(0.0, 0.0, 0.0))

        pts = np.asarray(points, dtype=np.float64)
        cloud_min = np.min(pts, axis=0)
        cloud_max = np.max(pts, axis=0)
        text_box = mesh.get_axis_aligned_bounding_box()
        text_extent = text_box.get_extent()
        text_min = text_box.get_min_bound()
        anchor = np.asarray(
            [
                0.5 * (cloud_min[0] + cloud_max[0]) - 0.5 * text_extent[0],
                cloud_max[1] + 0.050,
                cloud_min[2],
            ],
            dtype=np.float64,
        )
        mesh.translate(anchor - text_min)
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        return None


def _status_bgr(stage: str) -> tuple[int, int, int]:
    stage_u = str(stage).upper()
    if "NO FUSE" in stage_u or "SEED" in stage_u or "WAITING" in stage_u:
        return (30, 170, 255)
    if "TSDF" in stage_u:
        return (255, 160, 40)
    if "FUSED" in stage_u:
        return (80, 230, 100)
    return (230, 230, 230)


def _draw_status_panel(node: Open3DCloudViewer, points: Optional[np.ndarray]) -> None:
    if not node.show_status_panel:
        return
    data = node.status_panel_data(points)
    w = node.status_panel_width
    h = node.status_panel_height
    img = np.full((h, w, 3), (28, 28, 30), dtype=np.uint8)
    color = _status_bgr(str(data["stage"]))

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (75, 75, 80), 1)
    cv2.rectangle(img, (0, 0), (w, 58), color, -1)
    cv2.putText(
        img,
        str(data["stage"])[:28],
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )

    lines = [
        f"object: {data['object']}",
        f"views: {data['views']}   points: {data['points']}",
        f"topic: {data['topic']}",
    ]
    y = 100
    for line in lines:
        cv2.putText(
            img,
            str(line)[:48],
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        y += 42

    cv2.imshow("Fusion Status", img)
    cv2.resizeWindow("Fusion Status", w, h)
    cv2.waitKey(1)


def main(args=None) -> None:
    # Same GLFW workaround as grasp_viz_viewer (2026-08-07, log #38): with
    # WAYLAND_DISPLAY set, Open3D's GLFW picks the Wayland backend, which
    # fails/misbehaves on this machine. Force X11/XWayland before the first
    # create_window call initialises GLFW.
    import os as _os

    _os.environ.pop("WAYLAND_DISPLAY", None)
    _os.environ["XDG_SESSION_TYPE"] = "x11"

    rclpy.init(args=args)
    node = Open3DCloudViewer()

    vis = o3d.visualization.Visualizer()
    created = vis.create_window(
        window_name=node.window_name,
        width=node.width,
        height=node.height,
    )
    render = vis.get_render_option()

    if not created or render is None:
        display = os.environ.get("DISPLAY", "")
        wayland = os.environ.get("WAYLAND_DISPLAY", "")
        session_type = os.environ.get("XDG_SESSION_TYPE", "")
        node.get_logger().error(
            "Open3D failed to create a render window. "
            f"DISPLAY={display!r}, WAYLAND_DISPLAY={wayland!r}, XDG_SESSION_TYPE={session_type!r}. "
            "This usually means the current desktop session does not provide a compatible OpenGL context for Open3D. "
            "Try launching the viewer through XWayland, for example: "
            "`env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 ros2 run grasp_pipeline open3d_cloud_viewer`"
        )
        try:
            vis.destroy_window()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        return

    render.background_color = np.asarray([0.08, 0.08, 0.09], dtype=np.float64)
    render.point_size = float(node.point_size)

    cloud = o3d.geometry.PointCloud()
    vis.add_geometry(cloud)

    if node.show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=node.axes_size)
        vis.add_geometry(axes)

    if node.show_status_panel:
        cv2.namedWindow("Fusion Status", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fusion Status", node.status_panel_width, node.status_panel_height)

    status_mesh: Optional[o3d.geometry.TriangleMesh] = None

    def update_status_text(points_for_text: Optional[np.ndarray]) -> None:
        nonlocal status_mesh
        if not node.show_status_text:
            return
        new_mesh = _make_status_text_mesh(
            node.status_text(points_for_text),
            points_for_text,
            node.status_text_scale,
            node.status_text_flip,
        )
        if status_mesh is not None:
            vis.remove_geometry(status_mesh, reset_bounding_box=False)
            status_mesh = None
        if new_mesh is not None:
            status_mesh = new_mesh
            vis.add_geometry(status_mesh, reset_bounding_box=False)

    last_stamp_ns = -1
    last_log_time = time.monotonic()
    last_status = ""
    last_status_revision = -1
    update_period_s = 1.0 / node.update_hz
    camera_fitted = False

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)

            points, colors, frame_id, stamp_ns = node.consume_latest()
            _draw_status_panel(node, points)
            status_revision = node.status_revision()
            if points is not None and (
                stamp_ns != last_stamp_ns or status_revision != last_status_revision
            ):
                cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
                if colors is not None and len(colors) == len(points):
                    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
                else:
                    cloud.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3), dtype=np.float64) * 0.85)
                vis.update_geometry(cloud)
                update_status_text(points)
                last_stamp_ns = stamp_ns
                last_status_revision = status_revision
                if not camera_fitted and len(points) > 0:
                    vis.reset_view_point(True)
                    camera_fitted = True

                now = time.monotonic()
                status = node.cloud_status(points)
                if status != last_status or now - last_log_time > node.status_log_period_s:
                    node.get_logger().info(
                        f"viewer status: {status}; source={node.source_label}; "
                        f"frame={frame_id}; points={len(points)}; "
                        f"messages={node._received_count}; topic={node.topic}"
                    )
                    last_status = status
                    last_log_time = now
            elif points is None:
                now = time.monotonic()
                status = node.cloud_status(points)
                if now - last_log_time > node.status_log_period_s:
                    node.get_logger().info(
                        f"viewer status: {status}; source={node.source_label}; "
                        f"messages=0; topic={node.topic}"
                    )
                    last_status = status
                    last_log_time = now

            if not vis.poll_events():
                break
            vis.update_renderer()
            time.sleep(update_period_s)
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        if node.show_status_panel:
            try:
                cv2.destroyWindow("Fusion Status")
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
