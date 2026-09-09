#!/usr/bin/env python3
"""Live 3D grasp visualiser (Open3D).

Added 2026-08-06 at Majd's request, replacing the old meshcat workflow for the
single-view pipeline: one native window showing

  * the selected object's point cloud (``/yolo/selected_cloud``, light grey),
  * every GraspGen candidate as a small gripper glyph:
      - GREEN  = survived the whole filter chain (``/graspgen/viz/kept_poses``)
      - RED    = filtered out               (``/graspgen/viz/rejected_poses``)
      - BLUE   = the best grasp             (``/graspgen/best_pose``)

All inputs are in the camera frame, so no TF is needed. The window updates
whenever a new selection cycle publishes. Runs as its own process (launched by
``run_grasp_pipeline.launch.py``; disable with ``grasp_viz:=false``).

Implementation notes:
  * Open3D's legacy Visualizer + poll_events/update_renderer, with a sleep in
    the loop — the old ``open3d_cloud_viewer`` burned 100% CPU busy-waiting
    (SYSTEM_STATE_LOG #25); this one idles at a few percent.
  * Each grasp class is ONE merged LineSet (hundreds of separate geometries
    would crawl); glyphs are simple 4-segment gripper forks.
"""
from __future__ import annotations

import signal
import struct
import sys
import time

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

WINDOW = "Morty result | [G]reen kept [R]ed rejected [B]lue best [C]loud [A]ll | ORANGE = executing"

# Gripper glyph in the grasp frame (metres): stem, palm bar, two fingers (+z).
_GLYPH_PTS = np.array(
    # Fingers spread along X — per GraspGen's robotiq_2f_140.yaml the contact
    # points/normals are on the ±X axis (2026-08-07; they were wrongly drawn
    # along Y before, making every glyph look rolled 90°).
    [
        [0.0, 0.0, -0.03],   # 0 stem back
        [0.0, 0.0, 0.0],     # 1 wrist
        [-0.04, 0.0, 0.0],   # 2 palm left
        [0.04, 0.0, 0.0],    # 3 palm right
        [-0.04, 0.0, 0.045], # 4 fingertip left
        [0.04, 0.0, 0.045],  # 5 fingertip right
    ],
    dtype=np.float64,
)
_GLYPH_LINES = np.array([[0, 1], [2, 3], [2, 4], [3, 5]], dtype=np.int32)


def _quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def _poses_to_lineset(
    poses, color, scale: float = 1.0, flip_z: bool = False
) -> o3d.geometry.LineSet:
    """Merge one gripper glyph per pose into a single LineSet.

    ``flip_z`` (2026-08-06): `/graspgen/best_pose` is the TCP-converted pose —
    the pipeline applies a 180° "flipY" rotation (log: "[fixed] grasp→tcp
    rotation: flipY"), which inverts the frame's z relative to the raw grasp
    frames used for kept/rejected. Without compensating, the best glyph draws
    mirrored and looks like it approaches from BEHIND the object. flip_z
    mirrors the glyph so the drawn approach matches physical reality."""
    ls = o3d.geometry.LineSet()
    if not poses:
        return ls
    glyph = _GLYPH_PTS * scale
    if flip_z:
        glyph = glyph * np.array([-1.0, 1.0, -1.0])  # 180° about Y
    all_pts = []
    all_lines = []
    for i, p in enumerate(poses):
        R = _quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        t = np.array([p.position.x, p.position.y, p.position.z])
        pts = glyph @ R.T + t
        all_pts.append(pts)
        all_lines.append(_GLYPH_LINES + i * len(_GLYPH_PTS))
    ls.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    ls.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(poses) * len(_GLYPH_LINES), 1))
    )
    return ls


def _cloud_to_o3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    """Minimal XYZ (+optional rgb) PointCloud2 → Open3D, float32 fields."""
    pc = o3d.geometry.PointCloud()
    n = msg.width * msg.height
    if n == 0:
        return pc
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    offs = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"}.issubset(offs):
        return pc

    def f32(off):
        return data[:, off:off + 4].copy().view(np.float32).ravel()

    xyz = np.column_stack([f32(offs["x"]), f32(offs["y"]), f32(offs["z"])])
    ok = np.isfinite(xyz).all(axis=1)
    xyz = xyz[ok]
    # Trim depth outliers for display (2026-08-06): the raw selected cloud can
    # contain background pixels metres behind the object (z_range up to ~3 m
    # was observed). The grasp pipeline removes them before grasping, but here
    # they inflate Open3D's bounding box, which makes the object tiny and
    # blocks zooming in. Keep points within 0.35 m of the cloud median.
    if len(xyz) > 10:
        med = np.median(xyz, axis=0)
        near = np.linalg.norm(xyz - med, axis=1) < 0.35
        ok2 = np.zeros(len(ok), dtype=bool)
        ok2[np.flatnonzero(ok)[near]] = True
        ok = ok2
        xyz = xyz[near]
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if "rgb" in offs:
        rgb_u = data[:, offs["rgb"]:offs["rgb"] + 4].copy().view(np.uint32).ravel()[ok]
        cols = np.column_stack(
            [(rgb_u >> 16) & 0xFF, (rgb_u >> 8) & 0xFF, rgb_u & 0xFF]
        ).astype(np.float64) / 255.0
        pc.colors = o3d.utility.Vector3dVector(cols)
    else:
        pc.paint_uniform_color([0.75, 0.75, 0.75])
    return pc


class GraspVizViewer(Node):
    def __init__(self) -> None:
        super().__init__("grasp_viz_viewer")
        # Two-instance design (2026-08-07, Majd's final window spec): this
        # executable runs twice from the launch —
        #   "Morty Frames":       cloud_topic=/yolo/selected_cloud_rgb,
        #                         show_grasps=false, subscribe_tsdf=false
        #                         (the live per-capture frame, nothing else)
        #   "Morty Final Result": cloud_topic=/yolo/selected_cloud_rgb,
        #                         subscribe_tsdf=true, show_grasps=true
        #                         (TSDF wins after fusion; grasps + ORANGE)
        self.declare_parameter("cloud_topic", "/yolo/selected_cloud_rgb")
        self.declare_parameter("show_grasps", True)
        self.declare_parameter("subscribe_tsdf", True)
        self.declare_parameter("window_title", "")
        self._cloud_msg = None
        self._kept_msg = None
        self._rejected_msg = None
        self._best_msg = None
        self._dirty = False

        # Result cloud (2026-08-07, two-window layout): whichever arrives last
        # wins — the coloured single-view crop (/yolo/selected_cloud_rgb) or
        # the TSDF surface (/graspgen/tsdf_cloud), which lands after fusion in
        # multiview and therefore correctly replaces the crop.
        cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.show_grasps = bool(self.get_parameter("show_grasps").value)
        self._executing = False
        self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, 2)
        if bool(self.get_parameter("subscribe_tsdf").value):
            # Final-result mode: the fused TSDF surface replaces the per-frame
            # crop as soon as it is published (latest message wins).
            self.create_subscription(PointCloud2, "/graspgen/tsdf_cloud", self._on_cloud, 2)
        if self.show_grasps:
            self.create_subscription(PoseArray, "/graspgen/viz/kept_poses", self._on_kept, 2)
            self.create_subscription(PoseArray, "/graspgen/viz/rejected_poses", self._on_rej, 2)
            # Raw-frame best from the pipeline — NOT the TCP-converted
            # /graspgen/best_pose, whose flipY made the blue glyph mirror (#42).
            self.create_subscription(PoseArray, "/graspgen/viz/best_pose_raw", self._on_best, 2)
            # ORANGE executing marker: the click window publishes True while
            # the executor launch it spawned is running, and the pipeline
            # bridges the executor's ACTUAL feasible pick (which may not be
            # GraspGen's best) onto /graspgen/viz/executing_pose.
            from std_msgs.msg import Bool as _Bool

            self._exec_pose_msg = None
            self.create_subscription(_Bool, "/morty/executor_active", self._on_exec_active, 2)
            self.create_subscription(
                PoseArray, "/graspgen/viz/executing_pose", self._on_exec_pose, 2
            )
        # Camera capture viewpoint (any mode): orients the initial view.
        self._campose_msg = None
        self.create_subscription(
            PoseStamped, "/graspgen/viz/camera_pose", self._on_campose, 2
        )
        self.get_logger().info(
            f"Grasp viz up: cloud='{cloud_topic}', green=kept, red=filtered, blue=best. "
            "Window appears/updates after each object selection."
        )

    def _on_cloud(self, m):
        self._cloud_msg = m
        self._dirty = True

    def _on_kept(self, m):
        self._kept_msg = m
        # New selection cycle → the previous execution's orange is history.
        self._exec_pose_msg = None
        self._dirty = True

    def _on_rej(self, m):
        self._rejected_msg = m
        self._dirty = True

    def _on_best(self, m):
        self._best_msg = m
        self._dirty = True

    def _on_exec_active(self, m):
        # Orange PERSISTS after execution (Majd 2026-08-07: "I want to see it
        # until I choose the next object") — cleared in _on_kept when the
        # next selection cycle's grasps arrive.
        if bool(m.data) != self._executing:
            self._executing = bool(m.data)
            self._dirty = True

    def _on_exec_pose(self, m):
        self._exec_pose_msg = m
        self._dirty = True
        self.get_logger().info("ORANGE: executor's ACTUAL selected candidate (persists until next selection).")

    def _on_campose(self, m):
        self._campose_msg = m

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        # VisualizerWithKeyCallback so each grasp class can be shown/hidden
        # from the SAME window (2026-08-06, Majd's request):
        #   G = green kept, R = red rejected, B = blue best, C = cloud, A = all.
        self._show = {"kept": True, "rejected": True, "best": True, "cloud": True}
        vis = o3d.visualization.VisualizerWithKeyCallback()

        def _mk_toggle(name: str):
            def _cb(_v):
                self._show[name] = not self._show[name]
                self._dirty = True
                state = " ".join(f"{k}={'ON' if v else 'off'}" for k, v in self._show.items())
                self.get_logger().info(f"[toggle] {state}")
                return False
            return _cb

        def _all_on(_v):
            for k in self._show:
                self._show[k] = True
            self._dirty = True
            self.get_logger().info("[toggle] everything ON")
            return False

        vis.register_key_callback(ord("G"), _mk_toggle("kept"))
        vis.register_key_callback(ord("R"), _mk_toggle("rejected"))
        vis.register_key_callback(ord("B"), _mk_toggle("best"))
        vis.register_key_callback(ord("C"), _mk_toggle("cloud"))
        vis.register_key_callback(ord("A"), _all_on)
        title = str(self.get_parameter("window_title").value).strip() or WINDOW
        created = False
        for attempt in range(3):
            if vis.create_window(window_name=title, width=760, height=560):
                created = True
                break
            self.get_logger().warn(
                f"Open3D window creation failed (attempt {attempt + 1}/3), retrying..."
            )
            time.sleep(1.0)
        if not created:
            self.get_logger().error(
                "Could not create the Open3D window — grasp viz disabled for this run."
            )
            return
        # get_render_option() returns None when the GL context is degraded —
        # cosmetic settings only, never worth crashing over. (2026-08-06)
        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = np.array([0.12, 0.12, 0.12])
            opt.point_size = 2.5
        geoms: list = []
        first = True
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            if self._dirty:
                self._dirty = False
                for g in geoms:
                    vis.remove_geometry(g, reset_bounding_box=False)
                geoms = []
                if self._show["cloud"] and self._cloud_msg is not None:
                    pc = _cloud_to_o3d(self._cloud_msg)
                    if len(pc.points):
                        geoms.append(pc)
                if (
                    self._show["rejected"]
                    and self._rejected_msg is not None
                    and self._rejected_msg.poses
                ):
                    geoms.append(
                        _poses_to_lineset(self._rejected_msg.poses, [0.85, 0.15, 0.15], flip_z=True)
                    )
                if self._show["kept"] and self._kept_msg is not None and self._kept_msg.poses:
                    geoms.append(
                        _poses_to_lineset(self._kept_msg.poses, [0.10, 0.80, 0.20], flip_z=True)
                    )
                if self._show["best"] and self._best_msg is not None and self._best_msg.poses:
                    if self._exec_pose_msg is not None and self._exec_pose_msg.poses:
                        # ORANGE = the executor's ACTUAL feasible pick (TCP
                        # convention: z = approach, so no flip). No provisional
                        # marker before the executor reports, and it persists
                        # until the next object selection.
                        geoms.append(
                            _poses_to_lineset(
                                list(self._exec_pose_msg.poses), [1.0, 0.55, 0.05],
                                scale=2.0, flip_z=False,
                            )
                        )
                    geoms.append(
                        _poses_to_lineset(
                            list(self._best_msg.poses), [0.15, 0.35, 0.95], scale=1.6, flip_z=True
                        )
                    )
                for g in geoms:
                    vis.add_geometry(g, reset_bounding_box=first)
                if geoms and first:
                    # ONLY on the very first data (2026-08-07): re-applying
                    # this on every rebuild meant each G/R/B toggle yanked the
                    # user's orbit back to the preset ("why is it refreshing").
                    first = False
                    try:
                        fid = ""
                        if self._cloud_msg is not None:
                            fid = str(self._cloud_msg.header.frame_id)
                        ctr = vis.get_view_control()
                        if "optical" in fid or fid.startswith("camera"):
                            # Camera-frame cloud: view from the camera itself.
                            ctr.set_front([0.0, 0.0, -1.0])
                            ctr.set_up([0.0, -1.0, 0.0])
                        else:
                            # Anchor to the REAL capture viewpoint published
                            # by the pipeline (2026-08-07 — hand-tuned
                            # headings kept facing the wrong way). Eye sits
                            # where the camera was: front = campos - centroid.
                            c = None
                            for g in geoms:
                                if hasattr(g, "get_center") and hasattr(g, "points"):
                                    c = np.asarray(g.get_center())
                                    break
                            f = None
                            if (
                                c is not None
                                and self._campose_msg is not None
                                and str(self._campose_msg.header.frame_id) == fid
                            ):
                                p = self._campose_msg.pose.position
                                cam = np.array([p.x, p.y, p.z])
                                v = cam - c
                                if np.linalg.norm(v) > 1e-3:
                                    f = v / np.linalg.norm(v)
                            if f is None and c is not None and np.linalg.norm(c[:2]) > 1e-3:
                                f = np.array([-c[0], -c[1], 0.28])
                                f = f / np.linalg.norm(f)
                            if f is not None:
                                ctr.set_front([float(f[0]), float(f[1]), float(f[2])])
                                if c is not None:
                                    ctr.set_lookat([float(c[0]), float(c[1]), float(c[2])])
                            ctr.set_up([0.0, 0.0, 1.0])
                        ctr.set_zoom(0.7)
                    except Exception:
                        pass
            if not vis.poll_events():
                break  # window closed
            vis.update_renderer()
            time.sleep(0.03)  # ~30 Hz — do NOT busy-loop (see log #25)
        vis.destroy_window()


def main(args=None) -> int:
    # Force GLFW onto X11/XWayland (2026-08-06): with WAYLAND_DISPLAY set,
    # Open3D's GLFW picks the Wayland backend and window creation fails on
    # this machine. open3d_cloud_viewer.py documents exactly this workaround
    # in its own error message ("env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11").
    # Must happen before the first create_window call initialises GLFW.
    import os

    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"

    rclpy.init(args=args)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    node = GraspVizViewer()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
