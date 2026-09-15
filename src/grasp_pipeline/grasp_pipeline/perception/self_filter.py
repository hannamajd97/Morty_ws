"""Robot self-filter: remove the gripper from the camera's own data.

The wrist camera always sees the gripper's fingers, and every detector takes
them for objects on the table: the trained YOLO reported a finger as
``small_bottle`` at 0.88 (2026-09-15), YOLOE called it a "wedge", and their
depth pixels sit 10-20 cm from the lens, exactly where an object cloud is
clipped. Confidence thresholds cannot separate them, because the detector is
*certain*.

The robot knows where its own body is better than any detector. The URDF the
robot_state_publisher runs carries a collision hull per gripper link, and TF
carries each link's pose in the camera frame at every instant, open or
closed. So the hulls are projected into the image (about 1 ms per frame,
cached while the fingers do not move) and the projection is applied twice:

* detections whose box lies mostly under it are dropped before tracking, and
* depth pixels under it are zeroed before anything reads the frame, so a
  finger can never enter an object's point cloud or a neighbour box.

A fixed painted-out area would fail as soon as the gripper opens; a plain
near-depth cut would delete the object about to be grasped. This is the same
principle as MoveIt's robot self-filter on its sensors.

The hulls come from the running robot's ``robot_description`` (the same
meshes MoveIt uses for collision checking), fetched once on a helper thread
so the pipeline never waits for it; until they arrive the filter is inactive
and every consumer sees the raw data, exactly as before.
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import rclpy.time

# Mesh coordinates in the URDF are metres once ``scale`` is applied; points
# closer to the lens than this are behind the optical centre for practical
# purposes and cannot be projected.
_MIN_Z_M = 0.02
_PARAM_SERVICE = "/robot_state_publisher/get_parameters"


def _rot_from_quat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix from a ROS (x, y, z, w) quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _rot_from_rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF roll-pitch-yaw (fixed axes X, then Y, then Z) to a rotation."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _resolve_mesh_path(uri: str) -> str:
    """``package://pkg/path`` or ``file://`` to a filesystem path."""
    if uri.startswith("package://"):
        from ament_index_python.packages import get_package_share_directory
        pkg, _, rel = uri[len("package://"):].partition("/")
        return os.path.join(get_package_share_directory(pkg), rel)
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri


def hull_points_from_urdf(urdf_xml: str, links: Sequence[str]) -> Dict[str, np.ndarray]:
    """Convex-hull vertices (metres, in the link frame) of every collision
    mesh of each requested link, read from the robot's own URDF."""
    import trimesh
    from urdf_parser_py.urdf import URDF

    robot = URDF.from_xml_string(urdf_xml)
    wanted = set(links)
    out: Dict[str, np.ndarray] = {}
    for link in robot.links:
        if link.name not in wanted:
            continue
        cols = list(getattr(link, "collisions", None) or ([link.collision] if link.collision else []))
        pts: List[np.ndarray] = []
        for col in cols:
            geom = col.geometry
            filename = getattr(geom, "filename", None)
            if not filename:
                continue          # primitive shapes are not used on this gripper
            mesh = trimesh.load(_resolve_mesh_path(filename), force="mesh")
            v = np.asarray(mesh.vertices, dtype=np.float64)
            scale = getattr(geom, "scale", None)
            if scale is not None:
                v = v * np.asarray(scale, dtype=np.float64)
            if col.origin is not None:
                r = _rot_from_rpy(*(col.origin.rpy or (0.0, 0.0, 0.0)))
                v = v @ r.T + np.asarray(col.origin.xyz or (0.0, 0.0, 0.0), dtype=np.float64)
            pts.append(v)
        if pts:
            allp = np.vstack(pts)
            try:
                allp = np.asarray(trimesh.convex.convex_hull(allp).vertices, dtype=np.float64)
            except Exception:
                pass
            out[link.name] = allp
    return out


class GripperSelfFilter:
    """Owns the gripper hulls, the per-frame image mask and its two uses.

    All state is set once by the loader thread and then only read, so the
    depth callback, the detection thread and the overlay can share it.
    """

    def __init__(self, node: Any) -> None:
        self._node = node
        self._hulls: Optional[Dict[str, np.ndarray]] = None
        self._cache_key: Optional[Tuple] = None
        self._cache_mask: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._missing_tf_logged: set = set()
        self._dropped_total = 0
        self._last_drop_log = 0.0
        if self.enabled:
            threading.Thread(target=self._load_hulls, name="self_filter_urdf", daemon=True).start()

    # ---- parameters -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        try:
            return bool(self._node.get_parameter("self_filter_enabled").value)
        except Exception:
            return False

    def _param(self, name: str, default):
        try:
            v = self._node.get_parameter(name).value
            return default if v is None else v
        except Exception:
            return default

    @property
    def active(self) -> bool:
        return self.enabled and self._hulls is not None

    # ---- one-time load from the running robot -----------------------------
    def _fetch_robot_description(self) -> Optional[str]:
        """Ask the robot_state_publisher for its URDF. Runs on the helper
        thread with its own node, so the pipeline's executor is untouched."""
        from rcl_interfaces.srv import GetParameters
        from rclpy.executors import SingleThreadedExecutor

        helper = rclpy.create_node("grasp_self_filter_urdf_client")
        ex = SingleThreadedExecutor()
        ex.add_node(helper)
        cli = helper.create_client(GetParameters, _PARAM_SERVICE)
        try:
            while rclpy.ok():
                if not cli.wait_for_service(timeout_sec=5.0):
                    continue
                fut = cli.call_async(GetParameters.Request(names=["robot_description"]))
                ex.spin_until_future_complete(fut, timeout_sec=10.0)
                res = fut.result() if fut.done() else None
                if res is not None and res.values and res.values[0].string_value:
                    return res.values[0].string_value
                time.sleep(5.0)
        finally:
            ex.remove_node(helper)
            helper.destroy_node()
        return None

    def _load_hulls(self) -> None:
        log = self._node.get_logger()
        links = [str(s) for s in self._param("self_filter_links", [])]
        try:
            xml = self._fetch_robot_description()
            if not xml:
                return
            hulls = hull_points_from_urdf(xml, links)
        except Exception as e:  # noqa: BLE001 - any failure leaves the filter off
            log.warn(f"[self-filter] could not load gripper hulls ({e}); filter inactive")
            return
        missing = [name for name in links if name not in hulls]
        if missing:
            log.warn(f"[self-filter] no collision mesh in the URDF for {missing}; those links are not masked")
        if not hulls:
            log.warn("[self-filter] no gripper hulls found; filter inactive")
            return
        with self._lock:
            self._hulls = hulls
        log.info("[self-filter] gripper hulls loaded from the robot's URDF: "
                 + ", ".join(f"{k} ({len(v)} pts)" for k, v in hulls.items()))

    # ---- the mask -----------------------------------------------------------
    def mask(self, shape: Tuple[int, int], frame_id: str) -> Optional[np.ndarray]:
        """uint8 HxW image, 255 where the gripper is, or None when inactive
        or when TF cannot place a link yet. Cached while nothing moved."""
        hulls = self._hulls
        if hulls is None or not self.enabled or not frame_id:
            return None
        fb = self._node.fb
        if not (fb.fx and fb.fy):
            return None
        pad = int(self._param("self_filter_pad_px", 16))
        poses: List[Tuple[str, np.ndarray, np.ndarray]] = []
        key: List = [tuple(shape), str(frame_id), pad]
        for link in hulls:
            try:
                tf = self._node.tf_buffer.lookup_transform(frame_id, link, rclpy.time.Time())
            except Exception:
                if link not in self._missing_tf_logged:
                    self._missing_tf_logged.add(link)
                    self._node.get_logger().warn(f"[self-filter] no TF {frame_id} <- {link} yet; link skipped until it appears")
                continue
            q, t = tf.transform.rotation, tf.transform.translation
            quat = (round(q.x, 4), round(q.y, 4), round(q.z, 4), round(q.w, 4))
            trans = (round(t.x, 4), round(t.y, 4), round(t.z, 4))
            key.append((link, quat, trans))
            poses.append((link, _rot_from_quat(*quat), np.asarray(trans, dtype=np.float64)))
        if not poses:
            return None
        key_t = tuple(key)
        with self._lock:
            if key_t == self._cache_key and self._cache_mask is not None:
                return self._cache_mask
        h, w = int(shape[0]), int(shape[1])
        m = np.zeros((h, w), dtype=np.uint8)
        for link, r, t in poses:
            p = hulls[link] @ r.T + t
            p = p[p[:, 2] > _MIN_Z_M]
            if len(p) < 3:
                continue
            u = fb.fx * p[:, 0] / p[:, 2] + fb.cx
            v = fb.fy * p[:, 1] / p[:, 2] + fb.cy
            pts = np.stack([np.clip(u, -4 * w, 5 * w), np.clip(v, -4 * h, 5 * h)], axis=1).astype(np.int32)
            cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
        if pad > 0:
            m = cv2.dilate(m, np.ones((pad, pad), dtype=np.uint8))
        with self._lock:
            self._cache_key, self._cache_mask = key_t, m
        return m

    # ---- use 1: detections --------------------------------------------------
    def filter_detections(self, dets: List[Dict[str, Any]], shape: Tuple[int, int],
                          frame_id: str) -> List[Dict[str, Any]]:
        """Drop detections whose box lies mostly on the gripper. Anything in
        that image region that is not the gripper is behind it, hence
        occluded and not graspable, so no depth test is needed."""
        if not dets:
            return dets
        m = self.mask(shape, frame_id)
        if m is None:
            return dets
        thr = float(self._param("self_filter_drop_overlap", 0.3))
        h, w = m.shape[:2]
        kept: List[Dict[str, Any]] = []
        dropped: List[str] = []
        for d in dets:
            try:
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            except Exception:
                kept.append(d)
                continue
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            sub = m[y1:y2, x1:x2]
            frac = float((sub > 0).mean()) if sub.size else 0.0
            if frac >= thr:
                dropped.append(f"{d.get('label', '?')} {float(d.get('score', 0.0)):.2f} ({frac * 100:.0f}% gripper)")
            else:
                kept.append(d)
        if dropped:
            self._dropped_total += len(dropped)
            now = time.monotonic()
            if now - self._last_drop_log > 5.0:
                self._last_drop_log = now
                self._node.get_logger().info(
                    f"[self-filter] dropped {len(dropped)} detection(s) on the gripper: "
                    + "; ".join(dropped) + f"  (total {self._dropped_total})")
        return kept

    # ---- use 2: depth -------------------------------------------------------
    def depth_mask(self, shape: Tuple[int, int], frame_id: str) -> Optional[np.ndarray]:
        """Hook for the frame buffer: the mask to zero in every depth frame."""
        return self.mask(shape, frame_id)

    # ---- overlay ------------------------------------------------------------
    def draw_outline(self, bgr: np.ndarray, frame_id: str) -> np.ndarray:
        """The mask's outline in red on a copy of the frame, so the operator
        can see where the robot believes its own fingers are."""
        if not bool(self._param("self_filter_draw_overlay", True)):
            return bgr
        m = self.mask(bgr.shape[:2], frame_id)
        if m is None:
            return bgr
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = bgr.copy()
        cv2.drawContours(out, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
        return out
