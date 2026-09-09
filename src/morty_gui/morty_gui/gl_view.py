"""Embedded 3D view: object cloud + grasp glyphs (pyqtgraph.opengl).

Geometry and conventions are copied 1:1 from grasp_pipeline's
grasp_viz_viewer (the Open3D window), which was verified on the robot:
fingers spread along X per robotiq_2f_140.yaml, and ``flip_z`` mirrors
glyphs drawn from RAW grasp frames (kept/rejected/best) whose z points
fingers→wrist, while the executor's TCP pose (orange) draws unflipped.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui

_GLYPH_PTS = np.array(
    [
        [0.0, 0.0, -0.03],
        [0.0, 0.0, 0.0],
        [-0.04, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [-0.04, 0.0, 0.045],
        [0.04, 0.0, 0.045],
    ],
    dtype=np.float64,
)
_GLYPH_SEGS = np.array([[0, 1], [2, 3], [2, 4], [3, 5]], dtype=np.int64)

COLORS = {
    "kept": (0.10, 0.80, 0.20, 1.0),
    "rejected": (0.85, 0.15, 0.15, 0.6),
    "best": (0.15, 0.35, 0.95, 1.0),
    "exec": (1.00, 0.55, 0.05, 1.0),
}
FLIP_Z = {"kept": True, "rejected": True, "best": True, "exec": False}
SCALE = {"kept": 1.0, "rejected": 1.0, "best": 1.4, "exec": 2.0}


# ── Display-only frame correction ────────────────────────────────────
# The viewer data (cloud + grasp poses) arrives in camera_color_optical_frame,
# where +z points FORWARD into the scene and the physical "up" is ~ -y.
# pyqtgraph's GLViewWidget hardwires +z as screen-up (it orbits by
# elevation/azimuth about +z), so drawing optical-frame points raw tilts the
# whole scene by ~90 deg: upright objects look like they are lying down.
# Everything that enters the GL scene therefore passes through
# CloudGLView._to_view() first.  This is a RENDERING transform only — no data
# that leaves the GUI (recordings, .ply, published topics) is touched.
GL_UP = np.array([0.0, 0.0, 1.0])       # pyqtgraph GLViewWidget screen-up
SCENE_UP_OPTICAL = (0.0, -1.0, 0.0)     # physical up, in an optical frame


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def _rotation_a_to_b(a, b) -> np.ndarray:
    """Smallest proper rotation R (3x3) with ``R @ a_hat == b_hat``.

    Rodrigues about the common perpendicular a x b::

        R = I + sin(t) K + (1 - cos(t)) K^2 ,  K = skew(axis)

    Degenerate cases: a parallel to b -> identity; a antiparallel to b ->
    180 deg about an arbitrary but stable axis perpendicular to a (built from
    a's smallest component, which can never be parallel to a); a or b of
    zero length -> identity.

    For the default optical up this yields a rotation of -90 deg about x:
    (0,-1,0) -> (0,0,1) (scene up becomes screen up) and (0,0,1) -> (0,1,0)
    (the camera's forward axis becomes the horizontal depth direction).
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)                       # already aligned
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, axis)
        axis /= float(np.linalg.norm(axis))
        K = _skew(axis)
        return np.eye(3) + 2.0 * (K @ K)           # 180 deg: sin=0, cos=-1
    K = _skew(v / s)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


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


def _poses_to_segments(poses: List, flip_z: bool, scale: float) -> Optional[np.ndarray]:
    """One glyph per pose, flattened to a (M*8, 3) 'lines'-mode vertex array."""
    if not poses:
        return None
    glyph = _GLYPH_PTS * scale
    if flip_z:
        glyph = glyph * np.array([-1.0, 1.0, -1.0])
    out = []
    for p in poses:
        R = _quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        t = np.array([p.position.x, p.position.y, p.position.z])
        pts = glyph @ R.T + t
        out.append(pts[_GLYPH_SEGS.reshape(-1)])
    return np.vstack(out).astype(np.float32)


class CloudGLView(gl.GLViewWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setBackgroundColor((20, 20, 24))
        self._scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32), size=2.5, pxMode=True)
        self._scatter.setGLOptions("opaque")
        self.addItem(self._scatter)
        self._lines: Dict[str, gl.GLLinePlotItem] = {}
        for kind, color in COLORS.items():
            item = gl.GLLinePlotItem(mode="lines", width=3 if kind in ("best", "exec") else 1.5,
                                     color=color, antialias=True)
            item.setVisible(False)
            self.addItem(item)
            self._lines[kind] = item
        self._visible = {k: True for k in COLORS}
        self._view_initialised = False
        # Cached RAW (data-frame) geometry.  Everything is re-pushed through
        # _to_view() whenever the scene-up changes, so cloud, glyphs and
        # camera can never drift apart.
        self._campose: Optional[np.ndarray] = None
        self._cloud_pts: Optional[np.ndarray] = None
        self._cloud_rgba = (0.75, 0.75, 0.78, 1.0)
        self._segs: Dict[str, Optional[np.ndarray]] = {k: None for k in COLORS}
        self._scene_up = np.asarray(SCENE_UP_OPTICAL, dtype=np.float64)
        self._view_R = _rotation_a_to_b(self._scene_up, GL_UP)

    # ── display rotation ────────────────────────────────────────────
    def _to_view(self, pts: np.ndarray) -> np.ndarray:
        """Map data-frame coordinates (N,3) into the GL view frame.

        THE single place the display rotation is applied."""
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        return (arr @ self._view_R.T).astype(np.float32)

    def set_scene_up(self, up) -> None:
        """Declare which data-frame direction is physically UP.

        Defaults to ``SCENE_UP_OPTICAL`` = (0,-1,0), the nominal up of a
        camera optical frame.  SEAM (not wired in this task): the pipeline
        already measures the real table-plane normal in the SAME optical
        frame every cycle -- grasp_pipeline/perception/cloud_extraction.py,
        ``estimate_plane_svd`` oriented toward the object.  Publishing that
        normal and calling this setter would swap the nominal up for the
        measured one with no other change.  A zero / non-finite vector is
        ignored (the current up is kept)."""
        try:
            v = np.asarray(up, dtype=np.float64).reshape(3)
        except Exception:
            return
        n = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else 0.0
        if n < 1e-6:
            return
        self._scene_up = v / n
        self._view_R = _rotation_a_to_b(self._scene_up, GL_UP)
        self._reapply_view_rotation()

    def _reapply_view_rotation(self) -> None:
        """Re-push every cached geometry through the current rotation."""
        has_cloud = self._cloud_pts is not None and len(self._cloud_pts) > 0
        if has_cloud:
            self._scatter.setData(pos=self._to_view(self._cloud_pts),
                                  color=self._cloud_rgba)
        for kind, segs in self._segs.items():
            if segs is not None:
                self._lines[kind].setData(pos=self._to_view(segs))
        if has_cloud and self._view_initialised:
            self._frame_cloud(self._cloud_pts)
        self.update()

    def set_campose(self, campose: Optional[np.ndarray]) -> None:
        """Remember where the physical camera captured from — the initial
        view faces the object the same way (the Open3D viewer's lesson:
        anything else shows the object from the back)."""
        if campose is not None:
            self._campose = np.asarray(campose, dtype=np.float64)

    def set_cloud(self, pts: Optional[np.ndarray], cols01: Optional[np.ndarray]) -> None:
        if pts is None or len(pts) == 0:
            return
        self._cloud_pts = np.asarray(pts)
        if cols01 is not None and len(cols01) == len(pts):
            rgba = np.concatenate([cols01, np.ones((len(cols01), 1), dtype=np.float32)], axis=1)
        else:
            rgba = (0.75, 0.75, 0.78, 1.0)
        self._cloud_rgba = rgba
        self._scatter.setData(pos=self._to_view(self._cloud_pts), color=rgba)
        if not self._view_initialised:
            self._frame_cloud(self._cloud_pts)

    def _frame_cloud(self, pts: np.ndarray) -> None:
        """Immediately restore the canonical object-centred camera.

        ``pts`` is RAW (data-frame); the GL camera lives in the rotated view
        frame, so the cloud AND the stored camera pose are mapped through
        _to_view() here."""
        # Robust framing: median centre + 90th-percentile radius, so a
        # few stray background points cannot zoom the object away.
        pv = self._to_view(pts)
        c = np.median(pv, axis=0)
        r = float(np.percentile(np.linalg.norm(pv - c, axis=1), 90))
        self.opts["center"] = QtGui.QVector3D(float(c[0]), float(c[1]), float(c[2]))
        dist = max(0.3, 4.5 * max(0.05, r))
        if self._campose is not None:
            v = self._to_view(self._campose.reshape(1, 3))[0] - c
            d = float(np.linalg.norm(v))
            if d > 1e-6:
                az = math.degrees(math.atan2(v[1], v[0]))
                el = math.degrees(math.asin(max(-1.0, min(1.0, v[2] / d))))
                self.setCameraPosition(distance=dist, elevation=el, azimuth=az)
                self._view_initialised = True
                self.update()
                return
        self.setCameraPosition(distance=dist, elevation=30.0, azimuth=45.0)
        self._view_initialised = True
        self.update()

    def reset_view(self) -> None:
        # The old implementation only changed this flag, so nothing moved
        # until another cloud happened to arrive. Re-frame the cloud now.
        if self._cloud_pts is not None and len(self._cloud_pts):
            self._frame_cloud(self._cloud_pts)
        else:
            self.opts["center"] = QtGui.QVector3D(0.0, 0.0, 0.0)
            self.setCameraPosition(distance=2.0, elevation=30.0, azimuth=45.0)
            self._view_initialised = False
            self.update()

    def set_poses(self, kind: str, poses: List) -> None:
        segs = _poses_to_segments(poses, FLIP_Z[kind], SCALE[kind])
        self._segs[kind] = segs
        item = self._lines[kind]
        if segs is None:
            item.setVisible(False)
            return
        item.setData(pos=self._to_view(segs))
        item.setVisible(self._visible[kind])

    def set_kind_visible(self, kind: str, visible: bool) -> None:
        self._visible[kind] = bool(visible)
        item = self._lines[kind]
        has_data = item.pos is not None and len(item.pos) > 0
        item.setVisible(visible and has_data)
