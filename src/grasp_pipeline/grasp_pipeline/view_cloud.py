"""A light RealSense point cloud for RViz (2026-09-29).

Majd: "I don't see the RealSense point cloud." The camera's own cloud is
off since 2026-09-09 (it cost a full core; nothing in the pipeline or GUI
uses it), and a full cloud is ~5 MB — with the 212 KB kernel UDP buffers the
stack still runs with, such messages rarely arrive whole. This publishes the
frame buffer's newest colour+depth pair (the one pointing uses, FrameBuffer
.paired) as a thinned, coloured cloud: every `stride`-th pixel (4 → ~19 k
points, ~300 KB), once per `period_s`, in the colour optical frame.

It costs nothing while nobody listens (checked every tick), and it steps
aside when another node publishes the same topic — the MoveIt launch's own
downsampler runs there with camera_pointcloud:=true.
"""
from __future__ import annotations

import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


class ViewCloudPublisher:
    def __init__(self, node, fb, topic: str, period_s: float = 1.0, stride: int = 4,
                 max_range_m: float = 2.0, callback_group=None) -> None:
        self._node, self._fb, self._topic = node, fb, topic
        self._stride = max(1, int(stride))
        self._max_range = float(max_range_m)
        self._pub = node.create_publisher(
            PointCloud2, topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        node.create_timer(max(0.1, float(period_s)), self._tick, callback_group=callback_group)
        self._grid = None           # cached pixel grid for the current image size

    def _tick(self) -> None:
        if self._pub.get_subscription_count() == 0:
            return
        if self._node.count_publishers(self._topic) > 1:   # the MoveIt downsampler is on
            return
        bgr, depth, K, scale, hdr = self._fb.paired(max_dt_s=0.1)
        if bgr is None or depth is None or hdr is None or not all(K):
            return
        msg = self.build(bgr, depth, K, scale, hdr)
        if msg is not None:
            self._pub.publish(msg)

    def build(self, bgr: np.ndarray, depth: np.ndarray, K, scale, hdr) -> PointCloud2 | None:
        s = self._stride
        d = np.asarray(depth)[::s, ::s]
        z = d.astype(np.float32) / float(scale or 1000.0) if d.dtype == np.uint16 else d.astype(np.float32)
        h, w = z.shape
        if self._grid is None or self._grid[0].shape != (h, w):
            v, u = np.mgrid[0:h, 0:w]
            self._grid = (u.astype(np.float32) * s, v.astype(np.float32) * s)
        u, v = self._grid
        ok = np.isfinite(z) & (z > 0.05) & (z < self._max_range)
        if not ok.any():
            return None
        fx, fy, cx, cy = (float(k) for k in K)
        zz = z[ok]
        xyz = np.stack([(u[ok] - cx) / fx * zz, (v[ok] - cy) / fy * zz, zz], axis=1)
        c = np.asarray(bgr)[::s, ::s][ok].astype(np.uint32)             # colour at the same pixels (aligned)
        rgb = (c[:, 2] << 16) | (c[:, 1] << 8) | c[:, 0]
        buf = np.empty((len(zz), 4), np.float32)
        buf[:, :3] = xyz
        buf[:, 3] = rgb.view(np.float32)
        msg = PointCloud2()
        msg.header = hdr
        msg.height, msg.width = 1, len(zz)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(("x", "y", "z", "rgb"))]
        msg.is_bigendian, msg.is_dense = False, True
        msg.point_step, msg.row_step = 16, 16 * len(zz)
        msg.data = buf.tobytes()
        return msg
