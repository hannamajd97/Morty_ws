"""The fixed Astra as the pointing camera (2026-09-29).

Majd: pointing from either camera, chosen in Advanced -> Detection; "when 1
is selected then it should only be focused on it" — no fusion. The pointing
selector (point_selector.PointOrVoiceSelector) does not care which camera it
watches: it needs a colour frame, the depth of the same moment with its
intrinsics, and the objects' boxes in THAT image. For the RealSense these
come from the frame buffer and YOLO; this class supplies them for the Astra,
so the finger ray, the arrow and the vote are exactly the RealSense code.

Frames. The shuttle relays /astra/color/compressed_slow (JPEG 640x480) and
/astra/depth/compressed_slow (compressedDepth PNG 320x240, registered to the
colour camera: frame astra_color_optical_frame, intrinsics exactly half the
colour ones) at ~7 Hz each, stamped independently. Two LatestMessage readers
hold the last frames inside DDS — no Python callbacks, nothing decoded while
nobody points with the Astra. Each colour frame is its own "stream" (keyed
by its stamp), so the colour reader keeps the last 8 and the depth frame is
paired with the colour frame nearest in stamp (measured 2026-09-29: within
+-30 ms, worst 61 ms). The depth is upsampled x2 (nearest) to the colour
size and the colour intrinsics are used — the geometry of the RViz Astra
cloud that Majd's pose was tuned on. A frame older than max_age_s (by its
receipt time; the shuttle's clock is not ours) is never used: a frozen
driver (astra-driver-freezes) must not keep voting with its last picture.

Objects. The Astra runs no detector. Each RealSense detection's 3-D centre
(object_centres_3d, the selector's own rule) is moved into the Astra frame
with TF and handed to the selector as ``centre_3d``, so pointing uses the
object where the RealSense measured it and the saved Astra pose places it in
the Astra's view. The box shown on the Astra picture is the projection of
the RealSense box between the object's depth and the table behind it — if
the boxes sit on the objects, the pose is right. The index order is kept, so
the chosen index is the RealSense detection's index. An object without
RealSense depth (glass) cannot be placed and is left out.

Overlay. While a session runs, every processed frame is drawn with the
RealSense overlay's drawing (debug_image_publisher.draw_debug: boxes, hand,
arrow) and published as JPEG on overlay_topic while someone listens; the
GUI shows it in its Astra views.
"""
from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage
from ur3e_fast_observation.compressed_depth import decode_compressed_depth
from ur3e_fast_observation.latest_message import LatestMessage

from .debug_image_publisher import draw_debug
from .point_selector import deproject, object_centres_3d, project
from .registration import lookup_transform_matrix

ASTRA_FRAME = "astra_color_optical_frame"
# A box no pointer can reach: objects the Astra cannot place (no RealSense depth).
OFF_IMAGE_BOX = [-1.0e5, -1.0e5, -1.0e5 + 1.0, -1.0e5 + 1.0]


def _cdr_stamp(data: bytes) -> bytes:
    """LatestMessage stream key: the serialized header stamp (sec, nanosec
    right after the 4-byte CDR encapsulation header), so every frame is its
    own stream and the reader returns all of the last ``depth`` frames."""
    return bytes(data[4:12])


def _stamp_s(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


class AstraPointingCamera:
    """Frames, object boxes and the overlay for pointing with the Astra.

    One session: ``begin(dets, rs_sample)`` places the objects; then the
    selector's loop calls ``next_frame()`` (its frame provider) and reads
    ``sample`` (its depth provider); ``publish_overlay()`` after each frame.
    """

    def __init__(self, node, tf_buffer, *,
                 colour_topic: str = "/astra/color/compressed_slow",
                 depth_topic: str = "/astra/depth/compressed_slow",
                 info_topic: str = "/astra/color/camera_info",
                 overlay_topic: str = "/astra/pointing/compressed",
                 max_age_s: float = 0.5,
                 pair_dt_s: float = 0.08,
                 logger=None) -> None:
        self._tf = tf_buffer
        self._log = logger or node.get_logger()
        self.max_age_s = float(max_age_s)
        self.pair_dt_s = float(pair_dt_s)
        self._colour_rd = LatestMessage(node, CompressedImage, colour_topic, depth=8, stream_key=_cdr_stamp)
        self._depth_rd = LatestMessage(node, CompressedImage, depth_topic, depth=1)
        self._info_rd = LatestMessage(node, CameraInfo, info_topic, depth=1)
        self._pub = node.create_publisher(CompressedImage, overlay_topic, 1)
        self._colours: collections.deque = collections.deque(maxlen=8)   # (receipt, stamp_s, msg)
        self._depth: Optional[Tuple[float, float, Any]] = None           # (receipt, stamp_s, msg)
        self._info: Optional[Tuple[int, int, float, float, float, float]] = None
        self._used_stamp: Optional[float] = None
        self._drawn_stamp: Optional[float] = None
        # the current frame, in the selector's depth_provider format
        self.sample: Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float], float, Any]] = None
        self.dets: List[Dict[str, Any]] = []
        self.pointer_px: Optional[Tuple[float, float]] = None
        self.hand: Tuple[Any, Any] = (None, None)

    # ── frames ───────────────────────────────────────────────────────
    def _refresh(self) -> None:
        now = time.monotonic()
        for msg, age in self._colour_rd.take_streams(with_age=True):
            self._colours.append((now - age, _stamp_s(msg.header), msg))
        got = self._depth_rd.take_streams(with_age=True)
        if got:
            msg, age = got[-1]
            self._depth = (now - age, _stamp_s(msg.header), msg)
        info = self._info_rd.take()
        if info is not None and info.k[0] > 0:
            self._info = (int(info.width), int(info.height),
                          float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]))

    def alive(self) -> bool:
        """An Astra depth frame arrived within max_age_s and the intrinsics are known."""
        self._refresh()
        return (self._depth is not None and self._info is not None
                and time.monotonic() - self._depth[0] < self.max_age_s)

    def _K(self, width: int) -> Tuple[float, float, float, float]:
        w0, _h0, fx, fy, cx, cy = self._info
        s = float(width) / float(w0)
        return (fx * s, fy * s, cx * s, cy * s)

    def next_frame(self, wait_s: float = 0.3) -> Optional[np.ndarray]:
        """The selector's frame provider: waits up to wait_s for a depth frame
        newer than the last one handed out, pairs it with its colour frame and
        decodes both. Returns the colour image and sets ``sample``; None (and
        ``sample`` None) when no fresh pair came — the selector then sees no
        hand, and its vote only fades."""
        deadline = time.monotonic() + float(wait_s)
        while True:
            self._refresh()
            d, c = self._depth, None
            now = time.monotonic()
            if (d is not None and self._info is not None and d[1] != self._used_stamp
                    and now - d[0] < self.max_age_s):
                # its colour partner may still be on the way (the two streams
                # are relayed separately): keep waiting for it until the deadline
                fresh = [e for e in self._colours if now - e[0] < self.max_age_s + self.pair_dt_s]
                c = min(fresh, key=lambda e: abs(e[1] - d[1])) if fresh else None
                if c is not None and abs(c[1] - d[1]) <= self.pair_dt_s:
                    break
            if now >= deadline:
                self.sample = None
                return None
            time.sleep(0.01)
        self._used_stamp = d[1]
        bgr = cv2.imdecode(np.frombuffer(bytes(c[2].data), np.uint8), cv2.IMREAD_COLOR)
        depth = decode_compressed_depth(d[2])
        if bgr is None or depth is None:
            self.sample = None
            return None
        H, W = bgr.shape[:2]
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        self.sample = (bgr, depth, self._K(W), 1.0, d[2].header)
        return bgr

    # ── objects ──────────────────────────────────────────────────────
    def begin(self, dets: List[Dict[str, Any]], rs_sample) -> List[Dict[str, Any]]:
        """The session's objects for the selector: the RealSense detections in
        the same order, each with ``centre_3d`` in the Astra frame and its box
        projected into the Astra image (OFF_IMAGE_BOX when it cannot be
        placed). rs_sample = the frame buffer's (rgb, depth, K, scale, header)."""
        self.pointer_px, self.hand = None, (None, None)
        self._used_stamp = self._drawn_stamp = None
        self.dets = [dict(d, bbox=list(OFF_IMAGE_BOX), centre_3d=None) for d in dets]
        t_end = time.monotonic() + 1.0
        while self._info is None and time.monotonic() < t_end:
            self._refresh()
            time.sleep(0.05)
        try:
            _rgb, depth, K_rs, scale, hdr = rs_sample
        except (TypeError, ValueError):
            depth = None
        if self._info is None or depth is None or K_rs is None or not K_rs[0]:
            self._log.warn("[point] Astra: no Astra intrinsics or no RealSense depth — no object can be placed",
                           throttle_duration_sec=30.0)
            return self.dets
        depth_m = np.asarray(depth)
        if depth_m.dtype not in (np.float32, np.float64):
            depth_m = depth_m.astype(np.float32) / float(scale or 1000.0)
        K_rs = tuple(float(k) for k in K_rs)
        try:
            # no logger: its "falling back to the latest transform" note would
            # repeat every session; the arm stands still while it waits.
            T = lookup_transform_matrix(self._tf, ASTRA_FRAME, str(hdr.frame_id), hdr.stamp, 0.5)
        except Exception as exc:
            self._log.warn(f"[point] Astra: no transform {hdr.frame_id} -> {ASTRA_FRAME} ({exc}) — "
                           "is the Astra TF running (astra_restart_laptop.sh)?", throttle_duration_sec=30.0)
            return self.dets
        K_a = self._K(self._info[0])
        centres = object_centres_3d(dets, depth_m, K_rs)
        for out, d, c in zip(self.dets, dets, centres):
            if c is None:
                continue
            out["centre_3d"] = T[:3, :3] @ c + T[:3, 3]
            # the box: the RealSense box at the object's front, and as deep as
            # the object is wide (round and square things). The table or wall
            # behind it was used first — with the wrist camera looking level
            # that is the room, and the box grew far above the object.
            x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
            deep = min(0.25, (x2 - x1) * c[2] / K_rs[0])
            corners = [deproject(u, v, z, K_rs) for z in (c[2], c[2] + deep)
                       for u, v in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
            px = [project(T[:3, :3] @ p + T[:3, 3], K_a) for p in corners]
            if any(p is None for p in px):
                continue
            us, vs = [p[0] for p in px], [p[1] for p in px]
            out["bbox"] = [min(us), min(vs), max(us), max(vs)]
        placed = sum(1 for d in self.dets if d["centre_3d"] is not None)
        if placed < len(self.dets):
            self._log.info(f"[point] Astra: {placed}/{len(self.dets)} objects placed "
                           "(the others have no RealSense depth)", throttle_duration_sec=60.0)
        return self.dets

    # ── overlay ──────────────────────────────────────────────────────
    def publish_overlay(self, status_text: Optional[str] = None) -> None:
        """Draw the current frame (once) with the objects, the hand and the
        arrow, and publish it as JPEG — only while someone listens."""
        if self.sample is None or self._pub.get_subscription_count() == 0:
            return
        stamp = _stamp_s(self.sample[4])
        if stamp == self._drawn_stamp:
            return
        self._drawn_stamp = stamp
        vis = draw_debug(self.sample[0], self.dets, pointer_px=self.pointer_px,
                         hand_landmarks_px=self.hand[0], hand_connections=self.hand[1],
                         status_text=status_text, show_scores=False)
        ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        msg = CompressedImage()
        msg.header = self.sample[4]
        msg.format = "bgr8; jpeg compressed bgr8"
        msg.data = jpg.tobytes()
        self._pub.publish(msg)
