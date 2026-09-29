"""
debug_image_publisher.py
=========================

This module defines :class:`DebugImagePublisher` which publishes a
debug image over a ROS2 topic. It annotates detections on top of
a BGR image for visual debugging during grasp selection. The
annotated image is published as a sensor_msgs/Image.

Example::

    from grasp_pipeline.debug_image_publisher import DebugImagePublisher
    pub = DebugImagePublisher(node, topic="/debug/image")
    pub.publish(header, bgr_image, detections, selected_idx=2)

"""
#!/usr/bin/env python3
# grasp_pipeline/debug_image_publisher.py

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Callable
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import hashlib
import math
import random
import threading


def _stable_color_for_class(name: str) -> Tuple[int, int, int]:
    """
    Deterministically map a class name to a BGR color with decent brightness.
    """
    if not name:
        return (0, 255, 0)
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()
    r = int(h[:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)

    def boost(x: int) -> int:
        return max(64, min(255, int(0.6 * x + 80)))

    return (boost(b), boost(g), boost(r))  # BGR


def draw_debug(
    bgr,
    dets: Optional[List[Dict[str, Any]]],
    selected_idx: Optional[int] = None,
    pointer_px: Optional[Tuple[float, float]] = None,
    hand_landmarks_px: Optional[List[Tuple[float, float]]] = None,
    hand_connections: Optional[List[Tuple[int, int]]] = None,
    selected_mask: Optional[object] = None,
    *,
    status_text: Optional[str] = None,
    show_scores: bool = True,
    show_centers: bool = False,
    font_scale: float = 0.36,
    thickness: int = 2,
    mask_alpha: float = 0.45,
):
    """The annotated copy of ``bgr``: boxes, names, hand skeleton, pointing
    arrow, status lines. DebugImagePublisher's drawing, a function since
    2026-09-29 so the Astra pointing view (astra_pointing.py) shows the same
    picture on the Astra image."""
    vis = bgr.copy()
    H, W = vis.shape[:2]

    if selected_mask is not None:
        try:
            import numpy as np

            m = np.asarray(selected_mask).astype(bool)
            if m.shape[0] == H and m.shape[1] == W and mask_alpha > 0.0:
                overlay = vis.copy()
                overlay[m] = (0, 255, 0)
                a = float(max(0.0, min(1.0, mask_alpha)))
                vis = cv2.addWeighted(overlay, a, vis, 1.0 - a, 0.0)
                cnts, _ = cv2.findContours(
                    m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, cnts, -1, (0, 200, 0), 2)
        except Exception:
            pass

    for i, d in enumerate(dets or []):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        label = d.get("display", d.get("label", f"id{i}"))
        score = d.get("score", None)

        base_color = _stable_color_for_class(label)
        # Reach warning colours (2026-09-09): orange = edge of reach,
        # red = out of reach; green objects keep their class colour.
        reach_state = str(d.get("reach_state") or "")
        if reach_state == "red":
            base_color = (0, 0, 230)
        elif reach_state == "orange":
            base_color = (0, 140, 255)
        is_sel = selected_idx is not None and i == selected_idx
        rect_color = (0, 255, 255) if is_sel else base_color
        rect_thick = max(2, thickness + (1 if is_sel else 0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), rect_color, rect_thick)

        idx_center = (x1 + 8, max(0, y1 - 10))
        cv2.circle(vis, idx_center, 8, (36, 36, 36), -1)
        cv2.putText(
            vis,
            str(i),
            (idx_center[0] - 5, idx_center[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Who named it (2026-09-15, Majd): "ys" = the trained YOLO,
        # "q8b" = the naming model on a YOLOE box.
        source = str(d.get("source") or "")
        tag = f"{source} {label}" if source else label
        if is_sel:
            tag = "[CHOSEN] " + tag
        # Majd 2026-08-25: "I can't see which vitamin tube is
        # which" — show the cap colour + the product it means.
        cap_c = str(d.get("cap_colour") or "")
        if cap_c:
            product = {
                "yellow": "C", "orange": "C",
                "red": "B12", "purple": "B12",
                "blue": "Multi",
            }.get(cap_c, "")
            tag += f" [{cap_c}{('=' + product) if product else ''}]"
        if show_scores and isinstance(score, (int, float)):
            tag += f" {score:.2f}"
        if d.get("dist_note"):
            tag += f" {d['dist_note']}"
        if d.get("precomputed") == "lin":
            tag += " **"         # grasps computed AND a straight approach verified
        elif d.get("precomputed"):
            tag += " *"          # grasps already computed for this object (precompute cache)
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Keep long labels inside the image instead of clipping
        # right-edge objects such as the vitamin tube.
        tag_bg_x1 = max(0, min(x1, W - tw - 6))
        tag_bg_y1 = max(0, y1 - th - 6)
        tag_bg_x2, tag_bg_y2 = tag_bg_x1 + tw + 6, y1
        tag_bg = (36, 36, 36) if reach_state not in ("red", "orange") else (
            (0, 0, 120) if reach_state == "red" else (0, 70, 130))
        cv2.rectangle(vis, (tag_bg_x1, tag_bg_y1), (tag_bg_x2, tag_bg_y2), tag_bg, -1)
        cv2.putText(
            vis,
            tag,
            (tag_bg_x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if show_centers:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.drawMarker(
                vis,
                (cx, cy),
                rect_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
            )

    if hand_landmarks_px:
        try:
            pts = [(int(p[0]), int(p[1])) for p in hand_landmarks_px]
            if hand_connections:
                for a, b in hand_connections:
                    if 0 <= a < len(pts) and 0 <= b < len(pts):
                        cv2.line(vis, pts[a], pts[b], (255, 200, 0), 2, cv2.LINE_AA)
            for i, (u, v) in enumerate(pts):
                if 0 <= u < W and 0 <= v < H:
                    r = 4 if i == 8 else 3
                    col = (0, 255, 255) if i == 8 else (255, 255, 0)
                    cv2.circle(vis, (u, v), r, col, -1)
        except Exception:
            pass

    if pointer_px is not None:
        u, v = int(pointer_px[0]), int(pointer_px[1])
        # Draw ray line from fingertip to the extended target point
        if hand_landmarks_px and len(hand_landmarks_px) > 8:
            tip_u, tip_v = int(hand_landmarks_px[8][0]), int(hand_landmarks_px[8][1])
            cv2.line(vis, (tip_u, tip_v), (u, v), (0, 100, 255), 2, cv2.LINE_AA)
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u, v), 8, (0, 0, 255), -1)
            cv2.drawMarker(
                vis,
                (u, v),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.putText(
                vis,
                "target",
                (min(u + 10, W - 60), max(12, v - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    if status_text:
        try:
            lines = [line for line in str(status_text).split("\n") if line.strip()]
            y = 24
            for line in lines[:4]:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(vis, (8, max(0, y - th - 6)), (min(W - 1, 16 + tw), y + 6), (20, 20, 20), -1)
                cv2.putText(
                    vis,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += th + 12
        except Exception:
            pass
    return vis


class DebugImagePublisher:
    """
    Publishes a single debug image with:
      - YOLO boxes (per-class stable colors)
      - Index numbers on each bbox
      - Optional score text
      - Optional center markers
      - Highlighted [CHOSEN] bbox
      - Optional fingertip pixel (red dot + crosshair + 'finger' label)
    """

    def __init__(
        self,
        node,
        topic: str = "/yolo/debug_image",
        *,
        show_window: bool = False,
        window_name: str = "Grasp Debug",
        window_width: int = 1280,
        window_height: int = 720,
        on_click_index: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.node = node
        self.pub = node.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()
        self.show_window = bool(show_window)
        self.window_name = str(window_name)
        self.window_width = max(int(window_width), 0)
        self.window_height = max(int(window_height), 0)
        self.on_click_index = on_click_index
        self._window_ready = False
        self._lock = threading.Lock()
        self._last_dets: List[Dict[str, Any]] = []
        self._scale_x = 1.0
        self._scale_y = 1.0

        # NOTE (2026-08-06): the window is drawn INLINE, on whichever thread
        # calls publish() — i.e. the ROS executor thread. This is deliberate and
        # must not be "fixed" by moving it to a worker thread: as
        # communication_llm/mic_visualizer_node.py documents, OpenCV's window
        # event loop only repaints from the thread that created the window, and
        # on Linux only the main thread holds the X11 display connection. A
        # background GUI thread creates the window but never displays it.
        #
        # The trade-off: under XWayland `cv2.imshow`/`cv2.waitKey` can be slow to
        # return and will stall the executor (SYSTEM_STATE_LOG #21). If that
        # happens, launch with `show_debug_window:=false` and view the annotated
        # image on the /yolo/debug_image topic instead (rqt_image_view / RViz),
        # or run the desktop session on Xorg where the window behaves normally.

    def publish(
        self,
        header,
        bgr: Optional[cv2.UMat] | Optional[object],
        dets: Optional[List[Dict[str, Any]]],
        selected_idx: Optional[int] = None,
        pointer_px: Optional[Tuple[float, float]] = None,
        hand_landmarks_px: Optional[List[Tuple[float, float]]] = None,
        hand_connections: Optional[List[Tuple[int, int]]] = None,
        selected_mask: Optional[object] = None,
        *,
        status_text: Optional[str] = None,
        show_scores: bool = True,
        show_centers: bool = False,
        font_scale: float = 0.36,   # 0.42 until 2026-09-09 (Majd: "give it -1")
        thickness: int = 2,
        mask_alpha: float = 0.45,
        render_window: bool = True,
    ) -> None:
        """
        Draw onto the SAME debug image topic used in your RViz2 setup.
        """
        if bgr is None:
            return

        with self._lock:
            vis = draw_debug(
                bgr, dets, selected_idx, pointer_px, hand_landmarks_px, hand_connections,
                selected_mask, status_text=status_text, show_scores=show_scores,
                show_centers=show_centers, font_scale=font_scale, thickness=thickness,
                mask_alpha=mask_alpha)

            msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            msg.header = header
            self.pub.publish(msg)
            self._last_dets = [dict(d) for d in (dets or [])]

            if self.show_window and render_window:
                self._show_window(vis)

    def click_at(self, ux: int, uy: int) -> bool:
        """Hit-test a click at ORIGINAL-image pixel coords against the last
        published detections and fire ``on_click_index`` on a hit.

        Used by both the native cv2 window (via :meth:`_on_mouse`, after
        unscaling) and the ``<debug_image_topic>_mouse_left`` subscriber fed by
        rqt_image_view's "publish click location" checkbox — the working
        click-to-select path on this machine, where the cv2 window cannot run
        (see SYSTEM_STATE_LOG #28). Returns True if a detection was hit.
        (2026-08-06)"""
        if not self.on_click_index:
            return False
        try:
            hit_idx = None
            for i, d in enumerate(self._last_dets):
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                if x1 <= int(ux) <= x2 and y1 <= int(uy) <= y2:
                    hit_idx = i
                    break
            if hit_idx is not None:
                # Remember WHAT was clicked, not just its list position:
                # YOLO reorders detections between frames, so a bare index
                # can point at a different object by the time it is consumed
                # (2026-08-07 — clicked the maze puzzle, got the vitamin
                # tube). The node re-resolves this det by label+position.
                self.last_clicked_det = dict(self._last_dets[hit_idx])
                self.on_click_index(hit_idx)
                return True
        except Exception:
            pass
        return False

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ux = int(round(float(x) / max(self._scale_x, 1e-6)))
        uy = int(round(float(y) / max(self._scale_y, 1e-6)))
        self.click_at(ux, uy)

    def stop(self) -> None:
        """Close the debug window, if one was opened."""
        try:
            if self._window_ready:
                cv2.destroyWindow(self.window_name)
                self._window_ready = False
        except Exception:
            pass

    def _show_window(self, image_bgr) -> None:
        """Render a local OpenCV window with the same debug image.

        Called inline from :meth:`publish` — see the threading note in
        ``__init__``; this must run on the thread that owns the window."""
        try:
            if not self._window_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                if self.window_width > 0 and self.window_height > 0:
                    cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
                cv2.setMouseCallback(self.window_name, self._on_mouse)
                self._window_ready = True
                if self.node is not None:
                    self.node.get_logger().info(
                        f"Debug window '{self.window_name}' opened — click an object to select it."
                    )
            out = image_bgr
            if self.window_width > 0 and self.window_height > 0:
                h, w = image_bgr.shape[:2]
                sx = float(self.window_width) / float(max(w, 1))
                sy = float(self.window_height) / float(max(h, 1))
                s = min(sx, sy)
                if s > 0 and abs(s - 1.0) > 1e-3:
                    out = cv2.resize(
                        image_bgr,
                        (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                        interpolation=cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA,
                    )
                    self._scale_x = float(out.shape[1]) / float(max(w, 1))
                    self._scale_y = float(out.shape[0]) / float(max(h, 1))
                else:
                    self._scale_x = 1.0
                    self._scale_y = 1.0
            else:
                self._scale_x = 1.0
                self._scale_y = 1.0
            cv2.imshow(self.window_name, out)
            cv2.waitKey(1)
            self._window_fail_count = 0
        except Exception as e:
            # Don't give up on the first hiccup: a transient failure while the
            # compositor is still settling used to disable the window for the
            # whole session, and the single warning scrolled past unnoticed.
            # Retry a few times, then stop trying — but say so loudly.
            # (2026-08-06)
            self._window_fail_count = getattr(self, "_window_fail_count", 0) + 1
            self._window_ready = False
            if self.node is not None:
                self.node.get_logger().warn(
                    f"Debug window render failed ({self._window_fail_count}/5): {e}"
                )
            if self._window_fail_count >= 5:
                self.show_window = False
                if self.node is not None:
                    self.node.get_logger().error(
                        "Debug window DISABLED after 5 failures. View the annotated "
                        "image instead:  ros2 run rqt_image_view rqt_image_view "
                        "/yolo/debug_image"
                    )
