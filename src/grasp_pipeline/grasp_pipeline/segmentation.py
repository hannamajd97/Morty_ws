"""
segmentation.py
===============

Turn a detection bounding box into a precise per-pixel object mask. The pipeline
detects objects with YOLO (boxes), but to build a clean point cloud it needs the
exact object pixels, not the whole rectangle (which includes table/background).

Three interchangeable backends implement the same ``Segmenter.segment`` API, so
the pipeline can pick one via a parameter without changing call sites:

  * ``GrabCutSegmenter``        — classical OpenCV GrabCut. No model/GPU needed,
                                  decent on simple scenes, weaker on clutter.
  * ``UltralyticsSamSegmenter`` — Segment Anything (SAM) prompted by the bbox.
                                  Best boundaries, needs a SAM checkpoint.
  * ``UltralyticsYoloSegSegmenter`` — a YOLO segmentation model; picks the
                                  predicted mask whose box best overlaps (IoU)
                                  the requested bbox.

All backends return a boolean HxW mask the same size as the input image, or
None on failure (caller then falls back to the raw bbox).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np


def _clamp_bbox(bbox: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a bbox to image bounds and fix any inverted corners, so downstream
    slicing/rect math never goes out of range."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


class Segmenter:
    """Abstract backend interface: bbox -> boolean object mask."""
    def segment(self, bgr: np.ndarray, bbox_xyxy: Sequence[int]) -> Optional[np.ndarray]:
        """Return a boolean HxW mask in the same image size, or None on failure."""
        raise NotImplementedError


@dataclass
class GrabCutConfig:
    iters: int = 5            # GrabCut refinement iterations
    margin_px: int = 5        # shrink the seed rect inside the bbox (see segment())
    close_kernel: int = 7     # morphological close: fill small holes in the mask
    open_kernel: int = 3      # morphological open: remove small speckles


class GrabCutSegmenter(Segmenter):
    """Classical GrabCut: seeds foreground/background from the bbox rectangle and
    iteratively refines a mask using color statistics. No model or GPU required."""
    def __init__(self, cfg: GrabCutConfig | None = None) -> None:
        self.cfg = cfg or GrabCutConfig()

    def segment(self, bgr: np.ndarray, bbox_xyxy: Sequence[int]) -> Optional[np.ndarray]:
        import cv2

        if bgr is None or bgr.ndim != 3:
            return None
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = _clamp_bbox(bbox_xyxy, w, h)
        if x2 <= x1 or y2 <= y1:
            return None

        # GrabCut expects a rect (x, y, w, h) and works better if the rect is slightly inside the bbox.
        m = int(max(0, self.cfg.margin_px))
        rx1 = min(w - 1, x1 + m)
        ry1 = min(h - 1, y1 + m)
        rx2 = max(0, x2 - m)
        ry2 = max(0, y2 - m)
        if rx2 <= rx1 or ry2 <= ry1:
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        rect = (int(rx1), int(ry1), int(rx2 - rx1), int(ry2 - ry1))
        mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, int(self.cfg.iters), cv2.GC_INIT_WITH_RECT)
        except Exception:
            return None

        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

        # Light morphology to remove speckles and fill small holes.
        if self.cfg.close_kernel and self.cfg.close_kernel > 1:
            k = int(self.cfg.close_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        if self.cfg.open_kernel and self.cfg.open_kernel > 1:
            k = int(self.cfg.open_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

        return fg


class UltralyticsSamSegmenter(Segmenter):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = str(model_path)
        self.device = str(device)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        # Ultralytics tries to write config/cache; keep it out of $HOME by default.
        tmp = os.path.join(tempfile.gettempdir(), "Ultralytics")
        os.environ.setdefault("YOLO_CONFIG_DIR", tmp)
        os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", tmp)

        from ultralytics import SAM

        self.model = SAM(self.model_path)

    def segment(self, bgr: np.ndarray, bbox_xyxy: Sequence[int]) -> Optional[np.ndarray]:
        if bgr is None or bgr.ndim != 3:
            return None

        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = _clamp_bbox(bbox_xyxy, w, h)
        if x2 <= x1 or y2 <= y1:
            return None

        try:
            results = self.model(
                source=bgr,
                bboxes=[[x1, y1, x2, y2]],
                device=self.device,
                verbose=False,
            )
        except Exception:
            return None

        if not results:
            return None
        r0 = results[0]
        masks = getattr(r0, "masks", None)
        if masks is None:
            return None

        data = getattr(masks, "data", None)
        if data is None:
            return None

        # Ultralytics masks.data is typically [N, H, W] torch tensor.
        try:
            m0 = data[0]
            if hasattr(m0, "detach"):
                m0 = m0.detach()
            if hasattr(m0, "cpu"):
                m0 = m0.cpu()
            m0 = np.asarray(m0)
        except Exception:
            return None

        if m0.ndim != 2:
            return None

        if m0.shape[0] != h or m0.shape[1] != w:
            # Resize just in case; SAM should usually return full-res.
            import cv2

            m0 = cv2.resize(m0.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        return m0 > 0.5


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-Union of two xyxy boxes. Used by the YOLO-seg backend to
    pick which predicted mask best matches the requested detection box."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


class UltralyticsYoloSegSegmenter(Segmenter):
    """Use a YOLO segmentation model and select the mask best matching the given bbox."""

    def __init__(self, model_path: str, device: str = "cpu",
                 imgsz: int = 320, half: bool = True) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        # Match the live detector's settings. Without these Ultralytics
        # defaults to imgsz=640 fp32 and one selection cost 6.07 s on
        # this GTX 1080 (measured 2026-08-21) — the single biggest chunk
        # of click->motion latency.
        self.imgsz = int(imgsz) if int(imgsz) > 0 else 320
        self.half = bool(half) and str(device).startswith("cuda")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def segment(self, bgr: np.ndarray, bbox_xyxy: Sequence[int]) -> Optional[np.ndarray]:
        if bgr is None or bgr.ndim != 3:
            return None
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = _clamp_bbox(bbox_xyxy, w, h)
        if x2 <= x1 or y2 <= y1:
            return None

        try:
            results = self.model.predict(
                source=bgr, device=self.device, verbose=False,
                imgsz=self.imgsz, half=self.half)
        except Exception:
            return None
        if not results:
            return None
        r0 = results[0]
        masks = getattr(r0, "masks", None)
        boxes = getattr(r0, "boxes", None)
        if masks is None or boxes is None:
            return None
        data = getattr(masks, "data", None)
        xyxy = getattr(boxes, "xyxy", None)
        if data is None or xyxy is None:
            return None

        try:
            if hasattr(xyxy, "detach"):
                xyxy = xyxy.detach()
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu()
            box_np = np.asarray(xyxy, dtype=np.float32)
        except Exception:
            return None
        if box_np.ndim != 2 or box_np.shape[0] == 0:
            return None

        target = [float(x1), float(y1), float(x2), float(y2)]
        best_i = -1
        best_iou = -1.0
        for i in range(box_np.shape[0]):
            iou = _bbox_iou_xyxy(box_np[i].tolist(), target)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i < 0:
            return None

        try:
            m = data[best_i]
            if hasattr(m, "detach"):
                m = m.detach()
            if hasattr(m, "cpu"):
                m = m.cpu()
            m = np.asarray(m)
        except Exception:
            return None
        if m.ndim != 2:
            return None
        if m.shape[0] != h or m.shape[1] != w:
            import cv2

            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        return m > 0.5


def build_segmenter(
    backend: str,
    *,
    sam_model_path: Optional[str] = None,
    device: str = "cpu",
    grabcut_cfg: GrabCutConfig | None = None,
    imgsz: int = 320,
    half: bool = True,
) -> Segmenter:
    backend_norm = str(backend or "").strip().lower()
    if backend_norm in {"sam", "ultralytics_sam", "ultralytics"}:
        if not sam_model_path:
            raise ValueError("sam_model_path is required for ultralytics_sam backend")
        return UltralyticsSamSegmenter(sam_model_path, device=device)
    if backend_norm in {"yolo_seg", "yoloseg", "ultralytics_yolo_seg"}:
        if not sam_model_path:
            raise ValueError("sam_model_path is required for yolo_seg backend (path to YOLO seg .pt)")
        return UltralyticsYoloSegSegmenter(
            sam_model_path, device=device, imgsz=imgsz, half=half)
    if backend_norm in {"grabcut", "opencv_grabcut"}:
        return GrabCutSegmenter(grabcut_cfg)
    raise ValueError(f"Unknown segmentation backend: {backend}")
