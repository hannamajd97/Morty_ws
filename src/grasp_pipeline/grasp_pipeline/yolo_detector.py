"""
yolo_detector.py
================

This module wraps the Ultralytics YOLOv8x detector to perform
object detection on all COCO classes. The :class:`YoloWorldDetector`
provides a simple API to predict bounding boxes along with scores
and class labels.

Usage example::

    from grasp_pipeline.yolo_detector import YoloWorldDetector
    detector = YoloWorldDetector(model_path='yolov8x.pt', conf=0.25)
    dets = detector.predict(image_bgr)
    for d in dets:
        print(d['label'], d['score'], d['bbox'])

The returned detections are dictionaries with keys ``label``,
``score`` and ``bbox``. BBoxes are in the form [x1, y1, x2, y2].
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO


class YoloWorldDetector:
    def __init__(self, model_path: str = "yolov8x.pt",
                 classes=None,
                 conf: float = 0.25, imgsz: int = 640,
                 device: str = "cpu", half: bool = False,
                 per_class_min_conf: Dict[str, float] | None = None,
                 logger: Any | None = None) -> None:

        self.logger = logger
        self.device = str(device)
        self.model_path = str(model_path)

        self.model = YOLO(model_path)
        if self.logger:
            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                class_list = ", ".join(str(v) for _, v in sorted(names.items()))
            else:
                class_list = str(names)
            self.logger.info(f"YOLO detector model: {self.model_path}")
            self.logger.info(f"YOLO detector model classes: {class_list}")

        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.half = bool(half)
        # Remember the ORIGINAL global conf BEFORE any per-class lowering.
        # This is the floor used for classes WITHOUT an override — we
        # cannot use the (possibly lowered) `self.conf` for that because
        # then the lowering would silently relax every class.
        self.default_class_floor = float(conf)
        # Per-class minimum confidence override. The model is asked to
        # emit at the GLOBAL `conf` floor (we lower it if any per-class
        # override needs a value below the global). After inference we
        # filter: detection of class X is kept only if its score ≥ the
        # per-class threshold (or `default_class_floor` if no override).
        self.per_class_min_conf: Dict[str, float] = {
            str(k): float(v) for k, v in (per_class_min_conf or {}).items()
        }
        if self.per_class_min_conf:
            min_override = min(self.per_class_min_conf.values())
            if min_override < self.conf:
                if self.logger:
                    self.logger.info(
                        f"Lowering global YOLO conf from {self.conf:.3f} to "
                        f"{min_override:.3f} so per-class overrides can "
                        f"surface (per-class={self.per_class_min_conf}). "
                        f"Other classes still require >= {self.default_class_floor:.3f}."
                    )
                self.conf = float(min_override)

    def set_classes(self, classes) -> None:
        """No-op for API compatibility. YOLOv8x detects all COCO classes."""
        if self.logger:
            self.logger.info("set_classes ignored: YOLOv8x detects all 80 COCO classes")

    def predict(self, bgr_image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on an image and return detections.

        Detections include the class ``label``, detection ``score``
        and bounding box ``bbox`` (format [x1, y1, x2, y2]). The
        detections are sorted descending by confidence.

        :param bgr_image: HxWx3 BGR image (OpenCV format)
        :returns: List of detection dicts sorted by score.
        """
        if bgr_image is None or bgr_image.ndim != 3:
            return []

        res = None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                res = self.model.predict(
                    bgr_image,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
                last_error = None
                break
            except Exception as e:
                last_error = e
                message = str(e)
                if (
                    attempt == 0
                    and self.device.lower() in {"cuda", "cuda:0", "0"}
                    and ("no kernel image" in message.lower() or "cuda error" in message.lower())
                ):
                    if self.logger:
                        self.logger.warning(
                            "YOLO CUDA inference failed (likely GPU/torch mismatch). Falling back to CPU. "
                            "If you want GPU, install a torch build compatible with your GPU (e.g. sm_61 for GTX 1080)."
                        )
                    self.device = "cpu"
                    continue
                break

        if res is None:
            if self.logger and last_error is not None:
                self.logger.error(f"YOLO inference failed: {last_error}")
            return []

        out: List[Dict[str, Any]] = []
        names = res.names
        # default_class_floor was captured at construction BEFORE the
        # global conf was lowered for per-class overrides. Classes NOT
        # in per_class_min_conf must clear that original floor.
        for b in res.boxes:
            cls_id = int(b.cls.item())
            label = names.get(cls_id, str(cls_id))
            score = float(b.conf.item())
            floor = self.per_class_min_conf.get(label, self.default_class_floor)
            if score < floor:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append({"label": label, "score": score, "bbox": [x1, y1, x2, y2]})
        # sort high→low
        out.sort(key=lambda d: d["score"], reverse=True)
        return out
