"""Open-set detection: a box and a name for every object on the table,
including the ones the trained YOLO has never seen.

The trained YOLO knows ten objects and forces everything else into one of
them (a drink carton became ``wooden_maze_puzzle 0.87``, 2026-09-15). This
module gives the rest of the table a proper existence:

* **Proposals** — YOLOE (prompt-free) draws a box around every object in the
  frame. Measured on this machine: the ``m`` model at 416 px takes 26 ms and
  found 7 of 7 objects where the ``s`` model lost a translucent bottle.
  The boxes are cleaned with what the robot already knows: the gripper mask
  from the self-filter, the depth image (no depth or farther than the table
  = background), the reach envelope (out of reach = ignored, never named),
  and containment/overlap between proposals. Boxes the trained YOLO already
  owns are dropped, so known objects keep YOLO's name.
* **Tracks** — an unknown box is matched to the previous frame by overlap,
  so an object is named once when it appears and keeps its name until it
  leaves; naming never runs per frame.
* **Naming** — a stable, in-reach track is sent to the naming service
  (Qwen3-VL-8B behind llama-server, started by the GUI) as a crop of at most
  224 px with a 6-token answer: 0.15 s per object, the same names as the
  full-size crop (measured 2026-09-15; 160 px starts to guess, batching
  several objects in one picture is slower and worse). The request runs on
  its own thread; the detection thread never waits.

The result is a detection dict in the trained YOLO's format plus
``source="q8b"``; YOLO's own carry ``source="ys"``. They join the same
``detections_cache``, so the precompute cache, the overlay, pointing and
voice treat a named unknown object like any other. ``open_set_enabled``
false leaves the frame loop exactly as before.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ASSET = "yoloe-26m-seg-pf.pt"
_PROMPT = "What object is this? Answer with 1-3 words only."


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / ua if ua > 0 else 0.0


def _contained(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of box a that lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix / area if area > 0 else 0.0


def clean_name(text: str) -> str:
    """'A blue computer mouse.' -> 'blue computer mouse' (at most 3 words)."""
    t = re.sub(r"[^a-z0-9 \-]", " ", text.lower())
    words = [w for w in t.split() if w not in ("a", "an", "the", "of", "with", "is", "this")]
    return " ".join(words[:3]) or "object"


class _Track:
    __slots__ = ("bbox", "score", "first", "last", "name", "pending", "tries", "next_try", "reach")

    def __init__(self, bbox: List[int], score: float, now: float) -> None:
        self.bbox, self.score, self.first, self.last = bbox, score, now, now
        self.name: Optional[str] = None
        self.pending = False
        self.tries = 0
        self.next_try = 0.0
        self.reach = ""


class OpenSetDetector:
    """Owns the proposal model, the unknown-object tracks and the naming
    client. ``detect`` runs on the detection thread; naming replies land on
    the tracks from the worker thread under ``_lock``."""

    def __init__(self, node: Any) -> None:
        self._node = node
        self._model = None
        self._model_failed = False
        self._tracks: List[_Track] = []
        self._lock = threading.Lock()
        self._jobs: "queue.Queue[Tuple[_Track, np.ndarray]]" = queue.Queue()
        self._frame = 0
        self._last_props: List[Tuple[List[int], float]] = []
        self._service_warned = 0.0
        self._stat_t = time.monotonic()
        self._stat = {"frames": 0, "yoloe_ms": 0.0, "named": 0, "name_s": 0.0}
        threading.Thread(target=self._naming_worker, name="open_set_naming", daemon=True).start()

    # ---- parameters ---------------------------------------------------------
    def _p(self, name: str, default):
        try:
            v = self._node.get_parameter(name).value
            return default if v is None else v
        except Exception:
            return default

    @property
    def enabled(self) -> bool:
        return bool(self._p("open_set_enabled", False))

    # ---- model ----------------------------------------------------------------
    def _weights_path(self) -> str:
        p = str(self._p("open_set_weights", "") or "")
        if p:
            return os.path.expanduser(p)
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", _ASSET)

    def _ensure_model(self):
        if self._model is not None or self._model_failed:
            return self._model
        log = self._node.get_logger()
        try:
            from ultralytics import YOLOE
            path = self._weights_path()
            if not os.path.exists(path):
                from ultralytics.utils.downloads import attempt_download_asset
                log.info(f"[open-set] downloading {_ASSET} (73 MB, once)")
                got = attempt_download_asset(_ASSET)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                os.replace(str(got), path)
            t = time.monotonic()
            model = YOLOE(path)
            model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False,
                          device=getattr(self._node.detector, "device", 0),
                          imgsz=int(self._p("open_set_imgsz", 416)))
            self._model = model
            log.info(f"[open-set] YOLOE loaded from {path} in {time.monotonic() - t:.1f} s")
        except Exception as e:  # noqa: BLE001
            self._model_failed = True
            log.warn(f"[open-set] proposal model unavailable ({e}); unknown objects are not detected")
        return self._model

    # ---- proposals --------------------------------------------------------------
    def _propose(self, rgb: np.ndarray) -> List[Tuple[List[int], float]]:
        model = self._ensure_model()
        if model is None:
            return []
        t = time.monotonic()
        r = model.predict(rgb, verbose=False, device=getattr(self._node.detector, "device", 0),
                          conf=float(self._p("open_set_conf", 0.10)),
                          imgsz=int(self._p("open_set_imgsz", 416)))[0]
        self._stat["yoloe_ms"] += (time.monotonic() - t) * 1000.0
        self._stat["frames"] += 1
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                out.append(([int(v) for v in b.xyxy[0].tolist()], float(b.conf)))
        return out

    def _filter(self, props: List[Tuple[List[int], float]], depth: Optional[np.ndarray],
                shape: Tuple[int, int], frame_id: str, known: Sequence[Dict[str, Any]]) -> List[Tuple[List[int], float]]:
        h, w = shape
        min_px = int(self._p("open_set_min_box_px", 12))
        max_frac = float(self._p("open_set_max_box_frac", 0.25))
        max_depth = float(self._p("open_set_max_depth_m", 1.0))
        scale = float(getattr(self._node.fb, "depth_scale", 1000.0) or 1000.0)
        gmask = None
        sf = getattr(self._node, "_self_filter", None)
        if sf is not None:
            try:
                gmask = sf.mask(shape, frame_id)
            except Exception:
                gmask = None
        kept: List[Tuple[List[int], float]] = []
        known_boxes = [k["bbox"] for k in known if len(k.get("bbox", [])) == 4]
        for bb, c in sorted(props, key=lambda x: -x[1]):
            x1, y1, x2, y2 = max(0, bb[0]), max(0, bb[1]), min(w, bb[2]), min(h, bb[3])
            bw, bh = x2 - x1, y2 - y1
            if bw < min_px or bh < min_px or bw * bh > max_frac * w * h:
                continue
            if gmask is not None and (gmask[y1:y2, x1:x2] > 0).mean() >= 0.3:
                continue
            if depth is not None:
                patch = np.asarray(depth[y1:y2, x1:x2], dtype=np.float32) / scale
                valid = patch[(patch > 0.1) & (patch < 3.0)]
                if valid.size < 40 or float(np.median(valid)) > max_depth:
                    continue
            box = [x1, y1, x2, y2]
            if any(_iou(box, k) >= 0.4 or _contained(box, k) >= 0.8 for k in known_boxes):
                continue
            if any(_contained(box, k) >= 0.8 or _iou(box, k) > 0.5 for k, _ in kept):
                continue
            kept.append((box, c))
        return kept

    # ---- tracks + naming ----------------------------------------------------------
    def detect(self, rgb: np.ndarray, depth: Optional[np.ndarray], header: Any,
               known: Sequence[Dict[str, Any]], extra_boxes: Sequence[Sequence[int]] = ()) -> List[Dict[str, Any]]:
        """Unknown objects in this frame as detection dicts (``source="q8b"``).
        ``extra_boxes`` are YOLO boxes the identity check rejected: they are
        unknown objects too."""
        if not self.enabled:
            with self._lock:
                self._tracks.clear()
            return []
        frame_id = str(getattr(header, "frame_id", "") or "")
        shape = rgb.shape[:2]
        self._frame += 1
        every = max(1, int(self._p("open_set_every_n_frames", 1)))
        if self._frame % every == 0 or not self._last_props:
            self._last_props = self._propose(rgb)
        props = list(self._last_props) + [([int(v) for v in b], 1.0) for b in extra_boxes]
        props = self._filter(props, depth, shape, frame_id, known)
        now = time.monotonic()
        with self._lock:
            unmatched = list(self._tracks)
            present: List[_Track] = []
            for box, c in props:
                best, best_iou = None, 0.5
                for tr in unmatched:
                    s = _iou(box, tr.bbox)
                    if s > best_iou:
                        best, best_iou = tr, s
                if best is None:
                    best = _Track(box, c, now)
                    self._tracks.append(best)
                else:
                    unmatched.remove(best)
                    best.bbox, best.score, best.last = box, c, now
                present.append(best)
            keep_s = float(self._p("open_set_forget_s", 1.0))
            self._tracks = [tr for tr in self._tracks if now - tr.last <= keep_s]
            dets = [{"label": tr.name or "object", "score": tr.score, "bbox": list(tr.bbox),
                     "source": "q8b", "named": tr.name is not None, "_track": tr} for tr in present]
        # Reach verdict first: out-of-reach objects are never named and never shown.
        try:
            self._node._debug_renderer._annotate_distances(self._node, dets)
        except Exception:
            pass
        stable = float(self._p("open_set_stable_s", 0.2))
        out: List[Dict[str, Any]] = []
        for d in dets:
            tr: _Track = d.pop("_track")
            tr.reach = str(d.get("reach_state") or "")
            if tr.reach == "red":
                continue
            if tr.name is None and not tr.pending and now - tr.first >= stable and now >= tr.next_try:
                x1, y1, x2, y2 = tr.bbox
                tr.pending = True
                self._jobs.put((tr, rgb[y1:y2, x1:x2].copy()))
            if tr.name is not None or tr.tries >= 3:
                out.append(d)
        self._log_stats()
        return out

    def _naming_worker(self) -> None:
        while True:
            tr, crop = self._jobs.get()
            t = time.monotonic()
            name, err = self._ask(crop)
            with self._lock:
                tr.pending = False
                if name:
                    tr.name = name
                    tr.tries = 0
                    self._stat["named"] += 1
                    self._stat["name_s"] += time.monotonic() - t
                else:
                    tr.tries += 1
                    tr.next_try = time.monotonic() + 3.0
                    if tr.tries >= 3:
                        tr.name = "object"
            if err and time.monotonic() - self._service_warned > 30.0:
                self._service_warned = time.monotonic()
                self._node.get_logger().warn(f"[open-set] naming service: {err}")

    def _ask(self, crop: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        if crop.size == 0:
            return None, None
        side = int(self._p("open_set_crop_px", 224))
        h, w = crop.shape[:2]
        if max(h, w) > side:
            s = side / max(h, w)
            crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return None, "encode failed"
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()}},
            {"type": "text", "text": _PROMPT}]}],
            "max_tokens": int(self._p("open_set_name_tokens", 6)), "temperature": 0}
        url = str(self._p("open_set_naming_url", "http://127.0.0.1:8090/v1/chat/completions"))
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=float(self._p("open_set_naming_timeout_s", 10.0))) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            return clean_name(text), None
        except Exception as e:  # noqa: BLE001 - service down, timeout, bad reply
            return None, f"{type(e).__name__}: {str(e)[:80]}"

    def _log_stats(self) -> None:
        now = time.monotonic()
        if now - self._stat_t < 10.0:
            return
        s = self._stat
        if s["frames"]:
            with self._lock:
                n_tr = len(self._tracks)
                n_named = sum(1 for t in self._tracks if t.name)
            self._node.get_logger().info(
                f"[open-set] YOLOE {s['yoloe_ms'] / s['frames']:.0f} ms/frame over {s['frames']} frames | "
                f"{n_tr} unknown tracked, {n_named} named"
                + (f" | naming {s['name_s'] / s['named']:.2f} s each" if s["named"] else ""))
        self._stat = {"frames": 0, "yoloe_ms": 0.0, "named": 0, "name_s": 0.0}
        self._stat_t = now
