"""Identity check: is the object YOLO named really that object?

A closed-set detector answers with one of its classes whatever it sees, and
it is confident about it (a drink carton: ``wooden_maze_puzzle 0.87``; the
real tube: ``vitamin_c 0.94`` -- the confidence cannot tell them apart,
2026-09-15). A vision-language model asked "is this a maze?" rejected two of
three correct detections on the same table. What does separate them is a
plain visual comparison with photos of the real object: DINOv2 embeddings
(vision-only, made for instance similarity) gave 0.68-0.85 for correct
detections and 0.23-0.49 for wrong ones and unknown objects, twice the gap
CLIP gives, at 10 ms per crop and 0.1 GB.

References live in ``models/identity_refs/<yolo_label>/*.png``, crops of
every side of every known object (build them from the training set with
``tools/build_identity_refs.py``). A detection whose best similarity to its
class's references is below ``identity_threshold`` is rejected; its box is
handed to the open-set path, which names the object for what it is. A class
without references is never rejected, and with no references at all the
check is inactive, so the pipeline behaves as before.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_HUB_REPO = "facebookresearch/dinov2"
_HUB_MODEL = "dinov2_vits14"


class IdentityCheck:
    def __init__(self, node: Any) -> None:
        self._node = node
        self._model = None
        self._device = "cpu"
        self._refs: Dict[str, np.ndarray] = {}
        self._tried = False
        self._next_try = 0.0
        self._warned_empty = False
        self._rejected_total = 0
        self._last_log = 0.0

    def _p(self, name: str, default):
        try:
            v = self._node.get_parameter(name).value
            return default if v is None else v
        except Exception:
            return default

    @property
    def enabled(self) -> bool:
        return bool(self._p("identity_check_enabled", False))

    def _ref_dir(self) -> str:
        p = str(self._p("identity_ref_dir", "") or "")
        if p:
            return os.path.expanduser(p)
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "identity_refs")

    # ---- loading ----------------------------------------------------------------
    def _load(self) -> bool:
        if self._refs:
            return True
        # No references yet: look again every 30 s, so a folder built while
        # the pipeline runs (tools/enroll_identity_refs.py) is picked up.
        now = time.monotonic()
        if self._tried and now - self._next_try < 0.0:
            return False
        self._tried = True
        self._next_try = now + 30.0
        log = self._node.get_logger()
        ref_dir = self._ref_dir()
        classes = sorted(d for d in glob.glob(os.path.join(ref_dir, "*")) if os.path.isdir(d))
        if not classes:
            if not self._warned_empty:
                self._warned_empty = True
                log.info(f"[identity] no references in {ref_dir}; YOLO names are not checked "
                         "(build them with tools/enroll_identity_refs.py or build_identity_refs.py)")
            return False
        try:
            import torch
            t = time.monotonic()
            local = os.path.join(torch.hub.get_dir(), _HUB_REPO.replace("/", "_") + "_main")
            if os.path.isdir(local):
                model = torch.hub.load(local, _HUB_MODEL, source="local", verbose=False)
            else:
                model = torch.hub.load(_HUB_REPO, _HUB_MODEL, verbose=False)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(self._device).eval()
            for cdir in classes:
                files = sorted(f for f in glob.glob(os.path.join(cdir, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
                crops = [cv2.imread(f) for f in files]
                crops = [c for c in crops if c is not None and c.size]
                if crops:
                    # The captures show every object standing; a tube lying
                    # on its side scored 0.3 against them and was renamed by
                    # the naming model ("golden horizontal bar", 2026-09-16).
                    # DINOv2 is not rotation-invariant, so each reference is
                    # also embedded turned 90° either way and upside down.
                    turned = [cv2.rotate(c, r) for c in crops
                              for r in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180)]
                    self._refs[os.path.basename(cdir)] = self._embed(crops + turned)
            stats = [f"{k} ({len(e) // 4} refs x 4 orientations)" for k, e in self._refs.items()]
            if self._device == "cuda":
                torch.cuda.empty_cache()   # the reference batches' activations are not needed any more
            log.info(f"[identity] DINOv2 ready in {time.monotonic() - t:.1f} s; references: " + ", ".join(stats)
                     + f"; threshold {float(self._p('identity_threshold', 0.5)):.2f}")
        except Exception as e:  # noqa: BLE001
            log.warn(f"[identity] unavailable ({e}); YOLO names are not checked")
            self._refs = {}
        return bool(self._refs)

    def _embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        batch = []
        for c in crops:
            x = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            batch.append(((x - _MEAN) / _STD).transpose(2, 0, 1))
        with torch.no_grad():
            f = self._model(torch.from_numpy(np.stack(batch)).to(self._device)).float()
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    # ---- identify a crop nobody has named --------------------------------------------
    def identify(self, crop: np.ndarray) -> Tuple[str, float]:
        """The known object a crop looks like, with its similarity; ("", 0)
        when the check is inactive or nothing reaches the threshold. Used
        for open-set boxes before the naming model is asked (2026-09-16):
        a known object the trained YOLO missed, e.g. the tube lying down,
        keeps its real name instead of a guess."""
        if crop is None or crop.size == 0 or not self.enabled or not self._load():
            return "", 0.0
        try:
            e = self._embed([crop])[0]
        except Exception:
            return "", 0.0
        sims = {k: float((v @ e).max()) for k, v in self._refs.items()}
        best = max(sims, key=sims.get)
        return (best, sims[best]) if sims[best] >= float(self._p("identity_threshold", 0.6)) else ("", sims[best])

    # ---- the check ------------------------------------------------------------------
    def check(self, rgb: np.ndarray, dets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """(kept, rejected). Each checked detection gets ``identity_sim``."""
        if not dets or not self.enabled or not self._load():
            return dets, []
        thr = float(self._p("identity_threshold", 0.5))
        h, w = rgb.shape[:2]
        todo = [(i, d) for i, d in enumerate(dets) if str(d.get("label", "")) in self._refs and len(d.get("bbox", [])) == 4]
        if not todo:
            return dets, []
        crops = []
        for _, d in todo:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            crops.append(rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)])
        try:
            emb = self._embed(crops)
        except Exception as e:  # noqa: BLE001
            self._node.get_logger().warn(f"[identity] embedding failed ({e}); detections kept")
            return dets, []
        rejected_idx = set()
        rejected: List[Dict[str, Any]] = []
        notes: List[str] = []
        for (i, d), e in zip(todo, emb):
            sims = {k: float((v @ e).max()) for k, v in self._refs.items()}
            sim = sims[str(d["label"])]
            d["identity_sim"] = round(sim, 2)
            if sim >= thr:
                continue
            best = max(sims, key=sims.get)
            if best != d["label"] and sims[best] >= thr:
                # A known object under the wrong name (the B12 tube as
                # "tape_dispenser", 0.40, while 0.80 like vitamin_c): rename.
                notes.append(f"{d['label']} {float(d.get('score', 0)):.2f} is {best} ({sims[best]:.2f})")
                d["label"] = best
                d["identity_sim"] = round(sims[best], 2)
                continue
            rejected_idx.add(i)
            rejected.append(d)
            notes.append(f"{d['label']} {float(d.get('score', 0)):.2f} rejected (looks {sim:.2f} like it)")
        if notes:
            self._rejected_total += len(rejected)
            now = time.monotonic()
            if now - self._last_log > 5.0:
                self._last_log = now
                self._node.get_logger().info("[identity] " + "; ".join(notes) + f"  (rejected so far {self._rejected_total})")
        return [d for i, d in enumerate(dets) if i not in rejected_idx], rejected
