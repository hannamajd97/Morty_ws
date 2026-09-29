#!/usr/bin/env python3
# grasp_pipeline/point_selector.py
"""
point_selector.py
=================

Resolves WHICH detected object the user wants, via finger pointing.

Two geometries, one switch (``pointing_mode``):

``3d`` (default since 2026-09-03)
    The finger's direction in the ROOM. Index knuckle (landmark 5) and
    fingertip (8) are deprojected with the aligned depth the frame buffer
    already carries; the ray tip + t·(tip − knuckle) is scored against each
    detection's 3D centre (median depth inside its box) by the ANGLE between
    them — smallest angle inside a cone wins, and the object currently
    hovered keeps a small bonus (hysteresis) so two neighbours don't flicker.
    This is the geometry of Hamid's `run fingerversion.py` and of Kavousian
    et al. 2026 (RealSense D435i, 20 participants, 93 % correct selections).

``2d`` (the behaviour before 2026-09-03)
    The line wrist(0) → tip(8) in the PICTURE, extended; that pixel is tested
    against the boxes (inside → highest score; else snap to the nearest box
    edge within snap_px).

``3d`` falls back to ``2d`` for any frame in which the fingertip has no valid
depth (a hole on a thin finger is common), so it can never do worse than
``2d`` did. What both share — and what makes pointing safe for a real arm —
is that nothing is picked on one frame, and the hover feedback (on_hover /
on_pointer / on_hand drive the GUI overlay and tell the chat node what is
being pointed at). Feedback was the statistically significant factor in the
2026 study, so it stays.

The commit (``decision``, 2026-09-24): ``vote`` — every frame spreads its
evidence over the objects (3d: a Gaussian in the angle off the ray; 2d: the
pointed box at half weight), evidence fades with ``vote_tau_s``, and an
object is chosen once it holds ``vote_commit_s`` seconds of evidence and
``vote_margin`` times the runner-up. ``dwell`` — the old rule, the same
object for min_hold_ms with the pointer within steady_px; with the live
steady_px of 10 px the noisy 3-D pointer restarted the hold again and
again (pointing sessions of 25-35 s). ``begin()`` + ``step(t, frame, dets)``
are the whole decision, so the offline benchmark (pointing_bench.py)
replays recordings through exactly this code.

Hand tracking: MediaPipe's Tasks HandLandmarker in VIDEO mode when the model
file is present (it tracks between frames and re-runs the palm detector only
when the hand is lost), else the legacy mp.solutions.hands. Measured on this
laptop's CPU, 640×480 (2026-09-03): 16.7 vs 17.5 ms per frame; the whole 3D
part adds ~1.4 ms — the method is not what you wait for, the commit rule is.

History: this used to also host a Vosk voice path; voice selection now lives
in communication_llm and reaches the pipeline via ROS params and
/yolo/select_index. The class name and the `vosk_selector` argument are kept
so call sites don't break.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Callable, Sequence
import math
import os
import time
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None  # optional

# Ships with the package (7.8 MB, Google's float16 hand_landmarker).
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# The 21-point hand skeleton, for the overlay. Same for both APIs.
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),
]
WRIST, INDEX_MCP, INDEX_TIP = 0, 5, 8
INDEX_CHAIN = (5, 6, 7, 8)          # MCP → PIP → DIP → tip
FINGER_LEN_M = (0.04, 0.13)         # plausible MCP→tip length; longer = depth bled into the background


# ── 2D helpers ───────────────────────────────────────────────────────
def _bbox_center(b: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _point_in_bbox(p: Tuple[float, float], b: Sequence[float]) -> bool:
    x, y = p
    x1, y1, x2, y2 = b
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _bbox_edge_dist2(p: Tuple[float, float], b: Sequence[float]) -> float:
    """Squared distance from point to the bbox rectangle (0 if inside)."""
    x, y = p
    x1, y1, x2, y2 = b
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return dx * dx + dy * dy


# ── hand-tracking backends (same contract: normalised 21 points or None) ──
# ── 3D helpers (module level since 2026-09-29: astra_pointing.py measures the
#    objects with the same rule) ─────────────────────────────────────────
def deproject(u: float, v: float, z: float, K) -> np.ndarray:
    fx, fy, cx, cy = K
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def project(p: np.ndarray, K) -> Optional[Tuple[float, float]]:
    fx, fy, cx, cy = K
    if p[2] <= 1e-6:
        return None
    return (float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy))


def object_centres_3d(dets: List[Dict[str, Any]], depth_m: np.ndarray, K) -> List[Optional[np.ndarray]]:
    """Each detection's 3D centre: the box centre pixel at the median
    valid depth of the box's inner 60 % (edges belong to the table). A
    detection that carries ``centre_3d`` (this camera's frame, metres) is
    placed there as is — the Astra's objects, measured by the RealSense
    (astra_pointing.py)."""
    H, W = depth_m.shape[:2]
    out: List[Optional[np.ndarray]] = []
    for d in dets:
        if d.get("centre_3d") is not None:
            out.append(np.asarray(d["centre_3d"], dtype=np.float64))
            continue
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        ix1, iy1 = int(max(0, x1 + 0.2 * w)), int(max(0, y1 + 0.2 * h))
        ix2, iy2 = int(min(W, x2 - 0.2 * w)), int(min(H, y2 - 0.2 * h))
        roi = depth_m[iy1:iy2, ix1:ix2] if (ix2 > ix1 and iy2 > iy1) else np.empty(0)
        ok = roi[(roi > 0.05) & (roi < 3.0) & np.isfinite(roi)] if roi.size else roi
        if ok.size < 10:
            out.append(None)
            continue
        cu, cv_ = _bbox_center(d["bbox"])
        out.append(deproject(cu, cv_, float(np.median(ok)), K))
    return out


class _LegacyHands:
    name = "legacy mp.solutions.hands"
    world: Optional[List[Tuple[float, float, float]]] = None   # metric 3-D hand of the last frame

    def __init__(self, det_conf: float, trk_conf: float) -> None:
        sol = mp.solutions.hands if hasattr(mp, "solutions") else None
        if sol is None:
            from mediapipe.python import solutions as _s
            sol = _s.hands
        self._hands = sol.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                min_detection_confidence=float(det_conf),
                                min_tracking_confidence=float(trk_conf))

    def landmarks(self, rgb) -> Optional[List[Tuple[float, float]]]:
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.world = None
            return None
        w = getattr(res, "multi_hand_world_landmarks", None)
        self.world = [(p.x, p.y, p.z) for p in w[0].landmark] if w else None
        return [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]

    def close(self) -> None:
        self._hands.close()


class _TasksHands:
    name = "Tasks HandLandmarker (VIDEO)"
    world: Optional[List[Tuple[float, float, float]]] = None   # metric 3-D hand of the last frame

    def __init__(self, model_path: str, det_conf: float, trk_conf: float) -> None:
        from mediapipe.tasks import python as tp
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=tp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=float(det_conf),
            min_hand_presence_confidence=float(det_conf),
            min_tracking_confidence=float(trk_conf))
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def landmarks(self, rgb) -> Optional[List[Tuple[float, float]]]:
        # VIDEO mode wants strictly increasing millisecond timestamps.
        ts = int((time.monotonic() - self._t0) * 1000.0)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.hand_landmarks:
            self.world = None
            return None
        w = getattr(res, "hand_world_landmarks", None)
        self.world = [(p.x, p.y, p.z) for p in w[0]] if w else None
        return [(p.x, p.y) for p in res.hand_landmarks[0]]

    def close(self) -> None:
        self._lm.close()


PALM = (0, 5, 9, 13, 17)          # wrist + the four knuckles: a large, depth-friendly surface


class _DwellDecision:
    """The rule before 2026-09-24: the same object hovered for min_hold_ms
    with the pointer within steady_px. A noisy 3-D ray moves the pointer
    more than steady_px (10 px live), which restarted the hold again and
    again — pointing sessions of 25-35 s in the 09-2x logs."""
    name = "dwell"

    def __init__(self, min_hold_ms: int, steady_px: float) -> None:
        self.min_hold_s = float(min_hold_ms) / 1000.0
        self.steady2 = float(steady_px) ** 2
        self.idx = self.t_hold = self.px = None

    def leader(self) -> Optional[int]:
        return self.idx

    def update(self, t: float, hover: Optional[int], evidence: Dict[int, float],
               px: Optional[Tuple[float, float]]) -> Optional[int]:
        if px is None:                       # hand lost: the hold is kept (old behaviour)
            return None
        if hover is None:
            self.idx = self.t_hold = self.px = None
            return None
        if hover != self.idx:
            self.idx, self.t_hold, self.px = hover, t, px
            return None
        if self.px is not None and _dist2(px, self.px) > self.steady2:
            self.t_hold, self.px = t, px
        if self.t_hold is not None and t - self.t_hold >= self.min_hold_s:
            return hover
        return None


class _VoteDecision:
    """Evidence voting (2026-09-24, Majd: "whatever the object is, it should
    get it correctly"). Every frame spreads its evidence over the objects
    (3-D: by how far each lies off the finger ray; 2-D: the pointed box), in
    SECONDS of evidence so the frame rate does not matter; old evidence fades
    with time constant tau_s. An object is chosen once it holds commit_s of
    evidence AND margin times more than the runner-up — a noisy frame, a
    dropped frame or a jittering pointer no longer restarts anything, and two
    close candidates simply keep collecting until one clearly leads."""
    name = "vote"

    def __init__(self, tau_s: float, commit_s: float, margin: float) -> None:
        self.tau = max(0.05, float(tau_s))
        self.commit = float(commit_s)
        self.margin = max(1.0, float(margin))
        self.score: Dict[int, float] = {}
        self.t_last: Optional[float] = None

    def leader(self) -> Optional[int]:
        return max(self.score, key=self.score.get) if self.score else None

    def update(self, t: float, hover: Optional[int], evidence: Dict[int, float],
               px: Optional[Tuple[float, float]]) -> Optional[int]:
        dt = 0.05 if self.t_last is None else min(0.2, max(0.0, t - self.t_last))
        self.t_last = t
        keep = math.exp(-dt / self.tau)
        for k in list(self.score):
            self.score[k] *= keep
            if self.score[k] < 1e-4:
                del self.score[k]
        for k, w in evidence.items():
            self.score[k] = self.score.get(k, 0.0) + float(w) * dt
        if not self.score:
            return None
        ranked = sorted(self.score.values(), reverse=True)
        best, second = ranked[0], (ranked[1] if len(ranked) > 1 else 0.0)
        if best >= self.commit and best >= self.margin * second:
            return self.leader()
        return None


class PointOrVoiceSelector:
    """
    Choose an object index by finger pointing (MediaPipe). See the module
    docstring for the 2d / 3d geometries. `vosk_selector` is accepted (and
    ignored) so existing call sites don't break.
    """
    def __init__(self,
                 vosk_selector=None,
                 use_mediapipe: bool = True,
                 min_hold_ms: int = 450,
                 steady_px: float = 22.0,
                 pointer_pref_inside: bool = False,
                 snap_px: float = 80.0,
                 ray_extension: float = 1.5,
                 min_detection_confidence: float = 0.60,
                 min_tracking_confidence: float = 0.60,
                 logger=None,
                 on_pointer: Optional[Callable[[Optional[Tuple[float, float]]], None]] = None,
                 on_hand: Optional[Callable[[Optional[List[Tuple[float, float]]], Optional[List[Tuple[int, int]]]], None]] = None,
                 on_hover: Optional[Callable[[Optional[int]], None]] = None,
                 # 2026-09-03 — 3D pointing (see module docstring)
                 pointing_mode: str = "3d",
                 cone_deg: float = 20.0,
                 hysteresis: float = 0.8,
                 use_tasks_api: bool = True,
                 model_path: str = "",
                 depth_provider: Optional[Callable[[], Any]] = None,
                 # 2026-09-24 — evidence voting and the model-based ray
                 decision: str = "vote",
                 vote_tau_s: float = 0.8,
                 vote_commit_s: float = 0.45,
                 vote_margin: float = 2.0,
                 sigma_deg: float = 7.0,
                 ray_source: str = "depth",
                 evidence: str = "arrow") -> None:
        self.vosk = None  # intentionally ignored in the new architecture
        self.min_hold_ms = int(min_hold_ms)
        self.steady_px = float(steady_px)
        self.pointer_pref_inside = bool(pointer_pref_inside)
        self.snap_px = float(snap_px)
        self.ray_extension = float(ray_extension)
        self.logger = logger
        self.on_pointer = on_pointer
        self.on_hand = on_hand
        self.on_hover = on_hover
        self.pointing_mode = "3d" if str(pointing_mode).lower() == "3d" else "2d"
        self.cone_cos = math.cos(math.radians(float(cone_deg)))
        self.hysteresis = float(hysteresis)
        # depth_provider() -> (rgb, depth, (fx, fy, cx, cy), depth_scale, header)
        # — the frame buffer's latest(); depth is uint16 (÷ depth_scale → m)
        # or float32 metres. Without it, 3d silently behaves as 2d.
        self.depth_provider = depth_provider
        # "vote" (default since 2026-09-24) or "dwell" (the old rule) — see the
        # two decision classes; sigma = the finger ray's angular noise used to
        # spread each frame's evidence. ray_source "depth" = fit through the
        # finger's four depth samples; "model" = MediaPipe's metric 3-D hand
        # for the direction and the PALM's depth for the position (no depth
        # read on the thin finger) — off until a recording proves it better.
        self.decision = "dwell" if str(decision).lower() == "dwell" else "vote"
        self.vote_tau_s, self.vote_commit_s = float(vote_tau_s), float(vote_commit_s)
        self.vote_margin, self.sigma = float(vote_margin), math.radians(float(sigma_deg))
        self.ray_source = "model" if str(ray_source).lower() == "model" else "depth"
        # What a 3-D frame votes for (2026-09-25, Majd: "great when my finger
        # is next to the object, but from far away it doesn't get the object
        # my arrow is already on"). The arrow ends on _choose_by_ray's object
        # — (1 − cos θ)·distance, hysteresis — while the vote spread the
        # frame over the objects by angle alone; near the objects both agree,
        # from far away they part: the arrow sat on X while the evidence went
        # to a better-aligned object behind it, or split and never reached the
        # margin. "arrow": the frame's vote goes to the object the arrow is on,
        # so the choice is where the arrow has been. "angle": the 09-24 rule.
        self.evidence_mode = "angle" if str(evidence).lower() == "angle" else "arrow"
        self._decision = None

        self.tracker = None
        if bool(use_mediapipe) and mp is not None:
            path = model_path or DEFAULT_MODEL
            if bool(use_tasks_api) and os.path.isfile(path):
                try:
                    self.tracker = _TasksHands(path, min_detection_confidence, min_tracking_confidence)
                except Exception as e:  # older mediapipe without Tasks, bad model file …
                    self._log("warn", f"Tasks HandLandmarker unavailable ({e}); using the legacy API")
            if self.tracker is None:
                try:
                    self.tracker = _LegacyHands(min_detection_confidence, min_tracking_confidence)
                except Exception as e:
                    self._log("warn", f"MediaPipe hands API not available: {e}")
        self.use_mediapipe = self.tracker is not None
        if self.use_mediapipe:
            self._log("info", f"[point] tracker={self.tracker.name} mode={self.pointing_mode}"
                              f"{'' if self.depth_provider or self.pointing_mode == '2d' else ' (no depth provider → 2d)'}")

        self._ema: Optional[np.ndarray] = None       # 2d pointer pixel
        self._ema_alpha = 0.35
        self._ray: Optional[Tuple[np.ndarray, np.ndarray]] = None   # 3d (origin, dir)
        self._ray_alpha = 0.5
        self._last_hover: Optional[int] = None
        self._n_frames = 0
        self._n_3d = 0
        self._n_bleed = 0      # frames whose finger depth bled into the background (fell back to 2d)

    # ── small utilities ──────────────────────────────────────────────
    def _log(self, level: str, msg: str) -> None:
        if self.logger is not None:
            try:
                getattr(self.logger, level)(msg)
            except Exception:
                pass

    def _emit(self, cb, *args) -> None:
        if cb:
            try:
                cb(*args)
            except Exception:
                pass

    # ── hand → landmarks (pixels) ────────────────────────────────────
    def _landmarks_px(self, bgr) -> Optional[List[Tuple[float, float]]]:
        if self.tracker is None or bgr is None:
            self._emit(self.on_hand, None, None)
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            norm = self.tracker.landmarks(rgb)
        except Exception as e:
            self._log("warn", f"[point] hand tracker error: {e}")
            norm = None
        if not norm:
            self._emit(self.on_hand, None, None)
            return None
        H, W = bgr.shape[:2]
        pts = [(float(x * W), float(y * H)) for x, y in norm]
        self._emit(self.on_hand, pts, HAND_CONNECTIONS)
        return pts

    # ── 2d geometry (the pre-2026-09-03 behaviour) ───────────────────
    def _smooth(self, p: Tuple[float, float]) -> Tuple[float, float]:
        q = np.array(p, dtype=np.float32)
        self._ema = q if self._ema is None else (1.0 - self._ema_alpha) * self._ema + self._ema_alpha * q
        return (float(self._ema[0]), float(self._ema[1]))

    def _pointer_2d(self, lm: List[Tuple[float, float]]) -> Tuple[float, float]:
        # extended = tip + (tip - wrist) * (ray_extension - 1.0)
        (wx, wy), (tx, ty) = lm[WRIST], lm[INDEX_TIP]
        ext = self.ray_extension - 1.0
        return self._smooth((tx + (tx - wx) * ext, ty + (ty - wy) * ext))

    def _choose_by_point(self, pointer_px: Tuple[float, float], dets: List[Dict[str, Any]],
                         allowed: Optional[Sequence[int]] = None) -> Optional[int]:
        idxs = list(range(len(dets))) if allowed is None else list(allowed)
        if not idxs:
            return None
        inside = [i for i in idxs if _point_in_bbox(pointer_px, dets[i]["bbox"])]
        if inside:
            return max(inside, key=lambda i: float(dets[i]["score"]))
        if self.pointer_pref_inside:
            return None
        # Snap gate (2026-08-07, wrong-selection fix): a pointer outside
        # every bbox may only snap to a box within snap_px of its edge.
        best = min(idxs, key=lambda i: _bbox_edge_dist2(pointer_px, dets[i]["bbox"]))
        if _bbox_edge_dist2(pointer_px, dets[best]["bbox"]) > self.snap_px * self.snap_px:
            return None
        return best

    # ── 3d geometry ──────────────────────────────────────────────────
    @staticmethod
    def _depth_frame(sample) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """frame_buffer.latest() → (depth in metres, (fx, fy, cx, cy)) or None."""
        try:
            _rgb, depth, K, scale, _hdr = sample
        except Exception:
            return None
        if depth is None or K is None or K[0] in (None, 0):
            return None
        d = np.asarray(depth)
        if d.dtype != np.float32 and d.dtype != np.float64:
            d = d.astype(np.float32) / float(scale or 1000.0)
        return d, tuple(float(k) for k in K)

    @staticmethod
    def _depth_at(depth_m: np.ndarray, u: float, v: float, r: int = 3) -> Optional[float]:
        """Depth of the NEAREST surface in a (2r+1)² window (20th percentile).

        2026-09-08 (Majd: "the line goes way forward"): the median was wrong.
        A fingertip landmark sits on the edge of a thin finger, so half the
        window is the table/objects BEHIND it; the median then returned the
        background, the "finger" came out 20–30 cm long and tilted steeply
        away from the camera, and the ray overshot everything. The finger is
        the closest thing in its own window, so a low percentile is right;
        a single pixel is still avoided because holes are common."""
        H, W = depth_m.shape[:2]
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < W and 0 <= y < H):
            return None
        win = depth_m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        ok = win[(win > 0.05) & (win < 3.0) & np.isfinite(win)]
        return float(np.percentile(ok, 20)) if ok.size >= 3 else None

    _deproject = staticmethod(deproject)
    _project = staticmethod(project)

    def _ray_3d(self, lm: List[Tuple[float, float]], depth_m: np.ndarray, K) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Finger ray in camera coordinates, smoothed over frames: origin =
        fingertip, unit direction knuckle → tip. ray_source picks how it is
        measured (see _ray_3d_depth / _ray_3d_model)."""
        raw = (self._ray_3d_model(lm, depth_m, K) if self.ray_source == "model"
               else self._ray_3d_depth(lm, depth_m, K))
        if raw is None:
            return None
        tip, d = raw
        if self._ray is None:
            self._ray = (tip, d)
        else:
            o0, d0 = self._ray
            o = (1.0 - self._ray_alpha) * o0 + self._ray_alpha * tip
            dd = (1.0 - self._ray_alpha) * d0 + self._ray_alpha * d
            nn = float(np.linalg.norm(dd))
            self._ray = (o, dd / nn if nn > 1e-9 else d)
        return self._ray

    def _ray_3d_depth(self, lm: List[Tuple[float, float]], depth_m: np.ndarray, K) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Direction = the least-squares line through the index finger's four
        joints (MCP, PIP, DIP, tip) deprojected with their depth, oriented
        knuckle → tip. Four samples instead of two make one bad depth
        survivable; the MCP→tip length must be a finger's (4–13 cm) —
        anything longer means a depth bled into the background, and that
        frame falls back to 2d rather than aiming the ray into the table.
        None also when the finger is foreshortened below 2 cm (pointing
        straight at the lens)."""
        pts = []
        for i in INDEX_CHAIN:
            u, v = lm[i]
            z = self._depth_at(depth_m, u, v)
            if z is not None:
                pts.append(self._deproject(u, v, z, K))
        if len(pts) < 3 or self._depth_at(depth_m, *lm[INDEX_TIP]) is None:
            return None
        P = np.asarray(pts)
        mcp, tip = P[0], P[-1]
        span = float(np.linalg.norm(tip - mcp))
        if not (FINGER_LEN_M[0] <= span <= FINGER_LEN_M[1]):
            self._n_bleed += 1
            return None
        # principal axis of the joint chain (SVD), signed towards the tip
        c = P.mean(axis=0)
        d = np.linalg.svd(P - c, full_matrices=False)[2][0]
        if np.dot(d, tip - mcp) < 0:
            d = -d
        n = float(np.linalg.norm(d))
        if n < 1e-9 or span < 0.02:
            return None
        return tip, d / n

    def _ray_3d_model(self, lm: List[Tuple[float, float]], depth_m: np.ndarray, K) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Direction from MediaPipe's metric 3-D hand (world landmarks, camera-
        aligned axes, origin at the hand centre): the line through the index
        joints — the network's estimate, no depth pixel on the thin finger.
        Position from depth where it is reliable: the PALM (wrist + four
        knuckles, a large surface); the tip = palm centre + the model's
        palm→tip offset. Falls back to the depth fit when the model or the
        palm depth is missing."""
        W = getattr(self.tracker, "world", None) if self.tracker is not None else None
        if not W or len(W) < 21:
            return self._ray_3d_depth(lm, depth_m, K)
        Wa = np.asarray(W, dtype=np.float64)
        cam, wld = [], []
        for i in PALM:
            z = self._depth_at(depth_m, *lm[i], r=5)
            if z is not None:
                cam.append(self._deproject(lm[i][0], lm[i][1], z, K))
                wld.append(Wa[i])
        if len(cam) < 3:
            return self._ray_3d_depth(lm, depth_m, K)
        tip = np.mean(cam, axis=0) + (Wa[INDEX_TIP] - np.mean(wld, axis=0))
        F = Wa[list(INDEX_CHAIN)]
        span = float(np.linalg.norm(F[-1] - F[0]))
        if not (FINGER_LEN_M[0] <= span <= FINGER_LEN_M[1]):
            return None
        c = F.mean(axis=0)
        d = np.linalg.svd(F - c, full_matrices=False)[2][0]
        if np.dot(d, F[-1] - F[0]) < 0:
            d = -d
        n = float(np.linalg.norm(d))
        return (tip, d / n) if n > 1e-9 else None

    def _centres_3d(self, dets: List[Dict[str, Any]], depth_m: np.ndarray, K) -> List[Optional[np.ndarray]]:
        return object_centres_3d(dets, depth_m, K)

    def _choose_by_ray(self, origin: np.ndarray, direction: np.ndarray,
                       centres: List[Optional[np.ndarray]]) -> Optional[int]:
        """Hamid's rule: candidates inside the cone; score = (1 − cos θ)·dist
        (on-axis and near win); the current hover keeps a hysteresis bonus."""
        best_i, best_s = None, float("inf")
        for i, c in enumerate(centres):
            if c is None:
                continue
            v = c - origin
            dist = float(np.linalg.norm(v))
            if dist < 0.05:
                continue
            cos = float(np.dot(direction, v / dist))
            if cos < self.cone_cos:
                continue
            s = (1.0 - cos) * dist
            if i == self._last_hover:
                s *= self.hysteresis
            if s < best_s:
                best_i, best_s = i, s
        return best_i

    # ── one frame → (hover index, overlay pixel, mode used) ──────────
    def _ray_evidence(self, origin: np.ndarray, direction: np.ndarray,
                      centres: List[Optional[np.ndarray]]) -> Dict[int, float]:
        """How this frame's ray splits its evidence over the objects: a
        Gaussian in the angle off the ray (sigma = the ray's angular noise),
        inside the cone only, normalised to 1 — a clear aim gives its target
        almost all of it, an aim between two neighbours splits it."""
        lik: Dict[int, float] = {}
        for i, c in enumerate(centres):
            if c is None:
                continue
            v = c - origin
            dist = float(np.linalg.norm(v))
            if dist < 0.05:
                continue
            cos = float(np.clip(np.dot(direction, v / dist), -1.0, 1.0))
            if cos < self.cone_cos:
                continue
            lik[i] = math.exp(-0.5 * (math.acos(cos) / self.sigma) ** 2)
        tot = sum(lik.values())
        return {i: w / tot for i, w in lik.items()} if tot > 1e-9 else {}

    def _evaluate(self, bgr, dets: List[Dict[str, Any]]
                  ) -> Tuple[Optional[int], Optional[Tuple[float, float]], str, Dict[int, float]]:
        """One frame → (hover index, overlay pixel, mode, evidence per object)."""
        # 3d: find the finger in the colour frame that BELONGS to the depth
        # frame (FrameBuffer.paired, 2026-09-24) — the newest colour frame
        # and the newest depth frame were up to ~0.2 s apart while the hand
        # moved, so the joints' depths came from where the finger had been.
        sample = None
        if self.pointing_mode == "3d" and self.depth_provider is not None:
            try:
                sample = self.depth_provider()
                if sample is not None and sample[0] is not None and sample[1] is not None:
                    bgr = sample[0]
            except Exception:
                sample = None
        lm = self._landmarks_px(bgr)
        if lm is None:
            self._emit(self.on_pointer, None)
            return None, None, "none", {}
        self._n_frames += 1
        if self.pointing_mode == "3d" and sample is not None:
            dm = self._depth_frame(sample)
            if dm is not None:
                depth_m, K = dm
                ray = self._ray_3d(lm, depth_m, K)
                if ray is not None:
                    origin, direction = ray
                    centres = self._centres_3d(dets, depth_m, K)
                    idx = self._choose_by_ray(origin, direction, centres)
                    if self.evidence_mode == "arrow":
                        evidence = {idx: 1.0} if idx is not None else {}
                    else:
                        evidence = self._ray_evidence(origin, direction, centres)
                    if idx is None and any(c is None for c in centres):
                        # objects with no depth (glass, chrome) still get the 2d test
                        px2d = self._pointer_2d(lm)
                        idx = self._choose_by_point(px2d, dets, allowed=[i for i, c in enumerate(centres) if c is None])
                        if idx is not None:
                            evidence = {idx: 0.5}
                    # overlay: the line ends ON the object being pointed at (the
                    # ray point nearest its centre), else 15 cm ahead — a fixed
                    # 25 cm segment overshot every close object (2026-09-08).
                    reach = 0.15
                    if idx is not None and centres[idx] is not None:
                        reach = max(0.05, float(np.dot(centres[idx] - origin, direction)))
                    # The arrow starts at the fingertip SEEN in the image and
                    # points along the projected 3d ray (2026-09-24): drawing
                    # from the 2-D tip to the end of the SMOOTHED 3d ray bent
                    # the arrow sideways whenever the two tips differed.
                    o_px = self._project(origin, K)
                    e_px = self._project(origin + reach * direction, K)
                    tip = lm[INDEX_TIP]
                    px = ((tip[0] + e_px[0] - o_px[0], tip[1] + e_px[1] - o_px[1])
                          if (o_px and e_px) else tip)
                    self._n_3d += 1
                    self._emit(self.on_pointer, px)
                    return idx, px, "3d", evidence
        # 2d: the box under the extended pointer; half the weight of a 3-D
        # frame in the vote — it cannot tell neighbours in depth apart.
        px = self._pointer_2d(lm)
        self._emit(self.on_pointer, px)
        idx = self._choose_by_point(px, dets)
        return idx, px, "2d", ({idx: 0.5} if idx is not None else {})

    # ── one selection: begin(), then step() per frame ───────────────
    def begin(self) -> None:
        """Start a new selection: fresh decision, smoothing and counters."""
        self._decision = (_VoteDecision(self.vote_tau_s, self.vote_commit_s, self.vote_margin)
                          if self.decision == "vote" else _DwellDecision(self.min_hold_ms, self.steady_px))
        self._last_hover = None
        self._ray = None
        self._ema = None
        self._n_frames = self._n_3d = self._n_bleed = 0
        self._t_begin: Optional[float] = None

    def step(self, t: float, bgr, dets: List[Dict[str, Any]]
             ) -> Tuple[Optional[int], Optional[int], Optional[Tuple[float, float]], str]:
        """One frame at time t (seconds, any clock): (chosen index or None,
        hover index for the UI, overlay pixel, mode). The live loop passes
        wall time; the offline benchmark passes the recording's stamps — the
        same code decides in both."""
        if self._decision is None:
            self.begin()
        if self._t_begin is None:
            self._t_begin = t
        idx, px, mode, evidence = self._evaluate(bgr, dets)
        self._last_hover = idx if px is not None else None
        chosen = self._decision.update(t, idx, evidence if px is not None else {}, px)
        hover = None
        if px is not None:
            hover = self._decision.leader() if self.decision == "vote" else idx
        if chosen is not None:
            self._log("info", f"[point] chose index {chosen} by {self._decision.name} via {mode} after "
                              f"{t - self._t_begin:.1f} s ({self._n_3d}/{self._n_frames} frames had a 3d ray, "
                              f"{self._n_bleed} rejected for background depth)")
        return chosen, hover, px, mode

    # ── the live selection loop ──────────────────────────────────────
    def choose(self,
               bgr_frame,
               dets: List[Dict[str, Any]],
               names: List[str],
               frame_provider: Optional[Callable[[], Any]] = None,
               listen_seconds: float = 8.0,
               overall_timeout_s: float = 10.0,
               cancel: Optional[Callable[[], bool]] = None) -> Optional[int]:
        # Pointing-only selection. Voice arrives via ROS (communication_llm).
        # listen_seconds / names are kept in the signature for compatibility.
        del listen_seconds, names
        if not dets:
            return None

        try:
            return self._choose_loop(bgr_frame, dets, frame_provider, overall_timeout_s, cancel)
        finally:
            # Pointing is over (chosen, cancelled or timed out): take the arrow
            # and the hand skeleton off the overlay (2026-09-24: the last arrow
            # stayed drawn on the live image until the next pointing session).
            self._emit(self.on_pointer, None)
            self._emit(self.on_hand, None, None)

    def _choose_loop(self, bgr_frame, dets, frame_provider, overall_timeout_s, cancel) -> Optional[int]:
        start = time.time()
        self.begin()
        last_emitted: Optional[int] = -1  # sentinel distinct from None
        while time.time() - start < float(overall_timeout_s):
            # A click or voice choice arrived while we were watching the
            # hand: hand over at once (2026-09-09: a click used to wait for
            # this loop's 10 s timeout, 1.5-4 s felt by the operator).
            if cancel is not None and cancel():
                return None
            cur_frame = frame_provider() if callable(frame_provider) else bgr_frame
            chosen, hover = None, None
            if self.use_mediapipe:
                chosen, hover, _px, _mode = self.step(time.time(), cur_frame, dets)
            # Hover → chat_node can say what the user points at before commit.
            if self.on_hover and hover != last_emitted:
                self._emit(self.on_hover, hover)
                last_emitted = hover
            if chosen is not None:
                return chosen
            time.sleep(0.05)
        return None

    def close(self):
        try:
            if self.tracker is not None:
                self.tracker.close()
        except Exception:
            pass
