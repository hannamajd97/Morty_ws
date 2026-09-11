"""Grasp precompute cache (2026-09-11).

While nobody has selected an object, a worker thread runs the pick
pipeline's perception phases (segmentation, depth fusion, cloud, GraspGen,
filter chain) for every stable detection and stores the result. When the
operator selects an object whose stored result is still valid, the pick
replays the captured outputs and publishes the stored candidates at once
instead of spending 0.6–0.9 s on perception.

Nothing else changes: the worker publishes nothing, moves nothing, and
touches no node state the overlay, voice stack or GUI read. A selection
still walks ``PickPipeline.run``; a valid cache entry only shortens it.

Validity of an entry (all must hold at lookup time)
  * same display label, box centre moved < ``precompute_box_shift_px``,
    box size changed < ``precompute_size_tol`` (ratio);
  * median depth at the box centre moved < ``precompute_depth_tol_m``;
  * arm (base_link -> tool0) moved < ``precompute_pose_tol_m`` /
    ``precompute_pose_tol_deg`` since the capture;
  * the node's parameters are unchanged (hash of every parameter that is
    not a precompute switch);
  * the stored result holds at least one grasp (an empty one is recomputed,
    a fresh capture may see the object better).

Scheduling
  * a detection must be seen for ``precompute_stable_s`` before it is queued;
  * hovered (pointed-at) object first, then objects inside the reach
    envelope, then ``edge`` ones; ``too far`` objects are skipped;
  * at most ``precompute_max_objects`` entries;
  * the worker never runs while a selection is being served
    (``foreground()`` holds the model lock around the whole pick) and it
    ends its own job at the next phase boundary when a selection is made.
"""
from __future__ import annotations

import copy
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy.duration
import rclpy.time

from ..perception.neighbour_obstacles import publish_neighbour_obstacles


def _box_centre(b) -> Tuple[float, float]:
    return (float(b[0] + b[2]) * 0.5, float(b[1] + b[3]) * 0.5)


def _box_size(b) -> Tuple[float, float]:
    return (max(1.0, float(b[2] - b[0])), max(1.0, float(b[3] - b[1])))


def _boxes_agree(a, b, shift_px: float, size_tol: float) -> bool:
    ca, cb = _box_centre(a), _box_centre(b)
    if abs(ca[0] - cb[0]) > shift_px or abs(ca[1] - cb[1]) > shift_px:
        return False
    sa, sb = _box_size(a), _box_size(b)
    return abs(sa[0] - sb[0]) / sa[0] <= size_tol and abs(sa[1] - sb[1]) / sa[1] <= size_tol


def _pose_delta(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(translation m, rotation deg) between two 4x4 transforms."""
    dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    r = a[:3, :3].T @ b[:3, :3]
    c = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return dt, float(np.degrees(np.arccos(c)))


@dataclass
class _Track:
    key: str
    label: str
    box: List[int]
    first_seen: float
    last_seen: float
    reach_state: str = ""
    score: float = 0.0
    failed_t: float = 0.0          # last failed job; retried after a backoff


@dataclass
class CacheEntry:
    key: str
    label: str
    box: List[int]
    depth_m: float
    tool0: np.ndarray
    params_hash: str
    params_snapshot: Dict[str, Any]
    result: Dict[str, Any]
    t_done: float
    n_grasps: int = 0
    mask: Optional[np.ndarray] = field(default=None, repr=False)


class GraspPrecomputer:
    # Parameters that cannot change a stored result: the cache's own
    # switches, overlay/debug output, the selection flow (how an object is
    # chosen, not what is computed for it), and the neighbour boxes, which a
    # replay rebuilds from the live frame anyway. Everything else — YOLO,
    # segmentation, depth fusion, GraspGen request, the filter chain —
    # invalidates the store when it changes.
    _EXCLUDED_PARAM_PREFIXES = (
        "precompute_", "graspgen_precompute", "publish_debug", "debug_", "use_sim_time",
        "auto_select_label", "selection_", "require_selection", "repeat", "listen_seconds",
        "mediapipe_", "pointing_", "use_mediapipe_selection", "manual_select_topic",
        "speak", "tts_", "executor_cycle_done_topic", "suspend_inference_while_selecting",
        "neighbour_", "publish_grasp_candidates", "grasp_candidates_top_k",
    )

    def __init__(self, node, pick) -> None:
        self._node = node
        self._pick = pick
        self._state = threading.Lock()          # tracks / entries / hover
        self.model_lock = threading.Lock()      # one pipeline computation at a time
        self._tracks: Dict[str, _Track] = {}
        self._entries: Dict[str, CacheEntry] = {}
        self._hover_key: Optional[str] = None
        self._foreground = False
        self._wake = threading.Event()
        self._params_hash = ""
        self._params_hash_t = 0.0
        self._params_snapshot: Dict[str, Any] = {}   # values behind _params_hash
        self._worker = threading.Thread(target=self._loop, name="grasp-precompute", daemon=True)
        self._worker.start()

    # ── parameters ───────────────────────────────────────────────────────
    def _p(self, name: str, default):
        try:
            return self._node.get_parameter(name).value
        except Exception:
            return default

    @property
    def enabled(self) -> bool:
        return bool(self._p("graspgen_precompute", False))

    def _params_hash_now(self) -> str:
        """Hash of every parameter value except the cache's own switches,
        refreshed at most once a second (the GUI changes filter parameters
        at run time; a stored result computed under old ones is stale)."""
        now = time.monotonic()
        if now - self._params_hash_t < 1.0 and self._params_hash:
            return self._params_hash
        h = hashlib.md5()
        snap: Dict[str, Any] = {}
        try:
            for name, prm in sorted(self._node.get_parameters_by_prefix("").items()):
                if name.startswith(self._EXCLUDED_PARAM_PREFIXES):
                    continue
                snap[name] = repr(prm.value)
                h.update(f"{name}={snap[name]};".encode())
        except Exception:
            pass
        self._params_hash = h.hexdigest()
        self._params_snapshot = snap
        self._params_hash_t = now
        return self._params_hash

    def _params_diff(self, old: Dict[str, Any]) -> str:
        """Names of the parameters that differ from the current snapshot —
        so a 'parameters changed' line says which, not just that."""
        cur = self._params_snapshot
        names = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
        return ", ".join(names[:6]) + (" …" if len(names) > 6 else "")

    # ── observations from the detector thread ────────────────────────────
    def observe(self, dets: List[Dict[str, Any]]) -> None:
        """Called with every fresh detection list (the detector thread).
        Updates the identity tracks, drops entries whose object moved, and
        marks each detection dict with ``precomputed`` for the overlay."""
        if not self.enabled:
            if self._entries or self._tracks:
                with self._state:
                    self._entries.clear(); self._tracks.clear()
            return
        shift = float(self._p("precompute_box_shift_px", 8.0))
        size_tol = float(self._p("precompute_size_tol", 0.15))
        now = time.monotonic()
        with self._state:
            seen = set()
            for d in dets:
                key = str(d.get("display") or d.get("simplified_label") or d.get("label") or "")
                box = [int(v) for v in d.get("bbox", [])]
                if not key or len(box) != 4:
                    continue
                seen.add(key)
                tr = self._tracks.get(key)
                if tr is not None and _boxes_agree(tr.box, box, shift, size_tol):
                    tr.last_seen = now
                    tr.box = box
                else:
                    if tr is not None:
                        self._entries.pop(key, None)       # it moved: the stored grasps are wrong
                    self._tracks[key] = tr = _Track(key, str(d.get("simplified_label", d.get("label", ""))),
                                                    box, now, now)
                tr.reach_state = str(d.get("reach_state") or "")
                tr.score = float(d.get("score", 0.0))
                e = self._entries.get(key)
                d["precomputed"] = bool(e is not None and e.n_grasps > 0
                                        and _boxes_agree(e.box, box, shift, size_tol))
            for key in [k for k in self._tracks if k not in seen]:
                self._tracks.pop(key, None)
                self._entries.pop(key, None)
        self._wake.set()

    def bump(self, idx: Optional[int]) -> None:
        """Hovered / pointed-at object goes first in the queue."""
        names = getattr(self._node, "_current_names", None) or []
        key = str(names[idx]) if (idx is not None and 0 <= idx < len(names)) else None
        with self._state:
            self._hover_key = key
        if key:
            self._wake.set()

    def invalidate_all(self, reason: str) -> None:
        with self._state:
            n = len(self._entries)
            self._entries.clear()
        if n:
            self._node.get_logger().info(f"[precompute] dropped {n} stored result(s): {reason}")

    # ── the foreground pick ──────────────────────────────────────────────
    class _Foreground:
        def __init__(self, outer: "GraspPrecomputer") -> None:
            self._o = outer

        def __enter__(self):
            self._o._foreground = True
            self._o.model_lock.acquire()
            return self

        def __exit__(self, *exc):
            self._o.model_lock.release()
            self._o._foreground = False
            return False

    def foreground(self) -> "_Foreground":
        """Context manager for the real pick: a running background job ends
        at its next phase boundary (selection_made is set by then) and the
        pick starts as soon as it has released the model."""
        return self._Foreground(self)

    def lookup(self, sel: Dict[str, Any]) -> Tuple[Optional[CacheEntry], str]:
        """Stored result for this selection, or (None, why)."""
        if not self.enabled:
            return None, "off"
        label = str(sel.get("simplified_label", sel.get("label", "")))
        box = [int(v) for v in sel.get("bbox", [])]
        shift = float(self._p("precompute_box_shift_px", 8.0))
        size_tol = float(self._p("precompute_size_tol", 0.15))
        with self._state:
            cands = [e for e in self._entries.values()
                     if e.label == label and _boxes_agree(e.box, box, shift, size_tol)]
        if not cands:
            with self._state:
                pending = [k for k, t in self._tracks.items() if t.label == label and k not in self._entries]
            return None, (f"no stored result for '{label}'"
                          + (" (not computed yet)" if pending else " (moved or never stable)"))
        e = min(cands, key=lambda c: abs(_box_centre(c.box)[0] - _box_centre(box)[0])
                + abs(_box_centre(c.box)[1] - _box_centre(box)[1]))
        if e.n_grasps <= 0:
            return None, f"stored result for '{label}' had no grasps"
        if e.params_hash != self._params_hash_now():
            changed = self._params_diff(e.params_snapshot)
            self.invalidate_all(f"parameters changed: {changed}")
            return None, f"parameters changed since the capture ({changed})"
        d_now = self._depth_at(box)
        if d_now is None:
            return None, "no depth at the box centre now"
        if abs(d_now - e.depth_m) > float(self._p("precompute_depth_tol_m", 0.01)):
            return None, f"object depth moved {abs(d_now - e.depth_m) * 100:.1f} cm"
        t_now = self._tool0()
        if t_now is None:
            return None, "arm pose unavailable (tf)"
        dt, dr = _pose_delta(e.tool0, t_now)
        if dt > float(self._p("precompute_pose_tol_m", 0.002)) or dr > float(self._p("precompute_pose_tol_deg", 0.5)):
            self.invalidate_all(f"arm moved {dt * 1000:.1f} mm / {dr:.2f} deg since the capture")
            return None, "arm moved since the capture"
        return e, ""

    def replay_capture(self, entry: CacheEntry, sel: Dict[str, Any]) -> bool:
        """Publish what the fresh capture path publishes, from the stored
        result: selected Detection2D, mask outputs, the object clouds,
        centroids, and the neighbour boxes rebuilt from the LIVE frame (a
        neighbour may have moved since the capture)."""
        node = self._node
        res = entry.result
        capture_sel = dict(res["sel_capture"])
        if sel.get("header") is not None:
            capture_sel["header"] = copy.deepcopy(sel["header"])
        node._publish_detection_message(capture_sel)
        x1, y1, x2, y2 = capture_sel["bbox"]
        node.get_logger().info(
            f"Published selected '{capture_sel['label']}' bbox=({x1},{y1},{x2},{y2}) | depth=yes (cached)")
        if entry.mask is not None:
            node._publish_selected_mask_outputs(capture_sel, entry.mask, publish_outputs=True)
        obs0 = res.get("obs0") or {}
        cloud_msg = obs0.get("cloud_msg")
        if cloud_msg is not None:
            cloud_msg = copy.deepcopy(cloud_msg)
            cloud_msg.header = capture_sel["header"]
            node.pub_cloud.publish(cloud_msg)
            node.last_selected_cloud_msg = copy.deepcopy(cloud_msg)
            try:
                pts, cols = obs0.get("points"), obs0.get("colors")
                if cols is not None and pts is not None and len(cols) == len(pts):
                    rgb_msg = node._cloud_pub.create_cloud_xyzrgb(
                        cloud_msg.header, pts, np.ascontiguousarray(cols[:, ::-1]))
                    node.pub_cloud_rgb.publish(rgb_msg)
                    node.last_selected_cloud_rgb_msg = rgb_msg
            except Exception as e:
                node.get_logger().warn(f"[precompute] RGB cloud replay failed: {e}")
        live_sel = dict(capture_sel)
        live_sel["bbox"] = [int(v) for v in sel.get("bbox", capture_sel["bbox"])]
        publish_neighbour_obstacles(node, live_sel, node.fb.depth_frame)
        if bool(self._p("publish_graspgen_input_cloud", False)) and res.get("pts_send") is not None:
            try:
                from std_msgs.msg import Header
                h = Header(); h.stamp = node.get_clock().now().to_msg(); h.frame_id = res["input_cloud_frame"]
                node.pub_graspgen_cloud.publish(node._create_cloud_xyzrgb(h, res["pts_send"], res.get("cols_send")))
            except Exception as e:
                node.get_logger().warn(f"[precompute] input cloud replay failed: {e}")
        if res.get("centroid_msg") is not None:
            node.pub_centroid.publish(res["centroid_msg"]); node.last_centroid = res["centroid_msg"]
        if res.get("centroid_base_msg") is not None:
            m = res["centroid_base_msg"]; m.header.stamp = node.get_clock().now().to_msg()
            node.pub_centroid_base.publish(m); node.last_centroid_base = m
        return True

    # ── measurements ─────────────────────────────────────────────────────
    def _depth_at(self, box) -> Optional[float]:
        """Median depth (m) of a 13x13 window at the box centre — the same
        measure the overlay's D:NNcm uses."""
        depth = self._node.fb.depth_frame
        if depth is None or len(box) != 4:
            return None
        h, w = depth.shape[:2]
        u, v = (int(c) for c in _box_centre(box))
        patch = np.asarray(depth[max(0, v - 6):min(h, v + 7), max(0, u - 6):min(w, u + 7)], dtype=np.float64)
        patch = patch[patch > 0]
        if patch.size < 5:
            return None
        return float(np.median(patch)) / float(self._node.fb.depth_scale)

    def _tool0(self) -> Optional[np.ndarray]:
        try:
            tf = self._node.tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return None
        from tf_transformations import quaternion_matrix
        q = tf.transform.rotation
        m = quaternion_matrix([q.x, q.y, q.z, q.w])
        m[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        return m

    # ── the worker ───────────────────────────────────────────────────────
    def _may_run(self) -> bool:
        n = self._node
        if not self.enabled or self._foreground:
            return False
        if getattr(n, "selection_made", False) or getattr(n, "_awaiting_executor_cycle_done", False):
            return False
        if bool(self._p("registration_enabled", False)):
            return False          # multi-view sweeps move the arm; not cached
        return True

    def _next_job(self) -> Optional[_Track]:
        stable = float(self._p("precompute_stable_s", 0.2))
        cap = int(self._p("precompute_max_objects", 8))
        now = time.monotonic()
        with self._state:
            if len(self._entries) >= cap:
                return None
            todo = [t for k, t in self._tracks.items()
                    if k not in self._entries and now - t.first_seen >= stable
                    and now - t.last_seen < 1.0 and t.reach_state != "red"
                    and now - t.failed_t > 5.0]
            if not todo:
                return None
            hover = self._hover_key
            rank = {"green": 0, "": 0, "orange": 1}
            todo.sort(key=lambda t: (0 if t.key == hover else 1, rank.get(t.reach_state, 2), -t.score))
            return copy.copy(todo[0])

    def _loop(self) -> None:
        while True:
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            try:
                if not self._may_run():
                    continue
                job = self._next_job()
                if job is None:
                    continue
                try:
                    self._run_job(job)
                except Exception as e:
                    self._node.get_logger().warn(f"[precompute] job '{job.key}' error: {e}")
                    self._mark_failed(job.key)
            except Exception as e:
                self._node.get_logger().warn(f"[precompute] worker error: {e}")

    def _mark_failed(self, key: str) -> None:
        with self._state:
            t = self._tracks.get(key)
            if t is not None:
                t.failed_t = time.monotonic()

    def _run_job(self, tr: _Track) -> None:
        node = self._node
        header = node.fb.rgb_header
        if header is None:
            return
        t_before = self._tool0()
        d_before = self._depth_at(tr.box)
        if t_before is None or d_before is None:
            return
        params_hash = self._params_hash_now()
        params_snapshot = self._params_snapshot
        sel = {"label": tr.label, "simplified_label": tr.label, "display": tr.key,
               "score": tr.score, "bbox": list(tr.box), "header": copy.deepcopy(header)}
        t0 = time.monotonic()
        if not self.model_lock.acquire(timeout=0.05):
            return
        try:
            if not self._may_run():
                return
            res = self._pick._compute(sel, background=True)
        finally:
            self._pick._background = False
            self.model_lock.release()
        dt_job = time.monotonic() - t0
        if not res:
            if getattr(node, "selection_made", False):
                node.get_logger().info(f"[precompute] job '{tr.key}' dropped after {dt_job:.2f} s: operator selected")
            else:
                node.get_logger().warn(f"[precompute] job '{tr.key}' failed after {dt_job:.2f} s")
                self._mark_failed(tr.key)
            return
        t_after = self._tool0()
        if t_after is None:
            return
        dmove, rmove = _pose_delta(t_before, t_after)
        if dmove > float(self._p("precompute_pose_tol_m", 0.002)) or rmove > float(self._p("precompute_pose_tol_deg", 0.5)):
            node.get_logger().info(f"[precompute] job '{tr.key}' discarded: arm moved during the capture")
            return
        obs0 = res.get("obs0") or {}
        mask = obs0.get("selected_mask")
        entry = CacheEntry(key=tr.key, label=tr.label, box=list(tr.box), depth_m=d_before, tool0=t_before,
                           params_hash=params_hash, params_snapshot=params_snapshot,
                           result=res, t_done=time.monotonic(),
                           n_grasps=len(res.get("grasps") or []), mask=mask)
        with self._state:
            live = self._tracks.get(tr.key)
            if live is None or not _boxes_agree(live.box, tr.box, float(self._p("precompute_box_shift_px", 8.0)),
                                                float(self._p("precompute_size_tol", 0.15))):
                node.get_logger().info(f"[precompute] job '{tr.key}' discarded: object moved during the capture")
                return
            self._entries[tr.key] = entry
        node.get_logger().info(
            f"[precompute] '{tr.key}' ready in {dt_job:.2f} s: {entry.n_grasps} grasps kept "
            f"of {len(res.get('raw_grasps') or [])} (stored {len(self._entries)})")
