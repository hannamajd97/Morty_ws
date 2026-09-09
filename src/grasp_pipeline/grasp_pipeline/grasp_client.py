"""
grasp_client.py
================

Bridge between grasp_pipeline and GraspGen. ``send_points_to_graspgen`` takes a
point cloud and returns grasp candidates as a JSON-style dict. Two backends are
supported, chosen by the ``GRASPGEN_MODE`` environment variable:

  * ``inprocess`` (default) — load the GraspGen model into THIS process and run
    inference directly. No separate service / ZMQ port needed. The model loads
    once (lazily) and is cached for the life of the process.
  * ``zmq`` — talk to a standalone GraspGen service over a ZMQ REQ/REP socket
    (the original design; start it with ``start_graspgen.sh``). Kept as a
    fallback so the service-based setup still works unchanged.

Both backends return the SAME dict shape, so every caller
(``GraspGenClient``, ``pick_pipeline``, ``grasp_selection_node``) is unaffected::

    {"ok": True,
     "grasps": [{"matrix": [16 floats], "score": float}, ...],
     "telemetry": {...}}

Usage::

    from grasp_pipeline.grasp_client import send_points_to_graspgen
    resp = send_points_to_graspgen(points, endpoint="tcp://127.0.0.1:5555")
    if resp.get("ok"):
        grasps = resp.get("grasps", [])

Environment variables:
    GRASPGEN_MODE    "inprocess" (default) or "zmq"
    GRASPGEN_CONFIG  path to the gripper .yml (default: the robotiq_2f_140
                     symlink under <workspace>/models/graspgen/)
"""
# grasp_pipeline/grasp_client.py

from __future__ import annotations
from typing import Iterable, Dict, Any, Optional
import logging
import os
import numpy as np

# Module logger: grasp_gen's own load messages ("Loading generator
# checkpoint...") reach the console via python logging, so these lines land
# right next to them and explain what is actually happening.
_LOG = logging.getLogger(__name__)


# Default gripper config used by the in-process backend. Resolved relative to
# $HOME so it survives the workspace name (the models/ dir holds symlinks to the
# GraspGen checkpoints + yml). Override with GRASPGEN_CONFIG.
# Honour $WS_DIR (same convention as selection/params.py); fall back to the
# historical name. Alienware workspace is ~/morty_ws. (2026-08-06)
_DEFAULT_CONFIG = os.path.join(
    os.environ.get("WS_DIR", os.path.expanduser("~/morty_ws")),
    "models/graspgen/graspgen_robotiq_2f_140.yml",
)

# Module-level singleton: the GraspGen sampler is expensive to build (loads two
# checkpoints onto the GPU), so we create it once on first use and reuse it for
# every subsequent call in this process.
_SAMPLER = None
_SAMPLER_CONFIG = None


def _grasps_to_payload(grasps, scores) -> list:
    """Convert raw GraspGen output (Nx4x4 poses + scores) into the list-of-dict
    response format the pipeline expects. Mirrors grasps_to_payload() in the
    GraspGen service so the in-process and ZMQ backends return identical data."""
    def _np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    g = _np(grasps)
    s = _np(scores)
    if g.ndim == 3 and g.shape[1:] == (4, 4):
        g = g.reshape(g.shape[0], 16)
    elif g.ndim == 2 and g.shape[1] == 16:
        pass
    else:
        # Unexpected shape: pass through raw so nothing is silently dropped.
        return [{"raw": _np(row).astype("float32").tolist(), "score": float(sc)}
                for row, sc in zip(g, s)]
    return [{"matrix": row.astype("float32").tolist(), "score": float(sc)}
            for row, sc in zip(g, s)]


_SAMPLER_LOCK = __import__("threading").Lock()


def _get_sampler(config_path: str):
    """Lazily build (and cache) the in-process GraspGenSampler. Rebuilds only if
    the requested config changes.

    Thread-safe (2026-08-06): the background preload thread and a user's first
    selection can race here — without the lock both built the model
    simultaneously (double VRAM, double load time, duplicate log lines).

    2026-08-27: a caller that loses that race now SAYS so. The silent wait
    made grasp_gen's "Loading ... checkpoint" lines (printed by the preload
    thread mid-build) look like the first pick was loading the model a second
    time. Both threads share the ONE module-level _SAMPLER."""
    global _SAMPLER, _SAMPLER_CONFIG
    if _SAMPLER is not None and _SAMPLER_CONFIG == config_path:
        return _SAMPLER
    if not _SAMPLER_LOCK.acquire(blocking=False):
        _LOG.info(
            "GraspGen sampler is being built by another thread (startup "
            "preload) — waiting to reuse that same instance..."
        )
        _SAMPLER_LOCK.acquire()
    try:
        if _SAMPLER is not None and _SAMPLER_CONFIG == config_path:
            _LOG.info(
                "Reusing the GraspGen sampler the concurrent thread just "
                "built — no reload."
            )
            return _SAMPLER
        # Imported here (not at module top) so that callers using the ZMQ
        # backend, or merely importing this module, don't pay the heavy
        # GraspGen import cost.
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        _SAMPLER = GraspGenSampler(load_grasp_cfg(config_path))
        _SAMPLER_CONFIG = config_path
        return _SAMPLER
    finally:
        _SAMPLER_LOCK.release()


def sampler_is_warm() -> bool:
    """True once the in-process GraspGen sampler is built and cached — the
    module-level singleton shared by the startup preload and the pick path.
    Cheap and lock-free; used by the node to report warmup state honestly."""
    return _SAMPLER is not None


def _send_inprocess(points_xyz, threshold, num_grasps, topk_num_grasps,
                    min_grasps, max_tries, remove_outliers,
                    config_path: str) -> Dict[str, Any]:
    """In-process backend: run GraspGen inference directly, no ZMQ."""
    from grasp_gen.grasp_server import GraspGenSampler

    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return {"ok": False, "error": f"'points' must be Nx3, got {pts.shape}"}

    sampler = _get_sampler(config_path)
    # Same call signature the service uses; only the XYZ columns are needed.
    grasps, scores = GraspGenSampler.run_inference(
        pts[:, :3], sampler,
        threshold, int(num_grasps), int(topk_num_grasps),
        int(min_grasps), int(max_tries), bool(remove_outliers),
    )

    payload = _grasps_to_payload(grasps, scores)
    # scores may be a CUDA tensor; move to host before numpy ops.
    def _to_host(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)
    s_np = _to_host(scores) if scores is not None else np.array([])

    # Optional auto-viz: if GRASPGEN_SAVE_VIZ points at a directory, render the
    # cloud + grasps to a timestamped PNG there. Done in a subprocess so the
    # Open3D EGL renderer's known teardown-segfault on WSL can never crash the
    # pipeline. Wrapped in try/except so a viz failure never breaks grasping.
    viz_dir = os.environ.get("GRASPGEN_SAVE_VIZ", "").strip()
    if viz_dir:
        try:
            _save_viz_subprocess(pts[:, :3], payload, viz_dir)
        except Exception:
            pass  # viz is best-effort; never let it affect the grasp result

    return {
        "ok": True,
        "grasps": payload,
        "telemetry": {
            "points": {"n_points": int(pts.shape[0])},
            "inference": {
                "num_grasps_after_threshold": int(len(payload)),
                "best_score": float(s_np.max()) if s_np.size else None,
                "threshold": float(threshold),
                "backend": "inprocess",
            },
        },
    }


def _save_viz_subprocess(pts_xyz: np.ndarray, payload: list, viz_dir: str) -> None:
    """Render cloud + grasps to a PNG via grasp_debug_viewer in a subprocess.

    Runs out-of-process so Open3D's EGL teardown crash on WSL is contained.
    Writes <viz_dir>/grasp_<timestamp>.png and also updates
    <viz_dir>/latest.png so a viewer can always open the most recent.
    """
    import os as _os, json as _json, subprocess as _sp, tempfile as _tf, time as _t
    _os.makedirs(viz_dir, exist_ok=True)
    ts = _t.strftime("%Y%m%d_%H%M%S")
    out = _os.path.join(viz_dir, f"grasp_{ts}.png")
    latest = _os.path.join(viz_dir, "latest.png")
    # Hand the cloud + grasps to the renderer via a temp JSON.
    data = {"pc": np.asarray(pts_xyz, dtype=np.float32).tolist(), "grasps": payload}
    tmp = _tf.NamedTemporaryFile("w", suffix=".json", delete=False)
    _json.dump(data, tmp); tmp.close()
    # Honour $WS_DIR and look in tools/, where the viewer actually lives in this
    # layout (it is NOT at the workspace root). Falls back to the historical
    # location. (Edited 2026-08-06 — workspace is ~/morty_ws here.)
    _ws = _os.environ.get("WS_DIR", _os.path.expanduser("~/morty_ws"))
    viewer = _os.path.join(_ws, "tools", "grasp_debug_viewer.py")
    if not _os.path.isfile(viewer):
        viewer = _os.path.join(_ws, "grasp_debug_viewer.py")
    # Fire and forget; don't block the pipeline waiting on the render.
    _sp.Popen(
        ["python3", viewer, "--from-json", tmp.name, "--save", out, "--also", latest],
        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
    )


def _send_zmq(points_xyz, endpoint, threshold, colors, num_grasps,
              topk_num_grasps, min_grasps, max_tries, remove_outliers,
              save_debug, timeout_ms, fx, fy, cx, cy, depth_scale) -> Dict[str, Any]:
    """ZMQ backend: talk to a standalone GraspGen service (original design)."""
    import zmq
    # A REQ socket enforces strict send->recv ordering, matching the service's
    # REP loop. A fresh socket per call keeps each request independent and avoids
    # a stuck REQ state if a previous call timed out mid-exchange.
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(endpoint)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)

    payload: Dict[str, Any] = {
        "points": np.asarray(points_xyz, dtype=np.float32).tolist(),
        "threshold": float(threshold),
        "num_grasps": int(num_grasps),
        "topk_num_grasps": int(topk_num_grasps),
        "min_grasps": int(min_grasps),
        "max_tries": int(max_tries),
        "remove_outliers": bool(remove_outliers),
        "save_debug": bool(save_debug),
    }
    if colors is not None:
        payload["colors"] = np.asarray(colors).tolist()
    if fx is not None: payload["fx"] = float(fx)
    if fy is not None: payload["fy"] = float(fy)
    if cx is not None: payload["cx"] = float(cx)
    if cy is not None: payload["cy"] = float(cy)
    if depth_scale is not None: payload["depth_scale"] = float(depth_scale)

    sock.send_json(payload)
    try:
        return sock.recv_json()  # blocking until the service replies
    except Exception:
        return {"ok": False, "error": "timeout or recv error"}


def send_points_to_graspgen(points_xyz: Iterable[Any], endpoint: str = "tcp://127.0.0.1:5555",
                            threshold: float = 0.4,
                            colors: Optional[Iterable[Any]] = None,
                            num_grasps: int = 200, topk_num_grasps: int = -1,
                            min_grasps: int = 40, max_tries: int = 6,
                            remove_outliers: bool = True, save_debug: bool = False,
                            timeout_ms: int = 60000,
                            fx: Optional[float] = None, fy: Optional[float] = None,
                            cx: Optional[float] = None, cy: Optional[float] = None,
                            depth_scale: Optional[float] = None) -> Dict[str, Any]:
    """
    Generate grasps for a point cloud and return the response dict.

    Backend is selected by GRASPGEN_MODE ("inprocess" default, or "zmq").
    The return shape is identical for both backends:
        {"ok": bool, "grasps": [{"matrix": [...16], "score": float}], "telemetry": {...}}

    Arguments:
        points_xyz: iterable of 3D points (Nx3) in the camera frame
        endpoint: ZMQ address of the GraspGen service (only used in zmq mode)
        threshold: minimum confidence threshold for grasps
        colors: optional Nx3 RGB colors (zmq mode only; ignored in-process)
        num_grasps: how many grasp candidates to request
        topk_num_grasps: keep only top-K grasps if >0
        min_grasps: minimum grasps to return (retry until this)
        max_tries: maximum retries if too few grasps
        remove_outliers: ask GraspGen to clean outliers from the cloud
        save_debug: ask the service to save debug npy files (zmq mode only)
        timeout_ms: socket timeout (zmq mode only)
        fx, fy, cx, cy, depth_scale: camera intrinsics (zmq mode only)
    """
    mode = os.environ.get("GRASPGEN_MODE", "inprocess").strip().lower()

    if mode == "zmq":
        return _send_zmq(points_xyz, endpoint, threshold, colors, num_grasps,
                         topk_num_grasps, min_grasps, max_tries, remove_outliers,
                         save_debug, timeout_ms, fx, fy, cx, cy, depth_scale)

    # Default: in-process.
    config_path = os.environ.get("GRASPGEN_CONFIG", _DEFAULT_CONFIG)
    try:
        return _send_inprocess(points_xyz, threshold, num_grasps, topk_num_grasps,
                               min_grasps, max_tries, remove_outliers, config_path)
    except Exception as e:
        # Surface the error in the same shape callers already handle.
        return {"ok": False, "error": f"in-process GraspGen failed: {e}"}
