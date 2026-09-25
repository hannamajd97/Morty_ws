#!/usr/bin/env python3
"""Pointing benchmark: record labelled pointing, then score pointing methods offline.

Why (2026-09-24, Majd: "whatever the object is, it should get it correctly"):
pointing was tuned by eye. This records what the wrist RealSense saw while
the operator pointed at each object in turn, with the target of every time
window known, and replays it through ``PointOrVoiceSelector.begin/step`` —
the exact code the live pipeline runs — for several settings at once.

    pointing_bench_record [--rounds 2] [--seconds 5] [--no-speak]
        Robot at the observation pose, pipeline running, objects on the table.
        Morty says "Point at the <object>" for every detected object, left to
        right; point at it until the next prompt. Hand pointing selection is
        switched OFF on the pipeline for the duration (so pointing cannot
        start a pick) and restored afterwards.
        -> ~/morty_ws/experiments/pointing_bench/<date_time>/{bag/, labels.json}

    pointing_bench_eval <folder> [--variants dwell,vote,vote_model]
        Correct / wrong / no choice per variant, time to the choice, and how
        many frames had a 3-D finger ray.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PIPELINE_NODE = "/yolo_live_select_grasp"
TOPICS = {
    "color": "/camera/camera/color/image_raw",
    "depth": "/camera/camera/aligned_depth_to_color/image_raw",
    "depth_info": "/camera/camera/aligned_depth_to_color/camera_info",
    "dets": "/yolo/all_detections",
}
OUT_ROOT = os.path.expanduser("~/morty_ws/experiments/pointing_bench")

# The live pipeline's pointing settings, as declared in selection/params.py
# (the evaluation varies only the commit rule and the ray source).
LIVE = dict(min_hold_ms=400, steady_px=10.0, pointer_pref_inside=False, snap_px=80.0,
            ray_extension=1.5, min_detection_confidence=0.60, min_tracking_confidence=0.60,
            pointing_mode="3d", cone_deg=20.0, hysteresis=0.8, use_tasks_api=True, model_path="",
            vote_tau_s=0.8, vote_commit_s=0.45, vote_margin=2.0, sigma_deg=7.0)
VARIANTS = {
    "dwell": dict(decision="dwell", ray_source="depth"),        # before 2026-09-24
    "vote": dict(decision="vote", ray_source="depth"),          # live default: votes where the arrow is (09-25)
    "vote_angle": dict(decision="vote", ray_source="depth", evidence="angle"),   # 09-24: spread by angle
    "vote_model": dict(decision="vote", ray_source="model"),    # MediaPipe 3-D hand + palm depth
    "dwell_model": dict(decision="dwell", ray_source="model"),
}


def _spoken_names(targets: List[Dict[str, Any]]) -> List[str]:
    """Names the operator can act on: two vitamin tubes become "the vitamin
    tube with the yellow cap" (when the cap colours differ) or "the vitamin
    tube on the left / in the middle / on the right" (targets are sorted
    left to right)."""
    base = [str(d.get("display") or d.get("label")).replace("_", " ") for d in targets]
    out = list(base)
    for name in set(base):
        idx = [i for i, b in enumerate(base) if b == name]
        if len(idx) < 2:
            continue
        caps = [str(targets[i].get("cap_colour") or "").strip() for i in idx]
        if all(caps) and len(set(caps)) == len(caps):
            for i, c in zip(idx, caps):
                out[i] = f"{name} with the {c} cap"
        else:
            where = (["on the left", "on the right"] if len(idx) == 2 else
                     ["on the left"] + ["in the middle"] * (len(idx) - 2) + ["on the right"])
            for i, w in zip(idx, where):
                out[i] = f"{name} {w}"
    return out


def _stamp(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


# ════════════════════════════════════════════════════════════════ record
def record_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Guided, labelled pointing recording (see module doc).")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=5.0, help="pointing time per object")
    ap.add_argument("--lead", type=float, default=1.0, help="time after the prompt before the window starts")
    ap.add_argument("--no-speak", action="store_true", help="prompts in the terminal only")
    a = ap.parse_args(argv)

    import rclpy
    from rclpy.action import ActionClient
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import GetParameters, SetParameters
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("pointing_bench_recorder")
    dets: List[Dict[str, Any]] = []

    def _on_dets(msg):
        try:
            d = json.loads(msg.data)
            if isinstance(d, list) and d:
                dets[:] = d
        except ValueError:
            pass
    node.create_subscription(String, TOPICS["dets"], _on_dets, 10)
    t = time.time()
    while not dets and time.time() - t < 8.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not dets:
        print("No detections on /yolo/all_detections — is the pipeline running and the table in view?")
        node.destroy_node(); rclpy.shutdown()
        return 1
    targets = sorted(dets, key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2.0)
    names = _spoken_names(targets)
    print("Objects, left to right:", ", ".join(names))

    # Pointing must not start a pick while we record.
    get_cli = node.create_client(GetParameters, f"{PIPELINE_NODE}/get_parameters")
    set_cli = node.create_client(SetParameters, f"{PIPELINE_NODE}/set_parameters")

    def _call(cli, req, timeout=8.0):   # the pipeline node answers slowly under load
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
        return fut.result()

    def _set_pointing(on: bool) -> bool:
        p = Parameter(name="use_mediapipe_selection",
                      value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=bool(on)))
        r = _call(set_cli, SetParameters.Request(parameters=[p]))
        return bool(r and r.results and r.results[0].successful)

    r = _call(get_cli, GetParameters.Request(names=["use_mediapipe_selection"]))
    was_on = bool(r.values[0].bool_value) if (r and r.values) else None
    if was_on is None:
        print(f"Could not read {PIPELINE_NODE} use_mediapipe_selection — refusing to record while "
              "pointing could trigger a pick.")
        node.destroy_node(); rclpy.shutdown()
        return 1

    speak = None
    if not a.no_speak:
        try:
            from morty_interfaces.action import Speak
            speak = ActionClient(node, Speak, "/speak")
            if not speak.wait_for_server(timeout_sec=2.0):
                speak = None
        except Exception:
            speak = None

    def say(text: str) -> None:
        print(">>", text)
        if speak is None:
            return
        from morty_interfaces.action import Speak
        g = Speak.Goal(); g.text = text
        fut = speak.send_goal_async(g)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        h = fut.result()
        if h is not None and h.accepted:
            rf = h.get_result_async()
            rclpy.spin_until_future_complete(node, rf, timeout_sec=15.0)

    def wait(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    out = os.path.join(OUT_ROOT, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    bag = None
    windows: List[Dict[str, Any]] = []
    try:
        if was_on:
            _set_pointing(False)
            print("Hand pointing selection OFF on the pipeline while recording.")
        bag = subprocess.Popen(["ros2", "bag", "record", "-o", os.path.join(out, "bag")] + list(TOPICS.values()),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        wait(3.0)
        say(f"Pointing test. I will name {len(targets)} objects. Point at each one until I name the next.")
        for rnd in range(a.rounds):
            for i, (d, name) in enumerate(zip(targets, names)):
                say(f"Point at the {name}.")
                wait(a.lead)
                t0 = node.get_clock().now().nanoseconds * 1e-9
                wait(a.seconds)
                t1 = node.get_clock().now().nanoseconds * 1e-9
                windows.append({"round": rnd, "order": i, "label": d.get("label"), "display": name,
                                "bbox": d.get("bbox"), "t0": t0, "t1": t1})
        say("Done, thank you.")
    finally:
        if bag is not None:
            try:
                os.killpg(bag.pid, signal.SIGINT)
                bag.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(bag.pid, signal.SIGKILL)
                except Exception:
                    pass
        if was_on:
            print("Hand pointing selection restored:", "ON" if _set_pointing(True) else "FAILED — switch it on in the GUI")
        with open(os.path.join(out, "labels.json"), "w") as f:
            json.dump({"created": datetime.datetime.now().isoformat(), "objects": targets,
                       "windows": windows, "seconds": a.seconds, "lead": a.lead, "topics": TOPICS}, f, indent=2)
        node.destroy_node(); rclpy.shutdown()
    print(f"\nRecorded {len(windows)} windows -> {out}\nScore it:  pointing_bench_eval {out}")
    return 0


# ═══════════════════════════════════════════════════════════════ evaluate
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _messages(bag_dir: str):
    """(topic, msg) in recording order."""
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    with open(os.path.join(bag_dir, "metadata.yaml")) as f:
        storage = yaml.safe_load(f)["rosbag2_bagfile_information"].get("storage_identifier", "sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic in types:
            yield topic, deserialize_message(data, types[topic])


def eval_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score pointing methods on a pointing_bench recording.")
    ap.add_argument("folder")
    ap.add_argument("--variants", default="dwell,vote,vote_angle,vote_model")
    ap.add_argument("--hz", type=float, default=15.0, help="frames per second fed to the selectors (live ≈ 15)")
    ap.add_argument("--grace", type=float, default=2.0, help="seconds after a window a late choice still counts")
    a = ap.parse_args(argv)

    from cv_bridge import CvBridge
    from .point_selector import PointOrVoiceSelector

    meta = json.load(open(os.path.join(a.folder, "labels.json")))
    windows = meta["windows"]
    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    for v in names:
        if v not in VARIANTS:
            print(f"unknown variant {v!r}; known: {', '.join(VARIANTS)}")
            return 2
    bridge = CvBridge()
    cur = {"pair": None}
    sel = {v: PointOrVoiceSelector(use_mediapipe=True, depth_provider=lambda: cur["pair"],
                                   **LIVE, **VARIANTS[v]) for v in names}
    res: Dict[str, List[Dict[str, Any]]] = {v: [] for v in names}
    state = {v: {"w": None, "done": False} for v in names}

    K = None
    # The detections snapshot taken when recording started; the pipeline
    # publishes /yolo/all_detections only ~1.4 Hz and not at all while busy,
    # so a recording may hold none — the snapshot is the fallback.
    dets: List[Dict[str, Any]] = list(meta.get("objects") or [])
    colours: List[Tuple[float, np.ndarray]] = []
    last_fed = -1e9
    wi = 0

    def _close_window(v: str) -> None:
        st = state[v]
        if st["w"] is not None and not st["done"]:
            res[v].append({"window": st["w"], "chosen": None, "t": None,
                           "n3d": sel[v]._n_3d, "nframes": sel[v]._n_frames})
        st["w"], st["done"] = None, False

    for topic, msg in _messages(os.path.join(a.folder, "bag")):
        if topic == TOPICS["depth_info"]:
            K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            continue
        if topic == TOPICS["dets"]:
            try:
                d = json.loads(msg.data)
                if isinstance(d, list) and d:
                    dets = d
            except ValueError:
                pass
            continue
        if topic == TOPICS["color"]:
            colours.append((_stamp(msg.header), bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")))
            colours = colours[-6:]
            continue
        if topic != TOPICS["depth"] or K is None or not colours or not dets:
            continue
        ts = _stamp(msg.header)
        if ts - last_fed < 1.0 / a.hz:
            continue
        # open / close windows by time
        while wi < len(windows) and ts > windows[wi]["t1"] + a.grace:
            for v in names:
                if state[v]["w"] == wi:
                    _close_window(v)
            wi += 1
        if wi >= len(windows):
            break
        w = windows[wi]
        if ts < w["t0"]:
            continue
        cs, bgr = min(colours, key=lambda c: abs(c[0] - ts))
        if abs(cs - ts) > 0.04:
            continue
        last_fed = ts
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cur["pair"] = (bgr, depth, K, 1000.0 if depth.dtype == np.uint16 else 1.0, None)
        for v in names:
            st = state[v]
            if st["w"] != wi:
                _close_window(v)
                st["w"], st["done"] = wi, False
                sel[v].begin()
            if st["done"]:
                continue
            chosen, _hover, _px, _mode = sel[v].step(ts, bgr, dets)
            if chosen is not None:
                st["done"] = True
                tgt = max(range(len(dets)), key=lambda i: (_iou(dets[i]["bbox"], w["bbox"])
                                                           if dets[i].get("label") == w["label"] else -1.0))
                res[v].append({"window": wi, "chosen": chosen, "target": tgt,
                               "chosen_label": dets[chosen].get("label"), "t": ts - w["t0"],
                               "n3d": sel[v]._n_3d, "nframes": sel[v]._n_frames})
    for v in names:
        if state[v]["w"] is not None:
            _close_window(v)

    n = len(windows)
    print(f"\n{a.folder}\n{n} pointing windows ({len(meta['objects'])} objects × {max(w['round'] for w in windows) + 1} rounds)\n")
    print(f"{'variant':12s} {'correct':>8s} {'wrong':>6s} {'none':>5s} {'median s':>9s} {'slowest s':>9s} {'3-D ray frames':>15s}")
    for v in names:
        rows = res[v]
        ok = sum(1 for r in rows if r["chosen"] is not None and r["chosen"] == r.get("target"))
        bad = sum(1 for r in rows if r["chosen"] is not None and r["chosen"] != r.get("target"))
        none = n - ok - bad
        ts_ = sorted(r["t"] for r in rows if r["t"] is not None)
        f3 = sum(r["n3d"] for r in rows); ff = sum(r["nframes"] for r in rows)
        print(f"{v:12s} {ok:>5d}/{n:<2d} {bad:>6d} {none:>5d} {(ts_[len(ts_) // 2] if ts_ else float('nan')):>9.2f} "
              f"{(ts_[-1] if ts_ else float('nan')):>9.2f} {(100.0 * f3 / ff if ff else 0):>13.0f} %")
    print("\nwrong choices:")
    for v in names:
        for r in res[v]:
            if r["chosen"] is not None and r["chosen"] != r.get("target"):
                w = windows[r["window"]]
                print(f"  {v:12s} pointed at {w['display']!r:22s} got {r['chosen_label']!r} after {r['t']:.1f} s")
    json.dump(res, open(os.path.join(a.folder, "eval.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(eval_main())
