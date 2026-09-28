"""Final stage times of a trial from exact event times (2026-09-28).

The trial window stamps its events when the GUI's callback runs — usually
0.1-0.3 s late, at worst 3 s (trials 083 and 085, during the start-up load)
— and the stage times it writes at ⏹ inherit that. The same messages are
recorded by two processes that are never late: the trial's ROS bag (rosbag2
stamps every message on receipt, in its own process) and the VLA recorder
(the events in meta.json). This module moves every GUI event the stage times
use to its exact time — the matching message in the bag (same content,
nearest in time within 10 s), else in the VLA episode — and rebuilds the
timeline and the stage times with the SAME function the window uses
(scenario_experiments.build_timeline). The pipeline and executor records
(selection, background jobs, executor phases) carry their own stamps and
are used as they are.

It rewrites timeline.txt / timeline.json / events.json / trial.json, and the
trial's rows in mmhri_experiments.csv and mmhri_meta.csv under the lock the
trial window also takes — each CSV value only while it still holds the
window's preliminary number (a value typed by hand is never replaced).

    python3 -m morty_gui.trial_times <trial dir> [...]

Runs after every saved trial through morty_gui.trial_postprocess.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path

MMHRI_DIR = os.path.expanduser("~/morty_ws/experiments/mmhri")
CSV_LOCK = os.path.join(MMHRI_DIR, ".csv.lock")
MATCH_WINDOW_S = 10.0

_BAG_TOPICS = {
    "/speech/accepted_transcript": "transcript",
    "/robot/intent": "intent",
    "/graspgen/selection_cycle_id": "selection_cycle_id",
    "/grasp_executor_lin_llm/cycle_done": "cycle_done",
    "/speech/ai_said": "morty_said",
    "/speech/utterance_meta": "utterance",
    "/grasp_executor_lin_llm/interactive_status": "robot_status",
}


@contextlib.contextmanager
def csv_lock():
    """Exclusive lock shared with the trial window's CSV writes."""
    import fcntl
    os.makedirs(MMHRI_DIR, exist_ok=True)
    with open(CSV_LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _intent_key(text: str):
    try:
        d = json.loads(text)
    except Exception:
        return text
    return json.dumps({"a": d.get("action_intent"), "o": d.get("object_intent"),
                       "x": d.get("object_attributes")}, sort_keys=True)


def _key(source: str, text: str):
    if source == "intent":
        return _intent_key(text)
    if source == "cycle_done":
        parts = str(text).split("|")
        return (parts[0].lower(), parts[1] if len(parts) > 1 else "")
    return str(text).strip()


def bag_events(trial: Path) -> dict:
    """source -> [(wall_time, text)] from the trial's closed bag (receive stamps)."""
    bags = sorted(p for p in trial.iterdir() if p.is_dir() and p.name.startswith("bag") and (p / "metadata.yaml").exists())
    if not bags:
        return {}
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bags[0]), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics = [t for t in _BAG_TOPICS if t in types]
    if not topics:
        return {}
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    cls = {t: get_message(types[t]) for t in topics}
    out: dict = {}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        out.setdefault(_BAG_TOPICS[topic], []).append((t_ns * 1e-9, str(msg.data)))
    return out


def vla_events(trial: Path) -> dict:
    """source -> [(wall_time, text)] from the VLA episode's events."""
    try:
        ev = json.loads((trial / "meta.json").read_text()).get("events") or {}
    except Exception:
        return {}
    out: dict = {}
    for e in ev.get("transcript", []):
        out.setdefault("transcript", []).append((float(e["stamp"]), str(e.get("text", ""))))
    for e in ev.get("intent", []):
        out.setdefault("intent", []).append((float(e["stamp"]), json.dumps(
            {"action_intent": e.get("action_intent"), "object_intent": e.get("object_intent"),
             "object_attributes": e.get("object_attributes")})))
    for e in ev.get("cycle_done", []):
        out.setdefault("cycle_done", []).append(
            (float(e["stamp"]), ("done" if e.get("ok") else "failed") + "|" + str(e.get("cycle_id", ""))))
    for e in ev.get("interactive_status", []):
        out.setdefault("robot_status", []).append((float(e["stamp"]), str(e.get("status", ""))))
    return out


def arm_motion_starts(trial: Path, start_wall: float, speed_rad_s: float = 0.02, still_s: float = 0.5) -> list:
    """Wall times at which the arm STARTS moving (joint speed rises above
    speed_rad_s after at least still_s at rest) — from joints_full.npz (the
    bag, ~500 Hz) else steps.npz (10 Hz). For moves the executor does not
    stamp (look / drop poses), this is the only 'arm starts moving' time."""
    import numpy as np
    t = v = None
    if (trial / "joints_full.npz").exists():
        with np.load(trial / "joints_full.npz") as z:
            t, v = z["stamp"], z["vel"]
    elif (trial / "steps.npz").exists():
        with np.load(trial / "steps.npz") as z:
            if "joint_stamp" in z.files:
                t, v = z["joint_stamp"], z["joint_vel"]
    if t is None or len(t) < 3:
        return []
    moving = np.linalg.norm(np.nan_to_num(v), axis=1) > speed_rad_s
    out, last_move = [], -1e18
    for i in np.where(moving)[0]:
        if t[i] - last_move > still_s:
            out.append(float(t[i]))
        last_move = t[i]
    return [x for x in out if x >= start_wall - 1.0]


def exact_events(events: list, start_wall: float, sources: list) -> tuple:
    """Move each (t_rel, source, text) to the nearest exact event with the same
    content (first source that has one wins). Returns (events, report)."""
    moved, shifts, used = 0, [], {}
    out = []
    for t_rel, src, txt in events:
        t_gui = start_wall + float(t_rel)
        # A recorder can receive a message only AFTER it was sent, so every
        # recorder's time is an upper bound and the EARLIEST one is the best
        # (trial 070: the bag, still starting, got a transcript 2.8 s after
        # the GUI did). Per source the occurrence nearest the GUI's time.
        best = None
        for name, acc in sources:
            cands = [(abs(t - t_gui), t) for t, x in acc.get(src, []) if _key(src, x) == _key(src, txt)
                     and abs(t - t_gui) <= MATCH_WINDOW_S]
            if cands:
                t_src = min(cands)[1]
                if t_src < t_gui and (best is None or t_src < best[0]):
                    best = (t_src, name)
        if best is not None:
            shifts.append(best[0] - t_gui)
            used[best[1]] = used.get(best[1], 0) + 1
            moved += 1
            out.append((best[0] - start_wall, src, txt))
        else:
            out.append((float(t_rel), src, txt))
    out.sort(key=lambda e: e[0])
    report = {"moved": moved, "from": used,
              "gui_late_by_s": {"median": round(sorted(-s for s in shifts)[len(shifts) // 2], 3) if shifts else None,
                                "max": round(max((-s for s in shifts), default=0.0), 3) if shifts else None}}
    return out, report


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a or "") == str(b or "")


def _fmt(v) -> str:
    return "" if v is None else str(v)


def _update_csv(path: str, match, updates: dict) -> int:
    """Rewrite the LAST row that `match`es: column -> (old, new), each only
    while the row still holds `old`. Keeps the file's line endings."""
    if not os.path.exists(path):
        return 0
    raw = open(path, newline="").read()
    rows = list(csv.reader(io.StringIO(raw, newline="")))
    if not rows:
        return 0
    hdr = rows[0]
    idx = next((i for i in range(len(rows) - 1, 0, -1) if rows[i] and match(dict(zip(hdr, rows[i])))), None)
    if idx is None:
        return 0
    n = 0
    for col, (old, new) in updates.items():
        if col in hdr:
            c = hdr.index(col)
            if c < len(rows[idx]) and _same(rows[idx][c], old) and rows[idx][c] != _fmt(new):
                rows[idx][c] = _fmt(new)
                n += 1
    if n:
        term = "\r\n" if "\r\n" in raw else "\n"
        buf = io.StringIO(newline="")
        csv.writer(buf, lineterminator=term).writerows(rows)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            f.write(buf.getvalue())
        os.replace(tmp, path)
    return n


def _backup(paths) -> None:
    """First rewrite of a day keeps <file>.orig_<date> next to it."""
    import datetime as _dt
    import shutil
    tag = ".orig_" + _dt.date.today().strftime("%Y%m%d")
    for p in paths:
        p = Path(p)
        if p.exists() and not Path(str(p) + tag).exists():
            shutil.copy2(p, str(p) + tag)


def finalize(trial_dir) -> dict:
    trial = Path(trial_dir)
    tl_p = trial / "timeline.json"
    if not tl_p.exists():
        return {"trial": trial.name, "skipped": "no timeline.json (no MMHRI record)"}
    _backup([tl_p, trial / "timeline.txt", trial / "events.json", trial / "trial.json"])
    from .scenario_experiments import build_timeline
    tl = json.loads(tl_p.read_text())
    start, stop = float(tl["start_wall"]), float(tl["stop_wall"])
    # Always rebuild from the WINDOW'S OWN event times (kept as events_gui;
    # before the first run they are "events", or in the day's backup when an
    # earlier version already rewrote them) — never from a previous result.
    gui = tl.get("events_gui")
    if gui is None:
        backups = sorted(trial.glob("timeline.json.orig_*"))
        if backups:
            try:
                gui = json.loads(backups[0].read_text()).get("events")
            except Exception:
                gui = None
    if gui is None:
        gui = tl.get("events", [])
    events = [(float(e["t"]), e["source"], e["text"]) for e in gui]
    # Order: the VLA recorder first — it matched the publishers' own logs in
    # every check; the rosbag stamps on its callback too and was 3 s late in
    # trial 085 while it was still starting. Messages that carry their own
    # time (intents since 2026-09-28, the utterance meta) are placed by that
    # time in build_timeline whatever this does.
    sources = []
    v = vla_events(trial)
    if v:
        sources.append(("vla", v))
    b = {}
    try:
        b = bag_events(trial)
        if b:
            sources.append(("bag", b))
    except Exception as exc:  # noqa: BLE001 - a broken bag must not block the VLA fallback
        print(json.dumps({"bag_events": f"unreadable: {type(exc).__name__}: {exc}"}), flush=True)
    fixed, report = exact_events(events, start, sources)
    # the arm's own motion starts (joint data) — Interaction Time for moves the
    # executor does not stamp (look / drop poses)
    fixed = [e for e in fixed if e[1] != "arm_motion"]
    for tw in arm_motion_starts(trial, start):
        fixed.append((tw - start, "arm_motion", "starts"))
    # robot status messages the window did not record (trials before
    # 2026-09-28): from the VLA recorder, else the bag
    if not any(s == "robot_status" for _t, s, _x in fixed):
        for name, acc in sources:
            if acc.get("robot_status"):
                for t, x in acc["robot_status"]:
                    fixed.append((t - start, "robot_status", x))
                break
    fixed.sort(key=lambda e: e[0])
    # the utterance payloads carry speech_end; add them from the bag when the
    # window did not record them (trials before 2026-09-28)
    if not any(s == "utterance" for _t, s, _x in fixed):
        for t, x in (b or {}).get("utterance", []):
            fixed.append((t - start, "utterance", x))
        fixed.sort(key=lambda e: e[0])
    records = [(0.0, r) for r in (tl.get("selection_cycles", []) + tl.get("executor_cycles", []) + tl.get("background", []))]
    text, data, stage = build_timeline(fixed, records, start, stop, start)
    old = tl.get("stage_times") or {}
    data["events_gui"] = gui
    data["event_times"] = {**report, "note": "GUI event times moved to the bag's receive stamps (else the VLA "
                                             "recorder's); stage times rebuilt from them"}
    stage_out = dict(stage)
    tl_p.write_text(json.dumps(data, indent=1, default=str))
    (trial / "timeline.txt").write_text(text)
    ev_p = trial / "events.json"
    if ev_p.exists():
        try:
            ej = json.loads(ev_p.read_text())
            ej["timings"] = stage_out
            ej["events_exact"] = [{"t": round(t, 3), "source": s, "text": x} for t, s, x in fixed]
            ev_p.write_text(json.dumps(ej, indent=1))
        except Exception:
            pass
    tj_p = trial / "trial.json"
    csv_rows = 0
    if tj_p.exists():
        tj = json.loads(tj_p.read_text())
        old_intent = tj.get("Interaction Time (s)")
        old_total = tj.get("Total Time (s)")
        tj["Stage Times"] = {**stage_out, "event_times": data["event_times"]}
        # 2026-09-28 (Majd): Interaction Time = command given -> arm moves
        # (interaction_s); earlier versions wrote the understanding time here
        if (_same(old_intent, old.get("interaction_s")) or _same(old_intent, old.get("intent_s"))
                or old_intent in ("", None)):
            tj["Interaction Time (s)"] = _fmt(stage.get("interaction_s"))
        if _same(old_total, old.get("total_s")) or old_total in ("", None):
            tj["Total Time (s)"] = _fmt(stage.get("total_s"))
        tj_p.write_text(json.dumps(tj, indent=2, ensure_ascii=False))
        tid, cmd, date = str(tj.get("Trial ID", "")), str(tj.get("Command Used", "")), str(tj.get("Date", ""))
        with csv_lock():
            _backup([os.path.join(MMHRI_DIR, "mmhri_experiments.csv"), os.path.join(MMHRI_DIR, "mmhri_meta.csv")])
            csv_rows += _update_csv(os.path.join(MMHRI_DIR, "mmhri_experiments.csv"),
                                    lambda r: r.get("Trial ID", "").strip() == tid and r.get("Command Used", "") == cmd,
                                    {"Interaction Time (s)": (old_intent, tj["Interaction Time (s)"]),
                                     "Total Time (s)": (old_total, tj["Total Time (s)"])})
            csv_rows += _update_csv(os.path.join(MMHRI_DIR, "mmhri_meta.csv"),
                                    lambda r: r.get("Trial ID", "").strip() == tid and r.get("Date", "") == date,
                                    {k: (old.get(k), stage.get(k)) for k in
                                     ("intent_s", "yolo_s", "graspgen_s", "execution_s", "total_s")})
    return {"trial": trial.name, "event_times": data["event_times"], "csv_values_updated": csv_rows,
            "stage": {k: stage.get(k) for k in ("interaction_s", "speech_end_to_intent_s", "intent_s", "choice_s", "yolo_s",
                                                "graspgen_compute_s", "graspgen_wait_s", "graspgen_lead_s",
                                                "verify_compute_s", "command_to_motion_s", "total_s", "started_by")}}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    for a in argv:
        try:
            print(json.dumps(finalize(Path(a).expanduser())), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"trial": Path(a).name, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
