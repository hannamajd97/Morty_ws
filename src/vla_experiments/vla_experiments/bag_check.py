"""Is a trial's ROS bag really there and readable? (2026-09-30)

    verify_bag(bag_dir, required_topics=(), window=None) -> dict

metadata.yaml alone proves nothing: rosbag2 writes it at the end, but the
database can be missing, a Git LFS pointer stub (the pushed trials store
bags through LFS — a checkout without `git lfs pull` has 130-byte text
files where the databases were), corrupt, or cut short by a SIGKILL. This
checks, without ROS:

* every database metadata.yaml names exists and is not an LFS pointer;
* each opens read-only and passes SQLite's quick_check;
* message counts (per topic and in total) match the metadata;
* the required topics were recorded with at least one message;
* the recorded time span is plausible (inside ``window`` = (start, end)
  wall seconds, when given) and the rosbag2 process exited cleanly (the
  trial's bag_status.json, written by the GUI's BagLink, when present).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple

LFS_HEADER = b"version https://git-lfs"


def _metadata(bag_dir: Path) -> dict:
    import yaml
    with open(bag_dir / "metadata.yaml") as f:
        return (yaml.safe_load(f) or {}).get("rosbag2_bagfile_information", {})


def verify_bag(bag_dir, required_topics: Iterable[str] = (), window: Optional[Tuple[float, float]] = None) -> dict:
    bag_dir = Path(bag_dir)
    out = {"bag": str(bag_dir), "ok": False, "issues": [], "files": [], "topics": {}, "message_count": 0}
    if not (bag_dir / "metadata.yaml").exists():
        out["issues"].append("no metadata.yaml (the recorder did not finish)")
        return out
    try:
        meta = _metadata(bag_dir)
    except Exception as exc:  # noqa: BLE001
        out["issues"].append(f"metadata.yaml unreadable: {exc}")
        return out
    files = list(meta.get("relative_file_paths") or [])
    if not files:
        out["issues"].append("metadata.yaml names no database")
    want = {t["topic_metadata"]["name"]: int(t.get("message_count", 0))
            for t in meta.get("topics_with_message_count") or []}
    got: dict = {}
    total = 0
    t_min = t_max = None
    for rel in files:
        db = bag_dir / rel
        info = {"file": rel, "exists": db.exists()}
        out["files"].append(info)
        if not db.exists():
            out["issues"].append(f"{rel}: missing")
            continue
        info["bytes"] = db.stat().st_size
        with open(db, "rb") as f:
            head = f.read(64)
        if head.startswith(LFS_HEADER):
            out["issues"].append(f"{rel}: a Git LFS pointer, not the database (run `git lfs pull`)")
            continue
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            qc = con.execute("pragma quick_check").fetchone()[0]
            if qc != "ok":
                out["issues"].append(f"{rel}: sqlite quick_check: {qc}")
            for name, n, lo, hi in con.execute(
                    "select t.name, count(m.id), min(m.timestamp), max(m.timestamp) from topics t "
                    "left join messages m on m.topic_id = t.id group by t.id"):
                got[name] = got.get(name, 0) + int(n)
                total += int(n)
                if lo is not None:
                    t_min = lo if t_min is None else min(t_min, lo)
                    t_max = hi if t_max is None else max(t_max, hi)
            con.close()
        except sqlite3.DatabaseError as exc:
            out["issues"].append(f"{rel}: not a readable database ({exc})")
    out["topics"] = got
    out["message_count"] = total
    if files and not out["issues"]:
        if total != int(meta.get("message_count", total)):
            out["issues"].append(f"{total} messages in the database(s), metadata says {meta.get('message_count')}")
        for name, n in want.items():
            if got.get(name, 0) != n:
                out["issues"].append(f"{name}: {got.get(name, 0)} messages, metadata says {n}")
    for t in required_topics:
        if got.get(t, 0) == 0:
            out["issues"].append(f"required topic {t} has no messages")
    if t_min is not None:
        out["start"], out["end"] = t_min * 1e-9, t_max * 1e-9
        if window is not None:
            lo, hi = window
            if out["start"] < lo - 60.0 or out["end"] > hi + 600.0:
                out["issues"].append(f"recorded span {out['start']:.0f}..{out['end']:.0f} lies outside the trial "
                                     f"({lo:.0f}..{hi:.0f})")
    status_p = bag_dir.parent / "bag_status.json"
    if status_p.exists():
        try:
            st = json.loads(status_p.read_text())
            out["recorder"] = st
            if st.get("exit_code") not in (0, None) or st.get("stop_stage", 1) > 1:
                out["issues"].append(f"rosbag2 did not exit cleanly (exit code {st.get('exit_code')}, "
                                     f"stop stage {st.get('stop_stage')})")
        except ValueError:
            out["issues"].append("bag_status.json unreadable")
    out["ok"] = not out["issues"]
    return out


def trial_bags(trial_dir) -> list:
    """The bag folders of a trial (bag, bag_HHMMSS …)."""
    return sorted(p for p in Path(trial_dir).iterdir() if p.is_dir() and p.name.startswith("bag"))
