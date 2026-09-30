"""After a trial (2026-09-28): full-rate joints and the viewer, without anyone
having to remember them.

    python3 -m vla_experiments.postprocess <trial dir>

The MMHRI trial window starts this detached when a trial has been saved.
The bag closes and the recorder writes the episode asynchronously after ⏹,
so it first waits (at most 90 s) until the bag's metadata.yaml exists and, if
the trial recorded a VLA episode (images/), until steps.npz and meta.json are
there and no longer growing. Then:

1. joints_full.process — joints_full.npz/.csv from the bag, joint_acc and
   joint_current added to steps.npz;
2. episode_viewer.build — viewer.html.

Each step is independent: a trial without a bag still gets its viewer, a
trial without an episode still gets its full-rate joints. Output goes to
postprocess.log in the trial folder (the window redirects it there).

2026-09-30: "ready" is evidence, not the existence of files — the bag is
ready when the GUI's BagLink recorded the rosbag2 process's exit
(bag_status.json) and bag_check.verify_bag passes (databases present, not
LFS stubs, readable, counts match the metadata); the episode when its
recording_manifest.json says complete / incomplete (v3; a v2 episode: the
files stopped growing). The verdicts, and every step's error, go to
postprocess_status.json and the exit code is non-zero on any error, so the
GUI never reports a failed trial as verified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _stable(paths, settle_s: float = 1.0) -> bool:
    try:
        before = [p.stat().st_size for p in paths]
        time.sleep(settle_s)
        return before == [p.stat().st_size for p in paths]
    except OSError:
        return False


def _episode_state(trial: Path):
    """None = no episode expected; else "recording" / "complete" / … from the
    manifest (v3), or "written" / "writing" for a v2 episode."""
    man = trial / "recording_manifest.json"
    if man.exists():
        try:
            return json.loads(man.read_text()).get("state")
        except ValueError:
            return "unreadable manifest"
    if not (trial / "images").is_dir():
        return None
    ep_files = [trial / "steps.npz", trial / "meta.json"]
    return "written" if all(p.exists() for p in ep_files) and _stable(ep_files) else "writing"


def wait_ready(trial: Path, timeout_s: float = 90.0) -> dict:
    """Wait until the bag recorder has exited and the episode is saved (when
    expected); then say what is true — never "ready" by timeout."""
    t_end = time.time() + timeout_s
    want_bag = any(p.is_dir() and p.name.startswith("bag") for p in trial.iterdir())
    while True:
        bag_done = (not want_bag) or (trial / "bag_status.json").exists() or (
            # a trial recorded before bag_status.json existed: the metadata
            not (trial / "bag_record.log").exists() or any(
                (p / "metadata.yaml").exists() for p in trial.iterdir() if p.is_dir() and p.name.startswith("bag")))
        ep = _episode_state(trial)
        ep_done = ep in (None, "complete", "incomplete", "written", "error")
        if bag_done and ep_done:
            return {"bag": want_bag, "episode": ep, "timeout": False}
        if time.time() > t_end:
            return {"bag": want_bag, "bag_exit_recorded": bag_done, "episode": ep, "timeout": True}
        time.sleep(1.0)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    trial = Path(argv[0]).expanduser()
    t0 = time.time()
    state = wait_ready(trial)
    print(json.dumps({"postprocess": trial.name, "ready": state, "after_s": round(time.time() - t0, 1)}), flush=True)
    errors = []
    status = {"trial": str(trial), "ready": state}
    if state.get("timeout"):
        errors.append(f"not ready after 90 s: {state}")
    if state.get("episode") in ("incomplete", "error", "recording", "unreadable manifest"):
        errors.append(f"VLA episode {state.get('episode')}")
    # the bag: verified, not assumed
    from .bag_check import trial_bags, verify_bag
    status["bags"] = []
    for b in trial_bags(trial):
        v = verify_bag(b)
        status["bags"].append(v)
        print(json.dumps({"bag_check": v["bag"], "ok": v["ok"], "issues": v["issues"],
                          "messages": v["message_count"]}), flush=True)
        if not v["ok"]:
            errors.append(f"{b.name}: " + "; ".join(v["issues"]))
    try:
        from .joints_full import process
        r = process(trial)
        status["joints_full"] = r
        print(json.dumps(r), flush=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"joints_full failed: {type(exc).__name__}: {exc}")
        print(json.dumps({"joints_full": f"failed: {type(exc).__name__}: {exc}"}), flush=True)
    fmt = None
    try:
        fmt = json.loads((trial / "meta.json").read_text()).get("format") if (trial / "meta.json").exists() else None
    except Exception:
        pass
    from .recording_contract import RAW_FORMATS_WITH_STEP_LABELS
    if (trial / "steps.npz").exists() and fmt not in RAW_FORMATS_WITH_STEP_LABELS:
        print(json.dumps({"viewer": f"skipped ({fmt or 'v1'} episode: no per-step stamps)"}), flush=True)
    elif (trial / "steps.npz").exists():
        try:
            from .episode_viewer import build
            print(json.dumps({"viewer": str(build(trial))}), flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"viewer failed: {type(exc).__name__}: {exc}")
            print(json.dumps({"viewer": f"failed: {type(exc).__name__}: {exc}"}), flush=True)
    status["errors"] = errors
    status["ok"] = not errors
    write_status(trial, "vla", status)
    return 0 if not errors else 1


def write_status(trial: Path, part: str, status: dict) -> None:
    """Merge one part's verdict into <trial>/postprocess_status.json (atomic)."""
    p = trial / "postprocess_status.json"
    try:
        cur = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        cur = {}
    cur[part] = status
    cur["ok"] = all(v.get("ok", False) for k, v in cur.items() if isinstance(v, dict))
    cur["updated_unix"] = time.time()
    tmp = p.with_name(".postprocess_status.json.tmp")
    tmp.write_text(json.dumps(cur, indent=2, default=str))
    tmp.replace(p)


if __name__ == "__main__":
    sys.exit(main())
