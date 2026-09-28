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


def wait_ready(trial: Path, timeout_s: float = 90.0) -> dict:
    """Wait until the bag is closed and the episode written (when expected)."""
    t_end = time.time() + timeout_s
    want_bag = any(p.is_dir() and p.name.startswith("bag") for p in trial.iterdir())
    want_ep = (trial / "images").is_dir()
    while True:
        bag_ok = (not want_bag) or any((p / "metadata.yaml").exists() for p in trial.iterdir()
                                       if p.is_dir() and p.name.startswith("bag"))
        ep_files = [trial / "steps.npz", trial / "meta.json"]
        ep_ok = (not want_ep) or (all(p.exists() for p in ep_files) and _stable(ep_files))
        if bag_ok and ep_ok:
            return {"bag": want_bag, "episode": want_ep, "waited": False}
        if time.time() > t_end:
            return {"bag": want_bag and bag_ok, "episode": want_ep and ep_ok, "timeout": True}
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
    try:
        from .joints_full import process
        print(json.dumps(process(trial)), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"joints_full": f"failed: {type(exc).__name__}: {exc}"}), flush=True)
    fmt = None
    try:
        fmt = json.loads((trial / "meta.json").read_text()).get("format") if (trial / "meta.json").exists() else None
    except Exception:
        pass
    if (trial / "steps.npz").exists() and fmt != "morty_vla_raw_v2":
        print(json.dumps({"viewer": f"skipped ({fmt or 'v1'} episode: no per-step stamps)"}), flush=True)
    elif (trial / "steps.npz").exists():
        try:
            from .episode_viewer import build
            print(json.dumps({"viewer": str(build(trial))}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"viewer": f"failed: {type(exc).__name__}: {exc}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
