"""Everything that runs after a saved trial (2026-09-28), detached from the GUI.

    python3 -m morty_gui.trial_postprocess <trial dir>

The MMHRI trial window starts it after ⏹ (output: postprocess.log in the
trial folder). In order:

1. vla_experiments.postprocess — waits for the bag to close and the VLA
   episode to be written, then the full-rate joints (joints_full.npz/.csv,
   joint_acc / joint_current in steps.npz) and viewer.html;
2. morty_gui.trial_times.finalize — the stage times rebuilt from exact event
   times (the messages' own stamps, else the VLA recorder, else the bag),
   written into timeline.txt/.json, events.json, trial.json and the trial's
   CSV rows.

Each step is independent; a failure is logged and the next one still runs.
2026-09-30: and it is REPORTED — every step's verdict goes to
postprocess_status.json in the trial folder and the exit code is non-zero
when anything failed; the trial window shows it instead of "saved".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    trial = Path(argv[0]).expanduser()
    from vla_experiments.postprocess import write_status
    rc = 0
    try:
        from vla_experiments.postprocess import main as vla_post
        rc = vla_post([str(trial)]) or 0
    except Exception as exc:  # noqa: BLE001
        rc = 1
        write_status(trial, "vla", {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
        print(json.dumps({"vla_postprocess": f"failed: {type(exc).__name__}: {exc}"}), flush=True)
    try:
        from .trial_times import finalize
        r = finalize(trial)
        write_status(trial, "trial_times", {"ok": True, "result": r})
        print(json.dumps(r), flush=True)
    except Exception as exc:  # noqa: BLE001
        rc = 1
        write_status(trial, "trial_times", {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
        print(json.dumps({"trial_times": f"failed: {type(exc).__name__}: {exc}"}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
