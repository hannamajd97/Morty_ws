"""Move one recorded trial to another scenario — everywhere it is stored.

2026-09-29 (Majd moved 091, 110, 111 and 120 after recording). Since
2026-10-01 this is a shortcut of fix_trial (which corrects ANY form field,
also from the trial window's ✎ fix button): the scenario and, optionally,
the note. See fix_trial for what is rewritten and the safety rules.

    python3 -m morty_gui.relabel_trial 2026-09-29 120 S5 [--note "..."] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List

from . import excel_export as xl
from .fix_trial import EXPERIMENTS, fix_trial

NOTE_COL = "Failure Category / Notes"


def _trial_dir(date: str, tid: str) -> Path:
    day = EXPERIMENTS / ("experiments_" + _dt.date.fromisoformat(date).strftime("%d_%m_%Y"))
    hits = sorted(day.glob(f"trial_{tid}_*"))
    if len(hits) != 1:
        raise SystemExit(f"trial {tid} of {date}: {len(hits)} folders in {day}")
    return hits[0]


def relabel(date: str, tid: str, new_sid: str, note: str | None = None,
            dry_run: bool = False, xlsx: str = xl.DEFAULT_XLSX) -> List[str]:
    new = {"Scenario": new_sid}
    if note is not None:
        new[NOTE_COL] = note
    try:
        return fix_trial(_trial_dir(date, tid), new, dry_run=dry_run, xlsx=xlsx)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("trial", help="trial id, e.g. 120")
    ap.add_argument("scenario", help="new scenario id, e.g. S5")
    ap.add_argument("--note", default=None, help="Failure Category / Notes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    for line in relabel(a.date, a.trial.zfill(3), a.scenario.upper(), a.note, a.dry_run):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
