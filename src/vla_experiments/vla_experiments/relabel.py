"""Rebuild an episode's labels with the CURRENT shared rules (labels.py) —
never in place (2026-09-30).

    python3 -m vla_experiments.relabel <trial or episode dir> [...]            # dry run
    python3 -m vla_experiments.relabel --out <new dir> <trial dir> [...]      # write copies
    (or `vla_relabel ...` after a colcon build of vla_experiments)

What it uses: the events the episode was recorded with (meta.json "events")
and, for a legacy episode, its own intent-arrival instructions. A v3 episode
(the router's command lifecycle recorded) gets the lifecycle labels; a
legacy one keeps the legacy rule and gets its findings — relabelling cannot
invent execution evidence that was never recorded (for that, see
`vla_validate`, which reconstructs it from a trial's router log where the
log exists and reports what it cannot).

Without --out nothing is written. With --out <dir>, each episode's
meta.json + steps.npz are written to <dir>/<episode id>/ (images are
linked, not copied); the original folder is never modified. The new meta
records "relabelled_from" and the rules used.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

from .episode_recorder import MOTION_INTENTS, episode_task_instruction
from .labels import label_episode
from .raw_dataset import canonical_instruction
from .recording_contract import RAW_FORMATS_WITH_STEP_LABELS, SEGMENT_NAMES


def _intent_instructions(meta: dict) -> list:
    """The legacy instructions: one per motion intent, as the v2 recorder
    made them (plus the pointed picks it added at the selection)."""
    ev = meta.get("events") or {}
    t0 = float(meta.get("started_ros") or 0.0)
    trs = sorted(ev.get("transcript", []), key=lambda e: e["stamp"])
    ins = []
    for e in sorted(ev.get("intent", []), key=lambda e: e["stamp"]):
        act = str(e.get("action_intent") or "")
        if act not in MOTION_INTENTS:
            continue
        cand = [x for x in trs if x["stamp"] <= e["stamp"] + 1e-6 and e["stamp"] - x["stamp"] < 30.0]
        tr = cand[-1] if cand else None
        ins.append({"idx": 0, "t": e["stamp"] - t0, "stamp": e["stamp"], "action_intent": act,
                    "object": e.get("object_intent"), "attributes": e.get("object_attributes"),
                    "transcript": (tr or {}).get("text"), "transcript_stamp": (tr or {}).get("stamp"),
                    "instruction": canonical_instruction(act, e.get("object_intent"), e.get("object_attributes"),
                                                         fallback=(tr or {}).get("text", ""))})
    for sc in sorted(ev.get("selection", []), key=lambda e: e["stamp"]):
        label = str(sc.get("label") or "")
        before = [i for i in ins if i["stamp"] <= sc["stamp"]]
        covered = before and before[-1]["action_intent"] in ("pick", "pick_and_place") and not any(
            before[-1]["stamp"] < c["stamp"] < sc["stamp"] for c in ev.get("cycle_done", []))
        if label and not covered:
            ins.append({"idx": 0, "t": sc["stamp"] - t0, "stamp": sc["stamp"], "action_intent": "pick",
                        "object": label, "attributes": {"source": sc.get("source") or "selection"},
                        "transcript": None, "transcript_stamp": None,
                        "instruction": canonical_instruction("pick", label, {"source": sc.get("source")})})
    ins.sort(key=lambda i: i["stamp"])
    for k, i in enumerate(ins):
        i["idx"] = k
    return ins


def rebuild(ep_dir: Path, events: dict | None = None) -> tuple:
    """(new_meta, new_steps, old_texts, new_texts) — the episode relabelled
    by labels.label_episode from its events (or ``events`` given)."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text())
    with np.load(ep_dir / "steps.npz") as z:
        steps = {k: z[k] for k in z.files}
    ev = events if events is not None else (meta.get("events") or {})
    lab = label_episode(ev, steps, legacy_instructions=_intent_instructions(meta))
    steps["instruction_idx"] = lab["instruction_idx"]
    steps["segment"] = lab["segment"]
    old = [i.get("instruction", "") for i in meta.get("instructions", [])]
    meta = copy.deepcopy(meta)
    meta["instructions"] = lab["instructions"]
    meta["label_source"] = lab["label_source"]
    meta["commands"] = lab.get("commands", [])
    task = episode_task_instruction(lab["instructions"], ev) if lab["instructions"] else ""
    if task:
        meta["task"], meta["task_from"] = task, "intent"
    seg = steps["segment"].astype(int)
    q = meta.setdefault("quality", {})
    q["segments"] = {name: int((seg == code).sum()) for code, name in SEGMENT_NAMES.items()}
    q["task_steps"] = q["segments"]["task"]
    sem = {"label_source": lab["label_source"], "flags": lab["flags"],
           "verified": lab["label_source"] == "command_lifecycle" and not lab["flags"]}
    sem["review_required"] = not sem["verified"]
    q["semantic"] = sem
    new = [i["instruction"] + (" [superseded]" if i.get("superseded") else "") for i in lab["instructions"]]
    return meta, steps, old, new


def write_copy(src: Path, out_root: Path, meta: dict, steps: dict, how: str) -> Path:
    """<out_root>/<episode id>/: new meta + steps, images linked; src untouched."""
    dst = Path(out_root) / Path(src).name
    dst.mkdir(parents=True, exist_ok=False)
    meta = dict(meta)
    meta["relabelled_from"] = {"path": str(src), "when": _dt.datetime.now().isoformat(timespec="seconds"),
                               "how": how}
    np.savez_compressed(dst / "steps.npz", **steps)
    (dst / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if (Path(src) / "images").is_dir():
        os.symlink(Path(src).resolve() / "images", dst / "images")
    return dst


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    out = None
    if "--apply" in argv:
        print("--apply (in place) was removed on 2026-09-30: originals are never rewritten. "
              "Use --out <dir> to write relabelled copies.")
        return 2
    if "--out" in argv:
        k = argv.index("--out")
        out = Path(argv[k + 1]).expanduser()
        del argv[k:k + 2]
    if not argv:
        print(__doc__)
        return 2
    for a in argv:
        d = Path(a).expanduser()
        if not ((d / "meta.json").exists() and (d / "steps.npz").exists()):
            print(f"skip {d}: no meta.json / steps.npz")
            continue
        fmt = json.loads((d / "meta.json").read_text()).get("format")
        if fmt not in RAW_FORMATS_WITH_STEP_LABELS:
            print(f"{d.name}: skipped ({fmt or 'v1'} episode — recorded before per-step labels existed)")
            continue
        meta, steps, old, new = rebuild(d)
        changed = old != [x.replace(" [superseded]", "") for x in new]
        print(f"{d.name}: {'CHANGED' if changed else 'same'}  labels={meta['label_source']}  task={meta.get('task')!r}")
        if changed:
            print(f"    before: {old}")
            print(f"    after:  {new}")
        for f in meta["quality"]["semantic"]["flags"]:
            print(f"    ! {f}")
        if out is not None:
            print(f"    -> {write_copy(d, out, meta, steps, 'vla_relabel (labels.label_episode)')}")
    if out is None:
        print("(dry run — nothing written; add --out <dir> for relabelled copies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
