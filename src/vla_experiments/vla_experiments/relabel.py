"""Rebuild an episode's instructions and step labels from its saved events
(2026-09-28, after the data review of trials 078-085).

    python3 -m vla_experiments.relabel [--apply] <trial or episode dir> [...]
    (or `vla_relabel ...` after a colcon build of vla_experiments)

Every episode keeps the raw events it was recorded with (meta.json "events":
transcripts, intents, executor phases, cycle_done, ...), and its trial
timeline keeps the pipeline's selection records. When the labelling rules
change, the labels can therefore be rebuilt with the CURRENT rules instead of
re-recording:

* one instruction per motion intent (the recorder's MOTION_INTENTS, incl. the
  instant path's compounds), worded by canonical_instruction (positions and
  ordinals included), with the transcript that caused it;
* a pick chosen by pointing gets its own instruction at the selection;
* a pick replaced by another pick of the same object before the arm moved is
  "superseded" and its steps stay idle;
* instruction_idx / segment / phase per step, exactly as the recorder labels
  them at stop (EpisodeRecorder._label_steps);
* the episode task = the last pick before the arm's first motion.

Without --apply it only prints what would change. With --apply, meta.json and
steps.npz are rewritten; the first rewrite keeps the originals as
meta.json.orig_<date> / steps.npz.orig_<date>. Images and every recorded
value (joints, poses, stamps) are never touched — only the labels.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np

from .episode_recorder import MOTION_INTENTS, episode_task_instruction
from .raw_dataset import SEGMENT_IDLE, SEGMENT_RESET, SEGMENT_TASK, canonical_instruction


def _recorder_class():
    from . import episode_recorder as er
    return [c for c in vars(er).values() if isinstance(c, type) and "_label_steps" in vars(c)][0]


def _covered_by_spoken_pick(instructions: list, events: dict, stamp: float) -> bool:
    """A selection is the pick already asked for when the instruction active
    at that moment is a pick and no cycle ended in between — whatever the
    names (the pipeline says 'vitamin_tube' for a spoken 'vitamin c';
    trials 068 and 071 got a second, bogus pick from a name comparison)."""
    before = [i for i in instructions if i["stamp"] <= stamp]
    if not before or before[-1]["action_intent"] not in ("pick", "pick_and_place"):
        return False
    t_pick = before[-1]["stamp"]
    return not any(t_pick < c["stamp"] < stamp for c in events.get("cycle_done", []))


def rebuild(ep_dir: Path) -> tuple[dict, dict, list, list]:
    """(new_meta, new_steps, old_instruction_texts, new_instruction_texts)."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text())
    with np.load(ep_dir / "steps.npz") as z:
        steps = {k: z[k] for k in z.files}
    ev = meta.get("events") or {}
    t0 = float(meta.get("started_ros") or meta.get("started_unix") or 0.0)
    motion = set(MOTION_INTENTS)
    trs = sorted(ev.get("transcript", []), key=lambda e: e["stamp"])
    ins = []
    for e in sorted(ev.get("intent", []), key=lambda e: e["stamp"]):
        act = str(e.get("action_intent") or "")
        if act not in motion:
            continue
        cand = [x for x in trs if x["stamp"] <= e["stamp"] + 1e-6 and e["stamp"] - x["stamp"] < 30.0]
        tr = cand[-1] if cand else None
        ins.append({"idx": 0, "t": e["stamp"] - t0, "stamp": e["stamp"], "action_intent": act,
                    "object": e.get("object_intent"), "attributes": e.get("object_attributes"),
                    "transcript": (tr or {}).get("text"), "transcript_stamp": (tr or {}).get("stamp"),
                    "instruction": canonical_instruction(act, e.get("object_intent"), e.get("object_attributes"),
                                                         fallback=(tr or {}).get("text", ""))})
    # Pointed picks: the recorder's own selection events (since 2026-09-28),
    # else the trial timeline's selection records.
    sels = list(ev.get("selection", []))
    if not sels and (ep_dir / "timeline.json").exists():
        try:
            tl = json.loads((ep_dir / "timeline.json").read_text())
        except Exception:
            tl = {}
        end = t0 + float(meta.get("duration_s") or 0.0)
        for sc in tl.get("selection_cycles", []):
            st = float(sc.get("select") or sc.get("published") or 0.0)
            if t0 - 1.0 <= st <= end + 1.0:
                sels.append({"t": st - t0, "stamp": st, "label": sc.get("label"),
                             "source": sc.get("source"), "cycle_id": sc.get("cycle_id")})
        if sels:
            ev["selection"] = sels
    for sc in sorted(sels, key=lambda e: e["stamp"]):
        label = str(sc.get("label") or "")
        if not label:
            continue
        if _covered_by_spoken_pick(ins, ev, sc["stamp"]):
            continue
        ins.append({"idx": 0, "t": sc["stamp"] - t0, "stamp": sc["stamp"], "action_intent": "pick",
                    "object": label, "attributes": {"source": sc.get("source") or "selection"},
                    "transcript": None, "transcript_stamp": None,
                    "instruction": canonical_instruction("pick", label, {"source": sc.get("source")})})
    ins.sort(key=lambda i: i["stamp"])
    for k, i in enumerate(ins):
        i["idx"] = k
    fake = types.SimpleNamespace(events=ev, instructions=ins)
    _recorder_class()._label_steps(fake, steps)
    old = [i.get("instruction", "") for i in meta.get("instructions", [])]
    meta["instructions"] = ins
    meta["events"] = ev
    task = episode_task_instruction(ins, ev)
    if task:
        meta["task"], meta["task_from"] = task, "intent"
    meta["phase_names"] = getattr(fake, "_phase_names", meta.get("phase_names"))
    q = meta.setdefault("quality", {})
    q["task_steps"] = int((steps["segment"] == SEGMENT_TASK).sum())
    q["idle_steps"] = int((steps["segment"] == SEGMENT_IDLE).sum())
    q["reset_steps"] = int((steps["segment"] == SEGMENT_RESET).sum())
    q["n_instructions"] = len(ins)
    new = [i["instruction"] + (" [superseded]" if i.get("superseded") else "") for i in ins]
    return meta, steps, old, new


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    apply = "--apply" in argv
    dirs = [a for a in argv if a != "--apply"]
    if not dirs:
        print(__doc__)
        return 2
    tag = ".orig_" + _dt.date.today().strftime("%Y%m%d")
    for a in dirs:
        d = Path(a).expanduser()
        if not ((d / "meta.json").exists() and (d / "steps.npz").exists()):
            print(f"skip {d}: no meta.json / steps.npz")
            continue
        fmt = json.loads((d / "meta.json").read_text()).get("format")
        if fmt != "morty_vla_raw_v2":
            print(f"{d.name}: skipped ({fmt or 'v1'} episode — recorded before per-step labels existed)")
            continue
        meta, steps, old, new = rebuild(d)
        changed = old != [x.replace(" [superseded]", "") for x in new] or any("[superseded]" in x for x in new)
        print(f"{d.name}: {'CHANGED' if changed else 'same'}  task={meta.get('task')!r}")
        if changed:
            print(f"    before: {old}")
            print(f"    after:  {new}")
        if apply:
            for name in ("meta.json", "steps.npz"):
                if not (d / (name + tag)).exists():
                    shutil.copy2(d / name, d / (name + tag))
            meta["relabelled"] = _dt.datetime.now().isoformat(timespec="seconds")
            np.savez_compressed(d / "steps.npz", **steps)
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
    if not apply:
        print("(dry run — nothing written; add --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
