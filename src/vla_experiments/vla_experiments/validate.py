"""Read-only validation of recorded trials, and repair proposals written to a
SEPARATE folder (2026-09-30, recording contract v1).

    python3 -m vla_experiments.validate <experiments root | day dir | trial dir> [...]
            [--json report.json] [--repair <out dir>]
    (or `vla_validate ...` after a colcon build of vla_experiments)

Nothing under the given folders is ever modified. Per trial:

1. episode integrity — meta.json / steps.npz readable, arrays of one length,
   finite joints, images on disk for every step, first / last decodable, the
   recording manifest's state (v3). A trial without an episode is reported
   as dialogue-only (not a defect: the recorder keeps episodes only when
   the arm moved).
2. bag — bag_check.verify_bag: every database the metadata names present,
   not a Git LFS pointer, readable, counts matching the metadata, the arm's
   /joint_states recorded when an episode exists, the recorder's exit.
3. labels —
   * v3 (command lifecycle recorded): the stored labels against a fresh
     labels.label_episode run, and the recorder's review flags;
   * legacy: the findings of labels.legacy_findings, and a RECONSTRUCTION
     of the command lifecycle from authoritative evidence the trial kept —
     the router's own log lines (logs/voice_gui.log: queued, dispatched,
     goal sent, completed, refused, deferred, resume published, cycle ids,
     cycle_done) and the executor statuses / phase marks in meta.json —
     labelled with the same lifecycle rules. Every instruction is then
     classified:
       confident    executed, with start and end evidence, no flag
       review       reconstructed but a flag or a missing edge remains
       unrecoverable no evidence for it (no router log, nothing executed)
   Reconstruction never invents: an episode without a router log is
   unrecoverable, and a repaired copy is still marked review_required
   (its provenance says "reconstructed").

--repair <dir> writes, for each legacy episode with a reconstruction, a
relabelled copy to <dir>/<trial>/ (meta.json + steps.npz; images linked)
with the reconstructed events and their provenance. Originals stay as they
are; nothing is bulk-relabelled in place.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .bag_check import trial_bags, verify_bag
from .labels import label_episode, legacy_findings
from .recording_contract import COMMAND_EVENT_SCHEMA, RAW_FORMAT, RAW_FORMATS_WITH_STEP_LABELS, SEGMENT_TASK

ROUTER_LINE = re.compile(r"\[intent_router_node\]: (.*)$")
STAMP = re.compile(r"\[(\d{10}\.\d+)\]")


# ------------------------------------------------------------ integrity ----
def check_episode(ep_dir: Path) -> dict:
    """Can the episode be read, and is every step's evidence on disk?"""
    out = {"episode": False, "issues": []}
    if not (ep_dir / "meta.json").exists() or not (ep_dir / "steps.npz").exists():
        return out
    out["episode"] = True
    try:
        meta = json.loads((ep_dir / "meta.json").read_text())
        with np.load(ep_dir / "steps.npz") as z:
            steps = {k: z[k] for k in z.files}
    except Exception as exc:  # noqa: BLE001
        out["issues"].append(f"unreadable: {type(exc).__name__}: {exc}")
        return out
    out["format"] = meta.get("format")
    n = len(steps.get("t", []))
    out["num_steps"] = n
    lengths = {k: len(v) for k, v in steps.items()}
    if len(set(lengths.values())) > 1:
        out["issues"].append(f"step arrays of different lengths: {lengths}")
    if "joint_pos" in steps and not np.isfinite(steps["joint_pos"]).all():
        out["issues"].append("non-finite joint positions")
    params = ((meta.get("provenance") or {}).get("parameters") or {})
    for cam in ("exterior", "wrist"):
        files = sorted((ep_dir / "images" / cam).glob("*.jpg"))
        required = params.get(f"require_{cam}", True)
        if required and len(files) != n:
            out["issues"].append(f"{cam}: {len(files)} images for {n} steps")
        try:
            import cv2
            for f in (files[:1] + files[-1:]) if files else []:
                if cv2.imread(str(f)) is None:
                    out["issues"].append(f"{cam}: {f.name} does not decode")
        except ImportError:
            pass
    man = ep_dir / "recording_manifest.json"
    if man.exists():
        try:
            m = json.loads(man.read_text())
            out["manifest_state"] = m.get("state")
            if m.get("state") != "complete":
                out["issues"].append(f"recording manifest says {m.get('state')!r}")
        except ValueError:
            out["issues"].append("recording manifest unreadable")
    elif meta.get("format") == RAW_FORMAT:
        out["issues"].append("v3 episode without a recording manifest (interrupted save?)")
    return out


# --------------------------------------------------- legacy reconstruction ----
def _router_lines(trial: Path, t0: float, t1: float) -> List[tuple]:
    p = trial / "logs" / "voice_gui.log"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        m, s = ROUTER_LINE.search(line), STAMP.search(line)
        if m and s and t0 - 5.0 <= float(s.group(1)) <= t1 + 5.0:
            out.append((float(s.group(1)), m.group(1)))
    return out


def reconstruct_commands(trial: Path, meta: dict) -> Optional[List[dict]]:
    """Command lifecycle events rebuilt from the router's log lines, in the
    contract's format with source "router_log". None = no router log."""
    t0 = float(meta.get("started_ros") or 0.0)
    t1 = t0 + float(meta.get("duration_s") or 0.0)
    lines = _router_lines(trial, t0, t1)
    if not lines:
        return None
    evs: List[dict] = []
    seq = [0]
    cmds: Dict[str, dict] = {}
    queued: List[dict] = []                     # FIFO of commands waiting to be dispatched
    cur: Optional[dict] = None                  # the dispatched command
    pick: Optional[dict] = None                 # the command whose cycle runs
    deferred: Optional[dict] = None

    def new(action: str, stamp: float) -> dict:
        c = {"command_id": f"log-{len(cmds)}", "action": action, "attributes": {}, "route": "", "cycle_id": None,
             "_first": stamp}
        cmds[c["command_id"]] = c
        emit(c, "received", stamp)
        return c

    def emit(c: dict, state: str, stamp: float, reason: str = "", **kw) -> None:
        c.update({k: v for k, v in kw.items() if k in ("route", "cycle_id", "target", "action")})
        seq[0] += 1
        evs.append({"schema": COMMAND_EVENT_SCHEMA, "event_id": f"log-{seq[0]}", "seq": seq[0],
                    "source": "router_log", "command_id": c["command_id"], "state": state, "stamp": stamp,
                    "_cmd": c, "route": c.get("route", ""), "cycle_id": c.get("cycle_id"),
                    "target": c.get("target"), "reason": reason})

    for i, (t, msg) in enumerate(lines):
        nxt = lines[i + 1][1] if i + 1 < len(lines) else ""
        if (m := re.match(r"Queued intent '(\w+)'", msg)):
            c = new(m.group(1), t)
            emit(c, "queued", t)
            queued.append(c)
        elif (m := re.match(r"dispatching: action='(\w+)'", msg)):
            act = m.group(1)
            cur = next((c for c in queued if c["action"] in (act, "pick_and_place" if act == "pick" else act)), None)
            if cur is not None:
                queued.remove(cur)
            else:
                cur = new(act, t)
            if act in ("pick", "pick_and_place") and "Refusing" not in nxt:
                pick = cur
                emit(cur, "accepted", t, route="pick")
        elif msg.startswith("Refusing") and cur is not None:
            emit(cur, "rejected", t, msg)
            if pick is cur:
                pick = None
        elif (m := re.match(r"Sending MoveToNamedPose -> '(\w+)'", msg)) and cur is not None:
            if cur.get("route") != "named_pose":
                emit(cur, "accepted", t, f"goal {m.group(1)!r} sent", route="named_pose",
                     target={"pose": m.group(1)})
        elif msg.startswith("MoveToNamedPose completed") and cur is not None and cur.get("route") == "named_pose":
            if not nxt.startswith("home reached"):       # a two-leg drop continues
                emit(cur, "completed", t)
        elif msg.startswith("MoveToNamedPose failed") and cur is not None:
            emit(cur, "failed", t, msg)
        elif msg.startswith("Sending OperateGripper") and cur is not None:
            emit(cur, "accepted", t, route="gripper")
        elif (m := re.match(r"Gripper done: success=(\w+)", msg)) and cur is not None:
            emit(cur, "completed" if m.group(1) == "True" else "failed", t)
        elif (m := re.match(r"Dedup: dropping repeat intent '(\w+)'", msg)):
            emit(new(m.group(1), t), "rejected", t, "duplicate")
        elif (m := re.match(r"Follow-up '(\w+)' noted", msg)):
            deferred = new(m.group(1), t)
            emit(deferred, "deferred", t, "kept for the release pause")
        elif (m := re.match(r"Resume '([^']+)' published \((.*)\)", msg)):
            tok, why = m.group(1), m.group(2)
            if "follow-up" in why and deferred is not None:
                c, deferred = deferred, None
            else:
                c = new(tok.split("+")[0], t)
                c["attributes"] = {"resume_token": tok}
            c["_why"] = why
            emit(c, "accepted", t, why, route="executor_answer", cycle_id=(pick or {}).get("cycle_id"))
        elif (m := re.match(r"Point proposal (\d+) \(([^)]*)\): YES", msg)):
            c = new("pick", t)
            pick = c
            emit(c, "accepted", t, "pointed + yes", route="pick", target={"label": m.group(2), "source": "pointing"})
        elif (m := re.match(r"cycle_id=(\d+) \(target=", msg)):
            if pick is not None and pick.get("cycle_id") is None:
                emit(pick, "cycle", t, cycle_id=int(m.group(1)))
        elif (m := re.match(r"cycle_done received: '(\w+)\|(\d+)", msg)):
            if pick is not None and pick.get("cycle_id") == int(m.group(2)):
                emit(pick, "completed" if m.group(1) == "done" else "failed", t)
                pick = None
        elif msg.startswith("Reset:") or msg.startswith("Stop received"):
            c = new("reset" if msg.startswith("Reset") else "stop", t)
            emit(c, "accepted", t, route="reset" if msg.startswith("Reset") else "stop")
    _attach_intents(cmds, meta)
    # pointed picks the router never saw (before 2026-09-28): the selection record
    for sc in (meta.get("events") or {}).get("selection", []):
        if sc.get("source") == "pointing" and not any(c.get("cycle_id") == sc.get("cycle_id") for c in cmds.values()):
            c = new("pick", float(sc["stamp"]))
            emit(c, "accepted", float(sc["stamp"]), "pointed (selection record)", route="pick",
                 target={"label": sc.get("label"), "source": "pointing"}, cycle_id=sc.get("cycle_id"))
            done = [e for e in (meta.get("events") or {}).get("cycle_done", [])
                    if str(e.get("cycle_id")) == str(sc.get("cycle_id"))]
            if done:
                emit(c, "completed" if done[0]["ok"] else "failed", float(done[0]["stamp"]))
    # the command's final (intent-attached) form on every event
    for e in evs:
        c = e.pop("_cmd")
        e.update({"action": c["action"], "attributes": c.get("attributes", {}), "utterance": c.get("utterance", ""),
                  "object": c.get("object")})
        if e.get("target") is None and c.get("object"):
            e["target"] = {"label": c["object"], "source": "voice"}
    return evs


TOKEN_ACTION = {"home": "go_home", "drop_left": "drop_left", "drop_right": "drop_right", "release": "open_gripper",
                "release_home": "release_home", "stay": "stay", "here": "confirm_drop", "close": "close_gripper"}
ANSWER_ACTIONS = ("confirm_drop", "drop_left", "drop_right", "go_home", "open_gripper", "release", "release_home",
                  "stay", "close_gripper", "carry_release", "go_home_open", "drop_right_release", "drop_left_release",
                  "drop_right_release_home", "drop_left_release_home")


def _attach_intents(cmds: Dict[str, dict], meta: dict) -> None:
    """Give each reconstructed command the recorded intent that caused it —
    the last intent of the same action family at most 3 s before the
    command's first log line — for its action, object, attributes and words.
    The router log names only the action; the episode recorded the rest."""
    ev = meta.get("events") or {}
    intents = sorted(ev.get("intent", []), key=lambda e: e["stamp"])
    trs = sorted(ev.get("transcript", []), key=lambda e: e["stamp"])
    used = set()
    for c in cmds.values():
        first = c.get("_first")
        if first is None:
            continue
        act = c["action"]
        fam = ANSWER_ACTIONS if (c.get("route") == "executor_answer" or act in TOKEN_ACTION) else (act,)
        if act in ("pick", "pick_and_place"):
            fam = ("pick", "pick_and_place")
        cand = [i for i, e in enumerate(intents) if i not in used and e.get("action_intent") in fam
                and first - 3.0 <= float(e["stamp"]) <= first + 0.2]
        # 2026-09-30 (trial 137): two answers 72 ms apart — the router's line
        # names the intent it acted on ("... keep holding): close_gripper",
        # "'drop_right' names a side"); that one wins over the family's last
        why = str(c.get("_why") or "")
        named = [i for i in cand if re.search(r"(?<![\w])'?" + re.escape(str(intents[i].get("action_intent"))) + r"\b", why)]
        if named:
            cand = named
        if cand:
            e = intents[cand[-1]]
            used.add(cand[-1])
            c["action"] = e.get("action_intent") or act
            c["object"] = e.get("object_intent")
            c["attributes"] = {**(c.get("attributes") or {}),
                               **{k: v for k, v in (e.get("object_attributes") or {}).items() if v not in (None, "")}}
            tr = [x for x in trs if x["stamp"] <= float(e["stamp"]) + 1e-6 and float(e["stamp"]) - x["stamp"] < 30.0]
            c["utterance"] = tr[-1]["text"] if tr else ""
            c["intent_stamp"] = float(e["stamp"])
        elif act in TOKEN_ACTION:
            c["action"] = TOKEN_ACTION[act]
            if act == "here":
                c["attributes"] = {**(c.get("attributes") or {}), "destination": "here"}


def reconstruct_task_states(meta: dict, commands: List[dict]) -> List[dict]:
    """Executor phases rebuilt from the statuses and phase marks the episode
    recorded (a legacy episode has no task_state stream)."""
    ev = meta.get("events") or {}
    out = []
    cyc = None
    marks = sorted(ev.get("executor_phase", []), key=lambda e: e["stamp"])
    phase_of = {"motion_start": "approaching", "wait_destination_start": "awaiting_destination",
                "home_done": "idle", "cycle_done": "idle"}
    for e in marks:
        if e["name"] in phase_of:
            out.append({"stamp": float(e["stamp"]), "phase": phase_of[e["name"]], "cycle_id": e.get("cycle_id"),
                        "seq": len(out), "boot": "reconstructed"})
    status_of = {"paused": "awaiting_destination", "resumed": "carrying", "redirect": "carrying",
                 "awaiting_release": "awaiting_release", "releasing": "releasing",
                 "awaiting_home_confirm": "awaiting_home_confirm", "stayed": "idle", "keeping": "returning_home"}
    for e in sorted(ev.get("interactive_status", []), key=lambda e: e["stamp"]):
        head = str(e["status"]).split(":")[0]
        if head in status_of:
            prev = [x for x in marks if x["stamp"] <= e["stamp"]]
            cyc = prev[-1].get("cycle_id") if prev else cyc
            out.append({"stamp": float(e["stamp"]), "phase": status_of[head], "cycle_id": cyc,
                        "seq": len(out), "boot": "reconstructed"})
    # an accepted answer that sends the arm home: the executor moves at once
    for c in commands:
        if c.get("state") == "accepted" and c.get("route") == "executor_answer" and \
                str((c.get("attributes") or {}).get("resume_token", "")).split("+")[0] in ("home", "release_home"):
            prev = [x for x in marks if x["stamp"] <= c["stamp"]]
            out.append({"stamp": float(c["stamp"]) + 0.05, "phase": "returning_home",
                        "cycle_id": prev[-1].get("cycle_id") if prev else cyc, "seq": len(out), "boot": "reconstructed"})
    for e in ev.get("cycle_done", []):
        out.append({"stamp": float(e["stamp"]), "phase": "idle", "cycle_id": e.get("cycle_id"),
                    "seq": len(out), "boot": "reconstructed"})
    out.sort(key=lambda s: s["stamp"])
    for k, s in enumerate(out):
        s["seq"] = k
        try:
            s["cycle_id"] = int(s["cycle_id"]) if s["cycle_id"] not in (None, "") else None
        except (TypeError, ValueError):
            s["cycle_id"] = None
    return out


def classify(lab: dict, steps: dict) -> List[dict]:
    """Per reconstructed instruction: confident / review."""
    idx = lab["instruction_idx"]; seg = lab["segment"]
    out = []
    for ins in lab["instructions"]:
        k = ins["idx"]
        n = int(((idx == k) & (seg == SEGMENT_TASK)).sum())
        mine = [f for f in lab["flags"] if f"'{ins['instruction']}'" in f]
        out.append({"instruction": ins["instruction"], "steps": n, "outcome": ins.get("outcome"),
                    "class": "confident" if n >= 2 and not mine and ins.get("outcome") == "completed" else "review",
                    "end_evidence": ins.get("end_evidence"),
                    "why": mine or ([] if ins.get("outcome") == "completed" else [f"outcome {ins.get('outcome')}"])})
    return out


# ------------------------------------------------------------------ trial ----
def validate_trial(trial: Path) -> dict:
    rep: Dict = {"trial": str(trial), "name": trial.name}
    ep = check_episode(trial)
    rep["episode"] = ep
    required = ["/joint_states"] if ep["episode"] else []
    rep["bags"] = [verify_bag(b, required_topics=required) for b in trial_bags(trial)] if trial.is_dir() else []
    if not ep["episode"]:
        rep["labels"] = {"status": "no VLA episode (dialogue-only or not kept) — not a defect"}
        return rep
    meta = json.loads((trial / "meta.json").read_text())
    with np.load(trial / "steps.npz") as z:
        steps = {k: z[k] for k in z.files}
    ev = meta.get("events") or {}
    if meta.get("format") not in RAW_FORMATS_WITH_STEP_LABELS:
        rep["labels"] = {"status": f"{meta.get('format') or 'v1'} episode: no per-step labels to validate"}
        return rep
    if ev.get("command"):
        lab = label_episode(ev, steps)
        same = np.array_equal(lab["instruction_idx"], steps["instruction_idx"]) and \
            np.array_equal(lab["segment"], steps["segment"])
        rep["labels"] = {"status": "command_lifecycle", "stored_equals_recomputed": bool(same),
                         "flags": lab["flags"], "review_required": bool(lab["flags"]) or not same}
        return rep
    findings = legacy_findings(ev, meta.get("instructions") or [], steps["instruction_idx"])
    stored_then = [f"'{i['instruction']}' has then={(i.get('attributes') or {}).get('then')} but its text drops it"
                   for i in meta.get("instructions") or []
                   if (i.get("attributes") or {}).get("then") and "hand it over" not in i["instruction"]
                   and "go back home" not in i["instruction"]]
    rep["labels"] = {"status": "legacy_intent_arrival", "findings": findings + stored_then}
    cmds = reconstruct_commands(trial, meta)
    if cmds is None:
        rep["labels"]["reconstruction"] = {"status": "unrecoverable: no router log in this trial"}
        return rep
    ts = reconstruct_task_states(meta, cmds)
    ev2 = dict(ev, command=cmds, task_state=ts)
    lab = label_episode(ev2, steps)
    rep["labels"]["reconstruction"] = {
        "status": "reconstructed from logs/voice_gui.log + recorded executor statuses",
        "instructions": classify(lab, steps), "flags": lab["flags"],
        "unexecuted_commands": [c for c in lab["commands"] if not c.get("accepted")]}
    rep["_repair"] = (ev2, lab)
    return rep


def write_repair(trial: Path, out_root: Path, ev2: dict, lab: dict) -> Path:
    from .relabel import rebuild, write_copy
    meta, steps, _old, _new = rebuild(trial, events=ev2)
    meta["label_source"] = "reconstructed_from_router_log"
    q = meta.setdefault("quality", {})
    q.setdefault("semantic", {})
    q["semantic"].update({"verified": False, "review_required": True,
                          "flags": ["labels reconstructed from the router log and executor statuses (review)"]
                          + lab["flags"]})
    return write_copy(trial, out_root, meta, steps, "vla_validate --repair (reconstructed command lifecycle)")


def _trials(root: Path) -> List[Path]:
    root = Path(root).expanduser()
    if (root / "trial.json").exists() or (root / "meta.json").exists() or any(root.glob("bag*/metadata.yaml")):
        return [root]
    return sorted(p for p in root.rglob("trial_*") if p.is_dir() and "vla_previous" not in p.parts
                  and not p.parent.name.startswith("trial_"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    out_json = repair = None
    if "--json" in argv:
        k = argv.index("--json"); out_json = Path(argv[k + 1]); del argv[k:k + 2]
    if "--repair" in argv:
        k = argv.index("--repair"); repair = Path(argv[k + 1]).expanduser(); del argv[k:k + 2]
    if not argv:
        print(__doc__)
        return 2
    reports = []
    for a in argv:
        for t in _trials(Path(a)):
            r = validate_trial(t)
            reports.append(r)
            ep, lb = r["episode"], r.get("labels", {})
            bag = "; ".join(f"{Path(b['bag']).name}: "
                            + ("ok" + (f" (warning: {'; '.join(b['warnings'])})" if b.get("warnings") else "")
                               if b["ok"] else " / ".join(b["issues"])) for b in r["bags"]) or "no bag"
            print(f"{t.name}: episode={'yes' if ep['episode'] else 'no'}"
                  + (f" ({'ok' if not ep['issues'] else '; '.join(ep['issues'])})" if ep["episode"] else "")
                  + f" | bag: {bag} | labels: {lb.get('status')}")
            for f in lb.get("findings", [])[1:]:
                print(f"    ! {f}")
            rec = lb.get("reconstruction")
            if rec:
                print(f"    reconstruction: {rec['status']}")
                for c in rec.get("instructions", []):
                    print(f"      [{c['class']:9}] {c['instruction']!r} ({c['steps']} steps)"
                          + (f" — {'; '.join(c['why'])}" if c["why"] else ""))
            if repair is not None and "_repair" in r:
                print(f"    -> repaired copy: {write_repair(t, repair, *r['_repair'])}")
    if out_json:
        out_json.write_text(json.dumps([{k: v for k, v in r.items() if k != "_repair"} for r in reports],
                                       indent=2, default=str))
        print(f"report: {out_json}")
    if repair is None:
        print("(read-only — nothing written; --repair <dir> writes relabelled copies there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
