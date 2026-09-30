"""Per-step labels of an episode — ONE implementation for the recorder (at
stop), relabel, validate and the tests (2026-09-30, recording contract v1).

    label_episode(events, steps) -> dict

Two sources of truth, never mixed:

* ``command_lifecycle`` (raw format v3: meta events hold "command"): an
  instruction exists for every command the robot actually EXECUTED, and its
  steps are the ones inside that command's execution interval:
    - a pick: its grasp cycle's task_state from "approaching" (motion) to the
      first pause with a question / idle / stop — a failed pick ends where
      the recovery starts (that recovery is reset);
    - an answer to a pause (destination, release, home): the cycle's next
      motion phase after the answer was accepted, to the next pause / idle;
    - a named-pose move / gripper command: from "accepted" to "completed",
      the move trimmed to the arm's actual joint motion (acceptance is not
      motion);
    - a compound stays one instruction across its executed parts
      (release + requested return home, carry + hand-over + release).
  Queued, deferred, rejected and cancelled commands never label steps; they
  are listed in meta "commands" with their whole lifecycle.
  Other steps: hold (the robot waits holding / at a pause for the operator's
  next words — attached to the command whose pause it is), clarify (a
  pointing question is open, or a pick was refused for a which-one question
  — attached to the pick that resolves it), reset (automatic cleanup: the
  recovery after a failure, a reset / stop, an automatic close after a
  hand-over), assist (operator-annotated intervention), idle (the rest).
  Anything the evidence cannot place is FLAGGED for review, never invented.

* ``legacy_intent_arrival`` (raw v2 and older): the v2 recorder's rule,
  unchanged (instruction active from its intent's arrival), plus detectors
  for the known failure patterns (queued intents milliseconds apart, a
  compound's lost "then", an instruction said before a pick it did not
  cause). Legacy labels are always flagged as unverified.

The observation / action convention is in recording_contract.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .raw_dataset import canonical_instruction
from .recording_contract import (RAW_FORMAT, SEGMENT_ASSIST, SEGMENT_CLARIFY, SEGMENT_HOLD,
                                 SEGMENT_IDLE, SEGMENT_RESET, SEGMENT_TASK, TERMINAL_STATES)

MOTION_PHASES = ("approaching", "carrying", "releasing", "returning_home")
QUESTION_PAUSES = ("awaiting_destination", "awaiting_release", "awaiting_home_confirm")
END_PHASES = QUESTION_PAUSES + ("idle", "stopped", "no_grasp")
EXECUTING_ROUTES = ("pick", "executor_answer", "named_pose", "gripper")
JOINT_MOVING_RAD_S = 0.05       # arm speed that counts as motion (max over joints)
CLARIFY_WORDS = ("which", "leftmost", "rightmost", "i see two", "i see three", "point at")


# ---------------------------------------------------------------- commands ----
def commands_from_events(evs: List[dict]) -> List[dict]:
    """Command records from the lifecycle events: states in order (duplicates
    and out-of-order delivery removed by event_id / seq), the ACCEPTED form
    and the terminal verdict."""
    seen, by_id = set(), {}
    for e in sorted(evs, key=lambda e: (float(e.get("stamp") or 0.0), int(e.get("seq") or 0))):
        eid = e.get("event_id")
        if eid in seen:
            continue
        seen.add(eid)
        c = by_id.setdefault(e.get("command_id"), {"command_id": e.get("command_id"), "states": []})
        c["states"].append({"state": e.get("state"), "stamp": float(e.get("stamp") or 0.0),
                            "reason": e.get("reason", ""), "route": e.get("route", "")})
        for k in ("action", "attributes", "target", "route", "cycle_id", "parent_id", "question",
                  "question_id", "utterance", "utterance_id", "plan_id", "plan_index", "resume_token"):
            if e.get(k) not in (None, "", {}):
                c[k] = e.get(k)
        if e.get("state") == "received":
            c.setdefault("received", float(e.get("stamp") or 0.0))
        if e.get("state") == "accepted" and "accepted" not in c:
            c["accepted"] = float(e.get("stamp") or 0.0)
            c["accepted_action"] = e.get("action")
            c["accepted_attributes"] = e.get("attributes") or {}
        if e.get("state") in TERMINAL_STATES and "terminal" not in c:
            c["terminal"] = e.get("state")
            c["terminal_stamp"] = float(e.get("stamp") or 0.0)
            c["terminal_reason"] = e.get("reason", "")
    return sorted(by_id.values(), key=lambda c: c.get("received", c["states"][0]["stamp"]))


def _task_states(events: dict) -> List[dict]:
    out, seen = [], set()
    for e in sorted(events.get("task_state", []), key=lambda e: float(e.get("stamp") or 0.0)):
        key = (e.get("boot"), e.get("seq"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _cycle_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _motion_window(steps: dict, t_on: float, t_off: float):
    """(first, last) stamp inside [t_on, t_off] where the arm moves, or None."""
    st = steps["stamp_ros"]
    jp = steps.get("joint_pos")
    if jp is None or len(st) < 2:
        return None
    dt = np.diff(st)
    speed = np.zeros(len(st))
    ok = dt > 1e-6
    speed[1:][ok] = np.max(np.abs(np.diff(jp, axis=0)[ok]), axis=1) / dt[ok]
    sel = np.nonzero((st >= t_on) & (st <= t_off) & (speed > JOINT_MOVING_RAD_S))[0]
    if not sel.size:
        return None
    return float(st[max(sel[0] - 1, 0)]), float(st[sel[-1]])


def _interval(cmd: dict, ts: List[dict], steps: dict, events: dict, flags: List[str]):
    """(t_on, t_off, cut) of a command's execution, or None (no evidence).
    ``cut`` = where an automatic cleanup inside the interval starts."""
    route = cmd.get("route")
    name = f"command {cmd.get('accepted_action') or cmd.get('action')} ({cmd['command_id'][:8]})"
    if route == "pick":
        cyc = _cycle_int(cmd.get("cycle_id"))
        if cyc is None:
            flags.append(f"{name}: accepted but no grasp cycle started")
            return None
        own = [s for s in ts if _cycle_int(s.get("cycle_id")) == cyc]
        on = next((float(s["stamp"]) for s in own if s.get("phase") == "approaching"), None)
        if on is None:
            ms = [float(e["stamp"]) for e in events.get("executor_phase", [])
                  if e.get("name") == "motion_start" and _cycle_int(e.get("cycle_id")) == cyc]
            on = ms[0] if ms else None
        if on is None:
            flags.append(f"{name}: cycle {cyc} never started moving")
            return None
        failed = cmd.get("terminal") == "failed"
        ends = END_PHASES + (("returning_home",) if failed else ())
        end = next((s for s in own if float(s["stamp"]) > on and s.get("phase") in ends), None)
        if end is not None:
            return on, float(end["stamp"]), end.get("phase")
        off = cmd.get("terminal_stamp")
        if off is None:
            flags.append(f"{name}: no end of cycle {cyc} recorded — interval runs to the episode end")
            off = float(steps["stamp_ros"][-1]) + 1.0
            return on, off, None
        return on, off, cmd.get("terminal")
    if route == "executor_answer":
        a = cmd.get("accepted")
        cyc = _cycle_int(cmd.get("cycle_id"))
        later = [s for s in ts if float(s["stamp"]) >= a - 0.05
                 and (cyc is None or _cycle_int(s.get("cycle_id")) == cyc)]
        on = next((float(s["stamp"]) for s in later if s.get("phase") in MOTION_PHASES), None)
        if on is None:
            flags.append(f"{name}: answer accepted but the executor showed no motion after it")
            return None
        end = next((s for s in later if float(s["stamp"]) > on and s.get("phase") in END_PHASES), None)
        if end is None:
            flags.append(f"{name}: no pause / idle after the answer's motion — interval runs to the episode end")
            return on, float(steps["stamp_ros"][-1]) + 1.0, None
        return on, float(end["stamp"]), end.get("phase")
    if route in ("named_pose", "gripper"):
        a, z = cmd.get("accepted"), cmd.get("terminal_stamp")
        if z is None:
            flags.append(f"{name}: accepted but never completed")
            z = float(steps["stamp_ros"][-1]) + 1.0
        if route == "named_pose":
            # the server's result is the end: the next move may start right after it
            w = _motion_window(steps, a, z)
            if w is None:
                flags.append(f"{name}: goal accepted but the arm did not move")
                return None
            return w[0], w[1] + 1e-3, cmd.get("terminal")
        return a, z, cmd.get("terminal")
    return None


def _auto_close_cut(cmd: dict, t_on: float, t_off: float, events: dict) -> Optional[float]:
    """An automatic gripper close inside a command that did not ask for one
    (the executor closes after a hand-over on its way home): the task ends
    there and the rest is cleanup — "open the gripper" must not teach a close."""
    act = str(cmd.get("accepted_action") or cmd.get("action") or "")
    if act in ("pick", "pick_and_place", "close_gripper", "grasp"):
        return None
    for g in sorted(events.get("gripper_cmd", []), key=lambda e: e["stamp"]):
        if t_on < float(g["stamp"]) < t_off and str(g.get("cmd")) == "close":
            return float(g["stamp"])
    return None


def _executed_form(cmd: dict, flags: List[str]) -> None:
    """A compound whose later part the robot did NOT carry out as part of it
    is labelled with what it executed (trial 084: "open the gripper and go
    back home" — the executor opened, then ASKED "Should I go back home?";
    the return was a separate command). Never teach the requested action as
    done when the robot did something else."""
    act = str(cmd.get("accepted_action") or cmd.get("action") or "")
    attrs = dict(cmd.get("accepted_attributes") or cmd.get("attributes") or {})
    then = str(attrs.get("then") or "")
    end = cmd.get("end_evidence")
    wants_home = then == "release_home" or act == "release_home"
    wants_release = then in ("release", "release_home") or act in ("go_home_open", "carry_release")
    new_act, new_then = act, then
    if wants_home and end == "awaiting_home_confirm":
        new_act = "open_gripper" if act in ("release_home", "open_gripper", "release") else act
        new_then = "release" if then else ""
    if wants_release and end == "awaiting_release":
        new_act = {"go_home_open": "go_home", "carry_release": "confirm_drop"}.get(act, act)
        new_then = ""
    if (new_act, new_then) != (act, then):
        attrs["then"] = new_then or None
        was = canonical_instruction(act, cmd.get("object"), cmd.get("accepted_attributes") or cmd.get("attributes"))
        cmd["accepted_action"], cmd["accepted_attributes"] = new_act, attrs
        now = canonical_instruction(new_act, cmd.get("object"), attrs)
        cmd["requested_instruction"] = was
        flags.append(f"'{was}' was executed only as '{now}' (the executor stopped at {end}) — labelled as executed")


def _instruction_text(cmd: dict) -> tuple:
    act = str(cmd.get("accepted_action") or cmd.get("action") or "")
    attrs = dict(cmd.get("accepted_attributes") or cmd.get("attributes") or {})
    tgt = cmd.get("target") or {}
    obj = tgt.get("label") or cmd.get("object") or None
    for k in ("position", "colour", "ordinal", "ordinal_from"):
        if tgt.get(k) and not attrs.get(k):
            attrs[k] = tgt[k]
    text = canonical_instruction(act, obj, attrs, fallback=str(cmd.get("utterance") or ""))
    src = tgt.get("source") or ("voice" if cmd.get("utterance") else "")
    how = ("pointing (confirmed)" if src == "pointing" else
           "spoken qualifier" if any(tgt.get(k) for k in ("position", "colour", "ordinal")) or
           any(attrs.get(k) for k in ("position", "colour", "ordinal")) else
           "name only" if act in ("pick", "pick_and_place") else "")
    return text, how


def _label_lifecycle(events: dict, steps: dict) -> dict:
    st = steps["stamp_ros"]
    n = len(st)
    instr = np.full(n, -1.0)
    seg = np.full(n, float(SEGMENT_IDLE))
    flags: List[str] = []
    cmds = commands_from_events(events.get("command", []))
    ts = _task_states(events)

    # 1. the executed commands and their intervals
    executed = []
    for c in cmds:
        if "accepted" not in c or c.get("route") not in EXECUTING_ROUTES:
            continue
        iv = _interval(c, ts, steps, events, flags)
        if iv is None:
            continue
        on, off, reached = iv
        c["end_evidence"] = reached
        cut = _auto_close_cut(c, on, off, events)
        executed.append((on, off, cut, c))
    executed.sort(key=lambda x: x[0])
    for (a_on, a_off, _, a), (b_on, _, _, b) in zip(executed, executed[1:]):
        if b_on < a_off - 0.05:
            flags.append(f"overlapping executions: {a.get('accepted_action')} and {b.get('accepted_action')}"
                         f" at {b_on:.2f} — steps go to the later one")

    instructions = []
    for k, (on, off, cut, c) in enumerate(executed):
        _executed_form(c, flags)
        text, how = _instruction_text(c)
        instructions.append({
            "idx": k, "stamp": on, "end_stamp": off, "t": None,
            "command_id": c["command_id"], "parent_command_id": c.get("parent_id", ""),
            "action_intent": c.get("accepted_action") or c.get("action"),
            "object": (c.get("target") or {}).get("label") or c.get("object"),
            "attributes": c.get("accepted_attributes") or c.get("attributes") or {},
            "target": c.get("target"), "target_resolution": how,
            "route": c.get("route"), "cycle_id": c.get("cycle_id"),
            "question": c.get("question", ""), "question_id": c.get("question_id"),
            "transcript": c.get("utterance") or None, "utterance_id": c.get("utterance_id", ""),
            "plan_id": c.get("plan_id", ""), "received_stamp": c.get("received"),
            "accepted_stamp": c.get("accepted"),
            # the router's verdict where it gives one (named poses, gripper,
            # the pick's cycle_done); an answer to a pause is complete when
            # the executor reached its next pause / idle (end_evidence)
            "outcome": c.get("terminal") or ("completed" if c.get("end_evidence") in END_PHASES else "unknown"),
            "outcome_reason": c.get("terminal_reason", ""), "end_evidence": c.get("end_evidence"),
            "auto_cleanup_from": cut, "instruction": text,
            "requested_instruction": c.get("requested_instruction")})
        m = (st >= on) & (st < (cut if cut is not None else off))
        instr[m] = k
        seg[m] = SEGMENT_TASK
        if cut is not None:
            mc = (st >= cut) & (st < off)
            seg[mc] = SEGMENT_RESET
            instr[mc] = k
            flags.append(f"'{text}': automatic gripper close at {cut:.2f} labelled as cleanup, not as the command")

    # 2. holds: a question pause of the executor, outside any execution
    for i, s0 in enumerate(ts):
        if s0.get("phase") not in QUESTION_PAUSES:
            continue
        t0 = float(s0["stamp"])
        t1 = float(ts[i + 1]["stamp"]) if i + 1 < len(ts) else float(st[-1]) + 1.0
        m = (st >= t0) & (st < t1) & (seg == SEGMENT_IDLE)
        before = [ins for ins in instructions if ins["stamp"] <= t0]
        seg[m] = SEGMENT_HOLD
        if before:
            instr[m] = before[-1]["idx"]

    # 3. resets: a reset / stop command while the arm still moves after it (a
    #    reset goes home, a stop freezes) — never past the next executed
    #    command, never over a hold; a failed cycle's recovery
    starts = sorted(e[0] for e in executed)
    for c in cmds:
        if c.get("route") in ("reset", "stop") and "accepted" in c:
            t0 = c["accepted"]
            cap = min([s for s in starts if s > t0] + [t0 + 15.0])
            w = _motion_window(steps, t0, cap)
            t1 = min(w[1] + 1e-3, cap) if w else min(t0 + 0.5, cap)
            m = (st >= t0) & (st < t1) & (seg == SEGMENT_IDLE)
            seg[m] = SEGMENT_RESET
            instr[m] = -1
    for on, off, cut, c in executed:
        if c.get("terminal") == "failed":
            t1 = next((float(s["stamp"]) for s in ts if float(s["stamp"]) > off and s.get("phase") == "idle"),
                      off + 5.0)
            m = (st >= off) & (st < t1) & (seg != SEGMENT_TASK)
            seg[m] = SEGMENT_RESET
            instr[m] = -1

    # 4. clarification: a pointing question open, or a pick refused for "which one?"
    spans = []
    opened = {}
    for q in sorted(events.get("point_question", []), key=lambda e: e["stamp"]):
        if q.get("state") == "open":
            opened[q.get("id")] = float(q["stamp"])
        elif q.get("id") in opened:
            spans.append((opened.pop(q.get("id")), float(q["stamp"])))
    for c in cmds:
        if c.get("terminal") == "rejected" and any(w in str(c.get("terminal_reason", "")).lower() for w in CLARIFY_WORDS):
            nxt = [e for e in executed if e[0] > c["terminal_stamp"]]
            spans.append((c["terminal_stamp"], nxt[0][0] if nxt else c["terminal_stamp"] + 30.0))
    for t0, t1 in spans:
        m = (st >= t0) & (st < t1) & (seg == SEGMENT_IDLE)
        seg[m] = SEGMENT_CLARIFY
        nxt = [ins for ins in instructions if ins["stamp"] >= t1 - 0.05]
        if nxt:
            instr[m] = nxt[0]["idx"]

    # 5. operator-annotated assistance overrides everything
    assist = _annotation_spans(events, ("assist", "manual_assistance", "intervention"), float(st[-1]) if n else 0.0)
    for t0, t1 in assist:
        m = (st >= t0) & (st < t1)
        seg[m] = SEGMENT_ASSIST

    # semantic review flags
    for ins in instructions:
        if ins["target_resolution"] == "name only" and ins["route"] == "pick":
            rejected_which = any(c.get("terminal") == "rejected" and "which" in str(c.get("terminal_reason", "")).lower()
                                 for c in cmds if c.get("received", 0) < ins["stamp"])
            if rejected_which:
                flags.append(f"'{ins['instruction']}': the router asked which one, but the executed pick names "
                             "only the kind — target unresolved")
    for c in cmds:
        if c.get("state") == "received" and "terminal" not in c:
            pass
    return {"label_source": "command_lifecycle", "instructions": instructions,
            "instruction_idx": instr, "segment": seg, "flags": flags,
            "commands": [_command_summary(c) for c in cmds], "assist": assist}


def _annotation_spans(events: dict, kinds: tuple, t_end: float) -> list:
    out, open_ = [], {}
    for a in sorted(events.get("annotation", []), key=lambda e: e["stamp"]):
        k = str(a.get("kind", ""))
        if k not in kinds:
            continue
        edge = str(a.get("edge", "at"))
        if edge == "start":
            open_[k] = float(a["stamp"])
        elif edge == "end" and k in open_:
            out.append((open_.pop(k), float(a["stamp"])))
    out += [(t0, t_end + 1.0) for t0 in open_.values()]
    return out


def _command_summary(c: dict) -> dict:
    return {k: c.get(k) for k in ("command_id", "action", "accepted_action", "route", "cycle_id", "parent_id",
                                  "question", "question_id", "utterance", "utterance_id", "plan_id",
                                  "received", "accepted", "terminal", "terminal_stamp", "terminal_reason")} | {
        "states": [s["state"] for s in c["states"]]}


# ------------------------------------------------------------------ legacy ----
def _label_legacy(events: dict, steps: dict, instructions: List[dict]) -> dict:
    """The v2 recorder's rule, unchanged (EpisodeRecorder._label_steps before
    2026-09-30), plus the detectors of its known failure patterns."""
    st = steps["stamp_ros"]; n = len(st)
    instr = np.full(n, -1.0); seg = np.full(n, float(SEGMENT_IDLE))
    cds = sorted(e["stamp"] for e in events.get("cycle_done", []))
    starts = sorted(e["stamp"] for e in events.get("executor_phase", []) if e.get("name") == "motion_start")
    picks = ("pick", "pick_and_place")
    for k, ins in enumerate(instructions):
        t_on = ins["stamp"]
        t_off = instructions[k + 1]["stamp"] if k + 1 < len(instructions) else np.inf
        nxt = instructions[k + 1] if k + 1 < len(instructions) else None
        ins.pop("superseded", None)
        if (nxt is not None and ins["action_intent"] in picks and nxt["action_intent"] in picks
                and not any(t_on <= s < t_off for s in starts)):
            ins["superseded"] = True
            continue
        ends = [c for c in cds if c > t_on]
        if ends:
            t_off = min(t_off, ends[0] + 0.5)
        m = (st >= t_on) & (st < t_off)
        instr[m] = k; seg[m] = SEGMENT_TASK
    resets = [e["stamp"] for e in events.get("cycle_done", []) if not e["ok"]]
    resets += [e["stamp"] for e in events.get("interactive_status", []) if str(e["status"]).startswith("reset_home")]
    for r in sorted(resets):
        nxt = [i["stamp"] for i in instructions if i["stamp"] > r]
        t_off = nxt[0] if nxt else np.inf
        m = (st >= r) & (st < t_off)
        seg[m] = SEGMENT_RESET; instr[m] = -1
    return {"label_source": "legacy_intent_arrival", "instructions": instructions,
            "instruction_idx": instr, "segment": seg, "flags": legacy_findings(events, instructions, instr),
            "commands": [], "assist": []}


def legacy_findings(events: dict, instructions: List[dict], instr: np.ndarray) -> List[str]:
    """What the legacy labels cannot be trusted with (dry-run evidence)."""
    out = ["labels from intent ARRIVAL (no command lifecycle recorded): intervals unverified"]
    starts = sorted((e["stamp"], e.get("cycle_id")) for e in events.get("executor_phase", [])
                    if e.get("name") == "motion_start")
    for k, ins in enumerate(instructions):
        n_steps = int((instr == k).sum())
        nxt = instructions[k + 1] if k + 1 < len(instructions) else None
        if nxt is not None and nxt["stamp"] - ins["stamp"] < 1.0 and not ins.get("superseded"):
            out.append(f"'{ins['instruction']}' and '{nxt['instruction']}' arrived {1000 * (nxt['stamp'] - ins['stamp']):.0f} ms "
                       f"apart: the router queued the second — '{ins['instruction']}' got {n_steps} step(s), "
                       f"'{nxt['instruction']}' took its motion (needs review)")
        then = str((ins.get("attributes") or {}).get("then") or "")
        if then and then not in ("none", "null"):
            fixed = canonical_instruction(ins["action_intent"], ins.get("object"), ins.get("attributes"),
                                          fallback=ins.get("transcript") or "")
            if fixed != ins["instruction"]:
                out.append(f"'{ins['instruction']}' lost its then={then}: should read '{fixed}' (repairable)")
        if ins["action_intent"] not in ("pick", "pick_and_place"):
            t_off = nxt["stamp"] if nxt is not None else np.inf
            inside = [s for s, _c in starts if ins["stamp"] <= s < t_off]
            if inside:
                out.append(f"'{ins['instruction']}' spans a grasp motion start at {inside[0]:.2f} — "
                           "the pick it covers was not caused by it (needs review)")
    return out


# --------------------------------------------------------------------- API ----
def label_episode(events: dict, steps: dict, legacy_instructions: Optional[List[dict]] = None) -> dict:
    """Labels for ``steps`` (needs "stamp_ros", optionally "joint_pos") from
    the episode's ``events``. Returns label_source, instructions,
    instruction_idx, segment, flags, commands and assist spans; the
    caller writes them into steps.npz / meta.json."""
    if events.get("command"):
        out = _label_lifecycle(events, steps)
        out["format"] = RAW_FORMAT
    else:
        out = _label_legacy(events, steps, list(legacy_instructions or []))
    t0 = float(steps["stamp_ros"][0]) if len(steps["stamp_ros"]) else 0.0
    for ins in out["instructions"]:
        if ins.get("t") is None:
            ins["t"] = ins["stamp"] - t0
    return out
