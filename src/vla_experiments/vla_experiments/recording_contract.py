"""The recording contract (2026-09-30): event schema, lifecycle states, segment
classes, outcomes and the conventions every recorder, labeller, validator and
exporter shares. Pure Python, no ROS imports — tests and offline tools use it.

Why it exists
-------------
Until raw format v2 an instruction became active when its INTENT arrived on
/robot/intent. The intent router queues, deduplicates, defers, refuses or
answers a pause with those intents, so arrival is not execution: trial 114's
look_right and go_home arrived 8 ms apart and go_home took the look's motion;
trial 121's release_home, said before a pointed pick, labelled that pick's
approach, close and lift. From v3 the router publishes each command's
LIFECYCLE on /robot/command_event and the recorder labels steps from the
command that is actually executing (see labels.py).

Command event (JSON on /robot/command_event, std_msgs/String, reliable, depth 100)
-------------------------------------------------------------------------------
    schema        "morty.command_event.v1"
    event_id      unique id of this event (uuid4 hex)
    seq           per-publisher sequence number (detects loss / reordering)
    source        publishing node ("intent_router")
    boot          publisher start time (wall s) — a restart starts a new seq
    command_id    the command (from the intent's "command_id"; the router
                  mints one for an intent without it); stable across events
    parent_id     the command this one belongs to: an answer to a pause of a
                  pick belongs to that pick's command ("" = top level)
    plan_id / plan_index   several intents from one utterance ("go home and
                  close") share plan_id = the utterance id, in order
    utterance_id  the transcript that caused it ("" = none: GUI, pointing)
    utterance     its text (kept even when the command is rejected)
    question_id / question   the question the command answers ("release",
                  "destination", "home" from the executor's task state,
                  "point" from the router), "" = none
    cycle_id      the grasp cycle executing it (pick commands, answers)
    state         received | queued | deferred | accepted | cycle | completed
                  | failed | rejected | cancelled | superseded
    action / attributes / target   the ACCEPTED form of the command (after
                  the router's rewrites: a pick_and_place carried over from a
                  which-one question, "here", a follow-up plan) and the
                  resolved target {label, position, colour, source,
                  proposal_id, bbox, image_stamp, frame} when known
    route         how it executes: "executor_answer" (a pause of a grasp
                  cycle), "pick" (a pipeline+executor cycle), "named_pose",
                  "gripper", "reset", "stop", "none"
    reason        why (rejections, deferrals, failures)
    stamp         source time of the event, wall clock seconds (clock "system")

"accepted" means the executing component took the command (goal accepted,
resume token consumed, pick committed). It is NOT evidence that motion began:
motion start comes from the executor's task_state phases (approaching,
carrying, releasing, returning_home) or, for named-pose moves, from the arm's
joints between "accepted" and "completed".

Other recorded evidence (meta.json "events"), each with "stamp" (source time,
system clock unless named) and "recv" (recorder receipt, system clock):
    transcript / utterance (utterance_id, speech_start/end, published)
    intent            raw /robot/intent payloads, kept even when rejected
    command           the command events above
    task_state        every new executor task-state sample (seq, cycle_id,
                      phase, question, question_id, holding, object) —
                      recorded incrementally, as it happens
    point_question    the pipeline's pointing proposals (open / answered)
    selection         the pipeline's selection record (label, source,
                      cycle_id, bbox, image stamp, frame, proposal_id)
    executor_phase    the executor's detailed marks (published at cycle end)
    interactive_status, gripper_cmd, cycle_done, annotation, camera_check

Clock domains: "system" (wall clock of the laptop; ROS time equals it here —
no sim time), "camera_exterior" (the shuttle PC's clock: its offset to the
laptop is measured per episode from receipt minus source stamp, see
meta["alignment"]), "camera_wrist" / "robot" (laptop).

Observation / action boundary
-----------------------------
Step i holds the observation at tick time t_i (images: the frame whose source
stamp is closest to t_i within max_image_skew_s; state: the newest sample not
newer than t_i) and the action is the change to step i+1 WITHIN the same
training segment (the last step of a segment repeats zero motion with the
segment's last gripper target). A command change starts a new segment, so no
action crosses an instruction boundary or an excluded gap.

Compound commands
-----------------
A compound command stays ONE instruction across its executed sequence:
"open the gripper and go back home" (open_gripper + then=release_home) covers
the release AND the requested return home; "bring it home and hand it over"
(go_home + then=release) covers the carry, the hold for the hand-over and the
release. The parent plan (plan_id) and the executed parts (task_state phases)
stay in meta, so a converter may split them later.
"""
from __future__ import annotations

COMMAND_EVENT_SCHEMA = "morty.command_event.v1"
RAW_FORMAT = "morty_vla_raw_v3"
RAW_FORMATS_WITH_STEP_LABELS = ("morty_vla_raw_v2", RAW_FORMAT)

# Command lifecycle states, in the order they may occur.
STATES = ("received", "queued", "deferred", "accepted", "cycle", "completed",
          "failed", "rejected", "cancelled", "superseded")
TERMINAL_STATES = ("completed", "failed", "rejected", "cancelled", "superseded")

# Per-step segment codes (steps.npz "segment"). 0-2 are the v2 codes, kept.
SEGMENT_IDLE = 0      # setup, conversation, no command executing
SEGMENT_TASK = 1      # the accepted command executing, incl. its meaningful holds
SEGMENT_RESET = 2     # automatic cleanup / reset after a failure or a stop
SEGMENT_HOLD = 3      # the robot holds and waits for the operator's next words
SEGMENT_CLARIFY = 4   # a question about the target is open (which one? pointed?)
SEGMENT_ASSIST = 5    # the operator annotated manual assistance / intervention
SEGMENT_NAMES = {SEGMENT_IDLE: "idle", SEGMENT_TASK: "task", SEGMENT_RESET: "reset",
                 SEGMENT_HOLD: "hold", SEGMENT_CLARIFY: "clarify", SEGMENT_ASSIST: "assist"}

# Trial / episode / command outcomes.
OUTCOMES = ("success", "failure", "unknown", "interrupted")


def parse_stop_arg(arg: str) -> tuple:
    """(outcome, note, recognised) for the recorder's "stop <arg>".

    A bare or unknown argument is NOT a human success: before v3 "stop" with
    no argument (or a typo) saved outcome=success. Recognised: ok / success /
    yes; fail / failed / failure / no; unknown / ?; interrupted / abort;
    auto (decided by the recorder from the executor's verdict)."""
    a = (arg or "").strip().lower()
    if a in ("ok", "success", "succeeded", "yes", "true", "1"):
        return "success", "outcome: human", True
    if a in ("fail", "failed", "failure", "no", "false", "0"):
        return "failure", "outcome: human", True
    if a in ("unknown", "?"):
        return "unknown", "outcome: human said unknown", True
    if a in ("interrupted", "interrupt", "abort", "aborted"):
        return "interrupted", "outcome: interrupted", True
    if a == "auto":
        return "auto", "", True
    if not a:
        return "unknown", "outcome: unknown (stop without a verdict)", True
    return "unknown", f"outcome: unknown (unrecognised stop argument {arg!r})", False


def validate_command_event(e: dict) -> list:
    """Problems with one command event (empty = valid)."""
    out = []
    if not isinstance(e, dict):
        return ["not an object"]
    if e.get("schema") != COMMAND_EVENT_SCHEMA:
        out.append(f"schema {e.get('schema')!r}")
    for k in ("event_id", "command_id", "state", "stamp", "source"):
        if not e.get(k) and e.get(k) != 0:
            out.append(f"missing {k}")
    if e.get("state") not in STATES:
        out.append(f"unknown state {e.get('state')!r}")
    try:
        float(e.get("stamp"))
    except (TypeError, ValueError):
        out.append("stamp not a number")
    return out
