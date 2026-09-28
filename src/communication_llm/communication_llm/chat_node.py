"""
chat_node.py
============

The conversational core of the voice stack. ``chat_node`` is the ROS node that
sits between the user's speech (heard by ``speech_input_node`` and transcribed
to text) and the robot (commanded through ``intent_router`` and the executors).
It runs the LLM dialog, extracts a robot *intent* from each utterance, speaks
short status narrations through ``speak_action_server``, and publishes the
chosen intent so the grasp pipeline can react.

Main responsibilities:

* Subscribe to ``/speech/transcript`` (text from the user) and feed it to the
  LLM via :mod:`llm_service`.
* Apply the bounded-intent contract — only a fixed set of robot intents
  (``pick``, ``drop``, ``home``, ``enable multi-view``, ...) is accepted; the
  LLM's structured output is validated and unsupported intents are rejected
  before anything is published.
* Publish the validated intent on ``/robot/intent`` for ``intent_router_node``
  to dispatch.
* Speak short hard-coded narrations (see ``HARDCODED_NARRATIONS`` below) via
  the :mod:`morty_interfaces` ``Speak`` action so the user hears feedback
  immediately, before the slower LLM reply arrives.
* Drop the LLM's own ECHO of its TTS (when the mic picks up Morty speaking)
  so the system does not talk to itself.
* Track the most recent grasps / joint states for context the LLM may need.

The LLM provider is selected via env vars (``MORTY_LLM_MODEL``,
``OPENAI_API_KEY``, ``AWS_BEARER_TOKEN_BEDROCK``); see ``llm_service.py``.

This file does NOT directly drive the robot or perform grasping; it only
emits *intents*. The grasp pipeline and executors interpret them.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
import math
import os
import re
import threading
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.wait_for_message import wait_for_message
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import JointState

from morty_interfaces.action import ExecuteGrasp, Speak
from rcl_interfaces.msg import SetParametersResult

from . import llm_service as _llm_service
from . import voice_grammar
from .llm_service import INTENT_TOOL, LLMService, StopStream, extract_intent, intent_to_json
from .llm_service import _AdaptedFunctionCall


PICK_INTENTS = {"pick", "pickup", "pick_up", "pick up", "grasp", "take", "grab"}

# Short, hard-coded narrations spoken IMMEDIATELY when an intent is
# extracted (before the slower LLM final-reply round-trip). Keeps audio
# feedback in sync with action dispatch.
QUICK_NARRATION = {
    # NOT spoken pre-emptively — see CONFIRMED_NARRATION_ACTIONS below.
    # Kept in the table so multi-intent turn accounting still sees them.
    "open_gripper": "Opening the gripper.",
    "close_gripper": "Closing the gripper.",
    "go_home": "Going home.",
    "go_to_ready": "Going to ready.",
    "go_to_start": "Going to start.",
    "sleep": "Going to sleep position.",
    "look_left": "Looking left.",
    "look_right": "Looking right.",
    "look_up": "Looking up.",
    "drop_right": "Going to drop right.",
    "drop_left": "Going to drop left.",
    "enable_multiview": "Multi-view scan enabled.",
    "disable_multiview": "Single-view mode.",
    "stop": "Stopping.",
    # 2026-09-25: reset = the GUI's "Reset pick" by voice; spoken at once
    # with interrupt, like "Stopping." — nothing else follows it.
    "reset": "Resetting.",
    # System control — executed by the Morty GUI, which owns the launches.
    "start_driver": "Starting the robot driver.",
    "stop_driver": "Stopping the robot driver.",
    "start_pipeline": "Starting the grasp pipeline.",
    "stop_pipeline": "Stopping the grasp pipeline.",
    "start_all": ("Starting the whole system. Remember to press the "
                  "Play button on the teach pendant."),
    "stop_all": "Stopping everything.",
    # pick/place are intentionally absent — the LLM's contextual reply
    # ("Picking up the leftmost vitamin tube") is more useful than a generic.
}

# 2026-09-25 (Majd: "'I didn't catch that' must come from CODE and be spoken
# ALONE — never glued to a real answer"). The model marks a garbled turn with
# the tool action 'unclear'; this node then speaks this line by itself. A
# model sentence that says it anyway is never spoken (it marks the turn
# unclear instead).
# 2026-09-25: how long before a question opened an answer's measured start
# may lie and still count (see _route_minimal's freshness rule): speech_start
# includes the speech node's 300 ms pre-roll, so start + 0.3 s is where the
# voice really began; 0.35 s = "began at or after the question opened, 50 ms
# margin". Majd's "Now give it." measured -38 ms (voice +0.26 s) and was
# re-asked; a "yes" begun 0.5 s or more before stays an answer to something else.
_FRESH_TOLERANCE_S = 0.35
UNCLEAR_LINE = "I didn't catch that, could you repeat?"
# 2026-09-25 (review Y5): FIRST person only — "I didn't catch that", "Sorry,
# didn't catch it", "I did not quite catch that". A story's "the fisherman
# didn't catch it" is not the model saying it misheard.
_UNCLEAR_SENTENCE_RE = re.compile(
    r"(?:\bI\s+|(?:^|[.!?,;:\u2014-])\s*(?:sorry,?\s*)?)(?:didn'?t|did\s+not)\s+"
    r"(?:quite\s+)?catch\s+(?:that|it)\b", re.IGNORECASE)

# 2026-09-25 (review M3): the model's prose that CLAIMS a robot action —
# the end-of-turn action-echo list, now module level so the streamed
# sentences are judged by the same list, plus the pick claim. GWDG hands
# over its tool call only at the end of the stream, and Claude often writes
# its sentence before the tool block, so "Opening the gripper now." was
# spoken before the command existed. The router (on acceptance) and this
# node's QUICK lines (on dispatch) say what happens; the model's claim is
# never spoken. Leading acknowledgements ("Okay,", "Sure!") are skipped.
_ACTION_ECHOES = (
    "going home", "going to ready", "going to start",
    "going to the drop", "going to drop", "going to sleep",
    "looking left", "looking right", "looking up",
    "opening the gripper", "closing the gripper",
    "stopping",
    "picking up", "picking the", "grasping the",
)
_CLAIM_LEAD_RE = re.compile(
    r"^(?:(?:okay|ok|sure|alright|all right|got it|on it|yes|yep|right|no problem|now)"
    r"[\s,.!\u2014-]+)*(?:i'm |i am )?", re.IGNORECASE)


def _is_action_claim(sentence: str) -> bool:
    """2026-09-25 (review M3): True when ``sentence`` announces a robot
    action ("Opening the gripper now.", "Okay, going home.", "Picking up
    the small bottle.") — see _ACTION_ECHOES."""
    s = _CLAIM_LEAD_RE.sub("", (sentence or "").strip().replace("’", "'")).lower()
    return s.startswith(_ACTION_ECHOES)


# 2026-09-25 (review H3): the question said again when an answer to it is
# not fresh (begun before the question opened) — it is neither acted on nor
# handed to the LLM as consent.
_REASK_LINES = {
    "release": "Should I open the gripper?",
    "destination": "Where should I drop it? Left, right, or home?",
    "home": "Should I go back home?",
}
# The FAST-NEGATIVE lines (2026-08-28), shared since 2026-09-25 with the LLM
# path (review Y2): a NO the LLM turned into close_gripper / stay says them too.
_KEEP_HOLDING_LINE = "Okay, I'll keep holding it. Where should I take it?"
_STAY_LINE = "Okay, I'll stay here."
_TASK_STATE_TOPIC = "/grasp_executor_lin_llm/task_state"
# 2026-09-25 (Majd's G1): a transcript that follows the previous one within
# this many seconds, while that one has neither spoken nor dispatched, is
# its continuation — "Drop it on home position. Give me the object uh" was
# cut by the fragment "Andrew said." 0.4 s later: the first turn was dropped
# as interrupted and only the fragment reached the LLM ("didn't catch that").
_MERGE_WINDOW_S = 2.0
_HUSH_BY_MERGE = "the operator's sentence went on"


class _ActCountingPublisher:
    """2026-09-25 (Majd's G1): /robot/intent publisher that counts every
    intent this node dispatches (every publish site at once), for
    ChatIntentNode._hush_and_merge's "has the previous input acted yet?"."""

    def __init__(self, pub, on_publish) -> None:
        self._pub = pub
        self._on_publish = on_publish

    def publish(self, msg) -> None:
        self._on_publish()
        # 2026-09-28: every intent carries the wall time it was published
        # ("stamp"). Recorders stamp a message when their callback runs — the
        # GUI and the rosbag were measured up to 3 s late during a trial's
        # start-up load (trial 085: published +5.52 s, recorded +8.5 s) — so
        # trial timing reads this field. Additive: readers use named keys.
        try:
            d = json.loads(msg.data)
            if isinstance(d, dict) and "stamp" not in d:
                d["stamp"] = round(time.time(), 3)
                msg = String(data=json.dumps(d, separators=(",", ":")))
        except (TypeError, ValueError, AttributeError):
            pass
        self._pub.publish(msg)

    def __getattr__(self, name: str):
        return getattr(self._pub, name)

# 2026-09-25: the four live-settable voice switches (SPEC C1). Their
# defaults are Majd's decisions of 2026-09-25: the minimal grammar, bare
# destination answers on (EXT-DEST), a story yields to a new robot
# question, and a GWDG stream that neither spoke nor dispatched after 3 s is
# re-answered by Claude (EXT-GWDG-STALL).
_VOICE_PARAMS = ("voice_fast_layers", "voice_ext_dest", "story_yields_to_question",
                 "gwdg_stall_s")
# Actions whose quick line the router speaks after the executor accepted the
# answer ("Going to drop right." on resumed:) while a pick is in flight.
_DEST_ACTIONS = {"drop_right", "drop_left", "confirm_drop", "go_home"}
_HUSH_BY_QUESTION = "the robot asks a question"

# Actions whose narration is owned by intent_router's CONFIRMATION layer,
# never by this node.
#
# 2026-08-28, from a real session: the FAST layer matched "...Yes, open the
# gripper..." at 1787943598.924 and spoke "Opening the gripper." at
# 1787943600.806 — but the resume token it published landed on an executor
# that had already left the pause 4 ms earlier (a 'go_home' from the same
# utterance took it), so the gripper never moved. Morty announced an action
# he did not perform. The operator: "he said 'Opening the gripper' but he
# didn't open the gripper?"
#
# A gripper change is the one thing in this stack with an authoritative
# after-the-fact report, and intent_router already subscribes to BOTH
# channels while this node subscribes to neither:
#   * /grasp_executor_lin_llm/interactive_status 'released' — the executor
#     says the gripper opened at the drop pose;
#   * the OperateGripper action result (success / object_detected) from the
#     gripper adapter, for a standalone open/close.
# So intent_router speaks these lines, and only after one of those two
# signals arrives. When a resume token is swallowed it instead says what
# really happened ("I did not open the gripper...").
#
# This node stays SILENT for these actions on every path: the FAST layer,
# the LLM tool-call layer, and the LLM's own end-of-turn reply (which would
# otherwise re-introduce the same lie in prose).
CONFIRMED_NARRATION_ACTIONS = {
    "open_gripper",
    "close_gripper",
    "open",
    "close",
    "release",
    "release_home",
    "drop_held",
    "let_go",
    "not_holding",   # 2026-09-28: the router says what it does with it
}

# Words that should trigger an instant cancel before any LLM round-trip.
# Use word-boundary matching so "cancellation" doesn't fire and innocuous mentions
# inside long sentences don't cancel the wrong thing.
STOP_PATTERNS = {"stop", "cancel", "abort", "halt", "freeze", "stop it"}


def _is_stop_command(text: str) -> bool:
    import re
    t = (text or "").strip().lower()
    if not t:
        return False
    # short utterance like "stop" or "stop!" — fast path
    if re.fullmatch(r"[!\.\s]*(stop|cancel|abort|halt|freeze)[!\.\s]*", t):
        return True
    # explicit phrasing: "stop it", "stop the robot", "cancel that"
    if re.match(r"^(stop|cancel|abort|halt|freeze)\b", t):
        return True
    return False


def _is_redo_command(text: str) -> bool:
    """True for 'again', 'do it again', 'repeat', 'redo', 'one more time' etc."""
    import re
    t = (text or "").strip().lower()
    if not t:
        return False
    if re.fullmatch(
        r"[!\.\s]*(again|redo|repeat|once more|one more time|do it again|do that again)[!\.\s]*",
        t,
    ):
        return True
    if re.match(r"^(again|redo|repeat|do (it|that) again)\b", t):
        return True
    return False


# Fast-path direct command matcher.
#
# These regexes detect simple, unambiguous commands like "open the gripper",
# "go home", "look left fast" — phrases that don't need the LLM at all.
# When matched, chat_node publishes the intent JSON directly (~10ms) and
# triggers QUICK_NARRATION TTS, completely bypassing the OpenAI round-trip.
# Anything not matched still goes to the LLM for context-aware handling
# (jokes, picks, complex sentences, ambiguous phrasing).
# Deterministic PICKS (2026-08-26): Haiku kept refusing plain pick
# commands with fabrications ("I'm already grasping it — it should
# arrive in a moment") and inventing alias features for ASR garbles
# ("craftsform/class/trust for me is..."). Plain single-object picks now
# bypass the LLM entirely. Qualified picks (left/right/middle/ordinal)
# still go to the LLM — the router's disambiguation needs them.
# 2026-09-25 (Majd's B1): the verb and object tables moved to voice_grammar
# (PICK_VERB / PICK_OBJECTS, unchanged, with their history) — the minimal
# layer's instant whole-sentence pick reads the same table; one owner.
_FAST_PICK_VERB = voice_grammar.PICK_VERB
_FAST_PICK_OBJECTS: list[tuple[str, str, str | None]] = voice_grammar.PICK_OBJECTS
_FAST_PICK_BLOCKERS = (
    r"\b(?:don'?t|do not|never|stop|cancel|left|right|middle|center|"
    r"closest|nearest|first|second|third|last|leftmost|rightmost)\b"
)


def _match_fast_pick(text: str):
    """Return (label, colour|None) when the utterance is a plain pick of a
    known object; None when it needs the LLM (qualified/negated/unknown)."""
    import re
    t = (text or "").strip().lower()
    if not t or re.search(_FAST_PICK_BLOCKERS, t):
        return None
    if not re.search(_FAST_PICK_VERB, t):
        return None
    for obj_pat, label, colour in _FAST_PICK_OBJECTS:
        if re.search(obj_pat, t):
            return (label, colour)
    return None


_FAST_COMMAND_PATTERNS: list[tuple[str, str]] = [
    # Gripper open — bare "open" or "open it" trigger the action directly.
    # Whisper sometimes mistranscribes the speech as "open it" / "open
    # please" / "open up" so we accept any line whose meaningful content
    # is just "open" with optional fillers.
    (r"^\s*(?:please\s+)?open(?:\s+it|\s+up|\s+please)?\s*[!\.\?]*\s*$", "open_gripper"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?open(?: the)? gripper\s*[!\.\?]*\s*$", "open_gripper"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?release\s*[!\.\?]*\s*$", "open_gripper"),
    (r"^\s*greifer\s+öffnen\s*[!\.\?]*\s*$", "open_gripper"),
    (r"^\s*öffne(?:n)?(?: den)? greifer\s*[!\.\?]*\s*$", "open_gripper"),
    # Bare "auf" (German "open") as a single-word command.
    (r"^\s*auf\s*[!\.\?]*\s*$", "open_gripper"),
    # Gripper close — bare "close" or "close it" trigger directly.
    (r"^\s*(?:please\s+)?close(?:\s+it|\s+up|\s+please)?\s*[!\.\?]*\s*$", "close_gripper"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?close(?: the)? gripper\s*[!\.\?]*\s*$", "close_gripper"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?grip\s*[!\.\?]*\s*$", "close_gripper"),
    (r"^\s*greifer\s+schließen\s*[!\.\?]*\s*$", "close_gripper"),
    (r"^\s*schließe(?:n)?(?: den)? greifer\s*[!\.\?]*\s*$", "close_gripper"),
    # Bare "zu" (German "close") as a single-word command.
    (r"^\s*zu\s*[!\.\?]*\s*$", "close_gripper"),
    # Go home
    (r"^\s*(?:(?:yes|yeah|yep|sure|ok(?:ay)?|now|and(?:\s+now)?|then|so|also|alright|all right),?\s+)?(?:please\s+)?(?:can you\s+)?(?:let'?s\s+)?(?:now\s+)?go(?: back)?(?: to)?(?: the)? home(?: position)?(?:,?\s*now)?,?\s*(?:please)?(?:,?\s*now)?\s*(very slow(?:ly)?|slow(?:ly)?|normal|very fast|fast)?\s*[!\.\?]*\s*$", "go_home"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?return(?: to)? home\s*[!\.\?]*\s*$", "go_home"),
    # Bare "home" / "now home" / "home please" (2026-09-02: "Now home."
    # fell to the LLM, which — with the live pose unknown mid-move —
    # answered "You're already at home." while the arm sat at look-left).
    (r"^\s*(?:now|and|then|ok(?:ay)?|yes),?\s*home(?:,?\s*please)?\s*[!\.\?]*\s*$", "go_home"),
    # 2026-09-08: "Home position." (the answer to "Where should I drop
    # it? Left, right, or home?") fell to the LLM, which answered "You're
    # already at home" while the arm held the object over the table.
    (r"^\s*(?:the\s+|to\s+(?:the\s+)?|at\s+)?home(?:\s+position)?(?:,?\s*please)?\s*[!\.\?]*\s*$", "go_home"),
    (r"^\s*geh(?:e)?(?: nach)? (?:hause|home)\s*[!\.\?]*\s*$", "go_home"),
    # Go to ready / start
    # "go home and close/open the gripper" — two sequential actions
    (r"^.{0,40}?\b(?:go\s+)?(?:back\s+)?home\b.{0,40}?\bclose\b.*$", "go_home_close"),
    (r"^.{0,40}?\bclose\b.{0,30}?\bgripper\b.{0,40}?\bhome\b.*$", "go_home_close"),
    (r"^.{0,40}?\b(?:go\s+)?(?:back\s+)?home\b.{0,40}?\bopen\b.*$", "go_home_open"),
    # Compound destination + open (+home) in one sentence — deterministic,
    # because the LLM keeps dropping the 'then' attribute (2026-08-26:
    # "Dropping left and opening the gripper" ... then it asked anyway).
    # Order-agnostic (Majd 2026-08-26: "GO TO THE RIGHT DROP POSITION and
    # give me it and then go back home" — side-before-drop wording fell
    # through). Lookaheads accept drop/put and the side in ANY order;
    # the release verb and the trailing home keep their sequence.
    (r"^(?!.*\b(?:don'?t|do not|never|stay|wait)\b)(?!.*\b(?:grasp|grass|class|cross|crass|craft\w*|trust|pick|grab|fetch)\b)(?=.{0,90}?\b(?:drop|put)\b)(?=.{0,90}?\bright\b).{0,90}?\b(?:open|release|give|hand)\b.{0,60}?\bhome\b.*$", "drop_right_release_home"),
    (r"^(?!.*\b(?:don'?t|do not|never|stay|wait)\b)(?!.*\b(?:grasp|grass|class|cross|crass|craft\w*|trust|pick|grab|fetch)\b)(?=.{0,90}?\b(?:drop|put)\b)(?=.{0,90}?\bleft\b).{0,90}?\b(?:open|release|give|hand)\b.{0,60}?\bhome\b.*$", "drop_left_release_home"),
    (r"^(?!.*\b(?:don'?t|do not|never|stay|wait|home)\b)(?!.*\b(?:grasp|grass|class|cross|crass|craft\w*|trust|pick|grab|fetch)\b)(?=.{0,90}?\b(?:drop|put)\b)(?=.{0,90}?\bright\b).{0,90}?\b(?:open|release|give|hand)\b.*$", "drop_right_release"),
    (r"^(?!.*\b(?:don'?t|do not|never|stay|wait|home)\b)(?!.*\b(?:grasp|grass|class|cross|crass|craft\w*|trust|pick|grab|fetch)\b)(?=.{0,90}?\b(?:drop|put)\b)(?=.{0,90}?\bleft\b).{0,90}?\b(?:open|release|give|hand)\b.*$", "drop_left_release"),
    # Destination answers with decorations (Majd 2026-08-26: "Drop it on
    # the right side. And then open the gripper." emitted NOTHING and the
    # arm froze). Non-anchored on purpose: any drop/put + side wording
    # fires the side intent; the router routes it to the pause resume.
    (r"^(?!.*\b(?:don'?t|do not|never|stay|wait)\b)(?!.*\b(?:grasp|grass|class|cross|crass|craft\w*|trust|pick|grab|fetch)\b)(?=.{0,90}?\b(?:drop|put)\b)(?=.{0,90}?\bright\b).*$", "drop_right"),
    (r"^(?!.*\b(?:don'?t|do not|never|stay|wait)\b)(?!.*\b(?:grasp|grass|class|cross|crass|craft\w*|trust|pick|grab|fetch)\b)(?=.{0,90}?\b(?:drop|put)\b)(?=.{0,90}?\bleft\b).*$", "drop_left"),
    # "reset the pick" == stop/abort the current cycle (Majd 2026-08-25:
    # the GUI button concept, now speakable).
    (r"^\s*(?:please\s+)?reset(?: the)?(?: pick| grasp| cycle)?\s*[!\.\?]*\s*$", "stop"),
    (r"^\s*(?:please\s+)?(?:start over|cancel(?: the)?(?: pick| grasp)?)\s*[!\.\?]*\s*$", "stop"),
    # Compound release+home in ONE utterance (Majd 2026-08-25): opens
    # the gripper and goes straight home without the extra question.
    # Filler-tolerant (Majd 2026-08-26: "open the gripper AND GIVE ME
    # THE OBJECT and go back home" fell through to the LLM, which
    # dropped the home part — again). Up to ~60 chars of chatter may sit
    # between the open and the final "...and (go) (back) home", but a
    # negation/stay word anywhere kills the compound match.
    # ASR garbles the connector ("open the gripper SAYING go back home",
    # 2026-08-26) — any short filler before a TRAILING home counts.
    (r"^(?!.*\b(?:don'?t|do not|never|stay|wait)\b)\s*(?:.{0,30}?[\.!,]\s*)?(?:please\s+)?(?:can you\s+)?(?:open(?: the)?(?: gripper)?|release(?: it)?|give(?: it| me)*|hand(?: it)?(?: over)?)\b.{0,60}?\bhome\s*[!\.\?]*\s*$", "release_home"),
    # Tolerant single release: "Yes, open the gripper and give me this F
    # object" repeated the pause question instead of opening (2026-08-26)
    # — yes-prefixes and give-me suffixes are fine, "home" anywhere sends
    # it to the compound above instead.
    (r"^(?!.*\b(?:don'?t|do not|not|never|stay|wait|home|close)\b)\s*(?:.{0,30}?[\.!,]\s*)?(?:please\s+)?(?:can you\s+)?open(?: the)?(?: gripper)?\b(?:.{0,60}?\b(?:give|hand|object|me)\b.*)?\s*[!\.\?]*\s*$", "open_gripper"),
    # "detect left/right" wording lands as look poses first-try
    # (Majd 2026-08-25: "go to detect left" caused a clarifying question).
    (r"^\s*(?:please\s+)?(?:can you\s+)?(?:go(?: to)?(?: the)?\s+)?detect(?:ion)?\s+left(?: position)?\s*[!\.\?]*\s*$", "look_left"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?(?:go(?: to)?(?: the)?\s+)?detect(?:ion)?\s+right(?: position)?\s*[!\.\?]*\s*$", "look_right"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?go(?: to)?(?: the)? ready(?: position)?\s*(very slow(?:ly)?|slow(?:ly)?|normal|very fast|fast)?\s*[!\.\?]*\s*$", "go_to_ready"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?go(?: to)?(?: the)? start(?: position)?\s*[!\.\?]*\s*$", "go_to_start"),
    # Sleep / stow / park
    (r"^\s*(?:please\s+)?(?:can you\s+)?(?:sleep|stow|park|go to sleep(?: position)?)\s*[!\.\?]*\s*$", "sleep"),
    # Look left/right/up
    (r"^\s*(?:please\s+)?(?:can you\s+)?look(?: to(?: the)?)? left\s*(very slow(?:ly)?|slow(?:ly)?|normal|very fast|fast)?\s*[!\.\?]*\s*$", "look_left"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?look(?: to(?: the)?)? right\s*(very slow(?:ly)?|slow(?:ly)?|normal|very fast|fast)?\s*[!\.\?]*\s*$", "look_right"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?look(?: up| higher)\s*(very slow(?:ly)?|slow(?:ly)?|normal|very fast|fast)?\s*[!\.\?]*\s*$", "look_up"),
    (r"^\s*schau(?:e)? nach links\s*[!\.\?]*\s*$", "look_left"),
    (r"^\s*schau(?:e)? nach rechts\s*[!\.\?]*\s*$", "look_right"),
    (r"^\s*schau(?:e)? nach oben\s*[!\.\?]*\s*$", "look_up"),
    # Drop — only sided. A bare "drop" / "go to drop" intentionally
    # falls through to the LLM so it can ask "left or right?".
    (r"^\s*(?:please\s+)?(?:can you\s+)?drop(?: to(?: the)?)? right(?: side)?(?: position)?\s*(very slow(?:ly)?|slow(?:ly)?|normal|very fast|fast)?\s*[!\.\?]*\s*$", "drop_right"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?drop(?: to(?: the)?)? left(?: side)?(?: position)?\s*(very slow(?:ly)?|slow(?:ly)?|normal|very fast|fast)?\s*[!\.\?]*\s*$", "drop_left"),
    # Multi-view toggle
    (r"^\s*(?:please\s+)?(?:can you\s+)?(?:enable|turn on|switch on|activate) (?:the )?multi[\s\-]?view( scan| registration)?\s*[!\.\?]*\s*$", "enable_multiview"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?multi[\s\-]?view (?:on|enabled?)\s*[!\.\?]*\s*$", "enable_multiview"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?(?:disable|turn off|switch off|deactivate) (?:the )?multi[\s\-]?view( scan| registration)?\s*[!\.\?]*\s*$", "disable_multiview"),
    (r"^\s*(?:please\s+)?(?:can you\s+)?(?:single[\s\-]?view|one[\s\-]?shot)( only| mode)?\s*[!\.\?]*\s*$", "disable_multiview"),
]


_SEQUENCE_CONNECTOR = re.compile(
    r"(?:,\s*|\s)(?:and\s+)?then\b|\bafter\s+that\b|\bafterwards\b|\bund\s+dann\b|\bdanach\b|(?:,\s*|\s)dann\b",
    re.IGNORECASE)


def _match_fast_command(text: str) -> tuple[str, str] | None:
    """If text is a direct command we can route without the LLM, return
    (action_intent, speed). speed is "" if not specified. Returns None
    for anything that needs the LLM."""
    import re
    t = (text or "").strip().lower()
    if not t:
        return None
    # 2026-09-22: "Go to the drop position right, then go back home" was
    # matched here as drop_right alone and the second half was lost; the
    # LLM turns a sequence into one intent per step, so anything with a
    # sequence connector is its job, not this shortcut's.
    if _SEQUENCE_CONNECTOR.search(t):
        return None
    for pattern, action in _FAST_COMMAND_PATTERNS:
        m = re.match(pattern, t)
        if not m:
            continue
        speed = ""
        if m.groups():
            raw = (m.group(1) or "").strip().lower()
            if raw in {"very slow", "very slowly"}:
                speed = "very_slow"
            elif raw in {"slow", "slowly"}:
                speed = "slow"
            elif raw == "normal":
                speed = "normal"
            elif raw == "fast":
                speed = "fast"
            elif raw == "very fast":
                speed = "very_fast"
        return action, speed
    return None


# ── Deterministic POSITIVE answers at an open robot pause ────────────────
# 2026-09-08 (Majd's transcript): "Home position." at "Where should I drop
# it?" and "Now give me the object." at "Should I open the gripper?" both
# reached the LLM, which answered from its conversation memory ("you're
# already at home", "I just dropped it at the right position") instead of
# the robot's state. An answer to an open question is not a conversation:
# while a pause is open, a short destination or consent phrase is routed
# deterministically, exactly like the negative layer below. Negatives are
# left to that layer ("no, don't give it to me" must never open).
_PAUSE_DEST_STATE = "waiting for a drop destination"
_PAUSE_ANSWER_MAX_WORDS = 8
_DEST_HOME_RE = re.compile(r"\b(?:home|hause)\b", re.IGNORECASE)
_DEST_SIDE_RE = re.compile(r"\b(left|right|links|rechts)\b", re.IGNORECASE)
_CONSENT_RE = re.compile(
    r"(?:^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|ja|go\s+ahead|please|of\s+course|do\s+it)\b"
    r"|\b(?:give|hand)\s+(?:it|me|him|her|them|the\s+\w+)\b"
    r"|\blet\s+(?:it\s+)?go\b"
    r"|\b(?:you\s+can\s+|now\s+)?(?:open|release|drop)\b"
    r"|\bgib\s+(?:es|mir|ihn|sie)\b|\bloslassen\b)",
    re.IGNORECASE,
)


def _match_pause_answer(text: str, cycle_state: str) -> str | None:
    """Map a short POSITIVE answer at an open pause to the intent that
    resolves it; None when no pause is open, the phrase is negative (the
    negative layer owns it), or it is too long to be a plain answer.
        destination pause  -> "go_home" | "drop_left" | "drop_right"
        release pause      -> "open_gripper"
    """
    t = (text or "").strip()
    state = (cycle_state or "").lower()
    if not t or len(t.split()) > _PAUSE_ANSWER_MAX_WORDS:
        return None
    if _PAUSE_NEGATIVE_RE.search(t):
        return None
    if _PAUSE_DEST_STATE in state:
        home = bool(_DEST_HOME_RE.search(t))
        sides = {m.group(1).lower() for m in _DEST_SIDE_RE.finditer(t)}
        sides = {"left" if x in ("left", "links") else "right" for x in sides}
        if home and not sides:
            return "go_home"
        if len(sides) == 1 and not home:
            return "drop_" + sides.pop()
        return None            # both / neither / ambiguous → LLM
    if _PAUSE_RELEASE_STATE in state:
        return "open_gripper" if _CONSENT_RE.search(t) else None
    return None


# ── Deterministic NEGATIVE answers at an open robot pause ────────────────
# 2026-08-28 (Majd's loop transcript). The release/home pauses only ever
# recognised AFFIRMATIVE answers, so a negative fell through to the LLM,
# which just re-emitted the last status line:
#     Morty:    "I'm at the home position. Should I open the gripper?"
#     Operator: "No, close the gripper."   -> same line again
#     Operator: "No."                      -> same line again
#     Operator: "Don't open it. You don't have any object at the moment.
#                Close it."                -> same line again
# The only escape was "go back home". A pause must NEVER be left
# unanswered after an explicit negative.
#
# Deliberately NOT a catch-all "no": it only fires while a pause is
# actually open (see _PAUSE_RELEASE_STATE / _PAUSE_HOME_STATE), and the
# emitted intent is a no-op at the router when the pause has already
# closed (close_gripper without a pause just closes the — already
# closed — gripper; 'stay' without a pause is an unknown intent and is
# ignored).
_PAUSE_NEGATIVE_RE = re.compile(
    r"(?:"
    # leading "no" / "nope" / "nah" / "negative" / "no thanks"
    r"^\s*(?:no|nope|nah|negative|no\s+thanks?|no\s+thank\s+you)\b"
    # "don't open (it)" / "do not release" / "never give it to me" ...
    r"|\b(?:do\s*n[o']?t|do\s+not|never|dont)\s+"
    r"(?:open|release|drop|give|hand|let\s+go|unclench|loosen|move|go)\b"
    # keep-holding wording
    r"|\bkeep\s+(?:hold|holding|it|the\s+\w+|closed?)\b"
    r"|\b(?:hold|hang)\s+on\b"
    r"|\bhold\s+it\b"
    r"|\bnot\s+yet\b"
    r"|\bstay\b"
    r"|\bwait\b"
    # "close the gripper" / "close it" as the answer to "should I open?"
    r"|\bclose\s+(?:the\s+gripper|it|up)\b"
    r")",
    re.IGNORECASE,
)

# Substrings of the derived _cycle_state that mean "a pause is open".
_PAUSE_RELEASE_STATE = "waiting for release confirmation"
_PAUSE_HOME_STATE = "asked about going home"


def _match_pause_negative(text: str, cycle_state: str) -> str | None:
    """Pure decision helper: map a NEGATIVE operator answer given at an
    open robot pause to the action_intent that resolves that pause.

    Returns
        "close_gripper" — release pause ("should I open the gripper?")
                          answered NO ⇒ KEEP HOLDING. intent_router turns
                          this into the executor's 'stop' resume token,
                          which sets skip_release and carries the object
                          home.
        "stay"          — home-confirm pause ("should I go back home?")
                          answered NO ⇒ STAY at the drop pose.
        None            — no pause open, or the answer is not negative.
    """
    t = (text or "").strip()
    state = (cycle_state or "").lower()
    if not t:
        return None
    if _PAUSE_RELEASE_STATE not in state and _PAUSE_HOME_STATE not in state:
        return None
    if not _PAUSE_NEGATIVE_RE.search(t):
        return None
    if _PAUSE_RELEASE_STATE not in state:
        return "stay"
    # ── v2 (2026-08-31): stop collapsing compound answers ─────────────
    # "No, go back home and give me the object directly." used to become
    # plain keep-holding — the whole second half was discarded before the
    # LLM ever saw it (Majd: "that is wrong"). Inspect what FOLLOWS the
    # negative: a destination means REDIRECT while holding; a give/open
    # verb AFTER that destination means release on arrival.
    low = t.lower()
    # Adversarial review 2026-08-31 (findings 1+2): scan ONLY the text
    # AFTER the negative trigger — scanning the whole utterance made
    # "no, don't go home" redirect HOME and "stay right there" redirect
    # to drop-RIGHT (both measured). Additionally: a side word counts as
    # a destination only with a motion verb in front of it or "side"
    # after it, and any destination preceded by an un-punctuated
    # negation ("don't go home", "never go home") is rejected.
    m_neg = _PAUSE_NEGATIVE_RE.search(low)
    tail_all = low[m_neg.end():] if m_neg else low
    # The lookback runs over the FULL utterance prefix (not just the
    # tail): "never go home"'s negation lives inside the trigger match
    # itself. Bare "no"/"not" are deliberately NOT in this set — the
    # leading "no," of every answer would otherwise veto every
    # legitimate "no, go back home".
    _neg_off = m_neg.end() if m_neg else 0
    _NEG_LOOKBACK = (r"(?:do\s*n[o']?t|do\s+not|never|dont)"
                     r"(?:\s+\w+){0,2}\s*$")
    dest = None
    dest_end = -1
    # home: any "home" not under a preceding negation
    for m in re.finditer(r"\bhome\b", tail_all):
        if re.search(_NEG_LOOKBACK, low[:_neg_off + m.start()]):
            continue
        if m.end() > dest_end:
            dest, dest_end = "home", m.end()
    # left/right: require motion-verb context or a trailing "side" so
    # "stay RIGHT there" / "no left" can never move the arm
    for m in re.finditer(r"\b(left|right)\b", tail_all):
        pre = tail_all[max(0, m.start() - 26):m.start()]
        post = tail_all[m.end():m.end() + 8]
        if not (re.search(r"\b(?:go|going|take|bring|carry|put|drop|"
                          r"move|to)\b(?:\s+\w+){0,3}\s*$", pre)
                or re.match(r"\s*side\b", post)):
            continue
        if re.search(_NEG_LOOKBACK, low[:_neg_off + m.start()]):
            continue
        if m.end() > dest_end:
            dest, dest_end = m.group(1), m.end()
    if dest is None:
        return "close_gripper"
    tail = tail_all[dest_end:]
    give = bool(re.search(r"\b(?:open|give|hand|release)\b", tail)) and not \
        re.search(r"\b(?:do\s*n[o']?t|do\s+not|never|dont)\s+"
                  r"(?:open|give|hand|release)\b", tail)
    return f"redirect_{dest}_release" if give else f"redirect_{dest}"


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


# 2026-09-25 (SPEC C4): RESET emits the reset intent (it was stop, which
# froze the arm); PAUSE CONSENT names the executor's ROBOT TASK STATE as the
# authority (announcements are history); GARBLED input is the tool action
# 'unclear' with no text (this node speaks the line alone); new rules TALK
# WHILE ACTING (story only when asked in the same utterance) and RELEASE SIDE
# (a side at the release question is a drop, never an open).
# 2026-09-25 (Majd's F1): SAME-KIND OBJECTS — two of the same kind and no
# choice named: emit the pick without a position (the robot asks "leftmost or
# rightmost, or point"); the model asked itself, naming brands and cap
# colours ("Which vitamin tube—the Vitamin B12 (purple cap) or …?"). The
# scene rule no longer invites colours for disambiguation.
# 2026-09-25 (Majd's live test): RELEASE QUESTION — "Should I open the
# gripper?" is yes/no, never "Left or right?" (the model asked it twice after
# "Yes, open the gripper with this option."); at "Should I go back home?"
# definitely/absolutely = go_home and a named side = drop_left/drop_right.
SYSTEM_PROMPT = """You are Morty, a robot assistant at Hochschule Reutlingen.
Reply in the language the user spoke: German questions get German answers,
everything else gets English. Never mix languages in one reply. Keep replies
SHORT - one sentence max, two if necessary. The user hears you through a
speaker, so long replies cause feedback.

You are HAPPY TO CHAT about anything: jokes, general knowledge, science,
history, robotics theory, AI concepts, philosophy, your day, weather small
talk, anything. Just answer like a friendly assistant. DO NOT decline with
"I'm just a robot arm" / "that's outside my wheelhouse" / "I'm only built
for robot tasks". You are BOTH a friendly chatbot AND a robot operator —
both roles are valid. Only ROBOT ACTIONS (below) trigger the
intent_extractor tool; everything else is plain conversation, give an
actual answer.

If asked "who are you" / "introduce yourself" / "wer bist du", reply:
"I'm Morty, a robotics researcher at the University of Reutlingen."

If the user answers YES to your greeting ("Want to know what I can do?"),
or asks what you can do: give a SHORT menu, max two sentences, e.g.:
"I can see the objects on the table, grasp the one you name, carry it
left, right or home, and hand it to you. Just tell me what to pick." Do
not list every command word — keep it inviting, not a manual.

ACTIONS - when the user asks for a robot action, call the intent_extractor
tool with the right action_intent. Valid action_intent values:
pick, place, pick_and_place, confirm_drop, release, open_gripper,
close_gripper, stop, go_home, go_to_ready, go_to_start, sleep,
look_left, look_right, look_up, drop_right, drop_left,
enable_multiview, disable_multiview,
start_driver, stop_driver, start_pipeline, stop_pipeline, start_all,
stop_all, reset, unclear.

SYSTEM CONTROL - "start the driver", "start the pipeline", "start
everything" / "start the system" -> start_driver / start_pipeline /
start_all. Same with stop. "Stop everything" / "shut it all down" /
"stop the whole system" (even when the user lists components) -> call
the tool TWICE: first stop (halts motion instantly), then stop_all
(shuts down pipeline and driver). The Morty GUI executes these. Use the
LIVE SYSTEM STATUS to answer whether something is already running
instead of firing a redundant start.

TELEGRAPHIC GERMAN - short noun-only German is a command, not noise:
"Greifbare Objekte?" -> list the visible objects (auf Deutsch).
"Greifbare Objekte, Vitamin C" / bare "Vitamin C" while objects are
visible -> grasp the vitamin_c. Same pattern for any known object name.
Do NOT treat these as unclear.

SPEECH MISHEARINGS - the transcript comes from speech recognition and
"grasp" is often misheard. When the sentence names a known object and the
verb looks like "cross", "crasse", "grass", "glass", "kras", or similar,
treat it as "grasp" (= pick) and just do it - do NOT ask what "cross"
means. Same for "grip"/"grab". Only ask when the OBJECT is unclear.

OBJECTS - the detector knows these labels (use one as object_intent):
tape_dispenser, cd_case, toy_faucet, kitchen_towel, nutmeg_jar,
retail_box, small_bottle, wooden_maze_puzzle, vitamin_c.
Aliases: "tube"/"vitamin tube"/"vitamin C" -> vitamin_c;
"towel" -> kitchen_towel; "puzzle" -> wooden_maze_puzzle;
"jar" -> nutmeg_jar; "bottle" -> small_bottle; "faucet" -> toy_faucet;
"box" -> retail_box; "dispenser" -> tape_dispenser; "case" -> cd_case.

QUALIFIERS - if the user says "leftmost", "rightmost", "left", "right",
"middle", "biggest", "smallest", put it in object_attributes.position.
If they COUNT ("the second tube", "third one from the right"), set
ordinal (1-based) and ordinal_from ("left" unless they say from the
right). If they name a COLOUR ("the red one"), set colour - it also
disambiguates between several objects of the same type.

VITAMIN TUBES - all three products detect as the same object
"vitamin tube"; they differ ONLY by CAP COLOUR. When the user names a
product, set object_intent="vitamin tube" AND the matching colour:
  "Vitamin C" (or "the C one")            -> colour=yellow
  "Vitamin B" / "B12"                     -> colour=red
  "Multivitamin" / "Multi-Mineral" /
  "the multi one"                         -> colour=blue
If only ONE vitamin tube is visible, any of these names just picks it.

DESCRIBING THE SCENE - name objects by their PRODUCT NAME ONLY:
"a Vitamin C, a Vitamin B12, a small bottle, a tape dispenser". Do NOT
volunteer cap colours or long attribute lists (Majd 2026-08-26: "it
doesn't need to say all this information"). Mention a cap colour or a
position ONLY when the user asks for it.

SAME-KIND OBJECTS - when the scene shows MULTIPLE INSTANCES of the kind
the user asked for (e.g. two vitamin tubes) and the user did not say
which one, do NOT ask yourself and do NOT tell them apart by product name
or cap colour: emit the pick WITHOUT position or colour. The robot itself
asks "leftmost or rightmost, or point at it".

WHICH-ONE ANSWER - when the robot asked which one ("I see two vitamin
tubes. Could you say leftmost or rightmost ..."), that pick was REFUSED and
nothing runs. The user's next words naming a place ("Right most.", "The
rightmost vitamin.", "the left one", "the middle one") ARE the answer: emit
the pick of that object again WITH position (right -> rightmost, left ->
leftmost, middle -> middle) and no text. Never answer "Left or right?", never
ask them to confirm, never say it is grasping before you sent the pick.

AFTER EMITTING AN ACTION INTENT - reply with NOTHING (empty text). The
status announcements narrate everything; a verbal acknowledgment on top
is noise ("On it. On it. On it." — Majd, 2026-08-26). Speak after an
intent ONLY when you must add real information the announcements will
not say (e.g. answering a question asked in the same breath). NEVER say
"Done"/"Finished" — the robot has only STARTED.

START IS NOT STOP - "start the whole system" / "start everything" emits
ONLY start_all — NEVER a stop/stop_all first (2026-08-26: a stop intent
sneaked in before start_all and interrupted a running system). stop_all
is only for explicit shutdown requests. If the system is already
running, start_all is harmless — the readiness announcement will simply
confirm "the system is running and we are ready to grasp."

NO ALIAS FEATURE - you CANNOT "remember that X means Y" and must never
offer to. Utterances like "class for me is a vitamin tube", "cross for
me the small bottle", "grass/cast/casform for me ..." are SPEECH
RECOGNITION ERRORS for "grasp for me ..." — treat them as pick commands
for the named object, never as definitions. If the user says you
misheard, apologize in 5 words max and emit the pick intent.

DISAMBIGUATION ANSWERS - when the robot asked "leftmost or rightmost?",
map the user's answer to a position qualifier: leftmost/left -> leftmost,
rightmost/right -> rightmost, "the middle one" with 3+ objects ->
middle. With only TWO objects "middle" is impossible — ask exactly once
"left or right?" instead of emitting a doomed intent. Never reply just
"On it" to a disambiguation answer without emitting the qualified pick.

CHANGE OF TARGET - while a disambiguation question ("leftmost or
rightmost?", "which one?") is unanswered, NOTHING is being picked — the
robot is idle, waiting. If the user then asks for a DIFFERENT object
("grasp the tape dispenser", "dismiss the vitamin tube and take the
tape dispenser", "not that one, the X"), that REPLACES the previous
request: emit the pick intent for the NEW object immediately, no
questions about the old one. Never refuse a new pick because an old
question is unanswered. Only an object physically IN THE GRIPPER blocks
a new pick (then ask to drop it first).

RESET - "reset the pick" / "reset" / "start over" all mean: emit the
reset intent (the robot ends the pick and goes home, keeping any held
object; the selection is cleared). Never ask what "reset" means.

POSITION GLOSSARY - the arm has these DISTINCT named positions; never
mix the families:
  home                      <- "home", "go back", "start position"
  detect_left / detect_right / detect_higher  <- LOOK/observation poses:
        "look left/right/up", "detect left/right", "detection left",
        "scan left/right", "view left/right"  -> look_left / look_right /
        look_up intents
  drop_right_side / drop_left_side            <- DROP poses:
        "drop right/left", "drop position right/left" -> drop_right /
        drop_left intents
  sleep                     <- "sleep", "park", "stow"
A detect/look pose is NEVER a drop pose: detect_right != drop_right_side.
When the Robot-arm-state message names the current pose, treat the user
as "already there" ONLY if the requested position maps to EXACTLY that
pose name; otherwise emit the move intent without arguing. If a position
request is ambiguous between families, pick the family whose word the
user used ("detect"/"look" -> look pose, "drop" -> drop pose) instead of
asking.

DROP DESTINATION - if the user names a drop side ("drop right", "on the
left", "home"), put it in object_attributes.destination as drop_right,
drop_left, or home. If they say "drop" with no side, ask "Left or right?"
and do NOT emit an intent — never at "Should I open the gripper?" (see
RELEASE QUESTION).

PAUSE DESTINATION COMPOUND - if the destination answer ALSO says to open/
release/give/hand it over (and possibly go home) in the same sentence, in
ANY wording or order ("drop it right, open the gripper and go back home",
"drop it on the right side and give me the object then go back home",
"right side, open it, after that go home", "leg es rechts ab und dann nach
Hause"), emit ONE confirm_drop with destination AND the tool's `then`
field: "release_home" (open + straight home, no more questions) or
"release" (open on arrival, then the robot asks about home). The same
`then` goes on pick_and_place ("pick the bottle, put it on the right and
give it to me") and on drop_right / drop_left / go_home while holding. The
robot then never pauses again for the parts you already authorized.

NO WORDS WITHOUT ACTION - while any robot question is open (destination,
release, home), your reply MUST include the intent emission. Saying
"I'm dropping it on the right" WITHOUT emitting confirm_drop leaves the
robot frozen forever — the words do nothing, only the intent moves it.

PAUSE DESTINATION - while "Where should I drop it?" is open, ANY phrase
naming a side IS the answer: "move to the right side", "the right
position", "right", "rechts", "on the left" -> drop_right / drop_left /
home intent. Do NOT re-ask the question when the user already named a
side, in any wording.

RELEASE + HOME - if the user, at the release pause, combines opening and
going home in ONE reply — ANY wording: "and", "then", "after that",
"open the gripper, then go back home", "of course, open it and head
home" — emit exactly ONE intent: release_home. NEVER emit release
followed by go_home as two intents; that makes the robot ask again.
Emit action_intent release_home (opens, then goes home with NO extra
question). A plain "open"/"release" answer opens and then the robot will
ask "Should I go back home?" — when THAT question is open, "yes" /
"definitely" / "absolutely" / "go home" means emit go_home, "no" / "stay" /
"wait" means emit stay (the robot then stays where it is), and "go to the
left / right (side)" means drop_left / drop_right (it moves there instead
of home). Only a phrase naming BOTH sides ("the right, left side") may ask
which one.

RELEASE QUESTION - "Should I open the gripper?" is a YES/NO question: ANY
consent phrasing ("yes", "open", "opens the gripper", "give it to me",
"yes, open the gripper with this option") = open_gripper; a named side =
drop_left / drop_right; NEVER answer it with "Left or right?".

PAUSE CONSENT - when the robot is holding an object and waiting ("Where
should I drop it?" / "Should I open the gripper?"), ANY consent phrasing
("yes", "go ahead", "give me the object", "gib es mir", "let it go")
means: CALL the tool with open_gripper (or the named destination). Never
answer a consent with the same question again. The ROBOT TASK STATE is
AUTHORITATIVE; announcements are history. With no pick and an empty
gripper, no question is open. While the destination question is open,
emit the destination intent (drop_right / drop_left / home); never claim
you are not holding anything while that question is open.

DROP MEANS MOVE - drop_left / drop_right / confirm_drop / go_home only CARRY
the held object to that pose; it is released ONLY by open_gripper / release
(or a "then" of release). After "drop it on the right side" the robot is
STILL HOLDING the object until the gripper opens — the ROBOT TASK STATE
says what the gripper holds.

SENT IS NOT DONE - a command in the conversation (yours or the robot's fast
path) was only SENT; the robot may have refused or failed it (e.g. "I'm
still holding the X — where should I put it first?"). Never say a pick or a
move is running, finished or "already" happening unless the ROBOT TASK STATE
says so this turn; when it says idle, nothing is running — send the command
again or ask.

OPERATOR SAYS EMPTY - the gripper sensor can read "holding" when the
fingers closed on nothing. When the ROBOT TASK STATE says holding and the
user says the gripper is empty ("you're not holding anything", "you missed
it", "there's nothing in your hand"): send NO command, never argue and never
cite sensors — ask exactly once: "Are you sure? Then I'll open the gripper
and forget the <object>." If the user confirms (yes / sure / I'm sure),
send not_holding and write no text. If they say no, keep everything as it
is.

RELEASE SIDE - at "Should I open the gripper?", "drop it on the
right/left" -> drop_right / drop_left, never open_gripper / release.

COMPOUND PICK-AND-PLACE - "pick the X and put it on the right" -> ONE
pick_and_place intent with destination set. Don't split it into pick+place.

MULTIPLE SEPARATE ACTIONS - "go home and look left" -> call the tool
ONCE PER ACTION in order.

GARBLED INPUT - if the transcript is nonsensical (random numbers,
broken grammar, single junk word), do NOT fire a robot intent: call the
tool with unclear and write NO text.

TALK WHILE ACTING - only when one utterance asks for an action and for
talk ("grasp the bottle and tell me a story"): call the tool FIRST with
talk=true, then the story (at most 6 short sentences — the one exception
to SHORT); never talk for a plain command.

REPLY RULES after firing an action:
- Single intent: empty text reply (the system speaks its own narration).
- Single intent + a side joke/comment: reply with JUST the joke/comment.
- Multiple intents: ONE combined plan sentence, e.g. "Going home, then
  looking left." (this replaces the per-intent narrations).

NEVER duplicate the action name in text ("Picking the towel.") - the
system already says that.

SCENE QUESTIONS - "what do you see?": if you received a "Scene currently
shows: ..." developer message this turn, answer using ONLY that. If you
did NOT, say "I'm not getting detections - make sure the grasp pipeline
is running." Do NOT recite the vocabulary list as if it were the scene."""


class ChatIntentNode(Node):
    def __init__(self) -> None:
        super().__init__("communication_llm_chat_node")

        self.declare_parameter("model", os.environ.get("MORTY_LLM_MODEL", "gpt-4o-mini"))
        self.declare_parameter("intent_topic", "/robot/intent")
        self.declare_parameter("assistant_topic", "/robot/assistant_text")
        self.declare_parameter("terminal_input", True)
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("speak_action", "/speak")
        self.declare_parameter("speak_voice", "")
        self.declare_parameter("speak_interrupt_current", True)
        self.declare_parameter("speech_input_enabled", True)
        self.declare_parameter("transcript_topic", "/speech/transcript")
        self.declare_parameter("typed_transcript_topic", "/speech/transcript_typed")
        self.declare_parameter("mute_topic", "/speech/muted")
        self.declare_parameter("ai_speaking_topic", "/speech/ai_speaking")
        # Optional one-time greeting spoken when the node is fully ready.
        # Empty string disables. Default introduces Morty in German with a
        # short fact about Hochschule Reutlingen.
        # Short by default — the long Reutlingen / VISIR pitch is only
        # delivered when the user explicitly asks who Morty is, where he
        # is, or something about the university. The SYSTEM_PROMPT has the
        # full facts so the LLM can quote them on demand.
        self.declare_parameter(
            "startup_greeting",
            # Majd 2026-08-31: invite instead of an open question — short,
            # so it does not annoy on every run. A "yes" falls through to
            # the LLM (no pause is open at startup, and "yes" passes the
            # junk filter), which answers from the system prompt.
            "Hi! I'm Morty.",
        )
        # Minimum delay before attempting the greeting. The actual greeting
        # is gated on the Speak server being ready (Kokoro takes ~6s to
        # load), so this is a floor, not a fixed wait.
        self.declare_parameter("startup_greeting_delay_s", 2.0)
        # Mic-mute behaviour during TTS playback. With True (default), the
        # mic is silenced from "AI starts speaking" until "AI stops speaking"
        # so the recognizer can't pick up the AI's own voice. With False,
        # the mic stays open and you can interrupt the AI mid-sentence —
        # but the echo filter below is what makes that safe (it drops
        # transcripts that match what the AI just said).
        self.declare_parameter("mute_during_tts", True)
        # Echo suppression: ignore an incoming transcript if it matches
        # a chunk of text the AI spoke in the last `echo_window_s` seconds.
        # Match is fuzzy substring overlap on tokens. Set to 0 to disable.
        # Long greetings (~12s) plus AEC residual mean an 8s window misses
        # the tail of the AI's own speech and re-fires intents. 25s gives
        # plenty of overhead.
        self.declare_parameter("echo_window_s", 25.0)
        # Lower threshold = more aggressive echo dropping. 0.45 means
        # "if 45% of the words the mic heard appear in the AI's recent
        # utterance, treat as echo". Real user commands rarely cross
        # this against the AI's reply because they introduce new words.
        # Combined with the affirmation/short-control-word allowlists,
        # this catches the quiet AEC residue that 0.60 was letting
        # through while still passing real voice input.
        # 0.85, was 0.45: common words (hi/morty/how/can/you) push ANY
        # greeting-adjacent sentence past 0.45 — a real user command was
        # dropped at 62 % (Majd, 2026-08-12, and he set the bar: "at
        # least 85%"). True mic echo of a full sentence scores ~1.0.
        # 2026-09-23 (Majd: "text comparison — leave only what is very
        # reliable"): the word-overlap echo rule is OFF (0.0 disables
        # _is_likely_echo). speech_input_node's speaker fingerprint decides
        # who spoke; this rule vetoed real questions ("What kind of one?").
        self.declare_parameter("echo_min_overlap", 0.0)
        # Hard cooldown after the AI speaks: any non-affirmation
        # transcript arriving within this many seconds is treated as
        # echo. Longer = more aggressive. Was 1.5s; 2.5s catches the
        # tail of long TTS sentences where AEC residue arrives slowly.
        self.declare_parameter("echo_cooldown_s", 2.5)
        # Overlap-based echo drops only apply while the TTS is still
        # PLAYING plus this tail (mic capture closes ~silence_ms after
        # the sound stops + ~1 s Whisper; 2026-09-25: silence_ms is 800 ms
        # now, 500 ms for short answers, so 0.8 + 1.0 s still fits in the
        # 2.0 s tail). Beyond it, even a 100%-overlap
        # transcript is the USER quoting Morty's words back — which is
        # how humans talk ("the pipeline IS running, how?"). Majd,
        # 2026-08-21: "the echo should be only in the same seconds that
        # he speaks."
        self.declare_parameter("echo_active_tail_s", 2.0)
        self.declare_parameter("auto_execute_grasp", False)
        self.declare_parameter("execute_grasp_action", "/execute_grasp")
        self.declare_parameter("best_pose_topic", "/graspgen/best_pose_base_link")
        self.declare_parameter("candidates_topic", "/graspgen/candidate_poses_base_link")
        self.declare_parameter("grasp_pose_max_age_s", 5.0)
        self.declare_parameter("grasp_timeout_s", 90.0)
        # Scene awareness: subscribe to detections so the LLM knows what is visible.
        self.declare_parameter("scene_awareness", True)
        self.declare_parameter("detections_topic", "/yolo/all_detections")
        self.declare_parameter("scene_max_age_s", 3.0)
        # The grasp pipeline used to own its own TTS for selection prompts
        # ("Pick from: kitchen_towel, retail_box."). Now it publishes those
        # prompts as plain strings on this topic and we speak them through
        # the same Kokoro pipeline as the rest of Morty's voice.
        self.declare_parameter(
            "selection_prompt_topic", "/grasp_pipeline/selection_prompt"
        )

        self.model = str(self.get_parameter("model").value)
        # 2026-09-25 (Majd's G1): +1 per intent this node publishes and per
        # line it speaks (_note_act); _hush_and_merge compares it.
        self._act_serial = 0
        self._prev_input: dict | None = None
        self._turn_text = ""
        self._merged_text = ""
        self.intent_pub = _ActCountingPublisher(self.create_publisher(
            String,
            str(self.get_parameter("intent_topic").value),
            10,
        ), self._note_act)
        # Growing reply while sentences stream to speech (2026-09-23): the
        # GUI extends the current "Morty:" line from this and never shows a
        # reply sentence as a "(status)" line; /robot/assistant_text still
        # carries the finished reply once.
        self.assistant_partial_pub = self.create_publisher(String, "/robot/assistant_partial", 5)
        self.assistant_pub = self.create_publisher(
            String,
            str(self.get_parameter("assistant_topic").value),
            10,
        )
        # Filtered user transcripts: only the ones that survived echo
        # suppression, the junk-transcript guard, and the RMS gate. The
        # mic visualiser subscribes here instead of /speech/transcript so
        # the debug window only shows what Morty actually acted on, not
        # the noisy raw stream of dropped/echoed Whisper output.
        self.accepted_transcript_pub = self.create_publisher(
            String, "/speech/accepted_transcript", 10
        )
        # Conversation state for the visualizer / debug UIs:
        # "idle"       — waiting for next user input
        # "thinking"   — LLM round-trip in flight
        # "responding" — TTS is playing the reply
        self.state_pub = self.create_publisher(String, "/chat/state", 10)
        self._chat_state: str = "idle"

        self._tts_enabled = bool(self.get_parameter("tts_enabled").value)
        self._speak_voice = str(self.get_parameter("speak_voice").value)
        self._speak_interrupt = bool(self.get_parameter("speak_interrupt_current").value)
        self._startup_greeting = str(self.get_parameter("startup_greeting").value).strip()
        self._startup_greeting_delay_s = float(
            self.get_parameter("startup_greeting_delay_s").value
        )
        self._greeting_timer = None
        self._mute_during_tts = bool(self.get_parameter("mute_during_tts").value)
        self._echo_window_s = float(self.get_parameter("echo_window_s").value)
        self._echo_min_overlap = float(self.get_parameter("echo_min_overlap").value)
        self._echo_cooldown_s = float(self.get_parameter("echo_cooldown_s").value)
        self._echo_active_tail_s = float(
            self.get_parameter("echo_active_tail_s").value)
        self._speak_client: ActionClient | None = None
        self._mute_pub = None
        self._ai_speaking_pub = None
        if self._tts_enabled:
            self._speak_client = ActionClient(
                self, Speak, str(self.get_parameter("speak_action").value)
            )
            self._mute_pub = self.create_publisher(
                Bool, str(self.get_parameter("mute_topic").value), 10
            )
            # Loudspeaker on/off, so speech_input_node can cut Morty's
            # own words out of a mixed transcript BY TIME. Published
            # around every Speak goal regardless of mute_during_tts —
            # with the mic open (the default, for barge-in) this is the
            # only signal that says when the echo was physically
            # possible.
            self._ai_speaking_pub = self.create_publisher(
                Bool,
                str(self.get_parameter("ai_speaking_topic").value),
                10,
            )

        # Echo suppression: a small ring of (timestamp, lowercased text) entries
        # for things the AI recently spoke. Used to drop microphone transcripts
        # that look like the AI hearing itself. Populated both by our own
        # _speak() calls AND by every utterance broadcast on /speech/ai_said
        # (so we cover intent_router narrations too).
        self._recent_ai_speech: list[tuple[float, str]] = []
        # Raw text of the last few things spoken aloud through the robot's
        # voice (status-layer narrations AND own replies). Injected into
        # the LLM context every turn so it KNOWS what was announced and
        # never puzzles over its own echo ("starting Greif", 2026-08-12).
        self._recent_spoken: list[tuple[float, str]] = []
        # Why the last transcript was dropped — published on
        # /speech/rejected so the GUI can SHOW it (Majd, 2026-08-12).
        self._last_echo_reason: str = "echo of my own voice"
        self._rejected_pub = self.create_publisher(
            String, "/speech/rejected", 10
        )
        self.create_subscription(
            String,
            "/speech/ai_said",
            self._on_ai_said,
            10,
        )

        self._chat_lock = threading.Lock()
        # Generation counter for voice transcripts. Each new transcript bumps
        # this; an in-flight worker thread compares the value at the start of
        # its LLM call to the current value before publishing. If a newer
        # transcript arrived in the meantime, the worker drops its result.
        # Effect: latest user input wins, so a stack of slow LLM calls can't
        # pile up commands the user has already moved past.
        self._voice_generation = 0
        self._voice_gen_lock = threading.Lock()
        # Interrupted answers (2026-09-24, Majd: "when I talk while he is
        # talking he hears me, but keeps answering the old question").
        # Every LLM turn gets a number. A barge-in, or a newer utterance
        # accepted as input, hushes the answers up to the current one: its
        # stream is stopped, nothing more of it is spoken, its sentences
        # already queued at the speak server are cancelled, a robot command
        # it had not dispatched yet is not run, and the next turn tells the
        # model where it was cut off.
        self._answer_seq = 0
        self._hushed_seq = 0
        self._answer_running = False
        self._answer_goals: list = []            # [(answer_seq, ClientGoalHandle)]
        self._answer_goals_lock = threading.Lock()
        self._interrupt_note = ""

        # Memory of the last extracted intent JSON; used to satisfy
        # "do it again" / "repeat" without going through the LLM.
        self._last_intent_json: str | None = None
        # Per-turn flag: set True after _handle_intent_tool_call publishes
        # an intent. _handle_user_text_locked checks this and skips
        # speaking the LLM's slow final reply (we already gave fast
        # audio feedback via QUICK_NARRATION).
        self._intent_fired_this_turn = False

        if bool(self.get_parameter("speech_input_enabled").value):
            self.create_subscription(
                String,
                str(self.get_parameter("transcript_topic").value),
                self._on_transcript,
                10,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("typed_transcript_topic").value),
                self._on_typed_transcript,
                10,
            )

        # ── grasp execution ──────────────────────────────────────────────
        self._auto_execute_grasp = bool(self.get_parameter("auto_execute_grasp").value)
        self._grasp_pose_max_age_s = float(self.get_parameter("grasp_pose_max_age_s").value)
        self._grasp_timeout_s = float(self.get_parameter("grasp_timeout_s").value)
        self._latest_pose: PoseStamped | None = None
        self._latest_candidates: PoseArray | None = None
        self._latest_pose_received_at: float = 0.0
        self._grasp_lock = threading.Lock()
        self._active_grasp_handle = None

        self._execute_grasp_client = ActionClient(
            self,
            ExecuteGrasp,
            str(self.get_parameter("execute_grasp_action").value),
        )

        # The grasp pipeline publishes these latched, so use TRANSIENT_LOCAL
        # so we receive whatever was last published when we subscribe late.
        latched = _latched_qos()
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("best_pose_topic").value),
            self._on_best_pose,
            latched,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter("candidates_topic").value),
            self._on_candidates,
            latched,
        )
        # Readiness (2026-09-18, Majd: "tell me when it is ready to grasp,
        # only for the first run, and while it loads say please wait"):
        # the pipeline's precompute publishes /graspgen/readiness every second.
        self._said_loading = False
        self._said_ready = False
        self.create_subscription(String, "/graspgen/readiness", self._on_readiness, 10)
        # The executor's open question (2026-09-18: "Should I go back home?"
        # + "Yes, go back home now." went to the LLM, which answered "you're
        # already home" while the arm stood at the drop). While a question
        # is open, an affirmative IS the answer and skips the LLM.
        self._open_question = ""     # "awaiting_home_confirm" | "awaiting_release" | ""
        self.create_subscription(String, "/grasp_executor_lin_llm/interactive_status",
                                 self._on_interactive_status_question, 10)

        # Proactive scene narration: when new objects appear and the user
        # has been idle, Morty volunteers what he sees and asks if the
        # user wants something picked. Disabled with proactive_scene=false.
        # OFF by default. The "I can see X, Y, Z..." line was firing on its
        # own and stepping on the user. The LLM still gets the scene as
        # context every turn, so it can volunteer details when asked
        # ("what do you see?"); it just won't blurt the list unprompted.
        self.declare_parameter("proactive_scene", False)
        # User must be idle this long before Morty speaks up. Avoids
        # interrupting the conversation.
        self.declare_parameter("proactive_idle_s", 12.0)
        # Don't narrate the same scene more than once per cooldown window.
        self.declare_parameter("proactive_cooldown_s", 60.0)
        self._proactive_enabled = bool(self.get_parameter("proactive_scene").value)
        self._proactive_idle_s = float(self.get_parameter("proactive_idle_s").value)
        self._proactive_cooldown_s = float(
            self.get_parameter("proactive_cooldown_s").value
        )
        self._last_user_input_at = self.get_clock().now().nanoseconds / 1e9
        self._last_proactive_at: float = 0.0
        self._last_proactive_signature: str = ""
        if self._proactive_enabled:
            self.create_timer(2.0, self._proactive_tick)

        # Held-object memory: subscribe to the latched topic the router
        # publishes after each pick. We feed the value into the LLM as a
        # developer message so phrases like "drop it on the left" /
        # "place it back home" can resolve "it" without the user having
        # to repeat the object name.
        self._held_object_label: str = ""
        self.create_subscription(
            String,
            "/robot/held_object",
            self._on_held_object,
            latched,
        )
        # 2026-09-24 (Majd: "why does he still think he is holding an
        # object — it happens a lot"). The announcements the LLM reads as
        # robot status were every line Morty spoke, its OWN answers
        # included, and the cycle state followed only phrases it knew: after
        # the release, "Opening the gripper." and "I'm at the home position.
        # What's next?" matched none, the state stayed "HOLDING … waiting for
        # release confirmation", and the next answer ("I'm holding the
        # Multivitamin tube — drop it first") went into the feed and became
        # the evidence for the one after. Now the model's own lines stay
        # out of the feed, and the state follows the gripper topic above and
        # the executor's cycle end below.
        self._own_answer_lines: dict[str, float] = {}
        self.create_subscription(
            String, "/grasp_executor_lin_llm/cycle_done", self._on_cycle_done_state, 10)
        # 2026-09-25 (Majd: "one owner of task state = the executor"): the
        # executor's latched task state replaces every guess this node made
        # from spoken announcements — the cycle state, the open question and
        # the held object follow it while it is known. Until the first
        # sample arrives (old executor) everything behaves as before.
        self._task: dict | None = None
        self._task_qkey: tuple = (None, 0)      # (boot, question_id) of the last question
        self.create_subscription(
            String, "/grasp_executor_lin_llm/task_state", self._on_task_state,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # 2026-09-25: when the operator started speaking (review M1: it was
        # the speech end), per transcript (speech_input_node publishes it
        # right before /speech/transcript). A "yes" answers the open
        # question only if the question opened before that moment
        # ("fresh"); a yes begun before it makes Morty ask again (review H3).
        # 2026-09-25 (review M1): [received wall, text, speech_end,
        # speech_start (None if the meta has none), used].
        self._utt_meta: list[list] = []
        self._utt_meta_lock = threading.Lock()
        # 2026-09-25 (review M1): a meta was ever received — the speech node
        # publishes them; without any (an old node, Vosk) the minimal
        # layer cannot judge freshness and the legacy layers answer.
        self._utt_meta_seen = False
        self.create_subscription(String, "/speech/utterance_meta", self._on_utterance_meta, 10)
        # 2026-09-25: C7 — the router speaks the destination line once the
        # executor accepted it, so this node must know a pick of its own is
        # still in flight.
        self._pick_sent_mono = 0.0
        self._cycle_done_mono = 0.0
        # 2026-09-25 (review X2): a pick refused before the executor started
        # a cycle (the selector's "failed|0|…") ends the pending-pick clause
        # of _router_narrates too — it never produces a cycle_done.
        self.create_subscription(
            String, "/grasp_pipeline/selection_result", self._on_selection_result, 10)
        # 2026-09-25 (review H1): +1 at every stop / reset this node sends
        # (grammar or LLM). A transcript remembers the value it arrived
        # with; a turn that waited on _chat_lock across a stop is dropped
        # instead of running after it ("grasp the bottle" … "Stop!" must not
        # start the grasp once the stop's turn lets go of the lock).
        self._stop_epoch = 0
        # 2026-09-25 (review Y6/L3): every Speak goal gets a serial; answer
        # goals sent at or before _cut_serial are cancelled when the speak
        # server accepts them late (an 'unclear' turn cuts what it streamed
        # without hushing the turn, whose UNCLEAR_LINE must still play).
        self._speak_serial = 0
        self._cut_serial = 0
        # 2026-09-25: per-turn flags of the LLM turn machinery (SPEC C8);
        # reset at the start of every turn.
        self._talk_turn = False
        self._talk_seq = -1
        self._unclear_turn = False
        self._real_intent = False
        self._withheld_this_turn: set[str] = set()
        self._interrupt_line_turn = False
        self._stream_mark = 0
        # 2026-09-25 (Majd, live: "why does it still not know the position
        # it is in?"): utterances the instant paths handled, with what they
        # did, waiting to enter the LLM's history at its next turn — see
        # _note_instant / _flush_instant_notes.
        self._instant_notes: list = []
        self._instant_notes_lock = threading.Lock()
        self._instant_seq = 0
        self._hush_why = ""
        # 2026-09-25: the four voice switches (SPEC C1), live-settable.
        self.declare_parameter("voice_fast_layers", "minimal")      # minimal | legacy
        self.declare_parameter("voice_ext_dest", True)
        # 2026-09-28 (Majd, live: "tell me a story because I'm sick" — one
        # sentence, then the grasp's question cut the other four): off by
        # default; a talk the operator asked for is finished and the robot's
        # question is spoken after it (the speak server queues it).
        self.declare_parameter("story_yields_to_question", False)
        self.declare_parameter("gwdg_stall_s", 3.0)
        self._voice_fast_layers = "minimal"
        self._voice_ext_dest = True
        self._story_yields = False
        for _name in _VOICE_PARAMS:
            self._apply_voice_param(_name, self.get_parameter(_name).value)
        self.add_on_set_parameters_callback(self._on_set_voice_params)
        # What the user is currently pointing at via mediapipe (or ""
        # when no finger is over an object). The grasp pipeline emits
        # this only while it's running its selection cycle, so it tells
        # us both (a) selection is active right now and (b) which object
        # the finger is hovering over. The LLM uses this to resolve
        # "pick this one" / "this object" to a specific label.
        self._pointed_object_label: str = ""
        self.create_subscription(
            String,
            "/yolo/pointed_object",
            self._on_pointed_object,
            10,
        )
        # Current arm pose label — latched, updated whenever the router
        # successfully completes a named-pose move. Fed to the LLM each
        # turn so it can answer "where are you?" and avoid issuing
        # redundant moves to the same pose.
        self._current_pose_label: str = ""
        # Joint-space nearest-pose lookup state. Loaded once at startup
        # so when the arm is mid-cycle (current_pose_label is empty), we
        # can still tell the LLM "the arm is closest to <pose>" by L2
        # distance across all 6 joints.
        self._named_poses: dict[str, list[float]] = {}
        self._named_pose_joint_names: list[str] = []
        self._latest_joint_positions: dict[str, float] = {}
        # Default points at the file shipped with ur3e_fast_observation; can
        # be overridden by passing observation_poses_file:=... at launch.
        from ament_index_python.packages import get_package_share_directory
        _default_poses = os.path.join(
            get_package_share_directory("ur3e_fast_observation"),
            "config", "observation_poses.yaml",
        )
        self.declare_parameter("observation_poses_file", _default_poses)
        self._load_named_poses(
            str(self.get_parameter("observation_poses_file").value)
        )
        # NOTE: /joint_states is SAMPLED, not subscribed to — see the
        # liveness timer a few lines below and _sample_one_message().
        # Live system awareness (2026-08-11): Morty used to GUESS about the
        # driver/pipeline ("Yes, I'm running and ready!" with a dead driver).
        # These signals feed _system_status_summary(), injected per turn.
        self._last_joint_state_at = 0.0
        # Pendant program state — TRANSIENT_LOCAL publish-on-change; the
        # latched value is only meaningful while the publisher exists.
        self._program_running: bool | None = None
        self.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            self._on_program_running,
            latched,
        )
        # Spoken pendant reminder (Majd, 2026-08-12: "let it tell to press
        # play on the teach pendant"): once per not-running episode, when
        # the driver is up but the pendant program isn't playing.
        self._pendant_reminded = False
        self._program_stopped_since = 0.0
        self.create_timer(3.0, self._pendant_reminder_tick)
        # 2026-09-18: the router owns the "program running" verdict (the
        # latched flag can stay False after the External Control link
        # flapped; an active trajectory controller or a PLAYING dashboard
        # outrank it). Its verdict arrives every ~2 s; the raw flag above
        # is only the fallback while no router is running.
        self._router_status: dict = {}
        self._router_status_at = 0.0
        self.create_subscription(
            String, "/intent_router/robot_status", self._on_router_status, 1)
        # speech_input_node's speaker verdict (2026-09-23): when the voice
        # fingerprint already said "human", the text-overlap echo rule below
        # must not veto it ("What kind of one?" was dropped as 75 % overlap).
        self._speaker_verdict: tuple = (0.0, "", False)
        self.create_subscription(String, "/speech/speaker_verdict", self._on_speaker_verdict, 10)
        self.create_subscription(Bool, "/speech/barge_in", self._on_barge_in, 10)
        # /tf freshness — half-dead driver detection (2026-08-20).
        from tf2_msgs.msg import TFMessage
        self._tf_msg_type = TFMessage
        self._last_tf_at = 0.0
        # /joint_states (~494 Hz) and /tf are both high-rate, and this node
        # uses them ONLY for freshness stamps plus a coarse nearest-pose
        # distance. Two standing subscriptions cost ~1.0 of a CPU core
        # (2026-08-28), and that load is what made
        # scaled_joint_trajectory_controller abort a real grasp with
        # PATH_TOLERANCE_VIOLATED (0.2007 rad lag against a 0.2000 rad
        # tolerance). Both are now SAMPLED once per tick through momentary
        # subscriptions. The default period leaves 4x margin against the
        # 2.0 s driver_up window and 6x against the 3.0 s tf_alive window,
        # so a healthy driver can never look dead, while a driver that
        # really stops is still caught inside the same windows as before.
        # How many recent USER turns the LLM keeps (0 = unlimited).
        # 12 exchanges ≈ several minutes of dialogue — enough for "it"/
        # "that one" references and taught vocabulary, small enough that
        # calls stay ~2-3 s instead of climbing all afternoon.
        self.declare_parameter("llm_history_max_turns", 12)
        self._llm_history_max_turns = int(
            self.get_parameter("llm_history_max_turns").value)
        self.declare_parameter("liveness_sample_period_s", 0.5)
        self.declare_parameter("liveness_sample_timeout_s", 0.3)
        self._liveness_sample_period_s = float(
            self.get_parameter("liveness_sample_period_s").value
        )
        self._liveness_sample_timeout_s = float(
            self.get_parameter("liveness_sample_timeout_s").value
        )
        self.create_timer(
            max(0.05, self._liveness_sample_period_s), self._liveness_tick
        )
        self.create_subscription(
            String,
            "/robot/current_pose",
            self._on_current_pose,
            latched,
        )

        # Selection prompts coming out of grasp_pipeline. The pipeline used
        # to speak these itself (with its own embedded Kokoro/Vosk). We've
        # removed that and route the strings here instead, so all of
        # Morty's TTS is owned by one process and can't conflict with
        # itself over the same speaker / mic.
        self._last_selection_prompt: str = ""
        self._last_selection_prompt_at: float = 0.0
        self.create_subscription(
            String,
            str(self.get_parameter("selection_prompt_topic").value),
            self._on_selection_prompt,
            10,
        )

        # Scene awareness ----------------------------------------------------
        self._scene_awareness = bool(self.get_parameter("scene_awareness").value)
        self._scene_max_age_s = float(self.get_parameter("scene_max_age_s").value)
        self._scene_lock = threading.Lock()
        self._latest_scene: list[dict] = []
        self._latest_scene_at: float = 0.0
        if self._scene_awareness:
            self.create_subscription(
                String,
                str(self.get_parameter("detections_topic").value),
                self._on_scene_update,
                10,
            )

        if self._auto_execute_grasp:
            self.get_logger().warn(
                "auto_execute_grasp=TRUE — chat_node will fire ExecuteGrasp goals "
                "directly using whatever pose is currently on best_pose_topic. "
                "Do NOT enable this together with intent_router + grasp_pipeline; "
                "they would both try to drive the executor."
            )
        else:
            self.get_logger().info(
                "auto_execute_grasp=false — chat_node only publishes /robot/intent. "
                "Run intent_router_node + grasp_pipeline to actually pick. "
                "(Use -p auto_execute_grasp:=true only if you want chat_node to "
                "directly fire ExecuteGrasp without grasp_pipeline.)"
            )

        self.chat_agent = LLMService(tools=[INTENT_TOOL])
        self.chat_agent.add_developer_message(
            f"{SYSTEM_PROMPT}\n\nYou are currently running with model: {self.model}."
        )

        self._input_thread: threading.Thread | None = None
        if bool(self.get_parameter("terminal_input").value):
            self._input_thread = threading.Thread(target=self._terminal_loop, daemon=True)
            self._input_thread.start()
            self.get_logger().info("Terminal chat enabled. Type robot requests in this shell.")

        if (
            self._startup_greeting
            and self._tts_enabled
            and self._startup_greeting_delay_s >= 0.0
        ):
            # Tick every 1s starting after the configured delay; the
            # callback no-ops until the Speak server is ready, then
            # speaks once and cancels the timer.
            self._greeting_started_at = (
                self.get_clock().now().nanoseconds / 1e9
            )
            self._greeting_timer = self.create_timer(
                1.0,
                self._fire_startup_greeting,
            )

    def _fire_startup_greeting(self) -> None:
        # Don't speak until the configured delay has passed AND the Speak
        # action server is ready. Kokoro typically takes ~6s to load.
        # Times out after 30s to avoid an infinite tick if TTS never comes up.
        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - getattr(self, "_greeting_started_at", now)
        if elapsed < self._startup_greeting_delay_s:
            return
        if elapsed > 30.0:
            if self._greeting_timer is not None:
                # destroy, not cancel (2026-08-27): a cancelled rclpy timer
                # in Humble stays perpetually "ready" and turns spin() into
                # a hot loop — this one line burned a FULL CPU CORE.
                self.destroy_timer(self._greeting_timer)
                self._greeting_timer = None
            self.get_logger().warn(
                "Startup greeting gave up: Speak server never came up."
            )
            return
        if (
            self._speak_client is None
            or not self._speak_client.server_is_ready()
        ):
            return
        if self._greeting_timer is not None:
            self.destroy_timer(self._greeting_timer)  # not cancel — see above
            self._greeting_timer = None
        try:
            # Majd 2026-09-22: one sentence, not two back to back — the
            # greeting and the "please wait" line are spoken together.
            line = self._startup_greeting
            if not self._said_ready and not self._said_loading:
                self._said_loading = True
                line = (f"{self._startup_greeting.rstrip()} Please wait until the system "
                        "is ready. You can chat with me until we start.")
            self._speak(line)
            self.get_logger().info("Startup greeting sent.")
        except Exception as exc:
            self.get_logger().warn(f"Startup greeting failed: {exc}")

    def _terminal_loop(self) -> None:
        while rclpy.ok():
            try:
                user_text = input("User: ")
            except EOFError:
                return
            except KeyboardInterrupt:
                return

            user_text = user_text.strip()
            if not user_text:
                continue

            try:
                # 2026-09-25: typed terminal text takes the same route as
                # voice (stop/reset first, then the grammar or the LLM); it
                # was typed when it arrived, so that is its speech end.
                self._handle_voice_input(user_text, 0, typed_at=time.time(),
                                         epoch=self._stop_epoch)
            except Exception as exc:
                self.get_logger().error(f"LLM chat failed: {exc}")

    def _apply_voice_param(self, name: str, value: Any) -> str:
        """2026-09-25: apply one of the four voice switches (SPEC C1); returns
        "" when applied, else why the value is refused."""
        if name == "voice_fast_layers":
            v = str(value).strip().lower()
            if v not in ("minimal", "legacy"):
                return "voice_fast_layers must be 'minimal' or 'legacy'"
            self._voice_fast_layers = v
        elif name == "voice_ext_dest":
            self._voice_ext_dest = bool(value)
        elif name == "story_yields_to_question":
            self._story_yields = bool(value)
        elif name == "gwdg_stall_s":
            try:
                v = float(value)
            except (TypeError, ValueError):
                return "gwdg_stall_s must be a number"
            _llm_service.GWDG_STALL_S = max(0.0, v)
        return ""

    def _on_set_voice_params(self, params) -> SetParametersResult:
        """2026-09-25: the voice switches apply live (`ros2 param set`, the
        GUI); every other parameter passes untouched."""
        for p in params:
            if p.name in _VOICE_PARAMS:
                why = self._apply_voice_param(p.name, p.value)
                if why:
                    return SetParametersResult(successful=False, reason=why)
                self.get_logger().info(f"voice switch {p.name} = {p.value!r}")
        return SetParametersResult(successful=True)

    def _on_typed_transcript(self, msg: String) -> None:
        """Typed GUI input (2026-08-27): keyboard text cannot be an
        acoustic echo — route it past _is_likely_echo entirely. The
        adversarial review measured deliberate typed retries ("grasp the
        small bottle") being eaten as 'echo of my own voice'."""
        text = (msg.data or "").strip()
        if not text:
            return
        self._last_user_input_at = self.get_clock().now().nanoseconds / 1e9
        with self._voice_gen_lock:
            self._voice_generation += 1
            my_gen = self._voice_generation
            epoch = self._stop_epoch     # 2026-09-25 (review H1): the stop epoch it arrived in
        # 2026-09-25: typed text has no utterance meta; its arrival is its
        # "speech start" for the fresh-question rule.
        threading.Thread(
            target=self._handle_voice_input, args=(text, my_gen),
            kwargs={"typed_at": time.time(), "epoch": epoch}, daemon=True
        ).start()

    def _on_transcript(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        # STRIP before you DROP (2026-08-28). The mic catches the tail
        # of Morty's own narration and the operator's reply as ONE
        # Whisper transcript ("...Should I open the gripper? Yes, open
        # the gripper."); the whole-utterance filter below called that
        # 100 % echo and threw away the real command with it, so the
        # operator had to say everything twice. Remove just the echoed
        # fragment and carry on with what is left. Returns None for pure
        # echo, which then falls through to _is_likely_echo unchanged.
        # DISARMED 2026-08-28 after adversarial review. Two confirmed
        # defects, both WORSE than the bug this was meant to fix:
        #  P1 (critical) a transcript spanning TWO narrations had the
        #     first stripped and the SECOND handed to the FAST matchers —
        #     pure acoustic echo fired 'drop_right_release', i.e. the arm
        #     drove to drop-right and OPENED THE GRIPPER. Measured, not
        #     theoretical; it re-opened the narration->pick loop the echo
        #     filter exists to prevent.
        #  P2 (high) 4-word non-contiguous overlap was enough to strip, so
        #     real commands were destroyed: "grasp the vitamin tube again
        #     please" -> "again please"; and "open the gripper at the drop
        #     right position" -> "open the gripper" (releases in the wrong
        #     place). 5 false strips in a 300-pair corpus.
        # The helpers below are kept for a future, much narrower attempt
        # (require contiguity, re-run _is_likely_echo on the remainder,
        # and demand a command verb survives). Until then we DROP the
        # whole transcript as before: the operator repeats himself
        # occasionally, which is annoying but never moves the arm.
        _stripped = None
        if _stripped is not None:
            _dropped, _kept, _sc, _where = _stripped
            self.get_logger().info(
                f"ECHO stripped: dropped {_dropped!r} , kept {_kept!r} "
                f"({_where} fuzzy={_sc:.2f})"
            )
            # The echo is gone; what remains is the operator talking, so
            # it must NOT be re-judged by the whole-utterance filter.
            text = _kept
        # Drop transcripts that look like the AI hearing itself talk.
        # This MUST run before bumping the generation counter so an echoed
        # transcript can't displace a still-pending real user input.
        elif self._is_likely_echo(text):
            self.get_logger().info(f"Dropping echo-like transcript: {text!r}")
            try:
                self._rejected_pub.publish(String(
                    data=f"{self._last_echo_reason} :: {text}"
                ))
            except Exception:
                pass
            return
        # Mark the user as active so proactive scene narration backs off.
        self._last_user_input_at = self.get_clock().now().nanoseconds / 1e9
        # Bump generation; in-flight workers will see they've been superseded.
        with self._voice_gen_lock:
            self._voice_generation += 1
            my_gen = self._voice_generation
            epoch = self._stop_epoch     # 2026-09-25 (review H1): the stop epoch it arrived in
        # heard_at (2026-09-25): pairs this transcript with the utterance
        # meta published right before it (_speech_start_for).
        threading.Thread(
            target=self._handle_voice_input, args=(text, my_gen),
            kwargs={"heard_at": time.time(), "epoch": epoch}, daemon=True
        ).start()

    @staticmethod
    def _normalise_for_echo(text: str) -> set[str]:
        """Lowercase, strip ALL punctuation (including apostrophes) and
        return a set of word tokens.

        Crucially: we strip BOTH straight apostrophes (``'`` U+0027) and
        curly apostrophes (``’`` U+2019) the same way. The LLM tends to
        produce curly apostrophes ("I’ll wait"), while Whisper produces
        straight ones ("I'll wait"). Treating them differently broke the
        echo filter for short contraction-heavy lines, which then looped
        right back through the mic.
        """
        import re
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        return {tok for tok in cleaned.split() if tok}

    @staticmethod
    def _normalise_echo_str(text: str) -> str:
        """Normalise for the FUZZY echo comparison: lowercase, strip
        punctuation (straight and curly apostrophes fall out the same
        way, see :meth:`_normalise_for_echo`), collapse whitespace.
        Unlike ``_normalise_for_echo`` this keeps WORD ORDER (a plain
        string), which the sequence matcher below needs.
        Added 2026-08-27 for the fuzzy echo filter."""
        import re
        cleaned = re.sub(r"[^a-z0-9äöüß ]+", " ", text.lower())
        return " ".join(cleaned.split())

    @staticmethod
    def _fuzzy_echo_score(heard_norm: str, said_norm: str) -> float:
        """Similarity (0..1) between a mic transcript and one recent AI
        utterance, both already passed through ``_normalise_echo_str``.

        2026-08-27: the token-SET overlap below missed real garbled
        echo — TTS said "Starting grasp on the tape dispenser", Whisper
        re-heard it as "Start in grasp on the tape dispenser" (5/7 =
        0.71 set overlap, under the 0.85 bar), and the FAST-PICK regex
        fired ANOTHER pick — every narration looped into a new grasp.
        ``difflib.SequenceMatcher`` over WORD sequences separates the
        cases cleanly (measured): true garbled echo scores >= 0.77,
        reworded real commands ("grasp for me the small bottle" vs
        "Starting grasp on the small bottle") <= 0.67. Char-level
        ratios do NOT separate them (0.76 on that real command), which
        is why this matches words, not characters."""
        import difflib
        heard_w = heard_norm.split()
        said_w = said_norm.split()
        if not heard_w or not said_w:
            return 0.0
        # Full-utterance similarity on word tokens ("starting" vs
        # "start in" costs one token either way, order preserved).
        score = difflib.SequenceMatcher(None, heard_w, said_w).ratio()
        # DIRECTION MATTERS (Majd 2026-08-28: "he is detecting my words
        # also !!!"). The containment / windowed shortcuts below are
        # evidence of echo ONLY when the MIC HEARD LESS THAN Morty said
        # (heard subset of said) — Whisper catching just a fragment of a
        # long narration. In the OPPOSITE direction (said subset of
        # heard) they are worthless: the mic caught Morty's line AND the
        # operator's reply in one utterance, so of course all of Morty's
        # words are in there. Measured on the real failure —
        #   said  = "Should I go back home?"
        #   heard = "Should I go back home? Yes, go back home."
        # containment returned 1.0 and the best window returned 1.0, so
        # the transcript was dropped at "100% fuzzy match" and Majd's
        # real "Yes, go back home." died with it. The honest
        # full-utterance ratio for that pair is 0.714 — correctly under
        # BOTH the 0.75 in-playback bar and the 0.92 post-playback bar.
        # So: only take the shortcuts when heard is the shorter side.
        # Garbled same-length echo is untouched (it never relied on
        # them: "Start in grasp on the tape dispenser" vs "Starting
        # grasp on the tape dispenser" scores 0.769 on the full ratio).
        short_w, long_w = ((heard_w, said_w) if len(heard_w) <= len(said_w)
                           else (None, None))
        if short_w is not None and len(short_w) >= 4:
            # Substring/prefix echo: Whisper often catches only the TAIL
            # of a long narration ("Where should I drop it left right or
            # home" out of "Got the vitamin tube. Where should I drop
            # it? Left, right, or home?"). Word-bounded containment of
            # the shorter (>= 4 words) in the longer is a certain echo.
            if f" {' '.join(short_w)} " in f" {' '.join(long_w)} ":
                return 1.0
            # Best-aligned window: slide a window of len(short_w) words
            # over the longer side for garbled PARTIAL echo the full
            # ratio dilutes. Stricter 0.85 bar than the full-utterance
            # ratio: legitimate short replies already reach 0.75 on
            # their aligned window ("yes go back home" right after
            # "Should I go back home?" measured at exactly 0.75,
            # 2026-08-27) while garbled partial echo scores ~0.9+.
            n = len(short_w)
            if len(long_w) > n:
                win = max(
                    difflib.SequenceMatcher(
                        None, short_w, long_w[i:i + n]).ratio()
                    for i in range(len(long_w) - n + 1)
                )
                if win >= 0.85:
                    score = max(score, win)
        return score

    # ---- ECHO STRIPPING (2026-08-28) ---------------------------------
    # The mic routinely captures the TAIL of Morty's own narration
    # IMMEDIATELY followed by the operator's reply, and Whisper returns
    # BOTH as one transcript ("I'm at the drop right position. Should I
    # open the gripper? Yes, open the gripper."). The whole-utterance
    # filter below scored that at 100 % and DISCARDED it — including the
    # real command riding at the end — so the operator had to repeat
    # himself every single time. These knobs drive the STRIPPER that
    # removes only the echoed fragment and keeps the rest.
    #
    # Why these numbers (measured against the real session transcripts):
    #  * MIN_FRAGMENT_WORDS = 4 — the same evidence bar the containment
    #    rule in _fuzzy_echo_score already uses. A 2-3 word "match"
    #    ("drop it" out of "Where should I drop it?") is coincidence,
    #    not echo; stripping it would eat the verb of a real command
    #    ("drop it on the right side" -> "on the right side").
    #  * MIN_REMAINDER_WORDS = 2 — anything shorter and there is no real
    #    command left, i.e. it was PURE echo: we then strip nothing and
    #    let _is_likely_echo drop it exactly as before.
    #  * PREFIX_BAR = 0.80 — a prefix match is stronger evidence than a
    #    whole-utterance match (position is informative: the fragment
    #    must be contiguous inside the narration AND land at word 0 of
    #    the transcript), so it earns a bar above the 0.75 in-playback
    #    whole-utterance bar without reaching the 0.92 post-playback
    #    one. It is deliberately NOT lower: over-stripping silently
    #    deletes the front of a real command, which is worse than the
    #    status quo of dropping it loudly.
    #  * SUFFIX_BAR = 0.85 / SUFFIX_MIN_WORDS = 5 — the mirrored case
    #    (operator speaks first, mic then catches Morty starting to
    #    talk) is real and safety-relevant: "stop" + "Starting grasp on
    #    the vitamin tube" currently scores 0.92 and the STOP is eaten.
    #    But a trailing overlap is weaker evidence than a leading one
    #    (operators naturally END sentences with Morty's nouns), so it
    #    pays a stricter bar.
    #  * TAIL_S = 6.0 — the acoustic-timing discipline still applies, but
    #    a MIXED transcript by construction arrives LATER than a pure
    #    echo: it also contains the operator's reply (~2 s) plus the
    #    800 ms end-of-utterance silence (whisper_silence_ms; 500 ms for
    #    answers under ~1.2 s voiced — 2026-09-25) plus
    #    Whisper inference. The 3.0 s capture tail used for the
    #    whole-utterance decision is too tight for that; 6.0 s covers it
    #    while still refusing to strip long after the speaker went quiet
    #    (which is exactly when the operator is legitimately REPEATING
    #    Morty's words back).
    _ECHO_STRIP_MIN_FRAGMENT_WORDS = 4
    _ECHO_STRIP_MIN_REMAINDER_WORDS = 2
    # ...with one exception: a SINGLE remaining word is enough when it is
    # an emergency word. "Starting grasp on the vitamin tube. Stop." is
    # the operator aborting the pick the moment he hears the narration,
    # and a one-word remainder rule would otherwise throw that STOP away
    # with the echo. Acting on a false positive here is harmless (the
    # arm stops); missing a real one is not.
    _ECHO_STRIP_EMERGENCY_WORDS = {"stop", "stopp", "cancel", "abort", "halt"}
    _ECHO_STRIP_PREFIX_BAR = 0.80
    _ECHO_STRIP_SUFFIX_BAR = 0.85
    _ECHO_STRIP_SUFFIX_MIN_WORDS = 5
    _ECHO_STRIP_TAIL_S = 6.0
    # What remains after a strip must not itself look like the same
    # narration; if it does, the "mixed" transcript was really one
    # narration that our block alignment split, and it must die whole.
    _ECHO_STRIP_REMAINDER_ECHO_BAR = 0.75

    @staticmethod
    def _echo_word_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
        """Tokenise exactly like :meth:`_normalise_echo_str` (same kept
        alphabet, same apostrophe handling) but ALSO return each token's
        character span in the ORIGINAL string, so the remainder of a
        stripped transcript can be handed back VERBATIM — original
        casing and punctuation intact — instead of the normalised mush.
        """
        import re
        keep = re.compile(r"[a-z0-9äöüß]")
        words: list[str] = []
        spans: list[tuple[int, int]] = []
        cur: list[str] = []
        start = 0
        for idx, ch in enumerate(text):
            low = ch.lower()
            # A character whose lowercase is not exactly one character
            # (some unicode) is treated as a separator, like the regex
            # in _normalise_echo_str would.
            if len(low) == 1 and keep.match(low):
                if not cur:
                    start = idx
                cur.append(low)
            elif cur:
                words.append("".join(cur))
                spans.append((start, idx))
                cur = []
        if cur:
            words.append("".join(cur))
            spans.append((start, len(text)))
        return words, spans

    _ECHO_STRIP_LEAD_TOL = 2      # garbled words tolerated before the anchor
    _ECHO_STRIP_GAP_TOL = 2       # garbled words tolerated between anchors
    _ECHO_STRIP_LEAD_CHAR_BAR = 0.70

    @classmethod
    def _echo_fragment_span(
        cls, heard_w: list[str], said_w: list[str], from_end: bool = False,
    ) -> tuple[int, float] | None:
        """Longest run of ``heard_w`` anchored at its START (or at its
        END when ``from_end``) that aligns with a CONTIGUOUS chunk of
        ``said_w``. Returns ``(n_heard_words, score)`` or ``None``.

        This is the alignment/prefix search the stripper needs: exact
        string matching is useless because Whisper garbles the echo
        ("Starting" -> "Start in"). ``difflib``'s matching blocks give a
        monotone word-level alignment; we walk them outward from the
        anchor and keep extending while the unmatched gap stays small
        (ASR mangles the odd word), then stop — everything past the
        first real break belongs to the other speaker.

        The LEADING unmatched words (the ones sitting between the anchor
        end of the transcript and the first aligned word) get special
        treatment, and this is what keeps the stripper honest. They are
        absorbed into the fragment ONLY if they are a plausible ASR
        garbling of the narration words they displace, judged at
        CHARACTER level: "start in" vs "starting" scores 0.88 and is
        absorbed, while "yes" vs "i" scores 0.0 and is not — so
        "yes open the gripper and then go home" spoken back at "Should I
        open the gripper?" does not align at word 0 at all and is left
        completely alone. Without this gate a real command that merely
        BEGINS with an affirmation would have its verb stripped off.
        The returned score is the word-level ratio of the aligned BODY
        (leads excluded on both sides, since the lead was already
        judged on its own terms); gaps in the middle stay inside the
        body and are still paid for.
        """
        import difflib
        if not heard_w or not said_w:
            return None
        h = list(reversed(heard_w)) if from_end else list(heard_w)
        s = list(reversed(said_w)) if from_end else list(said_w)
        sm = difflib.SequenceMatcher(None, s, h, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        if not blocks:
            return None
        end = 0
        body_s0: int | None = None
        body_h0 = 0
        hi = 0
        for blk in blocks:
            gap = blk.b - end
            if body_s0 is None:
                if gap > cls._ECHO_STRIP_LEAD_TOL:
                    break
                body_s0, body_h0 = blk.a, blk.b
            elif gap > cls._ECHO_STRIP_GAP_TOL:
                break
            hi = blk.a + blk.size
            end = blk.b + blk.size
        if body_s0 is None or end <= 0 or hi <= body_s0:
            return None
        if body_h0:
            # Unmatched words sit at the anchor end of the transcript:
            # only absorb them if they look like a garbling of the
            # narration words they displace.
            lead_s = s[max(0, body_s0 - body_h0):body_s0]
            if not lead_s:
                return None
            if difflib.SequenceMatcher(
                    None, " ".join(h[:body_h0]), " ".join(lead_s)
            ).ratio() < cls._ECHO_STRIP_LEAD_CHAR_BAR:
                return None
        score = difflib.SequenceMatcher(
            None, h[body_h0:end], s[body_s0:hi], autojunk=False).ratio()
        return end, score

    def _strip_echo_fragment(
        self, transcript: str,
    ) -> tuple[str, str, float, str] | None:
        """Remove a recent-narration fragment riding at the START (or
        END) of ``transcript`` and return
        ``(dropped_text, kept_text, score, where)``.

        Returns ``None`` when nothing should be stripped — including the
        PURE-echo case, where the remainder would be empty or trivial.
        That deliberately leaves the whole transcript to
        :meth:`_is_likely_echo`, which drops it with the same log line
        and the same rejected-topic publish as before, so this change
        cannot weaken pure-echo rejection.
        """
        if self._echo_window_s <= 0.0 or self._echo_min_overlap <= 0.0:
            return None
        if not transcript or not self._recent_spoken:
            return None
        now = self.get_clock().now().nanoseconds / 1e9
        heard_w, spans = self._echo_word_spans(transcript)
        if len(heard_w) < (self._ECHO_STRIP_MIN_FRAGMENT_WORDS
                           + self._ECHO_STRIP_MIN_REMAINDER_WORDS):
            return None

        best: tuple[tuple, str, str, float, str] | None = None
        for ts, raw in list(self._recent_spoken):
            if not raw:
                continue
            # Acoustic-timing discipline, same physics as the
            # whole-utterance bars: only strip while this narration
            # could still have been sounding into the mic (plus the
            # operator's own reply and Whisper's latency).
            if now > ts + 0.06 * len(raw) + self._ECHO_STRIP_TAIL_S:
                continue
            said_w = self._normalise_echo_str(raw).split()
            if len(said_w) < self._ECHO_STRIP_MIN_FRAGMENT_WORDS:
                continue
            for from_end in (False, True):
                span = self._echo_fragment_span(heard_w, said_w, from_end)
                if span is None:
                    continue
                n, score = span
                min_words = (self._ECHO_STRIP_SUFFIX_MIN_WORDS if from_end
                             else self._ECHO_STRIP_MIN_FRAGMENT_WORDS)
                bar = (self._ECHO_STRIP_SUFFIX_BAR if from_end
                       else self._ECHO_STRIP_PREFIX_BAR)
                if n < min_words or score < bar:
                    continue
                n_rest = len(heard_w) - n
                if (n_rest == 1
                        and (heard_w[0] if from_end else heard_w[n])
                        in self._ECHO_STRIP_EMERGENCY_WORDS):
                    n_rest = self._ECHO_STRIP_MIN_REMAINDER_WORDS
                if n_rest < self._ECHO_STRIP_MIN_REMAINDER_WORDS:
                    # PURE ECHO: this one narration explains essentially
                    # the WHOLE transcript, so there is no operator
                    # command hiding in it. Bail out of the stripper
                    # entirely (not just this direction) and let
                    # _is_likely_echo drop it with the same log line and
                    # the same rejected-topic publish as before. Without
                    # this precedence rule the mirrored (suffix) search
                    # would happily cut "Start in grasp on the vitamin
                    # tube" down to "Start in" and hand that to the
                    # command layers.
                    return None
                if from_end:
                    cut = spans[len(heard_w) - n][0]
                    kept, dropped = transcript[:cut], transcript[cut:]
                else:
                    cut = spans[n][0]
                    dropped, kept = transcript[:cut], transcript[cut:]
                kept = kept.strip()
                dropped = dropped.strip()
                if not kept or not dropped:
                    continue
                # A narration our alignment merely SPLIT is not a mixed
                # transcript: if the remainder still scores as echo of
                # the same utterance, strip nothing (this is what stops
                # a half-recognised narration from reaching FAST-PICK
                # and launching another grasp).
                rest = self._fuzzy_echo_score(
                    self._normalise_echo_str(kept),
                    self._normalise_echo_str(raw))
                if rest >= self._ECHO_STRIP_REMAINDER_ECHO_BAR:
                    continue
                # Prefer the longest fragment; prefer a prefix over a
                # suffix at equal length (the prefix case is the
                # observed one); then the better score.
                key = (n, 0 if from_end else 1, score)
                if best is None or key > best[0]:
                    best = (key, dropped, kept,
                            score, "suffix" if from_end else "prefix")
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    def _echo_explained_by_recent_speech(self, heard_norm: str) -> bool:
        """True when Morty's OWN recent chunks, taken TOGETHER, account
        for essentially every word of ``heard_norm``.

        Why this exists (2026-08-28). Restricting the containment /
        window shortcuts in :meth:`_fuzzy_echo_score` to the
        heard-subset-of-said direction is what stops Majd's real replies
        being eaten — but it also releases the one case those shortcuts
        were accidentally catching: the mic swallowing TWO CONSECUTIVE
        narrations as a single transcript. Measured: "Going to drop
        right. I'm at the drop right position. Should I open the
        gripper?" scores only 0.400 against any ONE chunk, so the
        per-chunk loop cannot see it — and the adversarial review of
        2026-08-27 proved that transcript reaches the FAST matchers as
        ``drop_right_release``, i.e. the arm drives to drop-right and
        OPENS THE GRIPPER. This closes that hole without re-introducing
        the wrong-direction match.

        The discriminator is ACCOUNTING, not similarity: each recent
        chunk may explain AT MOST ONE span (Morty said that line once,
        so it can appear once). Whatever is left over was said by the
        OPERATOR. Majd's failing transcript leaves "yes go back home"
        unexplained -> not echo, exactly as he wants. Pure
        double-narration leaves nothing -> echo.

        Deliberately requires >= 2 contributing chunks: single-chunk
        full coverage is already the per-chunk loop's job, and keeping
        this narrow means it can only ever fire on the multi-narration
        case it was written for.
        """
        import difflib
        heard_w = heard_norm.split()
        if len(heard_w) < 4:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        covered = [False] * len(heard_w)
        contributors = 0
        for ts, raw in list(self._recent_spoken):
            if self._echo_window_s >= 10.0:
                deadline = ts + self._echo_window_s
            else:
                deadline = ts + 0.06 * len(raw) + 3.0
            if now > deadline:
                continue
            said_w = self._normalise_echo_str(raw).split()
            if not said_w:
                continue
            hit = False
            for blk in difflib.SequenceMatcher(
                    None, heard_w, said_w).get_matching_blocks():
                # >= 3 words: a 1-2 word run ("the", "drop it") is
                # coincidence, not evidence that Morty said it.
                if blk.size >= 3:
                    for k in range(blk.a, blk.a + blk.size):
                        covered[k] = True
                    hit = True
            if hit:
                contributors += 1
        if contributors < 2:
            return False
        uncovered = [w for w, c in zip(heard_w, covered) if not c]
        # >= 2 leftover words means the operator is in this transcript.
        if len(uncovered) >= 2:
            return False
        self._last_echo_reason = (
            "echo of my own voice (two of my own sentences run together)"
        )
        return True

    def _on_barge_in(self, msg: Bool) -> None:
        if bool(msg.data):
            self._hush_answer("the operator talked over it")

    def _reply_hushed(self) -> bool:
        return self._answer_seq <= self._hushed_seq

    def _hush_answer(self, why: str) -> None:
        """End the answer in progress and any of its sentences still queued
        at the speak server. Idempotent; router status lines are not ours
        and stay (the speak server drops those itself on a real barge-in)."""
        seq = self._answer_seq
        newly = seq > self._hushed_seq
        self._hushed_seq = seq
        if newly:
            self._hush_why = why     # 2026-09-25: the first reason wins (read by _finish_interrupted_turn)
        live = self._cancel_answer_goals(seq)
        if live or (newly and self._answer_running):
            self.get_logger().info(
                f"Answer {seq} interrupted ({why}): {live} queued sentence(s) cancelled; "
                "nothing more of it will be said.")

    def _cancel_answer_goals(self, seq: int, cut: bool = False) -> int:
        """2026-09-25: cancel the Speak goals of answers up to ``seq`` that are
        still queued or playing, without marking the answer hushed (factored
        out of _hush_answer: an 'unclear' turn cancels what it streamed and
        then speaks its one line). Returns how many were cancelled.

        ``cut`` (2026-09-25, review Y6): also cancel the answer goals already
        SENT but not yet accepted by the speak server — they were not in
        the list yet and played before the unclear line. Not by raising
        _hushed_seq, which would also swallow the UNCLEAR_LINE of the same
        turn: the goals sent so far are cut by their serial."""
        with self._answer_goals_lock:
            if cut:
                self._cut_serial = self._speak_serial
            live = [h for s_, h in self._answer_goals if s_ <= seq]
            self._answer_goals = [(s_, h) for s_, h in self._answer_goals if s_ > seq]
        for h in live:
            try:
                h.cancel_goal_async()
            except Exception:
                pass
        return len(live)

    def _finish_interrupted_turn(self, response: Any) -> None:
        """The operator talked over this answer: keep the history valid —
        every tool call gets its output, the dispatched one or 'not
        executed' — say nothing more, and leave the next turn a note."""
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call" and item.name == "intent_extractor":
                early = (self._early_dispatched.pop(item.call_id, None)
                         if getattr(self, "_early_dispatched", None) else None)
                self.chat_agent.add_function_call_output(
                    item, early if early is not None else json.dumps(
                        {"status": "not executed: the operator interrupted this answer"}))
        heard = " ".join(getattr(self, "_streamed", []) or []).strip()
        if self._hush_why == _HUSH_BY_MERGE:
            # 2026-09-25 (Majd's G1): nothing of it was said or done; the
            # operator's sentence continued and comes next as one message.
            # The whole sentence may have been handled without you (an
            # instant answer), so the note names it instead of promising a
            # next message.
            self._interrupt_note = (
                "Your previous answer was cut before anything was said or done: the "
                "operator's sentence went on. Their whole sentence was: "
                f"\"{getattr(self, '_merged_text', '')}\" — it was handled as ONE message; "
                "do not answer the cut part separately.")
        elif self._hush_why == _HUSH_BY_QUESTION:
            # 2026-09-25: a story told while a pick ran yields to the robot's
            # question (story_yields_to_question); the model must know the
            # operator did not interrupt it.
            self._interrupt_note = (
                "The robot asked a question and cut your story."
                + (f" The operator heard at most: \"{heard}\"." if heard else "")
                + " Do not continue the story unless they ask for it.")
        else:
            self._interrupt_note = (
                "The operator interrupted your previous answer"
                + (f"; they heard at most: \"{heard}\"" if heard else " before you said anything")
                + ". Answer their new message directly. Do not continue or repeat the old "
                  "answer unless they ask for it.")
        text = (getattr(response, "output_text", "") or "").strip()
        if text:
            print(f"Assistant (interrupted): {text}")
        self.get_logger().info(f"Answer {self._answer_seq} dropped after the interruption.")
        self._publish_chat_state("idle")

    def _on_speaker_verdict(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
            self._speaker_verdict = (time.monotonic(), str(d.get("text", "")), bool(d.get("human")))
        except Exception:
            pass

    def _is_likely_echo(self, transcript: str) -> bool:
        if self._echo_window_s <= 0.0 or self._echo_min_overlap <= 0.0:
            return False
        t_v, txt_v, human_v = self._speaker_verdict
        if human_v and txt_v == transcript and time.monotonic() - t_v < 5.0:
            self.get_logger().info("speaker gate said human — text echo rule skipped")
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        # Drop expired entries.
        self._recent_ai_speech = [
            (t, s)
            for (t, s) in self._recent_ai_speech
            if now - t <= self._echo_window_s
        ]
        if not self._recent_ai_speech:
            return False
        heard_set = self._normalise_for_echo(transcript)
        if not heard_set:
            return False

        # ---- FUZZY echo check (additive, 2026-08-27) -----------------
        # Compares whole word SEQUENCES (difflib) instead of token sets,
        # so Whisper-garbled echo like "Start in grasp on the tape
        # dispenser" (mic re-hearing "Starting grasp on the tape
        # dispenser") is caught even though set overlap (0.71) sat under
        # echo_min_overlap. Runs BEFORE the affirmation / allowlist
        # early-outs: pure echo "Where should I drop it left right or
        # home" contains "right" (an affirmation word) and would slip
        # through them. Only transcripts of >= 4 words are fuzzy-checked
        # so short solicited answers ("open the gripper") keep their
        # hard_allow protection below. This runs in _on_transcript,
        # i.e. before FAST-command/FAST-PICK matching and the LLM.
        heard_norm = self._normalise_echo_str(transcript)
        if len(heard_norm.split()) >= 4:
            for ts, raw in list(self._recent_spoken):
                # Effective lifetime of this chunk for the fuzzy check:
                # it must cover the whole TTS playback plus mic-capture/
                # Whisper latency. echo_window_s (25 s today) already
                # does; were it configured small (< 10 s), extend PER
                # CHUNK by an estimated speaking time (~0.06 s/char)
                # plus a 3 s capture+transcription tail instead of
                # blindly raising the global window (2026-08-27).
                if self._echo_window_s >= 10.0:
                    deadline = ts + self._echo_window_s
                else:
                    deadline = ts + 0.06 * len(raw) + 3.0
                if now > deadline:
                    continue
                # TIMING DISCRIMINATOR (2026-08-27, adversarial review):
                # a garbled acoustic echo and a genuine user RETRY can
                # score IDENTICALLY ("Start in grasp on the tape
                # dispenser" echo = 0.769; "pick up the tape dispenser
                # now" retry = 0.769). Similarity alone cannot separate
                # them — but physics can: acoustic echo only exists
                # while the speaker is playing (+ capture/transcription
                # tail). Within that acoustic window the 0.75 bar
                # applies; afterwards only a near-verbatim 0.92 match
                # (a delayed transcript flush) counts as echo, so
                # natural retries like "grasp the small bottle" (0.80)
                # pass once the narration has finished sounding.
                acoustic_until = ts + 0.06 * len(raw) + 3.0
                bar = 0.75 if now <= acoustic_until else 0.92
                score = self._fuzzy_echo_score(
                    heard_norm, self._normalise_echo_str(raw))
                if score >= bar:
                    self.get_logger().info(
                        f"ECHO rejected: fuzzy={score:.2f} "
                        f"heard={transcript!r} vs said={raw!r} "
                        f"({now - ts:.1f}s ago)"
                    )
                    self._last_echo_reason = (
                        f"echo of my own voice "
                        f"({int(score * 100)}% fuzzy match)"
                    )
                    return True
            # Multi-chunk fallback: no SINGLE chunk matched, but Morty's
            # recent lines together may still account for the whole
            # transcript (two narrations heard as one). See the helper.
            if self._echo_explained_by_recent_speech(heard_norm):
                self.get_logger().info(
                    f"ECHO rejected (multi-chunk coverage): "
                    f"heard={transcript!r}"
                )
                return True

        # Always-allow short control words: pass them through even if
        # they appear in a recent AI utterance, so the user can stop the
        # robot mid-narration. Includes the EXPECTED ANSWERS to the
        # two-pause questions — the user naturally answers "Should I open
        # the gripper?" with "open the gripper", which overlaps Morty's
        # own words by construction and was echo-dropped at 66 %
        # (Majd, 2026-08-12).
        # Two tiers (Majd, 2026-08-12): hard_allow bypasses EVERYTHING —
        # emergency words plus the SOLICITED answers to the two-pause
        # questions ("open the gripper" back at "Should I open the
        # gripper?"). verb_allow only bypasses the cooldown: command
        # verbs mark real instructions, but a bare fragment of Morty's
        # own narration ("Starting grasp.") must still die in the
        # subset/overlap checks — with "grasp" in the old single list it
        # reached the LLM, which then answered its own echo.
        hard_allow = {"stop", "cancel", "abort", "halt",
                      "open", "release", "close", "gripper",
                      "left", "right", "home",
                      "links", "rechts", "öffnen", "loslassen",
                      "auf", "zu"}
        verb_allow = {"again", "redo", "repeat",
                      "start", "pick", "grasp", "grab", "starte",
                      "greif", "nimm"}
        always_allow_short = hard_allow  # subset/short checks use this

        # Affirmation words. If the user's transcript contains one of
        # these, treat the whole utterance as a real user reply even when
        # it heavily overlaps with the AI's last line. Without this, the
        # filter eats "yes, the vitamin tube" right after Morty asked
        # "do you mean the vitamin tube?" because the object name pushes
        # overlap above 0.6.
        affirmation_words = {
            "yes", "yeah", "yep", "yup", "ja", "correct", "right",
            "okay", "ok", "sure", "please", "exactly", "confirm",
            "confirmed", "go",
        }
        if heard_set & affirmation_words:
            return False

        # Cooldown: if the AI spoke very recently, be suspicious — but
        # only DROP when the transcript actually shares words with what
        # was just said. Real echo (even Whisper-garbled) keeps SOME of
        # Morty's tokens; a zero-overlap utterance is the user talking
        # right after (or over) Morty. The old unconditional drop ate a
        # correctly-heard "Start, all." 0.87 s after a greeting
        # (Majd, 2026-08-12).
        if not (heard_set & (hard_allow | verb_allow)):
            youngest_ai = max(t for t, _ in self._recent_ai_speech)
            if now - youngest_ai <= self._echo_cooldown_s:
                best_overlap = max(
                    len(heard_set & said) / max(1, len(heard_set))
                    for _, said in self._recent_ai_speech
                )
                # 0.6, up from 0.25: legitimate replies reuse Morty's
                # words ("open the gripper" back at the release question)
                # and were dropped at 66 % — Majd suggested raising it.
                # Real full-sentence echo scores near 1.0.
                if best_overlap >= 0.6:
                    self.get_logger().info(
                        f"echo-drop reason=cooldown+overlap="
                        f"{best_overlap:.2f} ({now - youngest_ai:.2f}s "
                        f"since AI, limit={self._echo_cooldown_s:.2f}s)"
                    )
                    self._last_echo_reason = (
                        f"echo of my own voice ({int(best_overlap * 100)}% "
                        "word overlap, just after I spoke)"
                    )
                    return True
                self.get_logger().info(
                    f"cooldown PASS: zero-ish overlap ({best_overlap:.2f}) "
                    "— user speaking during/after AI speech."
                )

        # Short transcripts (1-3 tokens). Drop ONLY if the token set is
        # fully contained in a recent AI utterance AND not in the
        # always-allow control set.
        if len(heard_set) <= 3:
            if heard_set & always_allow_short:
                return False
            for ts, said in self._recent_ai_speech:
                if now - ts > self._echo_active_tail_s:
                    continue  # playback long over — user, not echo
                if said and heard_set <= said:
                    self.get_logger().info(
                        f"echo-drop reason=short-subset; heard={heard_set}; "
                        f"said={said} ({now - ts:.2f}s old)"
                    )
                    self._last_echo_reason = (
                        "echo of my own voice (short fragment of what I said)"
                    )
                    return True
            return False
        for ts, said in self._recent_ai_speech:
            if not said:
                continue
            if now - ts > self._echo_active_tail_s:
                # Playback (plus capture tail) is over: at this point a
                # high-overlap transcript is the user REUSING Morty's
                # words, which is normal conversation — let it through.
                continue
            overlap = len(heard_set & said) / len(heard_set)
            if overlap >= self._echo_min_overlap:
                self.get_logger().info(
                    f"echo-drop reason=overlap={overlap:.2f}; "
                    f"heard={heard_set}; said={said} ({now - ts:.2f}s old)"
                )
                self._last_echo_reason = (
                    f"echo of my own voice ({int(overlap * 100)}% word overlap)"
                )
                return True
        return False

    def _on_ai_said(self, msg: String) -> None:
        """Anyone publishing on /speech/ai_said (the speak action server
        broadcasts every accepted goal here) gets folded into the echo
        ring. This catches narrations originating in intent_router or
        any other process so the mic feedback filter actually sees
        them."""
        text = (msg.data or "").strip()
        if text:
            self._record_ai_speech(text)
            if text not in getattr(self, "_own_answer_lines", {}):
                self._track_robot_status(text)

    def _track_robot_status(self, text: str) -> None:
        """Feed status narrations into the LLM's world model (Majd
        2026-08-26: "the AI Morty doesn't know everything about the
        Status Morty — it should keep updated"). Keeps a short history
        of announcements plus a one-line derived cycle state; both are
        injected as developer context every turn."""
        now = self.get_clock().now().nanoseconds / 1e9
        hist = getattr(self, "_status_history", [])
        hist.append((now, text))
        self._status_history = hist[-6:]
        low = text.lower()
        if self._task is not None:
            # 2026-09-25: the executor's task state owns the cycle state; a
            # phrase read back from the speaker (a late "Starting grasp on…"
            # after the cycle failed) no longer edits it. The history above
            # and the start-all readiness hook stay.
            if low.startswith("starting the whole system"):
                self._announce_ready_until = now + 180.0
            return
        if low.startswith("starting grasp on"):
            self._cycle_state = f"BUSY — {text[9:].strip()} (pick running)"
        elif low.startswith("got the"):
            self._cycle_state = ("HOLDING the object at the grasp pose — "
                                 "waiting for a drop destination")
        elif ("at the drop position" in low
              or "should i open the gripper" in low):
            # The real narration is "I'm at the <drop left|drop right|
            # home> position. Should I open the gripper?" — the old
            # "at the drop position" test never matched ANY of the three
            # (2026-08-28), so the release pause was invisible to the
            # deterministic yes/no layers below and every answer fell
            # through to the LLM.
            self._cycle_state = ("HOLDING at the drop pose — waiting for "
                                 "release confirmation")
        elif low.startswith("should i go back home"):
            self._cycle_state = "at the drop pose, gripper open — asked about going home"
        elif low.startswith("the grasp failed") or "not grasping anything" in low:
            self._cycle_state = "IDLE — last pick FAILED, arm recovered"
        elif (low.startswith("stopping") or low.startswith("resetting")
              or "stopped by your command" in low
              # 2026-09-25: the router's reset outcome lines ("Reset — I'm
              # back home…", "Reset — but I couldn't…", "Pick reset.").
              or low.startswith("reset —") or low.startswith("reset -")
              or low.startswith("pick reset")):
            # Operator stop/reset ends the cycle (Majd 2026-08-26: after
            # "Reset" the AI still refused new picks with "a grasp is
            # currently running" — the tracker never saw the stop).
            self._cycle_state = ("IDLE — the last pick was STOPPED/RESET by "
                                 "the operator; a new pick can start now")
        elif low.startswith("going home"):
            self._cycle_state = "moving to home"
        elif (low.startswith("done — i'm back at the home position")
              or low.startswith("done - i'm back at the home position")
              or low.startswith("i've brought the object back home")
              or low.startswith("okay — staying at")
              or low.startswith("okay - staying at")
              or low.startswith("stopped. i'm holding position")):
            # v2 completion narrations (adversarial review 2026-08-31,
            # finding 5): without these rules the state stayed "waiting
            # for release confirmation" FOREVER after a cycle ended, so
            # minutes later an idle remark like "stay right there" could
            # fire the FAST-NEGATIVE layer and queue a REAL arm move.
            self._cycle_state = "IDLE — last pick finished"
        elif low.startswith("the gripper is open"):
            # Release confirmed by hardware: the release pause is over.
            # (The follow-up "Should I go back home?" — if any — arrives
            # right after and sets its own state.)
            self._cycle_state = "gripper open at the drop pose"
        elif low.startswith("okay — taking it to") or \
                low.startswith("okay - taking it to"):
            # Redirect acknowledged: the arm is carrying the held object;
            # the re-asked question on arrival reopens the release pause.
            self._cycle_state = "carrying the held object to a new position"
        elif low.startswith("starting the whole system"):
            # Any path that announces the system start arms the
            # readiness watcher (FAST and LLM alike).
            self._announce_ready_until = now + 180.0
        elif low.startswith("grasp sequence finished") or low.startswith("dropped"):
            self._cycle_state = "IDLE"
        if str(getattr(self, "_cycle_state", "")).startswith("IDLE"):
            self._open_question = ""     # a finished cycle has no open question (2026-09-24)

    def _record_ai_speech(self, text: str) -> None:
        """Remember a chunk of AI speech for echo suppression."""
        if self._echo_window_s <= 0.0 or not text:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        # Timestamp with the ESTIMATED END of playback, not the start:
        # ai_said fires when speech STARTS, but a 3-4 s narration is still
        # playing (and echoing into the mic) long after the 2.5 s cooldown
        # from its start expired — "Starting Greif" leak, 2026-08-12.
        # ~13 chars/s speaking rate + 1 s margin.
        est_end = now + max(1.5, len(text) / 13.0)
        # Dedupe (2026-08-27): the chat's own replies are recorded twice —
        # once at _speak() send time and once when speak_node broadcasts
        # them back on /speech/ai_said — halving the effective ring. Skip
        # an identical text recorded within the last 2.5 s.
        for prev_ts, prev_raw in self._recent_spoken[-3:]:
            if prev_raw == text and (now - prev_ts) <= 2.5:
                return
        # Store the *normalised token set* so matching is symmetric and fast.
        self._recent_ai_speech.append((est_end, self._normalise_for_echo(text)))
        if len(self._recent_ai_speech) > 16:
            self._recent_ai_speech = self._recent_ai_speech[-16:]
        self._recent_spoken.append((now, text))
        if len(self._recent_spoken) > 8:
            self._recent_spoken = self._recent_spoken[-8:]

    def _prune_llm_context(self) -> None:
        """Keep the LLM prompt SMALL and CURRENT before every turn.

        Why (2026-08-31, measured): llm_service never trims
        ``self.messages``, and every turn injects ~10 developer state
        blocks (clock, status, scene, arm pose, gripper...). They were
        never removed, so each call re-sent every stale copy — the
        stopwatch showed LLM calls GROWING 5.7s -> 8.4s -> 10.5s within
        minutes of a fresh session ("Tell me a joke" cost 10.55s of
        pure API time). Worse than slow, the stale copies are exactly
        the misinformation the prompt's FRESHNESS / "LIVE and
        OVERRIDES" rules were written to fight.

        Two rules, applied at the START of each turn, BEFORE the fresh
        injections:
        1. Drop every developer message except messages[0] (the base
           SYSTEM_PROMPT). All 11 other add_developer_message sites are
           per-turn live state, regenerated fresh a few lines later —
           audited 2026-08-31, see SYSTEM_STATE_LOG #277.
        2. Cap the conversation to the last ``llm_history_max_turns``
           user turns, cutting AT a user-message boundary so an
           assistant tool call is never orphaned from its output.
           Durable facts the operator taught in conversation ("Drus
           means vitamin tube") live in user/assistant text and survive
           as long as they are recent; the cap only sheds the old tail.
        """
        msgs = self.chat_agent.messages
        if len(msgs) <= 1:
            return
        before = len(msgs)
        head, rest = msgs[:1], msgs[1:]
        rest = [
            m for m in rest
            if not (isinstance(m, dict) and m.get("role") == "developer")
        ]
        user_idx = [
            i for i, m in enumerate(rest)
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        max_turns = int(getattr(self, "_llm_history_max_turns", 12))
        if max_turns > 0 and len(user_idx) > max_turns:
            rest = rest[user_idx[-max_turns]:]
        self.chat_agent.messages = head + rest
        dropped = before - len(self.chat_agent.messages)
        if dropped > 0:
            self.get_logger().info(
                f"[timing] pruned {dropped} stale message(s) from the LLM "
                f"context ({before} -> {len(self.chat_agent.messages)})")

    def _handle_voice_input(self, text: str, my_gen: int = 0,
                            typed_at: float | None = None,
                            heard_at: float | None = None,
                            epoch: int | None = None) -> None:
        # typed_at (2026-09-25): arrival time of typed input (GUI, terminal);
        # it stands in for the speech start in the fresh-question rule.
        # heard_at (2026-09-25): arrival time of a voice transcript.
        # epoch (2026-09-25, review H1): _stop_epoch when the input arrived;
        # a turn that reaches _chat_lock after a later stop/reset is dropped.
        if epoch is None:
            epoch = self._stop_epoch
        # Generation gate: if a newer transcript arrived while we were
        # waiting for the chat lock or running the LLM, don't act on this
        # stale input. Latest-wins behaviour for piled-up voice activity.
        def _superseded() -> bool:
            with self._voice_gen_lock:
                return my_gen != 0 and my_gen != self._voice_generation
        # Filter junk transcripts before paying the LLM round-trip.
        # Whisper sometimes emits artefacts like "...", "Fook bike",
        # "bedgy" — short, low-letter-ratio, no recognizable command.
        # The LLM call is the slowest step (~1-3s); skipping it for
        # garbage avoids piling work the user didn't actually request.
        cleaned = text.strip()
        # Short REAL answers must pass: "yes"/"no"/"ja" is exactly what the
        # gripper-release question waits for, and typed GUI messages ride
        # this same path ("yes" killed as junk — Majd, 2026-08-20).
        short_word_ok = cleaned.strip(".,!?").lower() in {
            "yes", "no", "ok", "ja", "yep", "nope", "go",
        }
        if not short_word_ok and (len(cleaned) < 4 or cleaned in {"...", "..", "."}):
            self.get_logger().info(f"Skipping junk transcript: {cleaned!r}")
            return
        if not any(c.isalpha() for c in cleaned):
            self.get_logger().info(f"Skipping non-alphabetic transcript: {cleaned!r}")
            return
        # Real input ends whatever answer is still being spoken — also when
        # the barge-in detector did not fire (the operator spoke between
        # two sentences, or only briefly). Junk ("Uh.") above does not.
        # 2026-09-25 (Majd's G1): a fragment continuing the previous input
        # (≤ 2 s, nothing said or done yet) runs as ONE input with it; the
        # grammar below (stop anywhere, …) reads the merged text. The hush
        # happens inside, with the reason the hushed turn must report.
        text, meta_key = self._hush_and_merge(text, heard_at)
        print(f"User (voice): {text}")
        # [timing] wall-clock decomposition of "why is Morty slow to
        # answer" (Majd 2026-08-31). One line per accepted transcript,
        # logged when the route completes; whisper's own cost is logged
        # by stt.py, so speech-end -> reply is fully accounted for.
        _t_handle0 = time.monotonic()
        # Broadcast the accepted transcript so the mic visualiser (and any
        # other debug UI) can show what Morty actually acted on, not the
        # raw firehose of dropped/echoed Whisper output.
        try:
            self.accepted_transcript_pub.publish(String(data=text))
        except Exception:
            pass
        # 2026-09-25 (Majd: "STOP FIRST — a stop word anywhere wins over every
        # other rule; exactly ONE stop is sent"). In both modes, before any
        # other layer: a stop word anywhere (not negated) is one stop intent
        # and "Stopping."; with a reset word too, or a whole reset phrase, it
        # is one reset intent and "Resetting.". Neither ever reaches the LLM
        # (it used to send a second stop through its own tool call).
        kind, action = voice_grammar.classify(text)
        if kind in ("stop", "reset"):
            self._note_stop_or_reset()       # 2026-09-25 (review H1/X2)
            if self._task is None:
                self._open_question = ""     # no task state: the old bookkeeping
            if kind == "stop":
                self._fire_stop()
                self._speak(QUICK_NARRATION["stop"], interrupt=True)
                _ij = json.dumps({"action_intent": "stop"})
            else:
                _ij = self._fire_reset()
                self._speak(QUICK_NARRATION["reset"], interrupt=True)
            self._note_instant(text, kind, _ij)   # 2026-09-25: into the LLM history
            self.get_logger().info(
                f"[timing] handled in {time.monotonic() - _t_handle0:.2f}s "
                f"(route=GRAMMAR {kind}): {text!r}")
            return
        # 2026-09-25 (review Y1): the minimal layer needs the executor's
        # LIVE task state (its only source of the open question). Without it
        # (an old executor, before the first sample, a dead executor) the
        # legacy layers answer exactly as before the overhaul — otherwise
        # every pause answer went to the LLM and "No." at the release
        # question ended in silence.
        if self._voice_fast_layers != "legacy" and self._live_task() is not None:
            self._route_minimal(text, typed_at, _t_handle0, _superseded, heard_at, epoch,
                                meta_key=meta_key)
            return
        self._route_legacy(text, _t_handle0, _superseded, epoch)

    def _note_act(self) -> None:
        """2026-09-25 (Majd's G1): this node spoke a line or dispatched an
        intent (see _hush_and_merge)."""
        self._act_serial += 1

    def _hush_and_merge(self, text: str, heard_at: float | None) -> tuple[str, tuple | None]:
        """Hush the answer in progress ("the operator said something new")
        and — 2026-09-25, Majd's G1 — join a voice transcript to the previous
        one when it arrived ≤ _MERGE_WINDOW_S after it and that input has
        neither spoken nor dispatched anything since it was accepted
        (_act_serial). The previous input's turn is then hushed with
        _HUSH_BY_MERGE (decided BEFORE the hush: the hushed stream may end
        its turn at once) or, still waiting for the chat lock, is dropped as
        superseded; if it acted between the check and the hush, this input
        stands alone. Returns the text to act on and, when merged, the
        (text, heard_at) of the FIRST fragment — its utterance meta carries
        the speech start of the whole sentence. Typed input ends a chain."""
        with self._voice_gen_lock:
            prev = self._prev_input
            cand = (heard_at is not None and prev is not None
                    and 0.0 <= heard_at - prev["heard_at"] <= _MERGE_WINDOW_S
                    and self._act_serial == prev["act"])
        own_turn = bool(cand and self._answer_running and self._turn_text == prev["text"])
        if own_turn:
            self._merged_text = prev["text"] + " " + text
        self._hush_answer(_HUSH_BY_MERGE if own_turn else "the operator said something new")
        with self._voice_gen_lock:
            if heard_at is None:
                self._prev_input = None
                return text, None
            if cand and self._act_serial != prev["act"]:
                cand = False            # it acted meanwhile: this input stands alone
            if cand:
                text = prev["text"] + " " + text
            self._prev_input = {"text": text, "heard_at": heard_at, "act": self._act_serial,
                                "key": prev["key"] if cand else (text, heard_at)}
        if not cand:
            return text, None
        self.get_logger().info(
            f"fragment {heard_at - prev['heard_at']:.1f}s after {prev['text']!r}, which had "
            f"not answered yet — one input: {text!r}")
        return text, prev["key"]

    def _route_legacy(self, text: str, _t_handle0: float, _superseded, epoch: int) -> None:
        """The pre-2026-09-25 layers (voice_fast_layers=legacy, or no live
        task state — review Y1), unchanged; factored out of
        _handle_voice_input (2026-09-25) so the minimal layer can fall back
        to it when the speech node sends no utterance metas."""
        # ── legacy layers, unchanged ─────────────────────────────────────
        # Fast path: if the transcript is a recognized direct command,
        # publish the intent ourselves and skip the LLM entirely. Cuts
        # latency from ~1-3s down to ~10ms and avoids piling up calls in
        # the chat lock when the user gives several commands in a row.
        fast = _match_fast_command(text) or self._answer_open_question(text)
        if fast is not None:
            action, speed = fast
            if action in ("go_home", "open_gripper", "release", "release_home", "stop", "close_gripper"):
                self._open_question = ""     # this answers the executor's question
            self._fire_fast_command(action, speed, text)
            self.get_logger().info(
                f"[timing] handled in "
                f"{time.monotonic() - _t_handle0:.2f}s (route=FAST "
                f"{action})")
            return
        # NEGATIVE answer at an open pause = deterministic action
        # (2026-08-28: "No, close the gripper." / "No." / "Don't open it."
        # each fell through to the LLM, which re-emitted "I'm at the home
        # position. Should I open the gripper?" — an endless loop with the
        # object still in the gripper). Runs AFTER _match_fast_command so
        # unambiguous direct commands keep priority.
        # POSITIVE answer at an open pause (destination / consent) —
        # deterministic, never the LLM (2026-09-08). Refuses negatives
        # itself, so the negative layer below keeps its priority on those.
        ans = _match_pause_answer(text, getattr(self, "_cycle_state", ""))
        if ans is not None:
            self._fire_fast_command(ans, "", text)
            self.get_logger().info(
                f"[timing] handled in {time.monotonic() - _t_handle0:.2f}s "
                f"(route=PAUSE-ANSWER {ans})")
            return
        neg = _match_pause_negative(text, getattr(self, "_cycle_state", ""))
        if neg is not None:
            _redirect = neg.startswith("redirect_")
            if _redirect:
                # v2: "no, <take it somewhere>[ and give it to me]".
                # Fire the destination intent (the executor redirects the
                # HELD object there via home and re-asks), and — for the
                # "...and give me it" form — an open_gripper intent right
                # behind it: the executor consumes it on arrival and
                # opens without re-asking. Same two-intent mechanism as
                # the go_home_open fast splitter.
                parts = neg.split("_")           # redirect,<dest>[,release]
                _dest_word = parts[1]
                _with_release = neg.endswith("_release")
                if _with_release:
                    # ONE atomic intent. Two back-to-back intents
                    # (destination, then open) would race the executor's
                    # single-slot resume mailbox: the open overwrites the
                    # destination and the gripper opens AT THE CURRENT
                    # POSE — the failure class the 08-27 adversarial
                    # review caught. carry_release carries both facts.
                    _payload = {"action_intent": "carry_release",
                                "object_intent": None,
                                "object_attributes":
                                    {"destination": _dest_word}}
                    self.intent_pub.publish(String(
                        data=json.dumps(_payload, separators=(",", ":"))))
                else:
                    _dest_intent = {"home": "go_home", "left": "drop_left",
                                    "right": "drop_right"}[_dest_word]
                    _payload = {"action_intent": _dest_intent,
                                "object_intent": None,
                                "object_attributes": None}
                    self.intent_pub.publish(String(
                        data=json.dumps(_payload, separators=(",", ":"))))
                _human = {"home": "home", "left": "the left side",
                          "right": "the right side"}[_dest_word]
                self.get_logger().info(
                    f"FAST-NEGATIVE redirect: {text!r} -> {neg}")
                self._cycle_state = f"moving to {_dest_word}"
                self._speak(
                    f"Okay — taking it to {_human} and handing it over."
                    if _with_release else f"Okay — taking it to {_human}.")
                return
            self._fire_pause_negative(neg, text)
            return
        # Bare affirmation + tracked robot state = deterministic action
        # (2026-08-26: "Yes, go back home" got "Already heading home" WORDS
        # with no action; the pause then hung until the user repeated).
        import re as _re
        if _re.match(r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?)(?:,?\s*please)?\s*[!\.\?]*\s*$",
                     text.strip(), _re.IGNORECASE):
            state = getattr(self, "_cycle_state", "") or ""
            if "asked about going home" in state:
                self._fire_fast_command("go_home", "", text)
                return
            if "waiting for release confirmation" in state:
                self._fire_fast_command("open_gripper", "", text)
                return
        # DESTINATION-PAUSE answers, deterministic (2026-09-02). Morty
        # asked "Where should I drop it? Left, right, or home?" and Majd
        # answered "on the right side." — no drop/put verb, so the FAST
        # drop patterns missed and the LLM took over, then reasoned from
        # the PREVIOUS cycle's object ("I already sent the vitamin tube
        # home. Do you want me to pick it back up...?") while the gripper
        # held a tape dispenser. A bare side/home answer at THIS pause
        # can only mean one thing; never let the LLM near it.
        _dstate = getattr(self, "_cycle_state", "") or ""
        if "waiting for a drop destination" in _dstate:
            _m = _re.match(
                r"^\s*(?:(?:yes|yeah|ok(?:ay)?),?\s*)?(?:(?:on|to|at|in)\s+)?"
                r"(?:the\s+)?(right|left)(?:\s+side)?(?:\s*,?\s*please)?"
                r"\s*[!\.\?]*\s*$", text.strip(), _re.IGNORECASE)
            if _m:
                self._fire_fast_command(
                    "drop_right" if _m.group(1).lower() == "right"
                    else "drop_left", "", text)
                return
        pick = _match_fast_pick(text)
        if pick is not None:
            self._fire_fast_pick(pick[0], pick[1], text)
            return
        if _superseded():
            self.get_logger().info(
                f"Superseded by newer transcript before LLM; dropping: {text!r}"
            )
            return
        try:
            # 2026-09-25 (review H1): re-checked once _chat_lock is held
            self.handle_user_text(text, superseded=_superseded, epoch=epoch)
        except Exception as exc:
            self.get_logger().error(f"LLM chat failed (voice path): {exc}")

    def _note_instant(self, text: str, what: str, intent_json: str | None = None) -> None:
        """2026-09-25 (Majd, live: at drop LEFT the model said "I'm already
        at the drop right position"; carrying the tape dispenser it said "I'm
        already home and my gripper is empty"). Utterances handled by the
        instant paths never reached the model's conversation history, so its
        memory jumped from the previous LLM turn to the new question — and it
        believed that, above the live state blocks (reproduced offline 2/2,
        fixed 2/2). Each instant turn is queued here (sub-commands of one
        utterance merge) and written into the history at the start of the
        next LLM turn (_flush_instant_notes) — not now, because an LLM turn
        may be running and its tool call / tool result pair must stay
        together. ``intent_json`` is what the fast path published (None =
        nothing was sent, ``what`` is then the line Morty said)."""
        if not text:
            return
        with self._instant_notes_lock:
            if self._instant_notes and self._instant_notes[-1][0] == text:
                self._instant_notes[-1][1].append((what, intent_json))
            else:
                self._instant_notes.append((text, [(what, intent_json)]))
            del self._instant_notes[:-12]

    @staticmethod
    def _instant_tool_args(intent_json: str | None) -> dict | None:
        """2026-09-25: a published fast-path intent as the model's own tool
        input (the flat INTENT_TOOL fields; legacy compounds such as
        drop_right_release_home become confirm_drop + destination + then).
        None when it has no tool form."""
        try:
            d = json.loads(intent_json or "")
        except (TypeError, ValueError):
            return None
        if not isinstance(d, dict):
            return None
        action = str(d.get("action_intent") or "")
        attrs = d.get("object_attributes") if isinstance(d.get("object_attributes"), dict) else {}
        m = re.fullmatch(r"drop_(left|right)_release(_home)?", action)
        if m:
            return {"action_intent": "confirm_drop", "destination": f"drop_{m.group(1)}",
                    "then": "release_home" if m.group(2) else "release"}
        if action == "go_home_open":
            return {"action_intent": "go_home", "then": "release"}
        allowed = INTENT_TOOL["parameters"]["properties"]["action_intent"].get("enum") or []
        if action not in allowed:
            return None
        args: dict = {"action_intent": action}
        if d.get("object_intent"):
            args["object_intent"] = d["object_intent"]
        for k in ("colour", "shape", "position", "ordinal", "ordinal_from", "speed", "destination", "then"):
            v = attrs.get(k)
            if v not in (None, "", [], {}):
                args[k] = v
        return args

    @staticmethod
    def _instant_result(arguments: str) -> str:
        """2026-09-25 (Majd, live: after "Drop it on the right side." the
        model said "I'm not holding anything — the bottle is already at the
        drop right position": in plain English "drop it" means let go, but
        drop_* only CARRIES the object there and the robot asks before
        opening). The tool result says what the command does."""
        try:
            args = json.loads(arguments)
        except (TypeError, ValueError):
            args = {}
        a = str(args.get("action_intent") or "")
        then = str(args.get("then") or "")
        if a in ("drop_left", "drop_right", "confirm_drop", "go_home") and not then:
            return ("sent: the arm MOVES there still holding the object — it is NOT released; "
                    "the robot asks before opening the gripper")
        if a in ("drop_left", "drop_right", "confirm_drop", "go_home"):
            return f"sent: the arm moves there, then {then.replace('_', ' ')}"
        if a in ("open_gripper", "release"):
            return "sent: the gripper opens and releases the held object"
        return "sent to the robot — the ROBOT TASK STATE says whether it runs (it may be refused)"

    def _flush_instant_notes(self) -> None:
        """2026-09-25: write the queued instant turns into the LLM history
        (called under _chat_lock at the start of an LLM turn, before the
        prune) in the SAME form as the model's own turns: the operator's
        words, then the tool call(s) the fast path made with their results.
        2026-09-25 (Majd, live): the first version wrote a text note
        ("[done instantly by the fast path: release_home]") and the model
        COPIED it — it answered "Now give me the object, then go back home."
        with that text and sent nothing. A real tool call in the history is
        what the model imitates correctly. A turn that sent nothing keeps
        the line Morty said."""
        with self._instant_notes_lock:
            notes, self._instant_notes = self._instant_notes, []
        for text, items in notes:
            self.chat_agent.add_user_message(text)
            calls, lines = [], []
            for what, intent_json in items:
                args = self._instant_tool_args(intent_json)
                if args is None:
                    if not intent_json and what:
                        lines.append(what)
                    continue
                self._instant_seq += 1
                calls.append(_AdaptedFunctionCall(
                    name=INTENT_TOOL["name"], arguments=json.dumps(args),
                    call_id=f"fastpath{self._instant_seq}"))
            if lines:
                self.chat_agent.messages.append({"role": "assistant", "content": " ".join(lines)})
            for c in calls:
                self.chat_agent.messages.append(c)
            for c in calls:
                self.chat_agent.messages.append({"type": "function_call_output", "call_id": c.call_id,
                                                 "output": json.dumps({"status": self._instant_result(c.arguments)})})

    def _fire_fast_pick(self, label: str, colour: str | None, text: str) -> None:
        """The FAST-PICK publish (2026-08-26), factored out 2026-09-25 so the
        minimal layer's instant whole-sentence pick (Majd's B1) sends the
        byte-identical payload. SILENCE — the router narrates the start."""
        payload = {
            "action_intent": "pick",
            "object_intent": label,
            "object_attributes": {"colour": colour} if colour else None,
        }
        intent_json = json.dumps(payload, separators=(",", ":"))
        self.intent_pub.publish(String(data=intent_json))
        self._last_intent_json = intent_json
        self._note_intent_sent("pick")     # 2026-09-25: bookkeeping for _router_narrates
        self._note_instant(text, f"pick {label}", intent_json)   # 2026-09-25: into the LLM history
        self.get_logger().info(
            f"FAST-PICK: {text!r} -> {label}"
            f"{f' ({colour})' if colour else ''}; published {intent_json}"
        )

    def _fire_pause_negative(self, neg: str, text: str) -> None:
        """A NO to the executor's open question (2026-08-28's FAST-NEGATIVE
        layer; 2026-09-25: factored out so the minimal grammar sends the same
        payload and says the same line). close_gripper = keep holding at the
        release question, stay = stay at the home question."""
        _payload = {"action_intent": neg, "object_intent": None,
                    "object_attributes": None}
        _intent_json = json.dumps(_payload, separators=(",", ":"))
        self.intent_pub.publish(String(data=_intent_json))
        self._note_instant(text, neg, _intent_json)   # 2026-09-25: into the LLM history
        self.get_logger().info(
            f"FAST-NEGATIVE: {text!r} at pause "
            f"({getattr(self, '_cycle_state', '')!r}) -> {neg}; "
            f"published {_intent_json}"
        )
        if neg == "close_gripper":
            # v2: the arm now STAYS at the pose with the question
            # open (it used to silently carry the object home), so
            # the cycle state must KEEP saying the release pause is
            # open — otherwise a later "go home" here would not be
            # recognised as a pause answer.
            self._speak(_KEEP_HOLDING_LINE)
        else:
            if self._task is None:       # 2026-09-25: the task state owns it when known
                self._cycle_state = "IDLE"
            self._speak(_STAY_LINE)

    def _route_minimal(self, text: str, typed_at: float | None, t0: float,
                       superseded, heard_at: float | None = None,
                       epoch: int | None = None, meta_key: tuple | None = None) -> None:
        """2026-09-25 (voice_fast_layers=minimal, Majd's item 3): after stop
        and reset, only three things skip the LLM, and only when the WHOLE
        utterance is that thing — a yes/no (or, with EXT-DEST, a bare side)
        answering the question the executor has open, and a standalone
        open/close. The question comes only from the live task state and
        must be fresh: opened before the operator STARTED speaking (review
        M1 — the speech end let a "Yes." begun before the question count).
        Picks, compounds and every other sentence go to the LLM; the old
        FAST-PICK and pattern layers do not run.

        Review H3: an answer that is NOT fresh is not acted on and is not
        handed to the LLM either (the model read it as consent and opened
        the gripper anyway) — Morty asks the question again."""
        task = self._live_task() or {}   # one snapshot: the executor thread replaces it
        question = str(task.get("question") or "")
        fresh = False
        start = None
        if question:
            if meta_key is not None:
                # 2026-09-25 (G1): a merged input began with its first fragment
                start = self._speech_start_for(meta_key[0], typed_at, meta_key[1], allow_used=True)
            else:
                start = self._speech_start_for(text, typed_at, heard_at)
            if start is None and not self._utt_meta_seen:
                # No utterance meta has EVER arrived: the speech node does not
                # send them (an old build, Vosk). Freshness cannot be judged,
                # and "not fresh" would re-ask forever — the legacy layers
                # answer, as before the overhaul.
                self.get_logger().info(
                    "no utterance metas from the speech node — legacy layers answer "
                    f"{text!r} at question {question!r}")
                self._route_legacy(text, t0, superseded,
                                   self._stop_epoch if epoch is None else epoch)
                return
            try:
                since = float(task.get("question_since") or 0.0)
            except (TypeError, ValueError):
                since = 0.0
            # 2026-09-25 (Majd, live: "Now give it." began 38 ms before the
            # release question opened — he answered as the arm arrived — and
            # was re-asked; speech_start also includes the 300 ms pre-roll):
            # an utterance begun up to _FRESH_TOLERANCE_S before the question
            # opened still answers it.
            fresh = start is not None and 0.0 < since <= start + _FRESH_TOLERANCE_S
        kind, action = voice_grammar.classify(text, question, fresh, self._voice_ext_dest)
        if kind == "negative" and action == "keep":
            # 2026-09-25 (Majd's I1): a NO to "Where should I drop it?" — the
            # executor keeps waiting for a destination, so nothing is sent.
            self._speak(_KEEP_HOLDING_LINE)
            self._note_instant(text, _KEEP_HOLDING_LINE)   # nothing sent: the line said
        elif kind == "negative":
            self._fire_pause_negative(action, text)
        elif kind in ("answer", "gripper"):
            self._fire_fast_command(action, "", text)
        if kind != "llm":
            self.get_logger().info(
                f"[timing] handled in {time.monotonic() - t0:.2f}s (route=GRAMMAR "
                f"{kind} {action}, question={question!r}, fresh={fresh})")
            return
        if not question and not task.get("holding"):
            # 2026-09-25 (Majd's B1: "picks via the LLM took 1.3-2.8 s"): a
            # WHOLE plain pick of a known object, with no question open and
            # nothing held, is published at once — the legacy FAST-PICK
            # payload, from the same tables (voice_grammar.match_pick).
            pick = voice_grammar.match_pick(text)
            if pick is not None:
                self._fire_fast_pick(pick[0], pick[1], text)
                self.get_logger().info(
                    f"[timing] handled in {time.monotonic() - t0:.2f}s (route=GRAMMAR pick "
                    f"{pick[0]}{f' ({pick[1]})' if pick[1] else ''})")
                return
        if (question and not fresh and voice_grammar.classify(
                text, question, True, self._voice_ext_dest)[0] in ("answer", "negative")):
            # 2026-09-25 (review H3): a bare answer begun before the question
            # opened (or whose start is unknown) answers something else —
            # ask again; nothing is published and the LLM is not asked.
            line = _REASK_LINES.get(question, "")
            self.get_logger().info(
                f"question {question!r} opened after this utterance began "
                f"(start={start}, since={task.get('question_since')}) — not a fresh "
                f"answer; asking again: {text!r}")
            if line:
                self._speak(line)
            return
        if superseded():
            self.get_logger().info(
                f"Superseded by newer transcript before LLM; dropping: {text!r}")
            return
        try:
            # 2026-09-25 (review H1): re-checked once _chat_lock is held
            self.handle_user_text(text, superseded=superseded, epoch=epoch)
        except Exception as exc:
            self.get_logger().error(f"LLM chat failed (voice path): {exc}")

    _FAST_MULTI = {
        # Compound utterances that mean TWO sequential actions — the LLM
        # keeps dropping the second one ("go home and close the gripper"
        # went home, never closed; 2026-08-26). The router queue
        # serializes them in order.
        "go_home_close": ("go_home", "close_gripper"),
        "go_home_open": ("go_home", "open_gripper"),
    }

    def _fire_fast_command(self, action: str, speed: str, text: str) -> None:
        """Publish an intent for a recognized direct command, bypass the LLM,
        and trigger the matching QUICK_NARRATION TTS line."""
        if action == "go_home_open" and self._held_object_label:
            # Holding an object: "go home and open" is ONE order to the
            # executor (destination home, release on arrival). Split into
            # two intents, the open fired standalone mid-carry and the
            # executor still asked at home (2026-09-18).
            intent_payload = {"action_intent": "go_home", "object_intent": None,
                              "object_attributes": {"then": "release", **({"speed": speed} if speed else {})}}
            intent_json = json.dumps(intent_payload, separators=(",", ":"))
            self.intent_pub.publish(String(data=intent_json))
            self._note_instant(text, "go_home then release", intent_json)   # 2026-09-25: into the LLM history
            self.get_logger().info(f"FAST: matched {text!r} -> go_home then release (holding); published {intent_json}")
            self._last_intent_json = intent_json
            if self._router_narrates("go_home"):
                # 2026-09-25 (Majd's B2): "Drop it home, give me the object."
                # at the destination question — the router announces it once
                # the executor accepted it (see _router_narrates).
                self.get_logger().info("Withholding 'Going home and opening the gripper there.' "
                                       "— intent_router narrates it on acceptance.")
            else:
                self._speak("Going home and opening the gripper there.")
            return
        if action in self._FAST_MULTI:
            for sub in self._FAST_MULTI[action]:
                self._fire_fast_command(sub, speed, text)
            return
        attrs: dict[str, Any] = {}
        if speed:
            attrs["speed"] = speed
        intent_payload = {
            "action_intent": action,
            "object_intent": None,
            "object_attributes": attrs or None,
        }
        intent_json = json.dumps(intent_payload, separators=(",", ":"))
        msg = String()
        msg.data = intent_json
        self.intent_pub.publish(msg)
        self._note_instant(text, action, intent_json)    # 2026-09-25: into the LLM history
        self.get_logger().info(
            f"FAST: matched {text!r} -> {action} (speed={speed or 'default'}); "
            f"published {intent_json}"
        )
        # Save for redo and play the quick narration line.
        if action not in {"stop", "cancel", "abort"}:
            self._last_intent_json = intent_json
        self._note_intent_sent(action)
        quick = QUICK_NARRATION.get(action)
        if self._router_narrates(action):
            # Reality, not intent: intent_router narrates this one once the
            # gripper (or the executor's 'released' event) actually reports
            # — and (2026-09-25) a destination said while a pick is in
            # flight once the executor accepted it ("Going to drop right."
            # on resumed:).
            self.get_logger().info(
                f"Withholding pre-emptive narration {quick!r} for {action!r} "
                "— intent_router speaks it only after the action is "
                "confirmed (2026-08-28 'said Opening the gripper but did "
                "not open the gripper')."
            )
        elif quick:
            self._speak(quick, interrupt=(action == "stop"))
        if action == "start_all":
            # Announce readiness when everything actually comes up
            # (Majd 2026-08-26): checked in _proactive_tick.
            self._announce_ready_until = (
                self.get_clock().now().nanoseconds / 1e9 + 180.0)

    def _set_mic_muted(self, muted: bool) -> None:
        if self._mute_pub is None:
            return
        msg = Bool()
        msg.data = bool(muted)
        self._mute_pub.publish(msg)

    def handle_user_text(self, user_text: str, superseded=None, epoch: int | None = None) -> None:
        # 2026-09-25: the stop fast path that lived here is gone — stop and
        # reset are decided once, before this, in _handle_voice_input (it
        # also sent a SECOND stop when the LLM then called the tool).
        # Fast-path: "do it again" — re-publish the last non-control intent
        # immediately so we don't pay the LLM round-trip cost. Still hand
        # the text to the LLM so it acknowledges in conversation.
        # 2026-09-25: legacy layers only (voice_fast_layers=legacy).
        if self._voice_fast_layers == "legacy" and _is_redo_command(user_text):
            self._fire_redo()
        with self._chat_lock:
            self._turn_text = user_text      # 2026-09-25 (G1): see _hush_and_merge
            # 2026-09-25 (review H1): this turn may have waited here while
            # another ran; a stop / reset sent since the input arrived, or a
            # newer transcript, makes it stale — "grasp the bottle" then
            # "Stop!" must never start the grasp after the stop.
            if epoch is not None and epoch != self._stop_epoch:
                self.get_logger().info(
                    f"A stop/reset came after this input; dropping it: {user_text!r}")
                return
            if superseded is not None and superseded():
                self.get_logger().info(
                    f"Superseded by newer transcript while waiting for the chat lock; "
                    f"dropping: {user_text!r}")
                return
            self._answer_seq += 1
            self._answer_running = True
            try:
                self._handle_user_text_locked(user_text)
            finally:
                self._answer_running = False

    def _fire_redo(self) -> None:
        if not self._last_intent_json:
            self.get_logger().info("REDO: nothing to repeat yet.")
            return
        try:
            msg = String()
            msg.data = self._last_intent_json
            self.intent_pub.publish(msg)
            self.get_logger().info(
                f"REDO: re-published last intent {self._last_intent_json}"
            )
        except Exception as exc:
            self.get_logger().warn(f"REDO: publish failed: {exc}")

    def _fire_stop(self) -> None:
        # Cancel chat_node's own ExecuteGrasp goal if any.
        with self._grasp_lock:
            handle = self._active_grasp_handle
        if handle is not None:
            try:
                handle.cancel_goal_async()
                self.get_logger().info("STOP: cancelled in-flight ExecuteGrasp.")
            except Exception as exc:
                self.get_logger().warn(f"STOP: cancel failed: {exc}")

        # Publish a stop intent so intent_router cancels its own goals too.
        try:
            stop_payload = json.dumps(
                {"action_intent": "stop", "object_intent": "", "object_attributes": {}}
            )
            msg = String()
            msg.data = stop_payload
            self.intent_pub.publish(msg)
            self.get_logger().info("STOP: published stop intent.")
        except Exception as exc:
            self.get_logger().warn(f"STOP: publish failed: {exc}")

    def _fire_reset(self) -> str:
        """2026-09-25 (Majd: "reset = what the GUI 'Reset pick' does: end the
        pick, go home keeping any held object, clear selections/queue — NOT
        a freeze"). Publishes the voice reset intent of SPEC §1.4 byte for
        byte; the router turns it into the executor reset. Returns the JSON."""
        payload = json.dumps({"action_intent": "reset", "object_intent": None,
                              "object_attributes": None}, separators=(",", ":"))
        try:
            self.intent_pub.publish(String(data=payload))
            self.get_logger().info(f"RESET: published {payload}")
        except Exception as exc:
            self.get_logger().warn(f"RESET: publish failed: {exc}")
        return payload

    def _publish_chat_state(self, state: str) -> None:
        """Broadcast the current conversation phase on /chat/state. The
        visualizer uses this to show 'thinking…' while we wait for the
        LLM, so the user has visible feedback during the round-trip."""
        if state == self._chat_state:
            return
        self._chat_state = state
        try:
            self.state_pub.publish(String(data=state))
        except Exception:
            pass

    def _add_robot_context(self) -> None:
        """What the LLM is told about the robot every turn: the scene, what
        the operator points at, the robot's own recent announcements, and —
        last, so they outrank everything before them — the live gripper and
        arm state (2026-09-24: the gripper state moved below the
        announcements, which no longer carry the model's own replies)."""
        scene = self._scene_summary()
        if scene:
            # Provide perception context to the LLM as a developer-role message
            # right before the user turn. Refreshed every turn so it tracks the
            # current scene rather than going stale in the chat history.
            self.chat_agent.add_developer_message(scene)

        if self._pointed_object_label:
            # The user's finger is currently over a detection. Tell the
            # LLM so phrases like "pick this one" / "this object" / "that
            # one" resolve to a specific label, and so the LLM can
            # acknowledge ("You're pointing at the wooden maze puzzle —
            # should I pick it?") if the user asks.
            pointed_human = self._pointed_object_label.replace("_", " ")
            self.chat_agent.add_developer_message(
                f"User pointing context: the operator is currently "
                f"pointing at the '{self._pointed_object_label}' "
                f"({pointed_human}) detection. If the user says 'pick "
                f"this one' / 'pick that' / 'this object' / 'grab this' "
                f"without naming an object, resolve to "
                f"'{self._pointed_object_label}'. If the user asks 'what "
                f"am I pointing at?', answer with '{pointed_human}'."
            )
        _st_hist = getattr(self, "_status_history", [])
        # 2026-09-25: with the executor's task state known, its block (last,
        # below) replaces the cycle-state clause here and both "Robot state:
        # gripper …" messages; the announcements stay as history.
        _task_known = self._task is not None
        _state_name = "ROBOT TASK STATE" if _task_known else "Robot state"
        if _st_hist:
            _now_s = self.get_clock().now().nanoseconds / 1e9
            _lines = "; ".join(
                f"[-{max(0.0, _now_s - _t):.0f}s] {_x}" for _t, _x in _st_hist)
            _cs = getattr(self, "_cycle_state", "unknown")
            if not self._held_object_label and "HOLDING" in _cs:
                _cs = "gripper open, holding nothing"   # the gripper topic outranks an old phrase
            self.chat_agent.add_developer_message(
                "ROBOT ANNOUNCEMENTS (what the robot itself said recently, "
                "newest last; your own replies are not in it): "
                + _lines + ("" if _task_known else ". Current cycle state: " + _cs)
                + ". Use them to know what the robot is doing — if they say "
                "a grasp is running, a grasp IS running even if you never "
                "commanded one this turn. The older a line, the more it may "
                f"be outdated: where a line disagrees with the '{_state_name}' "
                "or 'Robot arm state' messages below, the line is OLD and "
                "those messages are right. "
                "NO PARROTING: these status lines are CONTEXT, not a "
                "script. NEVER re-emit one verbatim (or near-verbatim) as "
                "your own reply — repeating 'I'm at the home position. "
                "Should I open the gripper?' back at the operator who just "
                "answered it is a bug, not an answer (2026-08-28: the same "
                "line went out four times while the operator said 'No', "
                "'No, close the gripper', 'Don't open it'). If the "
                "operator answered a question in the feed, ACT on their "
                "answer or say something NEW; if they said NO to opening "
                "the gripper, emit close_gripper — never re-ask. "
                f"FRESHNESS: the '{_state_name}' / 'Robot arm state' messages "
                "below describe the robot NOW and OVERRIDE anything said "
                "earlier — these announcements and your own previous "
                "replies included."
            )
        if _task_known:
            pass     # 2026-09-25: the ROBOT TASK STATE block below says what the gripper holds
        elif self._held_object_label:
            # Pronoun grounding: tell the LLM what's currently in the gripper
            # so "drop it on the left" / "stell es ab" can resolve "it".
            human_label = self._held_object_label.replace("_", " ")
            self.chat_agent.add_developer_message(
                f"Robot state: gripper is currently holding a {human_label}. "
                f"If the user says 'it' / 'that' / 'this one', resolve to "
                f"the {human_label}. This is LIVE and OVERRIDES anything "
                f"earlier in the conversation, including your own previous "
                f"replies."
            )
        else:
            # 2026-08-28: after a release the operator said "you are not
            # holding it anymore" and Morty kept insisting "I'm still
            # holding the small bottle" — the absence of a held-object
            # message left the last-mentioned object standing.
            self.chat_agent.add_developer_message(
                "Robot state: the gripper is EMPTY — it is holding "
                "NOTHING right now. This is LIVE and OVERRIDES anything "
                "earlier in the conversation, including your own previous "
                "replies. Never claim to be holding an object; if the user "
                "says 'it' without a fresh referent, ask which object."
            )
        # 2026-09-22 (Majd: "I sent the robot home with the GUI and he
        # still thinks he is at detect right"): the router's latched label
        # follows only the router's OWN named-pose moves; a GUI or executor
        # move leaves it stale. The joints are the truth. Nearest named
        # pose by joint-space distance: "at" below 0.15 rad, "near" below
        # 0.6 rad, otherwise between poses (mid pick cycle). The label is
        # a fallback for the moment before the first joint sample only.
        now_s = self.get_clock().now().nanoseconds / 1e9
        nearest = self._nearest_named_pose() if (now_s - self._last_joint_state_at) < 3.0 else None
        tail = (
            " This is the arm's joint pose, LIVE, and OVERRIDES anything said "
            "earlier in this conversation. Use it ONLY when the user asks about "
            "the arm's position ('which position are you at?', 'are you home?'); "
            "the generic 'where are you?' is about your location (Hochschule "
            "Reutlingen). A request to go somewhere is ALWAYS a move: emit the "
            "intent even when the arm is already there and NEVER answer "
            "'you're already at ...' — moving to a pose the arm is at is "
            "harmless, refusing is not."
        )
        if nearest is not None:
            name, dist = nearest
            human = name.replace("_", " ")
            if self._current_pose_label and self._current_pose_label != name and dist < 0.15:
                self.get_logger().warn(
                    f"stale pose label {self._current_pose_label!r}: joints say {name!r} "
                    f"({dist:.2f} rad) — the joints win.")
            if dist < 0.15:
                where = f"the arm is AT named pose '{name}' (say: at {human})"
            elif dist < 0.6:
                where = (f"the arm is NEAR named pose '{name}' ({dist:.2f} rad away), "
                         f"not exactly at it (say: near {human})")
            else:
                _cs = (getattr(self, "_cycle_state", "") or "").lower()
                place = ("at the grasp position over the table" if "grasp pose" in _cs else
                         "at the drop position" if "drop pose" in _cs else
                         f"between named poses, nearest '{name}' ({dist:.2f} rad away)")
                where = f"the arm is NOT at any named pose: it is {place} (do not claim a pose name)"
            self.chat_agent.add_developer_message(f"Robot arm state: {where}.{tail}")
        elif self._current_pose_label:
            self.chat_agent.add_developer_message(
                f"Robot arm state: last completed named move ended at "
                f"'{self._current_pose_label}' (no fresh joint state to confirm).{tail}")
        else:
            self.chat_agent.add_developer_message(
                "Robot arm state: unknown (no joint state yet). If asked which "
                "position the arm is at, say so honestly and do not guess." + tail)
        # 2026-09-25: LAST, so it outranks everything before it.
        self.chat_agent.add_developer_message(self._task_state_block())

    # 2026-09-25: the executor's phases in words, for the LLM's task-state block.
    _PHASE_WORDS = {
        "probing": "is starting (checking the grasp for the {obj})",
        "approaching": "is running: the arm is moving in to grasp the {obj}",
        "no_grasp": "missed: the gripper closed on nothing",
        "awaiting_destination": "is paused: holding the {obj} at the grasp pose",
        "carrying": "is running: carrying the {obj} to {dest}",
        "awaiting_release": "is paused: holding the {obj} at {dest}",
        "releasing": "is running: opening the gripper at {dest}",
        "awaiting_home_confirm": "is paused at the drop pose",
        "returning_home": "is running: the arm is returning home",
        "stopped": "was STOPPED by the operator: the arm is frozen where it was",
    }
    _QUESTION_WORDS = {
        "destination": ('OPEN QUESTION: "Where should I drop it? Left, right, or home?" — '
                        "answer with confirm_drop (destination drop_left / drop_right / home)"),
        # 2026-09-25 (Majd's live test): any consent = open_gripper; never "Left or right?"
        "release": ('OPEN QUESTION: "Should I open the gripper?" (yes/no) — any consent '
                    "(yes / open / give it to me) = open_gripper, no = close_gripper, a named "
                    'side = drop_left / drop_right; never ask "Left or right?"'),
        # 2026-09-25 (review Y2): stay, not stop — a NO keeps the arm where
        # it is with "Okay, I'll stay here." instead of "Stopping." + abort.
        "home": ('OPEN QUESTION: "Should I go back home?" — yes / definitely = go_home, '
                 "no / stay = stay, go to the left / right = drop_left / drop_right "
                 "(it moves there instead of home)"),
    }
    # 2026-09-25 (review X3): the no-grasp confirm is the home question with
    # no destination and nothing held — the arm is lifted above the grasp
    # spot, not at a drop pose.
    _NO_GRASP_HOME_WORDS = ("missed the grasp — lifted above the grasp position, "
                            "gripper empty, asking whether to go home")

    @staticmethod
    def _is_no_grasp_home(t: dict) -> bool:
        """2026-09-25 (review X3): the task state is the no-grasp home
        question (phase awaiting_home_confirm / no_grasp, question home, no
        destination, nothing held)."""
        return (str(t.get("question") or "") == "home"
                and str(t.get("phase") or "") in ("awaiting_home_confirm", "no_grasp")
                and not str(t.get("destination") or "") and not t.get("holding"))

    def _log_llm_context(self, user_text: str) -> None:
        """2026-09-28 (Majd, live: "The rightmost vitamin." after the robot's
        which-one question got "Left or right?" and no pick — not
        reproducible offline): append the model's exact input for this turn
        to ~/morty_ws/logs/llm_context.jsonl, the static system prompt
        replaced by a marker, so a wrong answer can be replayed. Off with
        MORTY_LLM_CONTEXT_LOG=0; a failure here never costs the turn."""
        if os.environ.get("MORTY_LLM_CONTEXT_LOG", "1") == "0":
            return
        try:
            msgs = []
            for m in self.chat_agent.messages:
                if isinstance(m, dict):
                    m = dict(m)
                    if m.get("content") == SYSTEM_PROMPT:
                        m["content"] = "<SYSTEM_PROMPT>"
                    msgs.append(m)
                else:
                    msgs.append({"type": getattr(m, "type", "?"), "name": getattr(m, "name", None),
                                 "arguments": getattr(m, "arguments", None),
                                 "call_id": getattr(m, "call_id", None),
                                 "text": getattr(m, "text", None),
                                 "content": (lambda c: c if isinstance(c, str) else
                                             [getattr(x, "text", str(x)) for x in (c or [])])(
                                     getattr(m, "content", None))})
            rec = {"t": time.time(), "model": self.model, "user": user_text, "messages": msgs}
            with open(os.path.expanduser("~/morty_ws/logs/llm_context.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug(f"llm context log failed: {exc}")

    def _task_state_block(self) -> str:
        """2026-09-25 (SPEC C3): the one developer message that tells the LLM
        the pick's state, built from the executor's task state — the only
        authority on the pick (the announcements feed is history and the
        model's own earlier replies are not evidence). Unknown until the
        executor reports (old executor): then the old held-object messages
        above are still sent."""
        t = self._task
        if t is None:
            return "ROBOT TASK STATE: unknown — the executor has not reported."
        phase = str(t.get("phase") or "idle")
        cid = t.get("cycle_id") or 0
        obj = str(t.get("object") or "").replace("_", " ") or "object"
        dest_name = str(t.get("destination") or "").replace("_", " ")
        dest = f"the {dest_name} pose" if dest_name else "the drop pose"
        if phase == "idle":
            head = "No pick is running (idle)"
        elif self._is_no_grasp_home(t):
            head = f"Pick #{cid} " + self._NO_GRASP_HOME_WORDS    # 2026-09-25 (review X3)
        else:
            words = self._PHASE_WORDS.get(phase, phase.replace("_", " "))
            head = f"Pick #{cid} " + words.format(obj=obj, dest=dest)
        q = str(t.get("question") or "")
        qtext = self._QUESTION_WORDS.get(q, "No question is open")
        if t.get("holding"):
            grip = f"Gripper: closed on the {obj} (not opened since; no sensor confirms contents)"
        else:
            grip = f"Gripper: {str(t.get('gripper') or 'unknown')}, holding NOTHING"
        last = t.get("last") or {}
        if isinstance(last, dict) and last.get("cycle_id"):
            reason = str(last.get("reason") or "")
            tail = (f"Last pick #{last.get('cycle_id')}: {last.get('outcome') or 'unknown'}"
                    + (f": {reason}" if reason else "")
                    + (", arm at home" if last.get("at_home") else ""))
        else:
            tail = "No pick has finished yet"
        return ("ROBOT TASK STATE (live, from the executor — the only authority on the "
                "pick; overrides the announcements, the conversation and your own "
                f"earlier replies): {head}. {qtext}. {grip}. {tail}.")

    def _handle_user_text_locked(self, user_text: str) -> None:
        # Reset per-turn flags. _handle_intent_tool_call sets
        # _intent_fired_this_turn=True if the LLM extracted an action; we
        # use that below to skip the slow final-reply TTS.
        self._intent_fired_this_turn = False
        # Per-turn buffer of (action, quick_narration_text) tuples. We
        # collect these as the LLM emits intents and decide AT THE END
        # whether to speak them — so a single-intent turn still gets
        # the snappy quick narration, but a multi-intent turn lets the
        # LLM's combined reply do the talking (no "Going home." +
        # "Going to drop right." stutter before the joke).
        self._pending_quick_narrations: list[tuple[str, str]] = []
        # 2026-09-25 (SPEC C8): this turn's talk / unclear / real-intent
        # marks, the router-narrated placeholders, and where the talk turn's
        # follow-up started streaming.
        self._talk_turn = False
        self._unclear_turn = False
        self._real_intent = False
        self._withheld_this_turn = set()
        self._interrupt_line_turn = False
        self._stream_mark = 0
        self._publish_chat_state("thinking")

        # Inject today's date / weekday / local time as a developer
        # message every turn. LLMs have a training cutoff and otherwise
        # confidently answer "what day is it?" with stale data — this
        # gives them an authoritative real-time anchor. Local wall clock
        # is good enough; if a timezone-aware answer is needed later we
        # can switch to zoneinfo and a configurable param.
        # Shed last turn's state blocks and the stale conversation tail
        # BEFORE injecting this turn's fresh state (see the docstring).
        # 2026-09-25: the turns the instant paths handled since the last LLM
        # turn go into the history first (see _note_instant).
        self._flush_instant_notes()
        self._prune_llm_context()
        now_local = _dt.datetime.now()
        self.chat_agent.add_developer_message(
            f"Real-time clock: today is {now_local.strftime('%A, %d %B %Y')}, "
            f"local time {now_local.strftime('%H:%M')}. Use these exact "
            f"values if the user asks about the date, day of the week, "
            f"or current time."
        )

        # Live component status FIRST so the LLM never invents system
        # state ("Yes, I'm running!" while the driver was down).
        self.chat_agent.add_developer_message(self._system_status_summary())

        # What the robot's voice recently said out loud (status narrations
        # + own replies) so the LLM (a) can answer "what's happening?" and
        # (b) recognises garbled mic echo of these lines instead of asking
        # the user what they meant.
        if self._recent_spoken:
            now_s = self.get_clock().now().nanoseconds / 1e9
            lines = "; ".join(
                f"[{max(0, int(now_s - t))}s ago] \"{s}\""
                for t, s in self._recent_spoken[-6:]
            )
            self.chat_agent.add_developer_message(
                "RECENTLY SPOKEN ALOUD through the robot's speaker (status "
                f"announcements and your own replies): {lines}. These were "
                "NOT said by the user. IMPORTANT: an ANSWER or COMMAND that "
                "reuses your words IS the user replying — 'yes, open the "
                "gripper' / 'give me the object' right after you asked "
                "'Should I open the gripper?' means CALL the intent tool "
                "(open_gripper), never re-ask the question. Treat a "
                "transcript as mic echo ONLY when it is a meaningless "
                "fragment of one of these lines with no command or answer "
                "content ('Starting Greif.'). When unsure, act on the "
                "user's words. You may reference these announcements when "
                "the user asks what is happening. They are the PAST: the "
                # 2026-09-25: with the task state known its block is the NOW
                + ("'ROBOT TASK STATE'" if self._task is not None else "'Robot state'")
                + " and 'Robot arm state' messages further below "
                "describe the robot NOW."
            )

        self._add_robot_context()
        if self._interrupt_note:
            self.chat_agent.add_developer_message(self._interrupt_note)
            self._interrupt_note = ""
        self.chat_agent.add_user_message(user_text)
        self._log_llm_context(user_text)
        try:
            _t_llm0 = time.monotonic()
            self._t_llm0 = _t_llm0
            self._early_dispatched = {}
            self._stream_buf = ""
            self._streamed: list[str] = []
            response = self.chat_agent.chat(model=self.model, on_tool_call=self._early_intent_call,
                                            on_text=self._on_reply_text,
                                            on_restart=self._on_llm_restart)
            cached = getattr(response, "cache_read_tokens", None)
            self.get_logger().info(
                f"[timing] LLM main call {time.monotonic() - _t_llm0:.2f}s"
                + (f" (prompt cache: {cached} tokens read)" if cached else ""))
        except Exception as exc:
            # OpenAI hiccup (timeout, network, rate limit). Don't deadlock
            # the rclpy executor — apologize and bail out of this turn.
            # Pop the user message we just added so the chat history stays
            # consistent (otherwise next turn's history has a user message
            # with no assistant reply, which can confuse the model).
            self.get_logger().warn(f"LLM call failed: {exc}")
            try:
                self.chat_agent.messages.pop()
            except Exception:
                pass
            self._speak("Sorry, I couldn't reach the language model just now. Please try again.")
            self._publish_chat_state("idle")
            return
        self.chat_agent.add_assistant_response(response)

        # Drain ALL intent_extractor function_calls. The model may emit several
        # in one response (compound commands like "pick X then go home") or
        # produce more in follow-up rounds after seeing prior outputs. Cap iters
        # so a buggy prompt can't loop forever.
        max_rounds = 4
        for _ in range(max_rounds):
            if self._reply_hushed():
                self._finish_interrupted_turn(response)
                return
            calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
                and item.name == "intent_extractor"
            ]
            if not calls:
                break
            only_unclear = True
            for call in calls:
                if self._call_action(call) != "unclear":
                    only_unclear = False
                early = self._early_dispatched.pop(call.call_id, None) if getattr(self, "_early_dispatched", None) else None
                if early is not None:
                    # Dispatched while the reply streamed; only the tool
                    # output still has to go into the history, in order.
                    self.chat_agent.add_function_call_output(call, early)
                    continue
                self._handle_intent_tool_call(call)
            if only_unclear and not self._real_intent:
                # 2026-09-25: an 'unclear' turn ends here — outputs recorded,
                # no follow-up call (it would only produce more words); the
                # end of the turn speaks UNCLEAR_LINE alone.
                break
            try:
                _t_llm1 = time.monotonic()
                if self._talk_turn:
                    # 2026-09-25: TALK WHILE ACTING — the story comes in the
                    # follow-up; stream it sentence by sentence like a plain
                    # answer, and remember where it started so the end of
                    # the turn does not repeat it.
                    self._stream_buf = ""
                    self._stream_mark = len(self._streamed)
                    response = self.chat_agent.chat(model=self.model, on_text=self._on_reply_text,
                                                    on_restart=self._on_llm_restart)
                else:
                    response = self.chat_agent.chat(model=self.model, on_restart=self._on_llm_restart)
                self.get_logger().info(
                    f"[timing] LLM follow-up call "
                    f"{time.monotonic() - _t_llm1:.2f}s")
            except Exception as exc:
                self.get_logger().warn(f"LLM follow-up call failed: {exc}")
                self._speak("Sorry, I lost the language model mid-thought. Please try again.")
                self._publish_chat_state("idle")
                return
            self.chat_agent.add_assistant_response(response)

        if self._reply_hushed():
            self._finish_interrupted_turn(response)
            return
        assistant_text = getattr(response, "output_text", "")
        if (not self._real_intent and not self._unclear_turn and assistant_text
                and _UNCLEAR_SENTENCE_RE.search(assistant_text.replace("’", "'"))):
            # 2026-09-25 (review M4): the model said "I didn't catch that" in
            # a sentence the stream never cut (the reply's last one) and
            # called no tool — dropping that sentence below left silence.
            # It is an unclear turn: the line is said from code.
            self._unclear_turn = True
            self.get_logger().info(
                f"model reply says it did not catch the input — unclear turn: {assistant_text!r}")
        if self._unclear_turn and not self._real_intent:
            # 2026-09-25 (Majd: "'I didn't catch that' must come from CODE and
            # be spoken ALONE"): whatever this turn streamed before the model
            # decided it was garbled is cancelled, and the one line is said.
            # cut=True (review Y6): also the sentences still on their way.
            cancelled = self._cancel_answer_goals(self._answer_seq, cut=True)
            print(f"Assistant: {UNCLEAR_LINE}")
            self.assistant_pub.publish(String(data=UNCLEAR_LINE))
            self.get_logger().info(
                f"unclear turn: {cancelled} streamed sentence(s) cancelled, "
                f"model text withheld ({(assistant_text or '').strip()!r}); speaking the line")
            self._speak(UNCLEAR_LINE, answer=True)
            self._publish_chat_state("idle")
            return
        # 2026-09-25: a model sentence saying "I didn't catch that" next to a
        # real command is dropped — never glued to a real answer.
        assistant_text = self._drop_marker_sentences(assistant_text)
        if not self._real_intent and assistant_text:
            # 2026-09-25 (Majd, live: "Now back home." → the model wrote
            # "Going home." WITHOUT a tool call; it was not spoken, but it
            # stayed in the conversation, so the next "Go back home." got
            # "You're already heading home" and again no command). A turn
            # that sent nothing keeps no action claim: not in the GUI text,
            # and not in the history the model reads next turn (the history
            # holds these very text items, so they are rewritten in place).
            stripped = self._drop_claim_sentences(assistant_text)
            if stripped != assistant_text:
                for item in (getattr(response, "output", None) or []):
                    if getattr(item, "type", None) == "message":
                        item.content = stripped or "(No command was sent — nothing moved.)"
                assistant_text = stripped
        if assistant_text:
            print(f"Assistant: {assistant_text}")
            msg = String()
            msg.data = assistant_text
            self.assistant_pub.publish(msg)

        trimmed = (assistant_text or "").strip()
        pending = list(self._pending_quick_narrations)
        n_intents = len(pending)
        # If EVERY intent this turn is confirmation-owned (gripper), this
        # node says nothing at all — not the quick line, and not the LLM's
        # own "Opening the gripper." prose, which is just the same
        # unearned claim in longer words. intent_router speaks once the
        # gripper actually reports, or says why it did not move.
        # The assistant text is still published above, so the GUI/chat
        # transcript keeps it; only the TTS is withheld.
        # 2026-09-25: "confirmation-owned" is now every action the router
        # narrates (_router_narrates, recorded per turn in
        # _withheld_this_turn) — except in a talk turn, whose prose is the
        # story the operator asked for.
        if (pending and all(a in self._withheld_this_turn for a, _ in pending)
                and not self._talk_turn):
            self.get_logger().info(
                "Withholding end-of-turn TTS: narration for "
                f"{[a for a, _ in pending]} is driven by intent_router "
                "confirmation, not by intent."
            )
            self._publish_chat_state("idle")
            return
        # Decide what to actually speak:
        #   - 0 intents this turn  -> just speak the LLM reply (if any)
        #   - 1 intent + rich reply -> speak quick narration first, then LLM reply
        #   - 1 intent + empty/echo reply -> speak quick narration only
        #   - >=2 intents + rich reply -> SKIP quick narrations; LLM reply covers it
        #   - >=2 intents + empty/echo reply -> speak quick narrations sequentially
        # 2026-09-25 (review M3): the list is module level (_ACTION_ECHOES),
        # shared with the streamed-sentence claim filter.
        reply_is_rich = bool(trimmed) and not any(
            trimmed.lower().startswith(a) and len(trimmed) <= 40
            for a in _ACTION_ECHOES
        )

        # sentences streamed early are not repeated (2026-09-25: in a talk
        # turn only those the follow-up streamed — the final text is its own)
        tail = self._unspoken_tail(assistant_text, since=self._stream_mark)
        if n_intents < 2:
            # 2026-09-25 (review M3): the model's action claims ("Going home.",
            # "Picking up the X.") are never spoken — the QUICK line (spoken
            # on dispatch) or the router (on acceptance) says what happens.
            # A multi-intent turn keeps its combined plan sentence: the
            # prompt makes it the replacement of the QUICK lines, and it is
            # spoken only now, after every command was published.
            tail = self._drop_claim_sentences(tail)
        if n_intents == 0:
            # Pure conversational reply, always speak it if non-empty.
            # 2026-09-25: except after an LLM stop/reset — "Stopping." /
            # "Resetting." was said at once and nothing else follows it.
            if tail and not self._interrupt_line_turn:
                self._speak(tail, answer=True)
        elif n_intents == 1:
            # Quick narration immediately, then the LLM reply if it adds
            # value. The action-echo check protects against the LLM
            # accidentally repeating the quick narration (e.g. "Going
            # home.") which would just bounce back through the mic.
            _, quick = pending[0]
            self._speak(quick, answer=True)
            if reply_is_rich and tail:
                self._speak(tail, answer=True)
        else:
            # Multi-intent: the SYSTEM_PROMPT instructs the LLM to write
            # a single combined plan announcement that REPLACES the
            # per-intent narrations. Trust any non-empty reply — the
            # action-echo check would mis-classify legitimate combined
            # plans like "Going home, then looking right." because they
            # naturally start with the first action name.
            if tail:
                self._speak(tail, answer=True)
            else:
                # LLM gave nothing — fall back to per-intent narrations
                # so the user still hears what's happening.
                for _, quick in pending:
                    self._speak(quick, answer=True)
        # LLM round-trip done — back to waiting state.
        self._publish_chat_state("idle")

    def _speak(self, text: str, *, interrupt: bool = False, answer: bool = False) -> None:
        """Send a text to the speak action server.

        ``interrupt=False`` (default) waits politely for whatever is
        currently being spoken to finish so we don't chop our own
        sentences off mid-word. Pass ``interrupt=True`` only for
        explicit cancel/stop narrations or when the user clearly
        wants to barge in.
        """
        if text:
            self._note_act()        # 2026-09-25 (Majd's G1): this input has now answered
        if not self._tts_enabled or self._speak_client is None or not text:
            return
        if not self._speak_client.server_is_ready():
            if not self._speak_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().warn(
                    "Speak action server not available; skipping TTS for this reply"
                )
                return

        # Remember what the AI is about to say so we can drop matching
        # transcripts caused by speaker→mic feedback.
        self._record_ai_speech(text)
        if answer:   # the model's own words: never robot status (see __init__)
            now_m = time.monotonic()
            self._own_answer_lines = {t: ts for t, ts in self._own_answer_lines.items()
                                      if now_m - ts < 120.0}
            self._own_answer_lines[text.strip()] = now_m
        # Mark "responding" so the visualizer shows we're speaking.
        self._publish_chat_state("responding")

        # Mute the mic only if mute_during_tts is on. With mute off, you can
        # interrupt the AI mid-sentence (the echo filter prevents feedback).
        if self._mute_during_tts:
            self._set_mic_muted(True)

        goal = Speak.Goal()
        goal.text = text
        goal.voice = self._speak_voice
        # Per-call interrupt flag overrides the global default. The global
        # default is kept (param-configurable) so behaviour is opt-in for
        # callers; the default `interrupt=False` here means LLM replies
        # and follow-ups queue politely instead of chopping each other off.
        goal.interrupt_current = bool(interrupt) and self._speak_interrupt
        seq = self._answer_seq if answer else 0     # answer=True: part of LLM turn `seq`, cancellable
        with self._answer_goals_lock:               # 2026-09-25 (review Y6): see _cut_serial
            self._speak_serial += 1
            serial = self._speak_serial

        send_future = self._speak_client.send_goal_async(goal)

        def _publish_ai_speaking(state: bool) -> None:
            if self._ai_speaking_pub is None:
                return
            try:
                self._ai_speaking_pub.publish(Bool(data=bool(state)))
            except Exception:
                pass

        def _on_goal_response(fut) -> None:
            handle = fut.result()
            if handle is None or not handle.accepted:
                if self._mute_during_tts:
                    self._set_mic_muted(False)
                return
            # Accepted -> audio is about to sound. The False is published
            # from the RESULT callback, which fires when playback has
            # actually finished, so the window covers the real speaker
            # time. Strictly paired: True only on accept, False only on
            # that goal's result, so the ref-count in stt.py stays sane
            # when the Speak server queues sentences back to back.
            _publish_ai_speaking(True)
            if seq:
                with self._answer_goals_lock:
                    self._answer_goals.append((seq, handle))
                # hushed while this goal was on its way — or (2026-09-25,
                # review Y6) cut by an 'unclear' turn after it was sent
                if seq <= self._hushed_seq or serial <= self._cut_serial:
                    handle.cancel_goal_async()

            def _on_result(_r) -> None:
                _publish_ai_speaking(False)
                if self._mute_during_tts:
                    self._set_mic_muted(False)
                if seq:
                    with self._answer_goals_lock:
                        self._answer_goals = [(s_, h) for s_, h in self._answer_goals if h is not handle]

            handle.get_result_async().add_done_callback(_on_result)

        send_future.add_done_callback(_on_goal_response)

    # A sentence ends at . ! ? followed by whitespace and a capital, digit or
    # quote — so "waited... for" and "Dr. Smith" do not split.
    _SENTENCE_END = re.compile(r"[.!?](?=\s+[A-ZÄÖÜ0-9\"“(])")
    _ABBREV = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|vs|e\.g|i\.e|Nr|z\.B|bzw|ca)\.$")

    def _on_reply_text(self, delta: str) -> bool:
        """Called from the Bedrock stream with each text delta (2026-09-23,
        Majd: "still slow"): a finished sentence is spoken at once while
        the model keeps writing — a story used to wait 3–9 s for its last
        word before the first was heard. Only while no command was
        dispatched in this turn: command replies stay with the end-of-turn
        rules (quick narration, withholding).

        2026-09-25: returns True when a sentence was spoken (the GWDG stall
        rule counts that as progress); a talk turn keeps streaming after its
        command was dispatched; an unclear turn streams nothing more, and a
        sentence saying "I didn't catch that" is never spoken — it marks the
        turn unclear and the end of the turn says the line from code."""
        if self._reply_hushed():
            raise StopStream()      # the operator is talking: end this reply now
        self._stream_buf += delta
        if self._unclear_turn:
            return False
        if self._early_dispatched and not self._talk_turn:
            return False
        spoke = False
        while True:
            cut = None
            for m in self._SENTENCE_END.finditer(self._stream_buf):
                if self._ABBREV.search(self._stream_buf[:m.end()]):
                    continue        # "Dr." / "z.B." — not a sentence end, look further
                cut = m.end()
                break
            if cut is None:
                return spoke
            sentence, self._stream_buf = self._stream_buf[:cut].strip(), self._stream_buf[cut:].lstrip()
            if _UNCLEAR_SENTENCE_RE.search(sentence.replace("’", "'")):
                self._unclear_turn = True
                self.get_logger().info(f"model said it did not catch the input — not spoken: {sentence!r}")
                return spoke
            if _is_action_claim(sentence):
                # 2026-09-25 (review M3): "Opening the gripper now." streamed
                # before the tool call exists (GWDG sends it at the end of the
                # stream) — never spoken; see _ACTION_ECHOES.
                self.get_logger().info(f"action claim in the model's prose — not spoken: {sentence!r}")
                continue
            if len(sentence) >= 8:
                if not self._streamed:
                    self.get_logger().info(
                        f"[timing] first sentence spoken {time.monotonic() - self._t_llm0:.2f}s into the LLM call")
                self._streamed.append(sentence)
                self.assistant_partial_pub.publish(String(data=" ".join(self._streamed)))
                self._speak(sentence, answer=True)
                spoke = True

    def _on_llm_restart(self) -> None:
        """2026-09-25: a fallback model re-answers this turn (GWDG stalled or
        failed before speaking or dispatching anything, or a Bedrock capacity
        retry): the half sentence the first stream left in the buffer must
        not be glued to the new answer."""
        self._stream_buf = ""
        # 2026-09-25 (review L1): the failed attempt's "I didn't catch that"
        # sentence (it never reached the speaker — a restart happens only
        # when nothing was spoken or dispatched) must not replace the
        # fallback model's answer with UNCLEAR_LINE. An 'unclear' TOOL call
        # counts as dispatched, so no restart follows one.
        self._unclear_turn = False
        self.get_logger().info("LLM restart: a fallback model re-answers this turn")

    def _drop_claim_sentences(self, text: str) -> str:
        """2026-09-25 (review M3): ``text`` without its action-claim
        sentences (_is_action_claim); unchanged when it has none."""
        if not text:
            return text
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        kept = [p for p in parts if not _is_action_claim(p)]
        if len(kept) != len(parts):
            self.get_logger().info(
                f"action claim(s) in the model's reply not spoken: "
                f"{[p for p in parts if _is_action_claim(p)]!r}")
        return " ".join(kept)

    def _drop_marker_sentences(self, text: str) -> str:
        """2026-09-25: ``text`` without its "I didn't catch that" sentences
        (unchanged when it has none)."""
        if not text or not _UNCLEAR_SENTENCE_RE.search(text.replace("’", "'")):
            return text
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        kept = [p for p in parts if not _UNCLEAR_SENTENCE_RE.search(p.replace("’", "'"))]
        return " ".join(kept)

    def _unspoken_tail(self, text: str, since: int = 0) -> str:
        """``text`` minus the sentences already spoken while streaming.
        2026-09-25: ``since`` = the first streamed sentence that belongs to
        ``text`` (a talk turn's follow-up; the earlier ones were another
        response's)."""
        rest = (text or "").strip()
        for sent in (getattr(self, "_streamed", []) or [])[since:]:
            i = rest.find(sent)
            if i < 0:
                break
            rest = rest[i + len(sent):].lstrip()
        return rest

    def _early_intent_call(self, item: Any) -> None:
        """Streaming callback: the intent tool call is complete before the
        model has finished its reply — dispatch it now (2026-09-18). The
        tool output is kept aside and written into the history by the
        drain loop, after the assistant message, so the order stays right."""
        if getattr(item, "name", "") != "intent_extractor":
            return
        if self._reply_hushed():
            self.get_logger().info("Robot command of an interrupted answer: not dispatched.")
            return
        try:
            self._last_tool_output = None
            self._handle_intent_tool_call(item, record_output=False)
            self._early_dispatched[item.call_id] = self._last_tool_output or "{}"
            self.get_logger().info(
                f"[timing] intent dispatched {time.monotonic() - getattr(self, '_t_llm0', time.monotonic()):.2f}s "
                "into the LLM call (before the reply finished)")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"early intent dispatch failed: {exc}")

    @staticmethod
    def _call_action(item: Any) -> str:
        """2026-09-25: the action_intent of one tool call ("" if unreadable)."""
        try:
            return str((json.loads(item.arguments) or {}).get("action_intent") or "").strip()
        except Exception:
            return ""

    def _note_intent_sent(self, action: str) -> None:
        """2026-09-25: remember when this node last sent a pick (C7: a
        destination said before the executor reports the new cycle is still
        the router's to announce). Review X1: a plain "pick" only — the
        router latches an early destination for a plain pick; a
        pick_and_place already carries its destination and is not latched."""
        if str(action or "").strip().lower() == "pick":
            self._pick_sent_mono = time.monotonic()

    def _note_stop_or_reset(self) -> None:
        """2026-09-25 (review H1/X2): this node sent a stop or reset — every
        input that arrived before it and still waits for the chat lock is
        stale (_stop_epoch), and no pick of ours is pending any more."""
        with self._voice_gen_lock:
            self._stop_epoch += 1
        self._pick_sent_mono = 0.0

    def _on_selection_result(self, msg: String) -> None:
        """2026-09-25 (review X2): the grasp selector refused the pick
        ("failed|0|…", the cycle_done grammar) before the executor began a
        cycle — no cycle_done follows, so the pending-pick clause of
        _router_narrates ends here."""
        if (msg.data or "").strip().lower().startswith("failed"):
            self._pick_sent_mono = 0.0

    def _plain_pick_pending(self) -> bool:
        """2026-09-25 (C7, review X2): this node sent a plain pick < 90 s ago
        and neither a cycle end, a selector refusal nor a stop/reset came
        since."""
        return bool(self._pick_sent_mono
                    and time.monotonic() - self._pick_sent_mono < 90.0
                    and self._cycle_done_mono < self._pick_sent_mono)

    def _live_task(self) -> dict | None:
        """2026-09-25 (review X4): the executor's task state, or None while it
        is unknown OR its publisher is gone — a dead executor's latched last
        sample must not be trusted (as the GUI checks). Used where a stale
        state would act: the open question and _router_narrates."""
        t = self._task
        if t is None:
            return None
        try:
            if self.count_publishers(_TASK_STATE_TOPIC) == 0:
                return None
        except Exception:  # noqa: BLE001 — a graph query failing is not "dead"
            pass
        return t

    def _router_narrates(self, action: str, dest: str = "") -> bool:
        """2026-09-25 (SPEC C7, Majd: "never announce what was not accepted").
        True when intent_router speaks this action's line after the robot
        confirmed it, so this node must stay silent: the gripper actions
        always. A destination action (drop_right / drop_left / confirm_drop
        / go_home) with a live task state (review M2: in every state except
        these two, nobody has accepted it yet and the router narrates on
        acceptance):
          * the home question is open → False for go_home, chat says
            "Going home." as before; True for a SIDE (drop_left /
            drop_right, confirm_drop whose ``dest`` names a side) —
            2026-09-25 (Majd's live test): the executor now moves to that
            side instead of home and the router narrates it on acceptance;
          * idle, no question → False ("Going home." as before), unless a
            plain pick of ours is pending (_plain_pick_pending, review X2):
            the router latches the destination and announces it;
          * probing / approaching → only when the pick asks for a
            destination (review X1: the router latches only then; a
            pick_and_place's "No, put it on the left" was left in silence);
          * any other question or phase → True.
        Without a live task state only the gripper actions, as before."""
        if action in CONFIRMED_NARRATION_ACTIONS:
            return True
        t = self._live_task()
        if t is None or action not in _DEST_ACTIONS:
            return False
        q = str(t.get("question") or "")
        if q == "home":
            d = str(dest or "").lower()
            return (action in ("drop_left", "drop_right")
                    or (action == "confirm_drop" and ("left" in d or "right" in d)))
        if q:
            return True
        phase = str(t.get("phase") or "")
        if phase == "idle":
            return self._plain_pick_pending()
        if phase in ("probing", "approaching"):
            return bool(t.get("asks_destination"))
        return True

    def _handle_intent_tool_call(self, item: Any, record_output: bool = True) -> None:
        # Single-call path: the LLM now returns the structured fields
        # directly via the tool call. No second extract_intent() round-trip.
        try:
            args = json.loads(item.arguments)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"intent_extractor args not JSON: {exc}")
            args = {}

        action = str(args.get("action_intent") or "").strip()
        if not action:
            self.get_logger().warn("intent_extractor returned empty action_intent")
            self._last_tool_output = '{"error": "empty action_intent"}'
            if record_output:
                self.chat_agent.add_function_call_output(item, self._last_tool_output)
            return

        if action == "unclear":
            # 2026-09-25: the model found the input garbled. Nothing is
            # published (the router never sees 'unclear'); the turn stops
            # streaming and its end speaks UNCLEAR_LINE alone — unless a real
            # command came in the same turn, which then wins.
            self._unclear_turn = True
            self._last_tool_output = '{"handled":"unclear"}'
            # 2026-09-25 (review L3/Y6): GWDG hands the tool call over only
            # at the end of its stream, after its prose was streamed — cut
            # that prose now (queued and still on its way), not only at the
            # end of the turn.
            cut = self._cancel_answer_goals(self._answer_seq, cut=True)
            self.get_logger().info(
                f"LLM marked the input unclear — nothing published ({cut} streamed sentence(s) cut)")
            if record_output:
                self.chat_agent.add_function_call_output(item, self._last_tool_output)
            return
        self._real_intent = True       # 2026-09-25: anything but 'unclear' is a real command
        if args.get("talk") in (True, "true", "True"):
            # 2026-09-25 (Majd: "a story while a grasp runs, only when asked
            # in that utterance"): the reply keeps streaming after this
            # dispatch; a new robot question may cut it (_on_task_state).
            self._talk_turn = True
            self._talk_seq = self._answer_seq

        # Stop narrations are time-critical — speak immediately,
        # interrupting whatever else was playing. Everything else gets
        # buffered for end-of-turn flushing so a multi-intent turn
        # ("go home then drop right and tell a joke") can be narrated
        # by the LLM as one coherent sentence rather than three
        # back-to-back fragments.
        # 2026-09-25 (review Y2): the live question decides what a NO means.
        live_q = str((self._live_task() or {}).get("question") or "")
        if action == "stop" and live_q == "home":
            # A stop word never reaches the LLM (the grammar sends it), so
            # an LLM stop at "Should I go back home?" is its reading of a NO
            # ("leave it there") — the same 'stay' as the deterministic NO:
            # the arm stays, "Okay, I'll stay here.", no abort.
            self.get_logger().info("LLM stop at the home question → stay")
            action = "stay"
            args["action_intent"] = "stay"
        quick = QUICK_NARRATION.get(action)
        if action == "reset":
            # 2026-09-25: an LLM reset is said at once, like stop, and
            # published as the exact voice reset intent (SPEC §1.4).
            self._note_stop_or_reset()       # 2026-09-25 (review H1/X2)
            self._interrupt_line_turn = True
            self._intent_fired_this_turn = True
            self._speak(quick, interrupt=True)
            self._last_tool_output = self._fire_reset()
            if record_output:
                self.chat_agent.add_function_call_output(item, self._last_tool_output)
            return
        if action == "close_gripper" and live_q == "release":
            # 2026-09-25 (review Y2): a NO the LLM understood ("Wait a
            # second.", "No, keep it for now") — say the FAST-NEGATIVE line
            # as HEAD did; withheld, it ended in silence.
            self._pending_quick_narrations.append((action, _KEEP_HOLDING_LINE))
        elif action == "stay":
            # 2026-09-25 (review Y2): the router answers the home pause with
            # it (and keeps holding at the release pause); elsewhere it is a
            # straggler the router swallows — nothing to say.
            line = {"home": _STAY_LINE, "release": _KEEP_HOLDING_LINE}.get(live_q, "")
            self._pending_quick_narrations.append((action, line))
            if not line:
                self._withheld_this_turn.add(action)
        elif self._router_narrates(action, str(args.get("destination") or "")):
            # Silent placeholder: keeps this intent in the per-turn
            # accounting (so a mixed turn still narrates its OTHER actions)
            # while saying nothing about the gripper here. _speak("") is a
            # no-op, and the all-confirmed case below withholds the LLM's
            # own reply too. 2026-09-25: also a destination the router
            # announces once the executor accepted it (_router_narrates).
            self._pending_quick_narrations.append((action, ""))
            self._withheld_this_turn.add(action)
        elif quick:
            if action == "stop":
                self._note_stop_or_reset()           # 2026-09-25 (review H1/X2)
                self._interrupt_line_turn = True     # 2026-09-25: nothing else follows it
                self._speak(quick, interrupt=True)
            else:
                self._pending_quick_narrations.append((action, quick))

        # Build intent JSON in the same shape the router expects.
        attrs = {
            "colour": (args.get("colour") or None),
            "shape": (args.get("shape") or None),
            "position": (args.get("position") or None),
            "ordinal": (args.get("ordinal") or None),
            "ordinal_from": (args.get("ordinal_from") or None),
            "speed": (args.get("speed") or None),
            "destination": (args.get("destination") or None),
            # 2026-09-24: the release/home plan said in the same sentence
            # (router: confirm_drop / drop_* / go_home / pick_and_place).
            "then": (args.get("then") or None),
        }
        # Drop the attrs dict if everything in it is None (so old consumers
        # that special-case object_attributes is None still work).
        attrs_payload = attrs if any(v for v in attrs.values()) else None
        intent_payload = {
            "action_intent": action,
            "object_intent": (args.get("object_intent") or None),
            "object_attributes": attrs_payload,
        }
        intent_json = json.dumps(intent_payload, separators=(",", ":"))

        output_msg = String()
        output_msg.data = intent_json
        self.intent_pub.publish(output_msg)
        self.get_logger().info(f"Published intent on {self.intent_pub.topic_name}: {intent_json}")
        self._note_intent_sent(action)
        # Mark that this turn produced an action so we can skip the slow
        # LLM final-reply TTS (the quick narration already covered it).
        self._intent_fired_this_turn = True

        # Remember for "do it again" fast-path. We deliberately ignore intents
        # that are themselves transient/control (stop, redo) — those aren't
        # meaningful to repeat.
        try:
            parsed = json.loads(intent_json) if isinstance(intent_json, str) else intent_json
        except Exception:
            parsed = {}
        action = str(parsed.get("action_intent") or "").strip().lower()
        if action and action not in {"stop", "cancel", "abort", "redo", "again", "repeat"}:
            self._last_intent_json = intent_json

        self._last_tool_output = intent_json
        if record_output:
            self.chat_agent.add_function_call_output(item, intent_json)

        # If auto-execute is on and the action looks like a pick, fire ExecuteGrasp.
        # We do this after publishing /robot/intent so any other subscribers still see it.
        try:
            intent_dict = json.loads(intent_json) if isinstance(intent_json, str) else intent_json
        except Exception:
            intent_dict = {}
        action = str(intent_dict.get("action_intent") or "").strip().lower()
        if self._auto_execute_grasp and action in PICK_INTENTS:
            self._maybe_execute_grasp_for_intent(intent_dict)

    # ── grasp execution helpers ──────────────────────────────────────────

    def _on_best_pose(self, msg: PoseStamped) -> None:
        with self._grasp_lock:
            self._latest_pose = msg
            self._latest_pose_received_at = self.get_clock().now().nanoseconds / 1e9

    def _tts_ready(self) -> bool:
        return self._speak_client is not None and self._speak_client.server_is_ready()

    def _on_interactive_status_question(self, msg: String) -> None:
        if self._task is not None:
            return      # 2026-09-25: the task state owns the open question when known
        d = str(msg.data or "").strip()
        self._open_question = d if d in ("awaiting_home_confirm", "awaiting_release") else ""
        self._open_question_t = time.monotonic()

    _AFFIRMATIVE = re.compile(
        r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please|go ahead|do it|alright|all right|of course|ja|klar)"
        r"(?:[\s,!.]+(?:please|now|go ahead|do it|sure|ok(?:ay)?))*\s*[!.?]*\s*$", re.I)

    def _answer_open_question(self, text: str):
        """(action, speed) when `text` answers the executor's open question."""
        q = getattr(self, "_open_question", "")
        # A question is only open until it is answered or its cycle ends
        # (both clear it) — and never longer than 60 s (2026-09-24: "Should
        # I go back home?" answered at 16:55:51 stayed "open", and a "Yeah."
        # in an unrelated chat 2.5 min later sent the arm home).
        if q and time.monotonic() - getattr(self, "_open_question_t", 0.0) > 60.0:
            self._open_question = q = ""
        if not q or not self._AFFIRMATIVE.match(text or ""):
            return None
        if q == "awaiting_home_confirm":
            return ("go_home", "")
        if q == "awaiting_release":
            return ("open_gripper", "")
        return None

    def _on_readiness(self, msg: String) -> None:
        """Majd 2026-09-18: at start 'please wait until the system is ready,
        you can chat until then'; once the first object is verified for
        execution, 'I'm ready to grasp now' — each once. The report comes
        every second, so a sentence is only marked as said once the speech
        server could take it (it loads ~8 s after this node; the first
        version lost the sentence to that gap)."""
        try:
            r = json.loads(msg.data)
        except Exception:
            return
        if self._said_ready or not self._tts_ready():
            return
        # Majd 2026-09-18: always "Hi! I'm Morty" first, then "please
        # wait". The greeting timer and this 1 Hz report both wait for the
        # speech server; whichever ticked first used to speak first. While
        # the greeting is still pending (timer alive) this handler stays
        # silent — _fire_startup_greeting speaks the wait line itself.
        if self._greeting_timer is not None:
            return
        objs = r.get("objects") or []
        verified = [o for o in objs if o.get("state") == "verified"]
        # Majd 2026-09-18 (evening): "it should be waiting until all of
        # them are done, not the first object" — the first scene is ready
        # when NO object is still waiting, computing or awaiting its
        # background check. A verified object implies GraspGen answered
        # and the executor planned it, whatever the model flag says; the
        # ones that ended failed / blocked / unreachable are named so the
        # operator knows not to ask for them.
        pending = [o for o in objs if o.get("state") in ("waiting", "computing", "grasps")]
        if verified and not pending:
            self._said_ready = True
            names = ", ".join(str(o.get("label", "")).replace("_", " ") for o in verified[:4])
            line = f"I'm ready to grasp now. {names} {'is' if len(verified) == 1 else 'are'} ready."
            rest = [str(o.get("label", "")).replace("_", " ") for o in objs if o not in verified]
            if rest:
                line += f" I can't reach {', '.join(rest[:3])} from here."
            self._speak(line)
        elif not self._said_loading:
            self._said_loading = True
            self._speak("Please wait until the system is ready. You can chat with me until we start.")

    def _on_candidates(self, msg: PoseArray) -> None:
        with self._grasp_lock:
            self._latest_candidates = msg

    # ── proactive scene narration ────────────────────────────────────────

    def _proactive_tick(self) -> None:
        # System-ready announcement after "start everything"
        # (Majd 2026-08-26: "let Morty say the system is ready").
        deadline = getattr(self, "_announce_ready_until", 0.0)
        if deadline:
            now = self.get_clock().now().nanoseconds / 1e9
            if now > deadline:
                self._announce_ready_until = 0.0
            else:
                driver_up = (now - self._last_joint_state_at) < 2.0
                scene_fresh = (now - getattr(self, "_latest_scene_at", 0.0)) < 5.0
                if driver_up and self._program_running and scene_fresh:
                    self._announce_ready_until = 0.0
                    self._speak("The system is running and we are ready to grasp.")
        """Periodic check: if there are visible objects, the user has been
        idle, and this scene hasn't been narrated recently, volunteer a
        one-line description and an offer to pick. No LLM call — uses the
        deterministic scene summary so it's fast and free."""
        if not self._proactive_enabled:
            return
        # Wait until Kokoro is up. Otherwise the first proactive tick
        # logs "Speak action server not available" and the line never
        # actually plays.
        if (
            self._tts_enabled
            and (self._speak_client is None or not self._speak_client.server_is_ready())
        ):
            return
        now = self.get_clock().now().nanoseconds / 1e9
        # Don't proactively narrate while an action is in flight or while
        # we're already speaking — that would step on real-time feedback.
        if self._intent_fired_this_turn:
            return
        # User idle gate.
        if now - self._last_user_input_at < self._proactive_idle_s:
            return
        # Cooldown between volunteers.
        if now - self._last_proactive_at < self._proactive_cooldown_s:
            return
        # Need a fresh scene with at least one object.
        with self._scene_lock:
            scene = list(self._latest_scene)
            scene_age = now - self._latest_scene_at if self._latest_scene_at else 1e9
        if scene_age > self._scene_max_age_s or not scene:
            return
        # Build a stable signature so we don't repeat the same line every
        # cooldown window if the scene hasn't actually changed.
        signature = ",".join(sorted(
            str(d.get("label") or d.get("display") or "") for d in scene
        ))
        if signature == self._last_proactive_signature and (
            now - self._last_proactive_at < max(self._proactive_cooldown_s * 3, 180.0)
        ):
            return
        # Compose the line. Reuse the existing summary helper so positional
        # qualifiers ("leftmost", "rightmost") match what the LLM already
        # gets as context.
        summary = self._scene_summary()
        if not summary:
            return
        # Strip the leading boilerplate so the spoken line sounds natural.
        spoken = summary.replace("Scene currently shows: ", "I can see ")
        if spoken.endswith("."):
            spoken = spoken[:-1]
        spoken += ". Would you like me to pick one of them?"
        self.get_logger().info(f"Proactive scene narration: {spoken!r}")
        self._speak(spoken)
        self._last_proactive_at = now
        self._last_proactive_signature = signature

    # ── grasp_pipeline selection prompts ─────────────────────────────────

    def _on_selection_prompt(self, msg: String) -> None:
        """Receive the grasp-pipeline's selection prompt for logging
        only. We deliberately do NOT speak it — the LLM already covers
        the same ground via proactive scene narration (built from the
        /yolo/all_detections feed) and via on-demand 'what do you see?'
        questions. Speaking both would double-narrate every scene."""
        text = (msg.data or "").strip()
        if not text:
            return
        self.get_logger().debug(f"grasp_pipeline prompt (suppressed): {text!r}")

    def _on_pointed_object(self, msg: String) -> None:
        """The grasp pipeline tells us which object the user's finger is
        currently over (or "" when the hand is away). We only log
        transitions, and inject the value into the LLM context each
        turn via _handle_user_text_locked."""
        new_label = (msg.data or "").strip()
        if new_label != self._pointed_object_label:
            human = new_label.replace("_", " ") if new_label else "<none>"
            self.get_logger().info(f"Pointed-object update: {human!r}")
            self._pointed_object_label = new_label

    # ── held-object memory ───────────────────────────────────────────────

    def _on_held_object(self, msg: String) -> None:
        new_label = (msg.data or "").strip()
        if self._task is not None:
            # 2026-09-25: the router derives /robot/held_object from the task
            # state, which this node reads directly — no second writer.
            return
        if new_label != self._held_object_label:
            self.get_logger().info(
                f"Held-object update: {self._held_object_label!r} -> {new_label!r}"
            )
            self._held_object_label = new_label
            cs = str(getattr(self, "_cycle_state", "") or "")
            if not new_label and "HOLDING" in cs:
                self._set_cycle_state("gripper open, holding nothing")
            elif new_label and "HOLDING" not in cs:
                self._set_cycle_state(f"HOLDING the {new_label.replace('_', ' ')}")

    def _on_cycle_done_state(self, msg: String) -> None:
        """The executor's cycle end ("done|id…", "failed|id|reason")."""
        self._cycle_done_mono = time.monotonic()     # 2026-09-25: for _router_narrates
        if self._task is not None:
            return      # 2026-09-25: the task state's 'last' says it (published first)
        outcome = (msg.data or "").strip().split("|", 1)[0].lower()
        if outcome == "done":
            self._set_cycle_state("IDLE — last pick finished")
        elif outcome == "failed":
            self._set_cycle_state("IDLE — last pick FAILED or stopped, arm recovered")

    def _set_cycle_state(self, state: str, from_task: bool = False) -> None:
        if self._task is not None and not from_task:
            return      # 2026-09-25: only the task state edits it while known
        if state != getattr(self, "_cycle_state", None):
            self.get_logger().info(f"Cycle state: {state}")
        self._cycle_state = state
        if state.startswith("IDLE") and not from_task:
            self._open_question = ""     # a finished cycle has no open question

    # ── executor task state (2026-09-25) ─────────────────────────────────

    # The legacy cycle-state phrases per task question / phase, so the old
    # matchers and the arm-pose hint keep working while the task is known.
    _TASK_CYCLE_STATE = {
        "destination": "HOLDING the object at the grasp pose — waiting for a drop destination",
        "release": "HOLDING at the drop pose — waiting for release confirmation",
        "home": "at the drop pose, gripper open — asked about going home",
    }

    def _on_task_state(self, msg: String) -> None:
        """2026-09-25 (SPEC §1.1, C2): the executor's latched task state, read
        as a LEVEL: a sample older than the last one (seq ≤ last, same boot)
        is dropped, the latched first sample is adopted silently. While it is
        known it alone sets the cycle state, the open question and the held
        object; a NEW question (new question_id) cuts a story this node is
        telling while a pick runs (story_yields_to_question)."""
        try:
            d = json.loads(msg.data or "")
        except Exception:
            return
        if not isinstance(d, dict):
            return
        prev = self._task
        try:
            seq = int(d.get("seq") or 0)
            prev_seq = int((prev or {}).get("seq") or 0)
        except (TypeError, ValueError):
            return
        if prev is not None and d.get("boot") == prev.get("boot") and seq <= prev_seq:
            return
        first = prev is None
        self._task = d
        q = str(d.get("question") or "")
        qkey = (d.get("boot"), d.get("question_id") or 0)
        new_question = bool(q) and qkey != self._task_qkey and not first
        self._task_qkey = qkey
        # The legacy view, derived.
        phase = str(d.get("phase") or "")
        if self._is_no_grasp_home(d):
            # 2026-09-25 (review X3): not "at the drop pose"; keeps the legacy
            # matchers' "asked about going home".
            cs = "missed the grasp — lifted above the grasp pose, gripper empty — asked about going home"
        elif q in self._TASK_CYCLE_STATE:
            cs = self._TASK_CYCLE_STATE[q]
        elif phase == "idle":
            outcome = str((d.get("last") or {}).get("outcome") or "") if isinstance(d.get("last"), dict) else ""
            cs = ("IDLE — last pick FAILED or stopped"
                  if outcome in ("failed", "stopped", "reset") else "IDLE — last pick finished")
        elif phase in ("probing", "approaching"):
            cs = "BUSY — pick running"
        elif phase == "returning_home":
            cs = "moving to home"
        elif phase == "stopped":
            cs = "STOPPED by the operator — the arm is frozen"
        else:
            cs = f"BUSY — pick running ({phase.replace('_', ' ')})"
        self._set_cycle_state(cs, from_task=True)
        oq = {"release": "awaiting_release", "home": "awaiting_home_confirm"}.get(q, "")
        if oq != self._open_question:
            self._open_question_t = time.monotonic()
        self._open_question = oq
        label = (str(d.get("object") or "") or "object") if d.get("holding") else ""
        if label != self._held_object_label:
            self.get_logger().info(f"Held-object update (task state): {self._held_object_label!r} -> {label!r}")
        self._held_object_label = label
        if first:
            self.get_logger().info(f"Task state adopted: phase={phase!r} question={q!r}")
        if new_question and self._story_yields and self._talk_seq == self._answer_seq:
            self._hush_answer(_HUSH_BY_QUESTION)

    def _on_utterance_meta(self, msg: String) -> None:
        """2026-09-25 (SPEC §1.5): remember the last 8 utterances' metas —
        review M1: with the speech START (None when the meta has none)."""
        try:
            d = json.loads(msg.data or "")
            text = str(d.get("text") or "").strip()
            end = float(d.get("speech_end") or 0.0)
            start = d.get("speech_start")
            start = float(start) if start not in (None, "") and float(start) > 0.0 else None
        except Exception:
            return
        self._utt_meta_seen = True
        with self._utt_meta_lock:
            self._utt_meta.append([time.time(), text, end, start, False])
            del self._utt_meta[:-8]

    def _speech_start_for(self, text: str, typed_at: float | None,
                          heard_at: float | None = None,
                          allow_used: bool = False) -> float | None:
        """2026-09-25 (SPEC C6, review M1): when the operator STARTED saying
        ``text``: typed input → its arrival; voice → the speech_start of the
        exact-text match among the last 8 utterance metas (≤ 10 s old; the
        meta is published just before the transcript, so wait ≤ 100 ms for
        it); no match, or a meta without a start → None, which is NOT fresh
        (the old fallback "now − 1.5 s" could lie later than the real
        speech and make a stale "Yes." look fresh).

        A meta answers one transcript only (it is marked used), and with the
        transcript's arrival known it must have arrived with it (≤ 2 s
        before): the harness caught a second "Yes." paired with the FIRST
        "Yes."'s meta when its own meta was a few ms late.

        ``allow_used`` (2026-09-25, G1): the first fragment of a merged
        input may already have been matched by its own turn."""
        if typed_at is not None:
            return typed_at
        want = (text or "").strip()
        deadline = time.monotonic() + 0.1
        while True:
            now = time.time()
            with self._utt_meta_lock:
                for entry in reversed(self._utt_meta):
                    rx, t, _end, start, used = entry
                    if (t == want and (allow_used or not used) and now - rx <= 10.0
                            and (heard_at is None or rx >= heard_at - 2.0)):
                        entry[4] = True
                        return start
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        return None

    def _on_current_pose(self, msg: String) -> None:
        new_label = (msg.data or "").strip()
        if new_label != self._current_pose_label:
            self.get_logger().info(
                f"Current pose update: {self._current_pose_label!r} -> {new_label!r}"
            )
            self._current_pose_label = new_label

    def _load_named_poses(self, path: str) -> None:
        """Parse observation_poses.yaml without bringing in PyYAML — the
        file is small and the format is fixed. We need joint_names plus
        the per-pose 'positions' arrays so we can compute joint-space
        distance from the live joint_states."""
        if not os.path.isfile(path):
            self.get_logger().warn(
                f"observation_poses_file '{path}' not found; "
                "nearest-pose answers will be unavailable."
            )
            return
        try:
            import yaml  # type: ignore
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse named-pose YAML: {exc}")
            return
        self._named_pose_joint_names = list(data.get("joint_names") or [])
        for name, entry in (data.get("poses") or {}).items():
            positions = entry.get("positions") if isinstance(entry, dict) else None
            if isinstance(positions, list) and len(positions) == len(
                self._named_pose_joint_names
            ):
                self._named_poses[name] = [float(v) for v in positions]
        self.get_logger().info(
            f"Loaded {len(self._named_poses)} named pose(s) for "
            f"nearest-pose lookup: {list(self._named_poses.keys())}"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        # JointState publishes the whole arm; we just snapshot the latest
        # per-joint angles so _nearest_named_pose can compute distance.
        self._last_joint_state_at = self.get_clock().now().nanoseconds / 1e9
        for name, position in zip(msg.name, msg.position):
            self._latest_joint_positions[name] = float(position)

    def _on_program_running(self, msg: Bool) -> None:
        self._program_running = bool(msg.data)

    def _on_router_status(self, msg: String) -> None:
        try:
            self._router_status = json.loads(msg.data)
        except Exception:
            return
        self._router_status_at = self.get_clock().now().nanoseconds / 1e9

    def _program_running_view(self) -> bool | None:
        """The router's verdict when fresh, else this node's raw flag."""
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._router_status_at < 6.0 and "program_running" in self._router_status:
            return self._router_status["program_running"]
        return self._program_running

    def _on_tf_msg(self, msg) -> None:
        self._last_tf_at = self.get_clock().now().nanoseconds / 1e9

    def _sample_one_message(self, msg_type, topic: str):
        """Take ONE message from *topic* through a MOMENTARY subscription.

        Returns the message, or None if nothing arrived inside
        ``liveness_sample_timeout_s``. Never raises: a liveness probe must
        not be able to take the node down, and a failed probe is
        indistinguishable from a silent topic, which the freshness windows
        already handle.

        Why not a standing subscription: rclpy pays a per-message wait-set
        cost even when the callback body is trivial, so subscribing to a
        ~494 Hz topic burns most of a CPU core to do nothing.

        Why rclpy's wait_for_message() and not the create/destroy busy-wait
        in ur3e_fast_observation's replay_node._capture_current_joints():
        that node runs a MultiThreadedExecutor, so a subscription in a
        dedicated ReentrantCallbackGroup can still be delivered while the
        caller blocks. THIS node runs rclpy.spin() — a SingleThreadedExecutor
        — where no callback can ever be delivered while another callback is
        on the stack, so that pattern would time out every single time.
        wait_for_message() builds its OWN private wait set and takes the
        message directly, bypassing the executor, and destroys the
        subscription in a finally: block so nothing can leak.
        """
        try:
            # Cheap local graph query. With no publisher at all (driver
            # not started — the common case) there is nothing to wait for,
            # so skip the bounded wait entirely rather than blocking this
            # node's only executor thread for nothing.
            if self.count_publishers(topic) == 0:
                return None
            ok, msg = wait_for_message(
                msg_type,
                self,
                topic,
                time_to_wait=max(0.0, self._liveness_sample_timeout_s),
            )
            return msg if ok else None
        except Exception:
            return None

    def _liveness_tick(self) -> None:
        """Sample /joint_states and /tf once per period and feed the
        SAME callbacks the standing subscriptions used to feed, so every
        downstream threshold, log line and status string is unchanged.

        Failures are silent: the freshness stamps stop advancing, which is
        exactly what a dead publisher always looked like.
        """
        # HALF-UP GUARD (2026-08-31). A publisher can exist while NO data
        # flows (driver process alive, robot off / e-stopped / cable out —
        # measured live: 2 publishers, 0 msgs in 6 s). Then every probe
        # burns its FULL 0.3 s timeout on this node's only executor
        # thread: 0.3 s of every 0.5 s tick = ~60% of the node stalled,
        # which Majd felt as "very slow with answering". Exponential
        # backoff (1 -> 2 -> 4 -> 8 -> 10 s cap) while the topic is
        # silent; any received message resets to full rate instantly.
        now_mono = time.monotonic()
        if now_mono < getattr(self, "_liveness_backoff_until", 0.0):
            return
        msg = self._sample_one_message(JointState, "/joint_states")
        if msg is None:
            # No joint states => the driver is down, and _system_status_summary
            # only ever reads tf_alive INSIDE its driver_up branch, so the /tf
            # probe would change no output. Skipping it also keeps the worst
            # case at ONE bounded wait per tick instead of two.
            if self.count_publishers("/joint_states") > 0:
                prev = getattr(self, "_liveness_backoff_s", 0.0)
                self._liveness_backoff_s = min(10.0, max(1.0, prev * 2.0))
                self._liveness_backoff_until = (
                    now_mono + self._liveness_backoff_s)
                if prev == 0.0:
                    self.get_logger().info(
                        "/joint_states has a publisher but no data (robot "
                        "off / e-stop?) — backing liveness probes off so "
                        "chat stays responsive.")
            return
        self._liveness_backoff_s = 0.0
        self._liveness_backoff_until = 0.0
        self._on_joint_state(msg)
        tf_msg = self._sample_one_message(self._tf_msg_type, "/tf")
        if tf_msg is not None:
            self._on_tf_msg(tf_msg)

    def _pendant_reminder_tick(self) -> None:
        """Speak 'press Play' once whenever the driver is up but the
        pendant program is not running (fresh start, protective stop,
        or the operator paused it). Re-arms when the program runs."""
        now = self.get_clock().now().nanoseconds / 1e9
        driver_up = (now - self._last_joint_state_at) < 2.0
        if not driver_up:
            # Driver down: next bring-up should remind again.
            self._pendant_reminded = False
            return
        running = self._program_running_view()
        if (self.count_publishers(
                "/io_and_status_controller/robot_program_running") == 0
                or running is None):
            return
        if running:
            self._pendant_reminded = False
            self._program_stopped_since = 0.0
            return
        # Start-up grace (2026-09-18): the GUI restarts the driver and
        # auto-Plays the pendant within ~12 s; a reminder spoken 3 s after
        # the driver came up ("press Play") was true for one moment and
        # wrong by the time it was heard. Remind only when the program has
        # stayed stopped for a while.
        if not self._program_stopped_since:
            self._program_stopped_since = now
        if now - self._program_stopped_since < 20.0:
            return
        if not self._pendant_reminded:
            self._pendant_reminded = True
            self._speak(
                "The robot driver is up, but the program on the teach "
                "pendant is not running. Please press Play so I can move."
            )

    def _system_status_summary(self) -> str:
        """One developer-message block of LIVE system state so the LLM
        reports facts instead of guessing. Publisher counts are cheap graph
        queries; freshness comes from our own subscriptions."""
        now = self.get_clock().now().nanoseconds / 1e9
        driver_up = (now - self._last_joint_state_at) < 2.0
        parts: list[str] = []
        tf_alive = (now - self._last_tf_at) < 3.0
        if driver_up and not tf_alive:
            parts.append(
                "robot driver: HALF-DEAD — joint data flows but the "
                "transform publisher is DEAD, so every grasp will "
                "miscompute; the operator must Stop and Start the driver"
            )
        elif driver_up:
            parts.append("robot driver: RUNNING (joint states flowing)")
        else:
            parts.append(
                "robot driver: NOT RUNNING — the arm cannot move at all; "
                "the operator must press 'Start driver' (or 'Start ALL') "
                "in the Morty GUI"
            )
        running = self._program_running_view()
        if (self.count_publishers(
                "/io_and_status_controller/robot_program_running") > 0
                and running is not None):
            parts.append(
                "pendant program: PLAYING" if running
                else "pendant program: STOPPED — motion commands will be "
                     "ignored until the operator presses Play on the "
                     "robot's teach pendant"
            )
        elif driver_up:
            parts.append("pendant program: unknown")
        camera_up = self.count_publishers(
            "/camera/camera/color/camera_info") > 0
        parts.append("camera: RUNNING" if camera_up
                     else "camera: NOT RUNNING (no RealSense frames)")
        pipeline_up = self.count_publishers(
            str(self.get_parameter("detections_topic").value)) > 0
        if pipeline_up:
            with self._scene_lock:
                age = (now - self._latest_scene_at
                       if self._latest_scene_at else 1e9)
            parts.append(
                "grasp pipeline: RUNNING" if age < 10.0
                else "grasp pipeline: RUNNING but detections are stale "
                     "(starting up, or inference is paused)"
            )
        else:
            parts.append(
                "grasp pipeline: NOT RUNNING — no object detection, no "
                "grasping; the operator must press 'Start pipeline' in the "
                "Morty GUI"
            )
        executor_up = self.count_publishers(
            "/grasp_executor_lin_llm/interactive_status") > 0
        parts.append(
            "grasp executor: RUNNING" if executor_up
            else "grasp executor: NOT RUNNING — picks would select a grasp "
                 "but the arm would never execute it"
        )
        return (
            "LIVE SYSTEM STATUS (ground truth, checked this second — "
            "answer status questions from THIS, never guess): "
            + "; ".join(parts) + ". "
            "If the user asks for motion or a pick while a required "
            "component is NOT RUNNING, say so and name what must be "
            "started instead of promising the action."
        )

    def _nearest_named_pose(self) -> tuple[str, float] | None:
        """Return (pose_name, l2_distance_rad) of the named pose closest
        to the current joint state, or None if we don't have enough data."""
        if not self._named_poses or not self._named_pose_joint_names:
            return None
        try:
            current = [
                self._latest_joint_positions[j]
                for j in self._named_pose_joint_names
            ]
        except KeyError:
            return None
        best_name = ""
        best_d = float("inf")
        for name, positions in self._named_poses.items():
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(current, positions)))
            if d < best_d:
                best_d = d
                best_name = name
        if not best_name:
            return None
        return best_name, best_d

    # ── scene awareness ──────────────────────────────────────────────────

    def _on_scene_update(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or "[]")
            if not isinstance(payload, list):
                return
        except json.JSONDecodeError:
            return
        with self._scene_lock:
            self._latest_scene = payload
            self._latest_scene_at = self.get_clock().now().nanoseconds / 1e9

    def _scene_summary(self) -> str:
        """Return a one-line description of currently visible objects, or "" if stale.

        Includes positional descriptors (leftmost/middle/rightmost) computed
        from bbox x-centres so the LLM can use them when the user says e.g.
        "pick the leftmost vitamin tube".
        """
        if not self._scene_awareness:
            return ""
        now = self.get_clock().now().nanoseconds / 1e9
        with self._scene_lock:
            age = now - self._latest_scene_at if self._latest_scene_at else 1e9
            scene = list(self._latest_scene)
        if age > self._scene_max_age_s:
            return ""
        if not scene:
            return "Scene currently shows: nothing detected."

        # Sort by bbox x-centre and assign positional labels.
        def _x_centre(det: dict) -> float:
            bbox = det.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            try:
                return (float(bbox[0]) + float(bbox[2])) * 0.5
            except (TypeError, ValueError, IndexError):
                return 0.0

        ordered = sorted(scene, key=_x_centre)
        n = len(ordered)
        # Count occurrences of each label so we can call out duplicates
        # explicitly. The LLM was confusing 'vitamin_tube (left)' and
        # 'vitamin_tube (right)' for ONE object with a left/right
        # qualifier instead of TWO distinct vitamin_tubes.
        from collections import Counter
        label_counts = Counter(
            (d.get("label") or d.get("display") or "object") for d in ordered
        )
        parts: list[str] = []
        for i, det in enumerate(ordered):
            label = det.get("label") or det.get("display") or "object"
            # 2026-08-25 (Majd: "don't know what is the vitamin B!"):
            # the LLM can only know WHICH tube is which product if the
            # scene text carries the cap colour — annotate it, with the
            # product name the colour implies.
            cap = str(det.get("cap_colour") or "")
            if cap:
                product = {
                    "yellow": "Vitamin C", "orange": "Vitamin C",
                    "red": "Vitamin B12", "purple": "Vitamin B12",
                    "blue": "Multivitamin",
                }.get(cap)
                label = (f"{label} [{cap} cap = {product}]"
                         if product else f"{label} [{cap} cap]")
            if n == 1:
                parts.append(label)
            elif n == 2:
                pos = "left" if i == 0 else "right"
                parts.append(f"{label} ({pos})")
            else:
                if i == 0:
                    pos = "leftmost"
                elif i == n - 1:
                    pos = "rightmost"
                else:
                    pos = "middle"
                parts.append(f"{label} ({pos})")
        # Tail-annotate the summary with a duplicate-count callout. This
        # is the unambiguous signal the LLM uses to trigger the multi-
        # instance disambiguation rule (it can't be confused by the
        # '(left)' / '(right)' qualifier on individual entries).
        dupes = [
            f"{c}× {lbl}" for lbl, c in label_counts.items() if c >= 2
        ]
        summary = "Scene currently shows: " + ", ".join(parts) + "."
        if dupes:
            summary += " MULTIPLE INSTANCES: " + ", ".join(dupes) + "."
        return summary

    def _describe_intent(self, intent_dict: dict) -> str:
        attrs = intent_dict.get("object_attributes") or {}
        colour = (attrs.get("colour") if isinstance(attrs, dict) else None) or ""
        obj = intent_dict.get("object_intent") or "object"
        if colour:
            return f"the {colour} {obj}"
        return f"the {obj}"

    def _maybe_execute_grasp_for_intent(self, intent_dict: dict) -> None:
        target_desc = self._describe_intent(intent_dict)

        with self._grasp_lock:
            pose = self._latest_pose
            candidates = self._latest_candidates
            received_at = self._latest_pose_received_at

        if pose is None:
            self._speak(
                f"I cannot pick {target_desc} yet. I do not have a grasp pose from perception."
            )
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self._grasp_pose_max_age_s > 0 and (now - received_at) > self._grasp_pose_max_age_s:
            self._speak(
                f"I cannot pick {target_desc}. The latest grasp pose is too old."
            )
            return

        if not self._execute_grasp_client.server_is_ready():
            if not self._execute_grasp_client.wait_for_server(timeout_sec=1.0):
                self._speak("The grasp executor is not running. I cannot move the robot.")
                return

        self._speak(f"Okay. Picking up {target_desc} now.")

        goal = ExecuteGrasp.Goal()
        goal.grasp_pose = pose
        if candidates is not None:
            goal.candidate_poses = candidates
        goal.timeout_s = float(self._grasp_timeout_s)

        self.get_logger().info(f"Sending ExecuteGrasp goal for {target_desc}")
        send_future = self._execute_grasp_client.send_goal_async(goal)

        def _on_goal_response(fut) -> None:
            handle = fut.result()
            if handle is None or not handle.accepted:
                self._speak("The executor rejected the grasp request.")
                return
            with self._grasp_lock:
                self._active_grasp_handle = handle
            handle.get_result_async().add_done_callback(self._on_grasp_result)

        send_future.add_done_callback(_on_goal_response)

    def _on_grasp_result(self, fut) -> None:
        with self._grasp_lock:
            self._active_grasp_handle = None
        try:
            result_wrapper = fut.result()
        except Exception as exc:
            self._speak(f"The grasp failed with an exception: {exc}")
            return
        if result_wrapper is None:
            self._speak("The grasp returned no result.")
            return
        result = result_wrapper.result
        if result.success:
            self._speak("Done.")
        else:
            self._speak(f"The grasp failed. {result.error}".strip())


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ChatIntentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
