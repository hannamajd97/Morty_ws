"""
voice_grammar.py
================

2026-09-25: the ONE deterministic voice grammar of chat_node (Majd: "stop
first; the deterministic layer shrinks to stop/reset, yes/no to the question
Morty has open, and open/close the gripper — only when the WHOLE utterance is
that command; everything else goes to the LLM").

Pure functions, no ROS: ``normalise`` and ``classify``. chat_node decides the
inputs (the open question comes only from the executor's task state, and
``fresh`` means the question opened before the operator STARTED speaking —
2026-09-25 review M1; it was "stopped") and
acts on the verdict. The regexes are the ones the design judge measured at
93/93 on the probe corpus (SPEC §4 C5); change them only together with
the scratch test that covers every MUST / MUST-NOT row. 2026-09-25 (review
H2/Y4): the stop test reads sentence punctuation as a clause boundary that a
negation cannot cross, and German "halt" counts only at a clause start.

Verdicts ``(kind, action)``:

    ("stop", "stop")          a stop word anywhere (not negated) — wins over all
    ("reset", "reset")        a stop word AND a reset word, or a whole reset phrase
    ("answer", <action>)      a positive answer to the open, fresh question:
                              open_gripper | go_home | drop_right | drop_left
    ("negative", <action>)    a negative answer: close_gripper (release question,
                              keep holding) | stay (home question)
    ("gripper", <action>)     a standalone open_gripper / close_gripper
    ("llm", "")               everything else

Accepted false positives (by design, stop must never be missed): "bus stop",
"why did you stop".
"""
from __future__ import annotations

import re

# ── building blocks (SPEC C5, verbatim) ────────────────────────────────────
FILL = r"(?:please|now|okay|ok|just|so|and|then|alright|uh|um|uhm|äh|hey|morty)"
AFF = r"(?:yes|yeah|yep|yup|sure|ok|okay|alright|all right|of course|go ahead|do it|ja|klar)"
NEG = r"(?:no|nope|nah|nein|not yet|no thanks|no thank you)"
HOME = r"(?:go )?(?:back )?(?:(?:to|at|on) )?(?:the )?(?:home(?: position)?|nach hause)"
STOPW = r"(?:stop|stopp|halt|freeze|abort|cancel|abbrechen)"
# 2026-09-25 (review Y4): the stop words that count ANYWHERE in the
# utterance. German "halt" is also a filler ("Dann nimm halt die Flasche.")
# and counts only at the start of a clause, after fillers ("Halt!", "Morty,
# halt!", "Wait, don't move! Halt!") — HEAD matched it only at the start.
STOPW_ANYWHERE = r"(?:stop|stopp|freeze|abort|cancel|abbrechen)"
RESETW = r"(?:reset|start over|zurücksetzen)"
# 2026-09-25 (review H2): sentence punctuation becomes this token in the
# stop test (_clauses); a negation never reaches across it, so "Don't. Stop!"
# and "No, don't! Stop!" are stops while "Don't stop, keep going." is not.
_BOUNDARY = r"\|"
VERB = r"(?:(?:drop|put|take|bring|place|move)(?: it)?(?: position)? )"


def _word(w: str) -> str:
    """2026-09-25: ``w`` as a whole word — an apostrophe or hyphen glues it to
    its neighbour ("non-stop" is not a stop, "stopping" is not a stop)."""
    return r"(?<![\w'-])" + w + r"(?![\w'-])"


def _neg(w: str) -> str:
    """2026-09-25: a negated ``w`` ("don't stop", "never ever reset") — the
    span is deleted before the stop test, so "Don't stop, keep going." is not
    a stop.

    2026-09-25 (review H2): only "ever" / "you" may stand between the
    negation and ``w`` (it was any word: "Don't move, stop!" and "Never
    mind, stop." were swallowed), and only single spaces — the clause
    boundary token of _clauses never matches, so a negation cannot reach
    into the next sentence ("Don't, cancel!", "No, don't! Stop!")."""
    return r"(?<![\w'-])(?:don't|dont|do not|never)(?: (?:ever|you))? " + w + r"(?![\w'-])"


def _whole(core: str) -> "re.Pattern[str]":
    """2026-09-25: the WHOLE normalised utterance is ``core``, with only
    filler words ("please", "okay", "uh", "morty" …) around it."""
    return re.compile(r"^(?:" + FILL + r" )*(?:" + core + r")(?: " + FILL + r")*$")


_WORD_STOP = re.compile(_word(STOPW_ANYWHERE))
# 2026-09-25 (review Y4): "halt" at the start of a clause, fillers allowed.
_CLAUSE_HALT = re.compile(r"(?:^|" + _BOUNDARY + r" )(?:" + FILL + r" )*halt(?![\w'-])")
_WORD_RESET = re.compile(_word(RESETW))
_NEG_STOP = re.compile(_neg(STOPW))
_NEG_RESET = re.compile(_neg(RESETW))

RESET = _whole(
    r"(?:" + HOME + r" (?:and )?(?:then )?)?" + RESETW
    + r"(?: (?:the )?(?:pick|grasp|cycle|task|everything|it|all))?"
    + r"(?: (?:and )?(?:then )?" + HOME + r")?")
YES = _whole(AFF + r"(?: " + AFF + r")*")
YES_HOME = _whole(r"(?:" + AFF + r" )*(?:let's )?" + HOME)
NO_REL = _whole(
    r"(?:" + NEG + r" )*(?:" + NEG + r"|wait|hold on|hold it|keep it|keep holding(?: it)?"
    r"|(?:don't|dont|do not) (?:open|release)(?: it| the gripper)?|not now)")
NO_HOME = _whole(
    r"(?:" + NEG + r" )*(?:" + NEG + r"|wait|stay(?: here| there)?"
    r"|(?:don't|dont|do not) go(?: back)?(?: home)?|not now)")
OPEN = _whole(
    r"(?:" + AFF + r" )*(?:you can |can you |could you )?"
    r"(?:open(?: it| up| the gripper| the hand| your hand)?|release(?: it| the object)?"
    r"|let (?:it )?go|greifer öffnen|öffne(?:n)?(?: den)? greifer|auf)")
CLOSE = _whole(
    r"(?:" + NEG + r" )*(?:you can |can you |could you )?"
    r"(?:close(?: it| up| the gripper| the hand| your hand)?|grip|greifer schließen"
    r"|schließe(?:n)?(?: den)? greifer|zu)")
DEST_SIDE = _whole(
    r"(?:" + AFF + r" )?" + VERB + r"?(?:(?:on|to|at|in|onto|over) )?(?:the )?"
    r"(left|right|links|rechts)(?: (?:side|one|position|drop|drop position))?")
DEST_HOME = _whole(r"(?:" + AFF + r" )?" + VERB + r"?" + HOME)

_SIDE = {"left": "left", "right": "right", "links": "left", "rechts": "right"}

LLM = ("llm", "")


def normalise(text: str) -> str:
    """2026-09-25: lowercase, curly apostrophe → straight, every character
    that is not a word character, whitespace, apostrophe or hyphen → space,
    whitespace collapsed. Whisper/Omni punctuation ("Stop. Reset.") and the
    LLM's curly quotes then never decide a verdict."""
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"[^\w\s'-]", " ", t)
    return " ".join(t.split())


def _clauses(text: str) -> str:
    """2026-09-25 (review H2): ``normalise`` for the STOP test only — every
    run of sentence punctuation (. , ! ? ; : …) becomes the boundary token
    "|" instead of a space. "No, don't! Stop!" → "no | don't | stop |": the
    negation pattern cannot cross the "|", so the stop survives. The other
    patterns keep matching the plain ``normalise`` text."""
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"[.,!?;:…]+", " | ", t)
    t = re.sub(r"[^\w\s'|-]", " ", t)
    return " ".join(t.split())


def classify(text: str, question: str = "", fresh: bool = False,
             ext_dest: bool = False) -> tuple[str, str]:
    """2026-09-25: the verdict for one utterance, in SPEC C5 order.

    1. Stop first: negated stop/reset spans are deleted; a remaining stop
       word means stop — or reset when a reset word also remains ("Stop.
       Reset. Go back home." is a reset: stop freezes, reset goes home, and
       the operator asked for both).
    2. A whole reset phrase → reset.
    3. Only while ``question`` is open AND ``fresh`` (it opened before the
       operator started speaking — a "yes" said to something else must not
       answer it): release yes/open → open_gripper, no/close → close_gripper;
       home yes/"go back home" → go_home, no/stay → stay; destination (only
       with ``ext_dest``, Majd's EXT-DEST) a bare side/home → drop_<side> /
       go_home.
    4. A whole open/close phrase → the standalone gripper command.
    5. Everything else → the LLM.
    """
    t = normalise(text)
    if not t:
        return LLM
    # 2026-09-25 (review H2/Y4): the stop test reads the clause-marked text
    # (a negation ends at sentence punctuation) and takes "halt" only at a
    # clause start.
    rest = " ".join(_NEG_RESET.sub(" ", _NEG_STOP.sub(" ", _clauses(text))).split())
    if _WORD_STOP.search(rest) or _CLAUSE_HALT.search(rest):
        return ("reset", "reset") if _WORD_RESET.search(rest) else ("stop", "stop")
    if RESET.match(t):
        return ("reset", "reset")
    if question and fresh:
        if question == "release":
            if YES.match(t) or OPEN.match(t):
                return ("answer", "open_gripper")
            if NO_REL.match(t) or CLOSE.match(t):
                return ("negative", "close_gripper")
        elif question == "home":
            if YES.match(t) or YES_HOME.match(t):
                return ("answer", "go_home")
            if NO_HOME.match(t):
                return ("negative", "stay")
        elif question == "destination" and ext_dest:
            m = DEST_SIDE.match(t)
            if m:
                return ("answer", "drop_" + _SIDE[m.group(1)])
            if DEST_HOME.match(t):
                return ("answer", "go_home")
    if OPEN.match(t):
        return ("gripper", "open_gripper")
    if CLOSE.match(t):
        return ("gripper", "close_gripper")
    return LLM
