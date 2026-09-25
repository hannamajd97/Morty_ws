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
                              keep holding) | stay (home question) | keep
                              (destination question: NOTHING is sent — the
                              question stays open; chat_node only says it keeps
                              holding, 2026-09-25 Majd's I1)
    ("gripper", <action>)     a standalone open_gripper / close_gripper
    ("llm", "")               everything else

2026-09-25 (Majd's B2): at the destination question a whole "<side|home>,
<give it to me / open / release …>[, and go back home]" is an answer too —
the legacy compound actions drop_<side>_release[_home] and go_home_open.

2026-09-25 (Majd's B1): ``match_pick`` — a WHOLE plain pick of a known
object ("Grasp for me the vitamin tube.", "Give me the small bottle.") for
chat_node's instant pick (only when no question is open and nothing is
held; chat_node decides that). Its verb and object tables are the legacy
FAST-PICK tables, which live here now (one owner; chat_node's legacy layer
reads the same objects).

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

# 2026-09-25 (Majd's B2): destination + release [+ back home] in ONE whole
# utterance at the destination question ("Drop it on the right side, give me
# the object and go back home.", "Drop it home, give me the object.") — the
# LLM answered the last one "I'm already heading home" with no tool call.
# Punctuation is gone after normalise, so ", " / ". " / " and " / " then "
# all join the parts.
RELEASE = (r"(?:give (?:me )?(?:it|the object|the objects)(?: to me)?|hand (?:it )?(?:over|to me)"
           r"|open(?: the gripper)?|release(?: it)?|let (?:it )?go)")
_CONN = r"(?:and )?(?:then )?"
DEST_COMPOUND = _whole(
    r"(?:" + AFF + r" )?(?:" + VERB + r"?(?:(?:on|to|at|in|onto|over) )?(?:the )?"
    r"(?P<side>left|right|links|rechts)(?: (?:side|one|position|drop|drop position))?"
    r"|(?P<home>" + VERB + r"?" + HOME + r"))"
    r" " + _CONN + RELEASE
    + r"(?P<back> " + _CONN + r"(?:go )?(?:back )?(?:to )?(?:the )?home(?: position)?)?")
# 2026-09-25 (Majd's I1): a whole NO to "Where should I drop it?" — the
# executor keeps waiting for a destination, so nothing is sent; the LLM
# used to answer "Don't drop it." with "Sorry, I'll pick it up instead."
NO_DEST = _whole(
    r"(?:" + NEG + r" )*(?:" + NEG + r"|wait|hold on|hold it|keep it|keep holding(?: it)?|not now"
    r"|(?:don't|dont|do not) (?:drop|release|open|put down)(?: it)?(?: yet)?"
    r"|(?:don't|dont|do not) put it down(?: yet)?)")

_SIDE = {"left": "left", "right": "right", "links": "left", "rechts": "right"}

# ── instant picks (2026-09-25, Majd's B1) ──────────────────────────────────
# The legacy FAST-PICK tables (chat_node, 2026-08-26 / 08-31), moved here
# unchanged so the minimal layer and the legacy layer share ONE table.
# Deterministic PICKS (2026-08-26): Haiku kept refusing plain pick commands
# with fabrications ("I'm already grasping it — it should arrive in a
# moment") and inventing alias features for ASR garbles ("craftsform/class/
# trust for me is..."). Qualified picks (left/right/middle/ordinal) still go
# to the LLM — the router's disambiguation needs them.
PICK_VERB = (
    # "glass"/"gras" joined 2026-08-31: Majd's habitual "Grasp for me
    # the X" comes back from Whisper as "Glass for me, the vitamin
    # tube" / "Glass for me is a small bottle" — measured 15.9 s and
    # 18.8 s to first reply via the LLM, vs ~0.02 s on this fast path.
    r"(?:grasp|grass|gras\b|glass|class|cross|crass|craft\w*|trust|"
    r"gas\b|drus\b|"  # 2026-08-31: "Gas for me is a vitamin C." / "Drus for me..."
    r"pick(?:\s*up)?|grab|fetch)"
)
PICK_OBJECTS: list[tuple[str, str, str | None]] = [
    (r"vitamin\s*c\b", "vitamin_tube", "yellow"),
    (r"vitamin\s*b(?:\s*12)?\b", "vitamin_tube", "red"),
    (r"multi\s*-?\s*(?:vitamin|mineral)\b", "vitamin_tube", "blue"),
    (r"vitamin\s+(?:tube|2)\b", "vitamin_tube", None),
    (r"tape\s+dispenser\b", "tape_dispenser", None),
    (r"small\s+bottle\b", "small_bottle", None),
    (r"nutmeg(?:\s+jar)?\b", "nutmeg_jar", None),
    (r"retail\s+box\b", "retail_box", None),
    (r"cd\s+case\b", "cd_case", None),
    (r"kitchen\s+towel\b", "kitchen_towel", None),
    (r"(?:wooden\s+)?maze(?:\s+puzzle)?\b", "wooden_maze_puzzle", None),
    (r"(?:toy\s+)?faucet\b", "toy_faucet", None),
]
# "give me / bring me / hand me / pass me X" is the same plain pick (B1).
_PICK_GIVE = r"(?:give|bring|hand|pass)"
_PICK_HEAD = re.compile(
    r"^(?:" + FILL + r" )*(?:(?:can|could) you )?(?:" + PICK_VERB + r"|" + _PICK_GIVE + r")"
    r"(?: for me| me)?(?: is)?(?: (?:the|a|an))?(?= )")
_FILL_WORD = re.compile(FILL)

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


def match_pick(text: str) -> tuple[str, str | None] | None:
    """2026-09-25 (Majd's B1: "picks via the LLM took 1.3-2.8 s"): (label,
    colour|None) when the WHOLE utterance is a plain pick of a known object —
    [fillers] VERB [for me|me] [is] [the|a|an] OBJECT [fillers/please] —
    else None. Anything more ("… and tell me a story", "… and go home"), a
    qualifier ("the leftmost …"), a prefix that is not a filler ("No, grasp
    …", "Yes, a crust for me …") or an object outside PICK_OBJECTS ("the
    phone") is the LLM's. A trailing "tube" is allowed ("the Vitamin C
    tube")."""
    t = normalise(text)
    m = _PICK_HEAD.match(t)
    if not m:
        return None
    words = t[m.end():].split()
    while words and _FILL_WORD.fullmatch(words[-1]):
        words.pop()
    obj = " ".join(words)
    for pat, label, colour in PICK_OBJECTS:
        if re.fullmatch(r"(?:" + pat + r")(?: tube)?", obj):
            return (label, colour)
    return None


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
       home yes/"go back home" → go_home, no/stay → stay; destination: a
       whole NO → keep (nothing is sent, I1); only with ``ext_dest``
       (Majd's EXT-DEST) a bare side/home → drop_<side> / go_home, and a
       side/home + release [+ home] → drop_<side>_release[_home] /
       go_home_open (B2).
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
        elif question == "destination":
            if NO_DEST.match(t):
                return ("negative", "keep")          # 2026-09-25 (I1): nothing is sent
            if ext_dest:
                m = DEST_COMPOUND.match(t)           # 2026-09-25 (B2)
                if m:
                    if m.group("home"):
                        return ("answer", "go_home_open")
                    side = _SIDE[m.group("side")]
                    return ("answer", f"drop_{side}_release" + ("_home" if m.group("back") else ""))
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
