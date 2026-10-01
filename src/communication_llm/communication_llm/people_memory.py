"""Morty remembers the people he talks to (2026-10-01).

Majd's design (2026-09-30 / 10-01):
  * Morty asks the person's NAME, spells it back letter by letter and asks
    whether it is right; the person corrects him until it is.
  * A name he already knows: "Oh okay, we have chatted before, <name>" —
    and what happens in this conversation is added to that person.
  * A new name: "Would you like me to remember your preferences?" — saved
    ONLY after the person agrees.
  * What is kept: the objects he grasped for them (so: which they like
    most) and what they told him about themselves.
  * He may ask about them ("When do you like to get your vitamin tube?") —
    ONLY while the robot is idle.

Split of work: the conversation (asking, spelling, the greeting, the
question) is the language model's, through the ``person_memory`` tool and
the PERSON message chat_node adds to every turn (``context``). Everything
that must not depend on the model is decided here: consent (``save`` is
refused unless the sentence that triggered it says yes), what is recorded
automatically (finished picks, ``record_pick``), when a question or the
name may be asked (``context``: robot idle, rate limits), and when a
conversation ends (``touch``: after ``session_timeout_s`` of silence the
next voice is a new conversation and the name is asked again).

The store is one JSON file OUTSIDE the repository (default
~/.morty/people.json) — personal data never lands in a git commit.
Writes are atomic (temp file + rename).
"""
from __future__ import annotations

import collections
import json
import os
import re
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

PEOPLE_TOOL: Dict[str, Any] = {
    "type": "function",
    "name": "person_memory",
    "description": (
        "Who you are talking to, and what you remember about people. "
        "identify: the person CONFIRMED the spelling of their name — returns whether you "
        "know them and what you remember. save: the person AGREED (in this message) that "
        "you remember their preferences. decline: they do not want to be remembered. "
        "remember: a saved person told you a preference or habit (fact = one short "
        "third-person sentence, e.g. 'likes the Vitamin C tube in the morning'). "
        "forget: they asked you to forget them. goodbye: they said goodbye / are leaving."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["identify", "save", "decline", "remember", "forget", "goodbye"]},
            "name": {"type": "string",
                     "description": "identify only: the name exactly as the person confirmed its spelling"},
            "fact": {"type": "string", "description": "remember only"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

# A yes to "Would you like me to remember your preferences?" — the words
# that may let ``save`` through (deterministic consent, never the model's
# word alone). German forms because Morty answers German in German.
_CONSENT = re.compile(
    r"(?<![\w'-])(?:yes|yeah|yep|yup|sure|ok|okay|alright|of course|please|go ahead|do it|"
    r"definitely|absolutely|why not|save|remember|ja|klar|gerne|gern|natürlich|bitte)(?![\w'-])",
    re.IGNORECASE)
_NEGATED = re.compile(r"(?<![\w'-])(?:no|nope|don't|dont|do not|not|never|nein|nicht)(?![\w'-])", re.IGNORECASE)
# Positive phrases that contain a negation word — removed before the
# negation test, so "Yes, I don't mind" / "No problem" / "Why not" count.
_POSITIVE_IDIOMS = re.compile(
    r"(?<![\w'-])(?:why not|(?:i )?(?:don't|dont|do not) mind|no problem|not a problem|"
    r"no worries|warum nicht|kein problem)(?![\w'-])", re.IGNORECASE)
# "Yes, that's my name" after Morty spelled it.
_CONFIRM = re.compile(r"(?<![\w'-])(?:correct|right|exactly|that's it|that is it|perfect|richtig|genau|"
                      r"stimmt)(?![\w'-])", re.IGNORECASE)
# What the model may pass as a name when it has none.
_NOT_A_NAME = {"unknown", "user", "none", "there", "friend", "morty", "operator", "guest", "person", "you"}


def spelled_in(name: str, reply: str) -> bool:
    """Morty's reply spelled ``name`` letter by letter ("M, A, J, D",
    "H-A-N-A-H", "M A J D")."""
    letters = [ch for ch in (name or "") if ch.isalpha()]
    if len(letters) < 2:
        return False
    pat = r"(?<![\w])" + r"[\s,.\-–—]{1,4}".join(re.escape(ch) for ch in letters) + r"(?![\w])"
    return re.search(pat, reply or "", re.IGNORECASE) is not None


# A no to the consent question, at the START of the answer.
_NO_START = re.compile(r"^\W*(?:no\b|nope\b|nah\b|nein\b|no thanks|not now|i'd rather not|"
                       r"i would rather not|don't|dont|do not|better not|lieber nicht)", re.IGNORECASE)
_FORGET = re.compile(r"(?<![\w'-])(?:forget|delete|remove|erase|vergiss|lösch\w*)(?![\w'-])", re.IGNORECASE)

_DEST_WORDS = (("left", "the left side"), ("right", "the right side"), ("home", "home"), ("here", "where it was"))


def _key(name: str) -> str:
    """The lookup key of a name: letters only, lower case ("Majd", "majd ",
    "M.A.J.D" are one person)."""
    return re.sub(r"[^\w]", "", name or "", flags=re.UNICODE).lower()


def says_yes(text: str) -> bool:
    """The person agreed in ``text``: a consent word and, once positive
    idioms are taken out, no negation ("Please don't" is a no)."""
    t = text or ""
    if not _CONSENT.search(t) and not _POSITIVE_IDIOMS.search(t):
        return False
    return not _NEGATED.search(_POSITIVE_IDIOMS.sub(" ", t))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime())


def _human_dest(dest: str) -> str:
    d = (dest or "").lower()
    for word, human in _DEST_WORDS:
        if word in d:
            return human
    return d.replace("_", " ")


class PeopleMemory:
    """One owner object: the store, the current conversation, the rules."""

    def __init__(self, path: str = "~/.morty/people.json", *,
                 session_timeout_s: float = 600.0,
                 question_gap_s: float = 180.0,
                 questions_per_conversation: int = 3,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.path = os.path.expanduser(path)
        self.session_timeout_s = float(session_timeout_s)
        self.question_gap_s = float(question_gap_s)
        self.questions_per_conversation = int(questions_per_conversation)
        self._log = log or (lambda _m: None)
        self.people: Dict[str, Dict[str, Any]] = {}
        self._load()
        self._new_conversation()
        self._last_heard = 0.0

    # ── store ────────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.people = dict(d.get("people") or {})
        except FileNotFoundError:
            self.people = {}
        except Exception as exc:  # noqa: BLE001 — a broken file must not stop the voice
            self._log(f"[people] could not read {self.path} ({exc}); starting empty, the file is left as it is")
            self.people = {}
            self._readonly = True

    def _write(self) -> None:
        if getattr(self, "_readonly", False):
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix=".people.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"version": 1, "people": self.people}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ── the conversation ─────────────────────────────────────────────
    def _new_conversation(self) -> None:
        self.current: Optional[Dict[str, Any]] = None   # {"key", "name", "saved", "declined"}
        self.name_asked = False
        self.consent_asked = False
        self.questions_asked = 0
        self.last_question_at = 0.0
        self.pending_events: List[Dict[str, Any]] = []   # picks before the person was saved
        self.last_reply = ""                               # what Morty said last turn
        # chat_node starts the model's history afresh at the next turn: the
        # next person must not inherit this one's name from the chat
        # (2026-10-01 live test: after the silence Morty greeted "Majd"
        # from the old history and never asked or looked him up).
        self.fresh_history = True

    def touch(self, now: Optional[float] = None) -> bool:
        """A person spoke. After session_timeout_s of silence this is a new
        conversation (True): whoever it is, the name is asked again."""
        now = time.monotonic() if now is None else now
        new = self._last_heard > 0.0 and now - self._last_heard > self.session_timeout_s
        if new:
            who = (self.current or {}).get("name")
            self._log(f"[people] {now - self._last_heard:.0f} s of silence — new conversation"
                      + (f" (was {who})" if who else ""))
            self._new_conversation()
        elif self._last_heard == 0.0:
            self.fresh_history = False       # the node's own start-up history is fresh already
        self._last_heard = now
        return new

    def heard(self, text: str) -> None:
        """What the person said this turn, before the model reads it: a
        clear NO right after Morty asked to remember them is a decline from
        code (live test: the model answered "No problem" to "No thanks."
        without calling decline)."""
        cur = self.current
        if (cur and not cur.get("saved") and not cur.get("declined") and self.consent_asked
                and _NO_START.search(text or "")):
            cur["declined"] = True
            self.pending_events.clear()
            self._log(f"[people] {cur['name']} declined ({text!r}) — nothing stored")

    def said(self, reply: str) -> None:
        """Morty's reply of this turn (the spelling check reads it next turn)."""
        self.last_reply = reply or ""

    def take_fresh_history(self) -> bool:
        """True once after a new conversation began (silence or goodbye)."""
        f, self.fresh_history = self.fresh_history, False
        return f

    def _entry(self) -> Optional[Dict[str, Any]]:
        if self.current and self.current.get("saved"):
            return self.people.get(self.current["key"])
        return None

    # ── the tool ─────────────────────────────────────────────────────
    def handle(self, args: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        """One person_memory call; the dict goes back to the model."""
        action = str(args.get("action") or "").strip().lower()
        if action == "identify":
            return self._identify(str(args.get("name") or ""), user_text)
        if action == "save":
            return self._save(user_text)
        if action == "decline":
            if self.current:
                self.current["declined"] = True
            self.consent_asked = True
            self.pending_events.clear()
            return {"ok": True, "note": "nothing is stored; do not ask again in this conversation"}
        if action == "remember":
            return self._remember(str(args.get("fact") or ""))
        if action == "forget":
            return self._forget(user_text)
        if action == "goodbye":
            who = (self.current or {}).get("name")
            self._new_conversation()
            return {"ok": True, "note": f"conversation with {who or 'this person'} ended"}
        return {"ok": False, "error": f"unknown action {action!r}"}

    def _identify(self, name: str, user_text: str = "") -> Dict[str, Any]:
        name = " ".join(name.split()).strip(" .,!?")
        parts = [w for w in re.split(r"[\s,.\-–—]+", name) if w]
        if len(parts) >= 2 and all(len(w) == 1 and w.isalpha() for w in parts):
            # the model passed the spelling ("H, A, N, A, H", live test): the name
            name = "".join(parts).capitalize()
        key = _key(name)
        if not key or key in _NOT_A_NAME:
            return {"ok": False, "error": "no real name — ask the person for their name"}
        # Majd's protocol, enforced here (2026-10-01 live test: the model
        # identified "Hana" straight from "My name is Hana." and "Unknown"
        # from "Hello there."): Morty spelled THIS name in his last reply
        # and the person confirmed it in this message.
        if not spelled_in(name, self.last_reply) or not (
                says_yes(user_text) or _CONFIRM.search(user_text or "")):
            return {"ok": False, "error": f"not confirmed — spell it back first ('{name} — "
                    + ", ".join(ch.upper() for ch in name if ch.isalpha())
                    + ". Is that correct?') and call identify only after they confirm"}
        self.name_asked = True
        if self.current and self.current.get("key") == key:
            # the same person confirmed again in this conversation
            e = self.people.get(key)
            return {"ok": True, "known": bool(e), "name": self.current["name"],
                    "note": "already identified in this conversation", **(self._summary(e) if e else {})}
        if self.current:
            # someone else now: the previous person's unsaved picks are theirs
            self.pending_events = []
            self.consent_asked = False
            self.questions_asked = 0
        e = self.people.get(key)
        if e is not None:
            prev_seen = e.get("last_seen", "")
            e["last_seen"] = _now_iso()
            e["conversations"] = int(e.get("conversations", 0)) + 1
            self.current = {"key": key, "name": e.get("name", name), "saved": True, "declined": False}
            self._flush_pending()
            self._write()
            self._log(f"[people] {e.get('name')} recognised (conversation {e['conversations']})")
            return {"ok": True, "known": True, "name": e.get("name", name), "last_seen": prev_seen,
                    "say": f"Oh okay, we have chatted before, {e.get('name', name)}!",
                    **self._summary(e)}
        self.current = {"key": key, "name": name, "saved": False, "declined": False}
        # the model asks for consent in this very reply ("next" below) — a NO
        # in the next message is then a decline (heard); live test: Morty
        # asked, "No thanks." went unnoticed and the question would come back
        self.consent_asked = True
        self._log(f"[people] new person {name!r} (not saved)")
        return {"ok": True, "known": False, "name": name,
                "next": "ask once whether you may remember their preferences; save only after a yes"}

    def _save(self, user_text: str) -> Dict[str, Any]:
        if not self.current:
            return {"ok": False, "error": "no confirmed name yet — ask for the name first"}
        if self.current.get("saved"):
            return {"ok": True, "note": "already saved"}
        if not says_yes(user_text):
            return {"ok": False, "error": "the person did not say yes in this message — "
                                          "ask them; nothing was saved"}
        key, name = self.current["key"], self.current["name"]
        self.people[key] = {"name": name, "created": _now_iso(), "last_seen": _now_iso(),
                            "conversations": 1, "facts": [], "picks": []}
        self.current["saved"] = True
        self.consent_asked = True
        self._flush_pending()
        self._write()
        self._log(f"[people] {name} saved (with their yes: {user_text!r})")
        return {"ok": True, "saved": True, "name": name}

    def _remember(self, fact: str) -> Dict[str, Any]:
        e = self._entry()
        if e is None:
            return {"ok": False, "error": "this person is not saved — nothing is stored"}
        fact = " ".join(fact.split()).strip()
        if not fact:
            return {"ok": False, "error": "empty fact"}
        facts = e.setdefault("facts", [])
        if any(f.get("text", "").lower() == fact.lower() for f in facts):
            return {"ok": True, "note": "already remembered"}
        facts.append({"text": fact, "at": _now_iso()})
        self._write()
        self._log(f"[people] {e.get('name')}: remembered {fact!r}")
        return {"ok": True, "remembered": fact}

    def _forget(self, user_text: str) -> Dict[str, Any]:
        if not self.current or not self.current.get("saved"):
            return {"ok": True, "note": "nothing was stored about this person"}
        if not _FORGET.search(user_text or ""):
            return {"ok": False, "error": "the person did not ask to be forgotten in this message"}
        name = self.current["name"]
        self.people.pop(self.current["key"], None)
        self._write()
        self.current = {"key": self.current["key"], "name": name, "saved": False, "declined": True}
        self._log(f"[people] {name} forgotten at their request")
        return {"ok": True, "forgotten": name}

    # ── what happened, recorded from code ────────────────────────────
    def record_pick(self, obj: str, destination: str, outcome: str) -> None:
        """A finished pick (any outcome) of this conversation: stored for a
        saved person at once, kept for one who may still agree, dropped for
        one who declined."""
        if not obj:
            return
        ev = {"object": obj, "destination": destination, "outcome": outcome, "at": _now_iso()}
        if self.current and self.current.get("declined"):
            return
        e = self._entry()
        if e is None:
            self.pending_events.append(ev)
            return
        e.setdefault("picks", []).append(ev)
        self._write()
        self._log(f"[people] {e.get('name')}: pick of {obj} ({outcome}) recorded")

    def _flush_pending(self) -> None:
        e = self._entry()
        if e is None or not self.pending_events:
            return
        e.setdefault("picks", []).extend(self.pending_events)
        self.pending_events = []

    # ── what the model is told ───────────────────────────────────────
    @staticmethod
    def _summary(e: Dict[str, Any]) -> Dict[str, Any]:
        picks = [p for p in e.get("picks", []) if p.get("outcome") == "done"]
        fav = collections.Counter(p["object"].replace("_", " ") for p in picks).most_common(3)
        last = [p["object"].replace("_", " ")
                + (f" to {_human_dest(p['destination'])}" if p.get("destination") else "")
                + ("" if p.get("outcome") == "done" else f" ({p.get('outcome') or 'unfinished'})")
                + f" on {p.get('at', '')}"
                for p in e.get("picks", [])[-3:]]
        return {"facts": [f.get("text", "") for f in e.get("facts", [])][-10:],
                "most_picked": [f"{o} ({n}x)" for o, n in fav],
                "last_picks": last}

    def note_reply(self, reply: str, offered: Dict[str, bool]) -> None:
        """After a turn: what the model asked, so it is not asked again too
        soon. ``offered`` = what the PERSON message allowed this turn."""
        r = (reply or "").lower()
        if "?" not in r:
            return
        if offered.get("name") and "name" in r:
            self.name_asked = True
        if offered.get("consent") and ("remember" in r or "save" in r or "preferences" in r):
            self.consent_asked = True
        if offered.get("question"):
            self.questions_asked += 1
            self.last_question_at = time.monotonic()

    def context(self, robot_idle: bool, now: Optional[float] = None) -> tuple:
        """(PERSON message, what it allowed) for this turn."""
        now = time.monotonic() if now is None else now
        offered = {"name": False, "consent": False, "question": False}
        busy = "" if robot_idle else " — NOT NOW: the robot is busy or a robot question is open"
        cur = self.current
        if cur is None:
            if self.name_asked:
                return ("PERSON: you do not know who you are talking to; you already asked their "
                        "name in this conversation — do not ask again unless they bring it up.", offered)
            offered["name"] = robot_idle
            return ("PERSON: you do not know who you are talking to yet. ASK THEIR NAME"
                    + (" now, once, if the moment fits (a greeting, small talk, after a finished "
                       "task) — never instead of doing a requested action" if robot_idle else busy)
                    + ".", offered)
        name = cur["name"]
        if cur.get("declined"):
            return (f"PERSON: {name} — does NOT want to be remembered: store nothing, do not ask "
                    "again, no personal questions.", offered)
        if not cur.get("saved"):
            if self.consent_asked:
                return (f"PERSON: {name} — new, not saved; you already asked whether you may "
                        "remember them — wait for their answer, do not ask again.", offered)
            offered["consent"] = robot_idle
            return (f"PERSON: {name} — new, not saved. ASK once whether you may remember their "
                    "preferences" + ("" if robot_idle else busy) + "; save only after a yes.", offered)
        e = self.people.get(cur["key"]) or {}
        s = self._summary(e)
        parts = [f"PERSON: {name} — saved; you have chatted before "
                 f"({e.get('conversations', 1)} conversation(s), first {e.get('created', '?')})."]
        if s["facts"]:
            parts.append("You remember: " + "; ".join(s["facts"]) + ".")
        if s["most_picked"]:
            parts.append("Picked for them most: " + ", ".join(s["most_picked"]) + ".")
        if s["last_picks"]:
            parts.append("Last picks: " + "; ".join(s["last_picks"]) + ".")
        allowed = (robot_idle and self.questions_asked < self.questions_per_conversation
                   and now - self.last_question_at > self.question_gap_s)
        offered["question"] = allowed
        if allowed:
            parts.append("PERSONAL QUESTION: allowed now — you MAY end your reply with ONE short "
                         "question about their habits or preferences with the objects (e.g. 'When do "
                         "you like to get your vitamin tube?'), only if it fits; store the answer with "
                         "person_memory remember.")
        else:
            parts.append("PERSONAL QUESTION: not now.")
        return (" ".join(parts), offered)
