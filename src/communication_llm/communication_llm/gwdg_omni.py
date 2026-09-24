"""Qwen3-Omni at GWDG (Academic Cloud) as Morty's speech-to-text.

Why this model (benchmark 2026-09-24, 24 clips of Majd's voice from 12
trial recordings, cut exactly like the live capture loop): Qwen3-Omni-30B-
A3B got "grasp" 3/3 and "gripper" 2/2 with no wrong-language output at a
median 0.29 s (p90 0.38 s); the local Parakeet got 1/3 and 0/2 at ~0.6 s
on the loaded CPU. It cannot run on this laptop (≈17 GB even at 4 bit,
the GTX 1080 has 8 GB), so it is called over HTTPS.

Why Parakeet stays loaded: GWDG caps the chat route per key (30/min,
200/h, 1000/day, 3000/month; every sentence Morty hears is one request)
and a network or server fault must never make Morty deaf. This object
answers ``None`` whenever it cannot give a transcript — no key, budget
used up, paused after a failure, timeout, HTTP error, or an answer that
is not a transcript — and the transcriber then uses Parakeet for that
sentence. A failure pauses Omni (``Retry-After`` or 60 s after HTTP 429,
30 s otherwise) so later sentences do not each wait for the timeout.

The GWDG transcription endpoint (``/audio/transcriptions``) is not used:
it serves one fixed Whisper whatever ``model`` says, returns HTTP 500 on
any prompt, and timed out on 13 of 24 test clips. Omni is reached through
``chat/completions`` with an ``input_audio`` part plus a context hint.

Privacy: every sentence that passes the capture gates is sent to GWDG, a
German academic data centre. Participants' consent must name it.
"""
from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import socket
import time
import urllib.error
import urllib.request
import wave
from typing import Callable, Optional

GWDG_CHAT_URL = "https://chat-ai.academiccloud.de/v1/chat/completions"
OMNI_MODEL = "qwen3-omni-30b-a3b-instruct"

# The benchmarked instruction (gwdg_bench.py, 2026-09-24) plus one clause
# for silence. Without it, 2 s of silence came back as "go back home" — the
# context hint turned into a command (tested 2026-09-24); "output nothing"
# did not help, an explicit marker did.
OMNI_INSTRUCTION = (
    "Transcribe the speech in this audio exactly, word for word, in the language spoken (English or German). "
    "Context: the speaker talks to a robot named Morty; typical words are grasp, pick up, gripper, drop it on "
    "the right/left side, go back home, tape dispenser, small bottle, vitamin tube, vitamin C, mouse, jacket. "
    "Do not correct or complete what was said. If the audio contains no clearly spoken words (silence, noise, "
    "breathing, a cough), output exactly [no speech] and nothing else. "
    "Output only the transcript."
)

# More words than this per second of audio (+ slack) cannot be a transcript
# of it: the model answered or described instead. Fast speech is ~3.5 w/s.
_MAX_WORDS_PER_S = 5.0
_WORD_SLACK = 4
_NO_SPEECH = "no speech"


class OmniTranscriber:
    """One sentence of 16-bit mono PCM in, its transcript out — or ``None``
    when the caller should fall back to the local engine."""

    def __init__(self,
                 api_key: str,
                 log: Callable[[str, str], None],
                 url: str = GWDG_CHAT_URL,
                 model: str = OMNI_MODEL,
                 timeout_s: float = 1.5,
                 day_cap: int = 950,
                 month_cap: int = 2900,
                 usage_path: str = "~/.morty/gwdg_omni_usage.json") -> None:
        self._key = (api_key or "").strip().strip('"')
        self._log = log
        self._url = url
        self._model = model
        self._timeout_s = float(timeout_s)
        self._day_cap = int(day_cap)
        self._month_cap = int(month_cap)
        self._usage_path = os.path.expanduser(usage_path)
        self._usage = self._load_usage()
        self._paused_until = 0.0
        self._budget_warned = ""

    @property
    def available(self) -> bool:
        return bool(self._key)

    # ---- budget (persisted: the monthly cap outlives node restarts) ----
    def _load_usage(self) -> dict:
        try:
            with open(self._usage_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_usage(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._usage_path), exist_ok=True)
            tmp = self._usage_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._usage, f)
            os.replace(tmp, self._usage_path)
        except OSError as exc:
            self._log("warn", f"[omni] could not save the usage counter: {exc}")

    def _budget_left(self) -> bool:
        today = _dt.date.today()
        day, month = today.isoformat(), today.strftime("%Y-%m")
        if self._usage.get("day") != day:
            self._usage.update(day=day, day_n=0)
        if self._usage.get("month") != month:
            self._usage.update(month=month, month_n=0)
        over = ("day" if self._usage["day_n"] >= self._day_cap else
                "month" if self._usage["month_n"] >= self._month_cap else "")
        if over and self._budget_warned != f"{over}:{day}":
            self._budget_warned = f"{over}:{day}"
            cap = self._day_cap if over == "day" else self._month_cap
            self._log("warn", f"[omni] {'daily' if over == 'day' else 'monthly'} budget of {cap} requests used — Parakeet only until it resets")
        return not over

    def _count_request(self) -> None:
        self._usage["day_n"] = self._usage.get("day_n", 0) + 1
        self._usage["month_n"] = self._usage.get("month_n", 0) + 1
        self._save_usage()

    def _pause(self, seconds: float, why: str) -> None:
        self._paused_until = time.monotonic() + seconds
        self._log("warn", f"[omni] {why} — Parakeet for the next {seconds:.0f} s")

    # ---- the request ------------------------------------------------------
    @staticmethod
    def _wav(pcm16: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm16)
        return buf.getvalue()

    def transcribe(self, pcm16: bytes, sample_rate: int) -> Optional[str]:
        if not self._key or time.monotonic() < self._paused_until or not self._budget_left():
            return None
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(self._wav(pcm16, sample_rate)).decode(),
                    "format": "wav"}},
                {"type": "text", "text": OMNI_INSTRUCTION},
            ]}],
        }
        req = urllib.request.Request(
            self._url, data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        self._count_request()
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as r:
                body = json.loads(r.read())
            text = str(body["choices"][0]["message"]["content"] or "").strip()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                try:
                    wait = float(exc.headers.get("Retry-After") or 60)
                except ValueError:
                    wait = 60.0
                self._pause(min(max(wait, 5.0), 3600.0), "GWDG rate limit (HTTP 429)")
            else:
                self._pause(30.0, f"HTTP {exc.code}")
            return None
        except (socket.timeout, TimeoutError):
            self._pause(30.0, f"no answer within {self._timeout_s:.1f} s")
            return None
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            if isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
                self._pause(30.0, f"no answer within {self._timeout_s:.1f} s")
            else:
                self._pause(30.0, f"request failed ({type(exc).__name__}: {exc})")
            return None
        dt = time.monotonic() - t0
        text = text.strip().strip('"').strip()
        if text.strip("[]()<>. ").lower() == _NO_SPEECH:
            text = ""
        audio_s = len(pcm16) / 2 / sample_rate
        if len(text.split()) > _MAX_WORDS_PER_S * audio_s + _WORD_SLACK:
            self._log("warn", f"[omni] {len(text.split())} words for {audio_s:.1f} s of audio is not a "
                              f"transcript ({text[:80]!r}) — using Parakeet for this sentence")
            return None
        self._log("info", f"[timing] omni {dt:.2f}s for {audio_s:.1f}s audio "
                          f"(today {self._usage['day_n']}, month {self._usage['month_n']})")
        return text
