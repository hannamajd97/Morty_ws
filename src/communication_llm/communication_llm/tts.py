"""Text-to-speech wrapper used by the Speak action server.

Wraps Kokoro neural TTS (preferred) with a pyttsx3 fallback. Adapted from the
TTS portion of grasp_pipeline.vosk_selector.VoskSelector. Once the speech code
is consolidated into communication_llm, vosk_selector should import from here
instead of carrying its own copy.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Callable, List, Optional

import numpy as np


# espeak-ng-data location fix.
#
# Kokoro's misaki/phonemize layer imports espeakng_loader, which ships its
# own libespeak-ng.so compiled on CI with a hardcoded fallback data path
# (/home/runner/work/...). That path doesn't exist on this machine, so the
# very first synth call crashes the whole process with:
#   "Error processing file '.../share/espeak-ng-data/phontab': No such file..."
#
# libespeak-ng respects the ESPEAK_DATA_PATH env var. Setting it BEFORE any
# import of misaki/kokoro/espeakng_loader points the lib at our system
# espeak-ng-data (apt: espeak-ng-data), which has the full phoneme tables.
def _ensure_espeak_data_path() -> None:
    if os.environ.get("ESPEAK_DATA_PATH"):
        return
    candidates = [
        "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
        "/usr/share/espeak-ng-data",
        "/usr/share/espeak-data",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "phontab")):
            # libespeak-ng wants the parent of espeak-ng-data, not the dir
            # itself, when given via espeak_Initialize(...path...). But the
            # ESPEAK_DATA_PATH env var is read as the parent dir. Confirmed
            # by checking espeak-ng source: it appends '/espeak-ng-data'
            # to the env var.
            parent = os.path.dirname(c.rstrip("/"))
            os.environ["ESPEAK_DATA_PATH"] = parent
            return


_ensure_espeak_data_path()


# Monkey-patch espeakng_loader so misaki/kokoro use the system espeak-ng
# data directory instead of the broken bundled one. This MUST run before
# any `from kokoro import ...` or `from misaki import ...`. Safe no-op
# if espeakng_loader isn't installed.
def _patch_espeakng_loader() -> None:
    try:
        import espeakng_loader as _enl  # type: ignore
    except Exception:
        return
    system_data = None
    for c in (
        "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
        "/usr/share/espeak-ng-data",
    ):
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "phontab")):
            system_data = c
            break
    if system_data:
        _enl.get_data_path = lambda *_a, **_kw: system_data  # type: ignore[assignment]


_patch_espeakng_loader()


# Heuristic: if a phrase contains German diacritics OR common German
# stopwords, treat the whole thing as German. Kokoro is English-only so
# German text needs to go through the pyttsx3/espeak fallback instead,
# otherwise the listener hears garbled English-phoneme German.
_DE_DIACRITICS = re.compile(r"[äöüÄÖÜß]")
_DE_STOPWORDS = re.compile(
    r"\b("
    r"ich|du|er|sie|es|wir|ihr|nicht|ist|bin|bist|sind|war|waren|"
    r"und|oder|aber|der|die|das|den|dem|des|ein|eine|einen|einem|"
    r"mit|von|zu|zur|zum|für|auf|nach|über|unter|hier|dort|"
    r"hallo|guten|tag|danke|bitte|schön|sehr|wie|was|wer|wo|wann|warum|"
    r"morty|greifer|öffnen|schließen|nimm|geh|schau|"
    r"ja|nein|alles|klar|gerne|gleich|jetzt|sofort|deutsch|englisch|"
    r"verstehe|spreche|kann|kannst|soll|ablegen|rechts|links|flasche|"
    r"gut|dir|dich|mir|mich|kein|keine|auch|noch|schon|wird|werde|"
    r"habe|hast|hat|mache|greife|hebe|lege"
    r")\b",
    flags=re.IGNORECASE,
)

# English markers to referee mixed/ambiguous text ("was", "die", "in" exist
# in both languages — a raw German-stopword count alone misfires).
_EN_STOPWORDS = re.compile(
    r"\b(the|is|are|you|your|it|to|of|and|for|with|this|that|now|please|"
    r"i|i'm|i'll|up|down|right|left|bottle|picking|grabbing|here)\b",
    flags=re.IGNORECASE,
)


def looks_german(text: str) -> bool:
    """Best-effort: True iff text is more likely German than English."""
    if not text:
        return False
    if _DE_DIACRITICS.search(text):
        return True
    de = len(_DE_STOPWORDS.findall(text))
    en = len(_EN_STOPWORDS.findall(text))
    return de >= 2 and de > en


_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_INLINE_CODE = re.compile(r"`([^`]+)`")
_MARKDOWN_BOLD = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_MARKDOWN_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)")
_MARKDOWN_STRIKE = re.compile(r"~~([^~]+)~~")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+", flags=re.MULTILINE)
_MARKDOWN_BULLET = re.compile(r"^\s*[-*+]\s+", flags=re.MULTILINE)
_MARKDOWN_NUMBERED = re.compile(r"^\s*\d+\.\s+", flags=re.MULTILINE)
_MULTI_BLANK = re.compile(r"\n{3,}")
_FENCED_CODE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", flags=re.DOTALL)


_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U0001F900-\U0001F9FF☀-➿⬀-⯿️]"
)


def strip_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting so a TTS engine doesn't read symbols aloud."""
    if not text:
        return text
    # Emoji: neither Kokoro nor Piper should try to phonemise these.
    text = _EMOJI.sub("", text)
    # Fenced code blocks -> contents only.
    text = _FENCED_CODE.sub(lambda m: m.group(1), text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _MARKDOWN_INLINE_CODE.sub(r"\1", text)
    text = _MARKDOWN_BOLD.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _MARKDOWN_ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _MARKDOWN_STRIKE.sub(r"\1", text)
    text = _MARKDOWN_HEADING.sub("", text)
    text = _MARKDOWN_BULLET.sub("", text)
    text = _MARKDOWN_NUMBERED.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()

try:
    import pyaudio  # type: ignore
    _PYAUDIO_AVAILABLE = True
    _PYAUDIO_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    pyaudio = None  # type: ignore
    _PYAUDIO_AVAILABLE = False
    _PYAUDIO_IMPORT_ERROR = repr(_exc)

try:
    from kokoro import KPipeline as _KokoroPipeline  # type: ignore
    _KOKORO_AVAILABLE = True
    _KOKORO_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    _KokoroPipeline = None  # type: ignore
    _KOKORO_AVAILABLE = False
    _KOKORO_IMPORT_ERROR = repr(_exc)

try:
    import pyttsx3 as _pyttsx3  # type: ignore
except Exception:
    _pyttsx3 = None  # type: ignore


class KokoroTTS:
    """Synthesise and play speech with Kokoro, falling back to pyttsx3.

    The class is thread-safe via an internal lock. ``stop()`` requests early
    abort during the next chunk boundary; in-flight audio in the playback
    buffer will still finish playing.
    """

    def __init__(
        self,
        voice: str = "bf_emma",
        volume_gain: float = 1.6,
        lang_code: str = "a",
        repo_id: str = "hexgrad/Kokoro-82M",
        sample_rate: int = 24000,
        logger: Any = None,
        force_pyttsx3: bool = False,
    ) -> None:
        self._voice = str(voice)
        self._volume_gain = float(volume_gain)
        self._sample_rate = int(sample_rate)
        self._logger = logger

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._kokoro = None
        self._pa = None
        self._fallback = None

        # Piper neural German voice (Kokoro is English-only; espeak German
        # sounds robotic — Majd wants both languages neural, 2026-08-11).
        # Loaded lazily on the first German utterance (~1.6 s once).
        self._piper_de = None
        self._piper_de_tried = False
        self._piper_de_rate = int(sample_rate)
        # Default is the MEDIUM voice: with the full stack live (YOLO,
        # camera, chat) the high model synthesises at ~1× realtime, adding
        # seconds before Morty starts talking; medium is ~3× faster and
        # still neural. Set MORTY_PIPER_DE_MODEL to the -high model for
        # demo-quality speech when latency doesn't matter.
        self._piper_de_model = os.environ.get(
            "MORTY_PIPER_DE_MODEL",
            os.path.expanduser("~/morty_ws/models/piper/de_DE-thorsten-medium.onnx"),
        )
        self._piper_de_load_lock = threading.Lock()
        # Preload in the background so the FIRST German reply doesn't pay
        # the ~1.6 s model load on top of LLM latency (Majd: "takes too
        # long"). _ensure_piper_de is idempotent via _piper_de_tried.
        threading.Thread(target=self._ensure_piper_de, daemon=True).start()

        if force_pyttsx3:
            self._log("info", "force_pyttsx3=true — skipping Kokoro init, using pyttsx3 only.")
        else:
            if not _KOKORO_AVAILABLE:
                self._log("warn", f"kokoro import unavailable: {_KOKORO_IMPORT_ERROR}")
            if not _PYAUDIO_AVAILABLE:
                self._log("warn", f"pyaudio import unavailable: {_PYAUDIO_IMPORT_ERROR}")

            if _KOKORO_AVAILABLE and _PYAUDIO_AVAILABLE:
                try:
                    self._kokoro = _KokoroPipeline(lang_code=lang_code, repo_id=repo_id)
                    self._pa = pyaudio.PyAudio()
                    self._log("info", f"Kokoro TTS initialised (voice='{self._voice}').")
                except Exception as exc:
                    self._kokoro = None
                    self._log("warn", f"Kokoro init failed: {exc}; will try pyttsx3.")

        # Always init the pyttsx3 fallback when available — even when Kokoro
        # is up — because Kokoro is English-only and we route German text
        # through pyttsx3 (which uses espeak-ng under the hood and supports
        # German pronunciation).
        if _pyttsx3 is not None:
            try:
                self._fallback = _pyttsx3.init()
                try:
                    rate = self._fallback.getProperty("rate")
                    self._fallback.setProperty("rate", int(rate * 0.9))
                except Exception:
                    pass
                self._fallback_default_voice_id: Optional[str] = None
                self._fallback_german_voice_id: Optional[str] = None
                try:
                    self._fallback_default_voice_id = (
                        self._fallback.getProperty("voice")
                    )
                    for v in self._fallback.getProperty("voices") or []:
                        langs = [
                            (l.decode() if isinstance(l, (bytes, bytearray)) else str(l)).lower()
                            for l in (getattr(v, "languages", []) or [])
                        ]
                        name = (getattr(v, "name", "") or "").lower()
                        vid = getattr(v, "id", "") or ""
                        if "de" in langs or "german" in name or "/de" in vid:
                            self._fallback_german_voice_id = vid
                            break
                except Exception:
                    pass
                self._log(
                    "info",
                    f"pyttsx3 TTS fallback initialised "
                    f"(german_voice={self._fallback_german_voice_id or 'none'}).",
                )
            except Exception as exc:
                self._fallback = None
                self._log("error", f"pyttsx3 fallback failed: {exc}")

    # ── public API ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._kokoro is not None or self._fallback is not None

    def set_voice(self, voice: str) -> None:
        self._voice = str(voice)

    def set_volume_gain(self, gain: float) -> None:
        self._volume_gain = float(gain)

    def stop(self) -> None:
        """Request early termination of any in-flight speak()."""
        self._stop_event.set()

    def speak_blocking(
        self,
        text: str,
        on_phase: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Synthesise and play ``text``. Returns True on success.

        ``on_phase`` is called with "synthesizing", "playing", "finished" so
        an action server can surface phase feedback without coupling to TTS
        internals.
        """
        if not text:
            return True
        text = strip_markdown_for_tts(text)
        if not text:
            return True
        # Language routing: German replies go to the Piper neural German
        # voice (espeak via _fallback_say only if Piper is unavailable);
        # everything else goes to Kokoro. The chat layer answers in the
        # user's language since 2026-08-11, so both paths are live.
        german = looks_german(text)
        with self._lock:
            self._stop_event.clear()
            try:
                if german and self._pa is not None:
                    if on_phase:
                        on_phase("synthesizing")
                    wav = self._synth_piper_de(text)
                    if wav is not None:
                        if self._stop_event.is_set():
                            if on_phase:
                                on_phase("finished")
                            return True
                        if on_phase:
                            on_phase("playing")
                        self._play_kokoro(wav, rate=self._piper_de_rate)
                        if on_phase:
                            on_phase("finished")
                        return True
                    if self._stop_event.is_set():
                        if on_phase:
                            on_phase("finished")
                        return True
                    return self._fallback_say(text, on_phase, german=True)
                if self._kokoro is not None:
                    if on_phase:
                        on_phase("synthesizing")
                    wav = self._synth_kokoro(text)
                    if wav is None:
                        if self._stop_event.is_set():
                            # Interrupted mid-synthesis by a newer Speak
                            # goal — do NOT re-speak this text through the
                            # pyttsx3 voice (Majd heard exactly that).
                            if on_phase:
                                on_phase("finished")
                            return True
                        return self._fallback_say(text, on_phase)
                    if self._stop_event.is_set():
                        if on_phase:
                            on_phase("finished")
                        return True
                    if on_phase:
                        on_phase("playing")
                    self._play_kokoro(wav)
                    if on_phase:
                        on_phase("finished")
                    return True
                return self._fallback_say(text, on_phase)
            except Exception as exc:
                self._log("warn", f"TTS failed: {exc}")
                return False

    def close(self) -> None:
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    # ── internals ────────────────────────────────────────────────────────

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        if level == "info":
            self._logger.info(msg)
        elif level == "warn":
            self._logger.warning(msg)
        elif level == "error":
            self._logger.error(msg)
        else:
            self._logger.info(msg)

    def _synth_kokoro(self, text: str) -> Optional[np.ndarray]:
        if self._kokoro is None:
            return None
        try:
            chunks: List[np.ndarray] = []
            for _, _, audio in self._kokoro(str(text), voice=self._voice):
                if self._stop_event.is_set():
                    break
                if audio is None:
                    continue
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                chunks.append(np.asarray(audio, dtype=np.float32))
            if not chunks:
                return None
            wav = np.concatenate(chunks) * self._volume_gain
            np.clip(wav, -1.0, 1.0, out=wav)
            return wav
        except Exception as exc:
            self._log("warn", f"Kokoro synth failed: {exc}")
            return None

    def _ensure_piper_de(self) -> None:
        """One-time load of the Piper German voice (preloaded in the
        background at init; the lock makes a German utterance arriving
        mid-load wait ~1 s for Piper instead of dropping to espeak)."""
        with self._piper_de_load_lock:
            self._ensure_piper_de_locked()

    def _ensure_piper_de_locked(self) -> None:
        if self._piper_de_tried:
            return
        self._piper_de_tried = True
        self._piper_de_rate = self._sample_rate
        try:
            from piper import PiperVoice
        except Exception as exc:
            self._log("warn", f"piper-tts not importable ({exc}); German uses espeak.")
            return
        model = self._piper_de_model
        if not os.path.isfile(model) or not os.path.isfile(model + ".json"):
            self._log(
                "warn",
                f"Piper German model missing at {model}(.json); German uses espeak.",
            )
            return
        try:
            self._piper_de = PiperVoice.load(model, model + ".json")
            self._piper_de_rate = int(self._piper_de.config.sample_rate)
            self._log(
                "info",
                f"Piper German voice loaded ({os.path.basename(model)}, "
                f"{self._piper_de_rate} Hz).",
            )
        except Exception as exc:
            self._piper_de = None
            self._log("warn", f"Piper German load failed: {exc}; German uses espeak.")

    def _synth_piper_de(self, text: str) -> Optional[np.ndarray]:
        """Synthesise German text with Piper; float32 [-1,1] or None."""
        self._ensure_piper_de()
        if self._piper_de is None:
            return None
        try:
            raw = b"".join(self._piper_de.synthesize_stream_raw(text))
            if not raw:
                return None
            wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            # Piper already uses the full range — apply the gain but
            # renormalise instead of hard-clipping (which distorts audibly).
            wav = wav * self._volume_gain
            peak = float(np.abs(wav).max() or 0.0)
            if peak > 0.95:
                wav *= 0.95 / peak
            # Piper ends the moment the last phoneme does — append real
            # trailing silence so the buffered tail is never speech
            # (German "eats last words", 2026-08-11).
            tail = np.zeros(int(self._piper_de_rate * 0.3), dtype=np.float32)
            return np.concatenate([wav, tail])
        except Exception as exc:
            self._log("warn", f"Piper German synth failed: {exc}")
            return None

    def _play_kokoro(self, wav: np.ndarray, rate: Optional[int] = None) -> None:
        if self._pa is None or wav is None or wav.size == 0:
            return
        rate = int(rate or self._sample_rate)
        pcm16 = (wav * 32767.0).astype(np.int16)
        # A generous device buffer (~400 ms) gives the audio backend slack so
        # it doesn't underrun between our chunk writes. Without this, WSLg's
        # RDP-bridged PulseAudio sink drains mid-utterance and crackles.
        frames_per_buffer = max(1024, int(rate * 0.4))
        stream = self._pa.open(
            format=pyaudio.paInt16,  # type: ignore[attr-defined]
            channels=1,
            rate=rate,
            output=True,
            frames_per_buffer=frames_per_buffer,
        )
        # Stream the audio in chunks so a mid-utterance stop() (interrupt
        # request from a new Speak goal) actually halts playback instead of
        # waiting for the whole buffer to drain. 200 ms windows stay responsive
        # while feeding the device buffer fast enough to avoid underruns.
        chunk_samples = max(256, int(rate * 0.2))
        try:
            for i in range(0, pcm16.size, chunk_samples):
                if self._stop_event.is_set():
                    break
                stream.write(pcm16[i : i + chunk_samples].tobytes())
            if not self._stop_event.is_set():
                # Pad the tail with silence before stopping. stop_stream()
                # through the Pulse backend drops whatever is still queued in
                # the ~400 ms device buffer — without padding that's the last
                # word of the utterance (2026-08-11 'eaten last word' bug).
                # 2× the device buffer: at 0.45 s (≈1× buffer) the Pulse
                # backend still clipped Piper's tail (2026-08-11).
                pad = np.zeros(int(rate * 0.85), dtype=np.int16)
                stream.write(pad.tobytes())
        finally:
            try:
                # If we bailed early, drop the queue rather than waiting
                # for it to finish playing.
                if self._stop_event.is_set():
                    stream.stop_stream()
                else:
                    stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _fallback_say(
        self,
        text: str,
        on_phase: Optional[Callable[[str], None]],
        german: bool = False,
    ) -> bool:
        if self._fallback is None:
            self._log("warn", "No TTS backend available.")
            return False
        # Explicit log so the operator can tell when the OLD pyttsx3 voice
        # is being used (e.g. after Kokoro synth returned None mid-session).
        self._log(
            "warn",
            f"TTS fallback (pyttsx3) used for utterance: {text[:60]!r}",
        )
        # Switch the engine voice to German for German text and back to the
        # default afterwards. Doing it inside the lock-protected speak path
        # so concurrent calls don't race on the engine voice.
        target_voice: Optional[str] = None
        if german and getattr(self, "_fallback_german_voice_id", None):
            target_voice = self._fallback_german_voice_id
        elif not german and getattr(self, "_fallback_default_voice_id", None):
            target_voice = self._fallback_default_voice_id
        if target_voice:
            try:
                self._fallback.setProperty("voice", target_voice)
            except Exception:
                pass
        if on_phase:
            on_phase("playing")
        self._fallback.say(text)
        self._fallback.runAndWait()
        if on_phase:
            on_phase("finished")
        return True
