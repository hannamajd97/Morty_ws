"""Continuous offline speech-to-text using Vosk + pyaudio.

Wraps a single mic stream with utterance segmentation (KaldiRecognizer's
built-in VAD via AcceptWaveform) and emits each recognized utterance
through a callback. A ``mute()`` context blocks recognition while audio
output (TTS) is active, preventing self-feedback loops.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

# Non-Latin scripts (Cyrillic, Arabic, CJK, Hangul, Greek, Hebrew): their
# presence in a transcript is hard evidence of Whisper hallucinating on
# noise in an EN/DE lab — used by the language gate in _transcribe().
_NON_LATIN_RE = re.compile(
    "[Ͱ-ϿЀ-ӿ֐-׿؀-ۿ"
    "　-ヿ一-鿿가-힯]"
)

try:
    import pyaudio  # type: ignore
    _PYAUDIO_AVAILABLE = True
except Exception as _exc:
    pyaudio = None  # type: ignore
    _PYAUDIO_AVAILABLE = False
    _PYAUDIO_ERR = repr(_exc)
else:
    _PYAUDIO_ERR = None


def _compute_rms_normalised(pcm16_bytes: bytes) -> float:
    """Return a 0..1 normalised RMS amplitude for a PCM-16 mono chunk.

    The mic visualiser uses this to draw a live level bar so the user
    can confirm the STT is hearing them. Returns 0.0 on empty or
    malformed input.
    """
    if not pcm16_bytes:
        return 0.0
    try:
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        # int16 max is 32767. Soft-cap at half-scale so normal speech
        # fills most of the bar without saturating.
        return max(0.0, min(1.0, rms / 16384.0))
    except Exception:
        return 0.0


def _compute_peak_envelope(
    pcm16_bytes: bytes, n_bins: int = 8
) -> np.ndarray:
    """Return ``n_bins`` peak-amplitudes (signed, normalised to ±1) for a
    PCM-16 mono chunk. The mic visualiser appends these to a ring buffer
    and renders them as mirrored vertical bars so the user can SEE the
    audio waveform, not just a slow RMS line.
    """
    if not pcm16_bytes:
        return np.zeros(n_bins, dtype=np.float32)
    try:
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if samples.size == 0:
            return np.zeros(n_bins, dtype=np.float32)
        # Split into n_bins equal-ish slices and take the signed peak
        # (max absolute value, preserving sign) for each.
        bins = np.array_split(samples, n_bins)
        out = np.empty(n_bins, dtype=np.float32)
        for i, b in enumerate(bins):
            if b.size == 0:
                out[i] = 0.0
                continue
            idx = int(np.argmax(np.abs(b)))
            out[i] = float(b[idx]) / 32768.0
        return out
    except Exception:
        return np.zeros(n_bins, dtype=np.float32)

try:
    from vosk import Model as _VoskModel, KaldiRecognizer as _KaldiRecognizer  # type: ignore
    _VOSK_AVAILABLE = True
except Exception as _exc:
    _VoskModel = None  # type: ignore
    _KaldiRecognizer = None  # type: ignore
    _VOSK_AVAILABLE = False
    _VOSK_ERR = repr(_exc)
else:
    _VOSK_ERR = None

try:
    import webrtcvad  # type: ignore
    _VAD_AVAILABLE = True
except Exception:
    webrtcvad = None  # type: ignore
    _VAD_AVAILABLE = False

try:
    from faster_whisper import WhisperModel  # type: ignore
    _WHISPER_AVAILABLE = True
    _WHISPER_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    WhisperModel = None  # type: ignore
    _WHISPER_AVAILABLE = False
    _WHISPER_IMPORT_ERROR = repr(_exc)


class VoskTranscriber:
    """Continuously transcribe microphone input.

    Call :meth:`start` to begin a background thread. ``on_transcript`` is
    invoked once per recognized utterance with the cleaned text.
    Use :meth:`set_muted` to disable recognition while audio is playing.
    """

    def __init__(
        self,
        model_dir: str,
        on_transcript: Callable[[str], None],
        sample_rate: int = 16000,
        frames_per_buffer: int = 4000,
        mic_index: Optional[int] = None,
        mic_device_name: Optional[str] = None,
        vad_enabled: bool = False,
        vad_aggressiveness: int = 2,
        on_audio_level: Optional[Callable[[float], None]] = None,
        on_state: Optional[Callable[[str], None]] = None,
        on_audio_waveform: Optional[Callable[[np.ndarray], None]] = None,
        logger: Any = None,
    ) -> None:
        if not _VOSK_AVAILABLE:
            raise RuntimeError(f"vosk unavailable: {_VOSK_ERR}")
        if not _PYAUDIO_AVAILABLE:
            raise RuntimeError(f"pyaudio unavailable: {_PYAUDIO_ERR}")

        self._logger = logger
        # Per-chunk peak-normalised RMS (0..1) and listener-state callbacks.
        # Used by mic_visualizer_node to draw a live level bar so the user
        # can see that the STT is actually hearing them.
        self._on_audio_level = on_audio_level
        self._on_state = on_state
        # Downsampled peak envelope (signed, normalised) for the waveform.
        self._on_audio_waveform = on_audio_waveform
        self._model = _VoskModel(model_dir)
        self._sample_rate = int(sample_rate)
        self._frames_per_buffer = int(frames_per_buffer)
        # If a name is given, resolve it to an index (case-insensitive
        # substring match against pyaudio device names). Wins over mic_index.
        if mic_device_name:
            resolved = self._resolve_device_index(mic_device_name)
            if resolved is None:
                msg = (
                    f"mic_device_name {mic_device_name!r} not found; "
                    "falling back to mic_index / system default."
                )
                if logger is not None:
                    logger.warning(msg)
                self._mic_index = (
                    mic_index if (mic_index is None or mic_index >= 0) else None
                )
            else:
                self._mic_index = resolved
                if logger is not None:
                    logger.info(
                        f"Selected mic device {mic_device_name!r} -> index {resolved}"
                    )
        else:
            self._mic_index = (
                mic_index if (mic_index is None or mic_index >= 0) else None
            )
        self._on_transcript = on_transcript

        # WebRTC VAD: optional gate that skips Vosk when frame is silence.
        self._vad = None
        self._vad_frame_size_samples = 0
        if vad_enabled:
            if not _VAD_AVAILABLE:
                if logger is not None:
                    logger.warning(
                        "vad_enabled=True but webrtcvad not installed; "
                        "running without VAD gate."
                    )
            else:
                # webrtcvad supports 8/16/32 kHz and 10/20/30 ms frames only.
                if sample_rate not in (8000, 16000, 32000, 48000):
                    if logger is not None:
                        logger.warning(
                            f"VAD requires sample_rate in {{8k,16k,32k,48k}}; "
                            f"got {sample_rate}. Disabling VAD."
                        )
                else:
                    self._vad = webrtcvad.Vad(int(vad_aggressiveness))
                    # Use 30ms frames at the configured sample rate for VAD.
                    self._vad_frame_size_samples = int(sample_rate * 0.03)
                    if logger is not None:
                        logger.info(
                            f"VAD enabled (aggressiveness={vad_aggressiveness}, "
                            f"30ms frames @ {sample_rate} Hz)."
                        )

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._muted = threading.Event()

    @staticmethod
    def _resolve_device_index(name_substring: str) -> Optional[int]:
        """Find a pyaudio input device whose name contains name_substring."""
        try:
            pa = pyaudio.PyAudio()
            try:
                target = name_substring.lower()
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if int(info.get("maxInputChannels", 0)) <= 0:
                        continue
                    if target in str(info.get("name", "")).lower():
                        return i
            finally:
                pa.terminate()
        except Exception:
            pass
        return None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_muted(self, muted: bool) -> None:
        if muted:
            self._muted.set()
        else:
            self._muted.clear()

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        if level == "info":
            self._logger.info(msg)
        elif level == "warn":
            self._logger.warning(msg)
        elif level == "error":
            self._logger.error(msg)

    def _loop(self) -> None:
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._frames_per_buffer,
                input_device_index=self._mic_index,
            )
            stream.start_stream()
            rec = _KaldiRecognizer(self._model, self._sample_rate)
            self._log(
                "info",
                f"VoskTranscriber listening (rate={self._sample_rate}, "
                f"mic_index={self._mic_index})",
            )

            last_state = ""
            while not self._stop.is_set():
                data = stream.read(
                    self._frames_per_buffer, exception_on_overflow=False
                )
                # Emit live audio level + downsampled peak envelope for
                # the mic visualiser even when we're muted — the user
                # can still see the mic itself is alive (just gated).
                if self._on_audio_level is not None:
                    try:
                        self._on_audio_level(_compute_rms_normalised(data))
                    except Exception:
                        pass
                if self._on_audio_waveform is not None:
                    try:
                        self._on_audio_waveform(_compute_peak_envelope(data))
                    except Exception:
                        pass
                if self._muted.is_set():
                    # Discard audio captured during TTS playback so the AI
                    # doesn't recognize its own voice.
                    rec.Reset()
                    if last_state != "muted":
                        last_state = "muted"
                        if self._on_state is not None:
                            try:
                                self._on_state("muted")
                            except Exception:
                                pass
                    continue
                # Optional VAD gate: only feed Vosk frames that contain speech.
                # Saves CPU and reduces nonsense recognitions from background
                # noise. WebRTC VAD wants 10/20/30 ms PCM-16 frames; we slice
                # the buffer into 30 ms windows and check each.
                voice_now = self._has_voice(data) if self._vad is not None else True
                new_state = "speech" if voice_now else "idle"
                if new_state != last_state:
                    last_state = new_state
                    if self._on_state is not None:
                        try:
                            self._on_state(new_state)
                        except Exception:
                            pass
                if self._vad is not None and voice_now is False:
                    rec.Reset()
                    continue
                if rec.AcceptWaveform(data):
                    text = self._extract_text(rec.Result())
                    if text:
                        try:
                            self._on_transcript(text)
                        except Exception as exc:
                            self._log("error", f"on_transcript callback failed: {exc}")
        except Exception as exc:
            self._log("error", f"Vosk loop error: {exc}")
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass

    def _has_voice(self, pcm16_bytes: bytes) -> bool:
        """Return True if any 30ms sub-frame contains speech."""
        if self._vad is None or not pcm16_bytes:
            return True
        # 2 bytes per sample (int16, mono)
        bytes_per_frame = self._vad_frame_size_samples * 2
        if bytes_per_frame <= 0 or len(pcm16_bytes) < bytes_per_frame:
            return True  # too small to evaluate; let it through
        for offset in range(0, len(pcm16_bytes) - bytes_per_frame + 1, bytes_per_frame):
            chunk = pcm16_bytes[offset:offset + bytes_per_frame]
            try:
                if self._vad.is_speech(chunk, self._sample_rate):
                    return True
            except Exception:
                return True  # on any error, fall back to letting it through
        return False

    @staticmethod
    def _extract_text(result_json: str) -> str:
        try:
            return str(json.loads(result_json).get("text", "")).strip()
        except Exception:
            return ""


class WhisperTranscriber:
    """Continuously transcribe with faster-whisper.

    Whisper is not a streaming recogniser — it consumes complete audio
    chunks. So we use WebRTC VAD to mark speech onset / offset, buffer the
    PCM between them, and feed each utterance to Whisper as it finishes.

    Same external interface as VoskTranscriber: start() / stop() /
    set_muted(); on_transcript receives one cleaned string per utterance.
    """

    def __init__(
        self,
        model_size: str,
        on_transcript: Callable[[str], None],
        sample_rate: int = 16000,
        frames_per_buffer: int = 1600,  # 100 ms chunks
        mic_index: Optional[int] = None,
        mic_device_name: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
        vad_aggressiveness: int = 2,
        silence_ms_to_finalise: int = 600,
        max_utterance_s: float = 12.0,
        min_utterance_ms: int = 250,
        min_rms: float = 0.012,
        initial_prompt: str = "",
        on_audio_level: Optional[Callable[[float], None]] = None,
        on_state: Optional[Callable[[str], None]] = None,
        on_audio_waveform: Optional[Callable[[np.ndarray], None]] = None,
        on_reject: Optional[Callable[[str, str], None]] = None,
        word_time_cut: bool = True,
        tts_lead_s: float = 0.15,
        tts_tail_s: float = 0.25,
        logger: Any = None,
        engine: str = "whisper",
        parakeet_model: str = "nemo-parakeet-tdt-0.6b-v3",
        vad_engine: str = "webrtc",
        silero_onset: float = 0.5,
        silero_offset: float = 0.35,
        speaker_gate: bool = False,
        morty_voice_path: str = "",
        speaker_gate_threshold: float = 0.35,
        on_barge_in: Optional[Callable[[float], None]] = None,
        barge_in_min_ms: int = 600,
    ) -> None:
        self._on_reject = on_reject
        # ---- SPEAKER GATE + BARGE-IN (2026-09-23) -------------------
        # Morty's voices are fixed (Kokoro bf_emma, Piper de_DE-thorsten);
        # an ECAPA-TDNN fingerprint of them (models/morty_voice_ecapa.npy,
        # built from trial recordings in this room) tells his echo from
        # ANY human: on 19 trials Morty scored >= 0.50 against it, every
        # human segment <= 0.10, 12 ms per check on the GPU. Utterances
        # that match are dropped before Whisper; a human voice heard while
        # Morty plays triggers on_barge_in (speak_node stops playback).
        self._speaker_gate = bool(speaker_gate)
        self._morty_voice_path = os.path.expanduser(morty_voice_path or "")
        self._speaker_thr = float(speaker_gate_threshold)
        self._barge_thr = min(0.25, self._speaker_thr)   # short windows of Morty's echo scored 0.14-0.49; whole sentences 0.50-0.86
        self._on_barge_in = on_barge_in
        self._barge_in_min_ms = int(barge_in_min_ms)
        self._spk_encoder = None
        self._spk_ref = None
        self._spk_failed = False
        self._last_speaker_score: Optional[float] = None
        # ---- WORD-TIME ECHO CUT (Majd 2026-08-28) ------------------
        # Whisper knows when every word was spoken; we know exactly when
        # Morty's loudspeaker was playing. Intersect the two and the
        # echo can be removed by TIME instead of guessed at by text
        # similarity. This is what makes "...Left, right or home? Drop
        # it home." recoverable: the word "home" appears twice, 1.3 s
        # apart, and no text-similarity test can ever separate those two
        # -- but the clock separates them trivially.
        self._word_time_cut = bool(word_time_cut)
        self._tts_lead_s = float(tts_lead_s)
        self._tts_tail_s = float(tts_tail_s)
        # [start, end] pairs in time.time() units; end is None while a
        # sentence is still playing. _ai_depth ref-counts overlapping
        # Speak goals (the server queues them) so nested starts/stops
        # collapse into ONE union window instead of tearing.
        self._ai_windows: list = []
        self._ai_texts: list = []      # (wall time, token set) of recent playback sentences
        self._ai_depth = 0
        self._ai_lock = threading.Lock()
        if not _WHISPER_AVAILABLE:
            raise RuntimeError(f"faster-whisper unavailable: {_WHISPER_IMPORT_ERROR}")
        if not _PYAUDIO_AVAILABLE:
            raise RuntimeError(f"pyaudio unavailable: {_PYAUDIO_ERR}")
        if not _VAD_AVAILABLE:
            raise RuntimeError(
                "WhisperTranscriber requires webrtcvad for VAD-based segmentation. "
                "pip install --user webrtcvad"
            )

        self._logger = logger
        # Live audio-level + state hooks for the mic visualiser.
        self._on_audio_level = on_audio_level
        self._on_state = on_state
        # Signed peak envelope per chunk (mirrored waveform bars).
        self._on_audio_waveform = on_audio_waveform
        self._sample_rate = int(sample_rate)
        self._frames_per_buffer = int(frames_per_buffer)
        # Resolve mic name → index, same logic as Vosk variant.
        if mic_device_name:
            resolved = VoskTranscriber._resolve_device_index(mic_device_name)
            if resolved is None:
                if logger is not None:
                    logger.warning(
                        f"mic_device_name {mic_device_name!r} not found; "
                        "falling back to mic_index / system default."
                    )
                self._mic_index = mic_index if (mic_index is None or mic_index >= 0) else None
            else:
                self._mic_index = resolved
                if logger is not None:
                    logger.info(
                        f"Selected mic device {mic_device_name!r} -> index {resolved}"
                    )
        else:
            self._mic_index = mic_index if (mic_index is None or mic_index >= 0) else None

        self._on_transcript = on_transcript
        self._language = language
        self._silence_ms = int(silence_ms_to_finalise)
        self._max_utterance_s = float(max_utterance_s)
        self._min_utterance_ms = int(min_utterance_ms)
        self._min_rms = float(min_rms)
        self._initial_prompt = initial_prompt or None

        self._vad = webrtcvad.Vad(int(vad_aggressiveness))
        # ---- VAD ENGINE (2026-09-23, Majd approved Silero) ------------
        # "silero": the neural detector bundled with faster-whisper
        # (assets/silero_vad.onnx, CPU, ~1 ms per 100 ms chunk), fed each
        # 100 ms capture chunk whole with carried LSTM state; onset/offset
        # hysteresis instead of WebRTC's per-frame yes/no. Fewer segments
        # from motor and room noise, softer word endings kept. The loop's
        # speech_ms / silence_ms bookkeeping is untouched — the barge-in
        # check depends on it. "webrtc" stays for A/B.
        self._vad_engine = str(vad_engine or "webrtc").lower()
        self._silero = None
        self._silero_state = None
        self._silero_on = float(silero_onset)
        self._silero_off = float(silero_offset)
        self._silero_speaking = False
        if self._vad_engine == "silero":
            try:
                from faster_whisper.vad import get_vad_model
                self._silero = get_vad_model()
                self._silero_state = self._silero.get_initial_state(batch_size=1)
                self._log("info", f"Silero VAD ready (onset {self._silero_on:.2f}, offset {self._silero_off:.2f}).")
            except Exception as exc:
                self._log("warn", f"Silero VAD unavailable ({exc}); using WebRTC VAD.")
                self._vad_engine = "webrtc"
        # 30 ms frames at the configured rate (WebRTC VAD only accepts 10/20/30 ms).
        self._vad_frame_size_samples = int(sample_rate * 0.03)

        if logger is not None:
            logger.info(
                f"Loading faster-whisper model {model_size!r} "
                f"(device={device}, compute_type={compute_type}). "
                "First load may take 30-60s while the model downloads."
            )
        # ---- ENGINE (2026-09-23) --------------------------------------
        # "parakeet": NVIDIA parakeet-tdt-0.6b-v3 through onnx-asr on the
        # GPU — 25 European languages with per-utterance detection, no
        # prompt, no word times. Benchmarked on 9 trial clips of 2–8 s
        # against Whisper large-v3-turbo int8 (both on the GTX 1080, one at
        # a time, warmed up): 52 ms vs 641 ms per utterance, texts equal,
        # German perfect, and it returns '' where Whisper hallucinated
        # ("Thank you.", "Hold on."). cudnn_conv_algo_search=DEFAULT is
        # load-bearing: the default search re-runs for every new clip
        # length and cost 5–9 s per utterance in the first benchmark.
        self._pk = None
        self._model = None
        self._engine = str(engine or "whisper").lower()
        if self._engine == "parakeet":
            import onnx_asr
            providers = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
                         if device in ("auto", "cuda") else []) + ["CPUExecutionProvider"]
            self._pk = onnx_asr.load_model(parakeet_model, providers=providers)
            for secs in (2, 4):                      # warm the two common shapes
                self._pk.recognize((np.random.randn(16000 * secs) * 0.003).astype(np.float32))
            self._log("info", f"Parakeet STT ready ({parakeet_model}, providers {[p if isinstance(p, str) else p[0] for p in providers]}).")
        else:
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        if self._speaker_gate:
            self._speaker_ready()      # load now, not on the first utterance (1.3 s)
        if logger is not None:
            # Faster-whisper resolves "auto" to "cuda" if it can load
            # libcublas + libcudnn; otherwise it silently falls back to
            # "cpu". Reading the resolved device off the CTranslate2
            # model so the operator knows which it got.
            resolved_device = "?"
            try:
                resolved_device = self._model.model.device
            except Exception:
                pass
            logger.info(
                f"faster-whisper model loaded on device={resolved_device!r}."
            )

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._muted = threading.Event()

        # Whisper inference runs on a SEPARATE worker thread so it never
        # blocks the audio-capture loop. Without this, every transcribe()
        # call freezes the mic for 1-3 s, samples pile up in the hardware
        # buffer, and the NEXT VAD segment starts with stale audio mixed
        # with the new speech → Whisper hallucinates ("Masses, flowers...").
        # The queue has maxsize=1 so we drop old utterances if the worker
        # is still busy with a previous one — better to lose one than to
        # stack delayed/corrupted replies.
        self._utterance_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self._worker_thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        # Worker first so it's ready when the audio loop starts queuing.
        self._worker_thread = threading.Thread(
            target=self._transcribe_worker,
            name="whisper_worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Unblock the worker if it's waiting on the queue.
        try:
            self._utterance_q.put_nowait((b"", 0.0))
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def _reject(self, reason: str, text: str) -> None:
        """Report a dropped utterance so the operator can SEE why nothing
        happened (Majd, 2026-08-12: 'it doesn't show me why!')."""
        if self._on_reject is None:
            return
        try:
            self._on_reject(reason, text)
        except Exception:
            pass

    def _transcribe_worker(self) -> None:
        """Drain the utterance queue, run Whisper, emit transcripts.

        Runs on a separate thread so the audio-capture loop is never
        blocked by Whisper inference. Inference takes 0.5-2 s on CPU for
        base.en; while it runs, the audio loop keeps reading samples,
        and the queue (maxsize=1) silently drops new utterances if they
        arrive faster than we can transcribe them.
        """
        while not self._stop.is_set():
            try:
                buf, utt_start_wall, skip_echo_cut = self._utterance_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not buf or self._stop.is_set():
                continue
            try:
                text = self._transcribe(buf, utt_start_wall, skip_echo_cut)
            except Exception as exc:
                self._log("error", f"whisper worker transcribe failed: {exc}")
                continue
            if not text:
                continue
            try:
                self._on_transcript(text)
            except Exception as exc:
                self._log(
                    "error", f"on_transcript callback failed: {exc}"
                )

    def set_muted(self, muted: bool) -> None:
        if muted:
            self._muted.set()
        else:
            self._muted.clear()

    # ── live tuning (2026-08-28) ─────────────────────────────────────────
    # The GUI's "Voice filters" tab used to need a voice restart because
    # these three numbers were constructor-only. They are read AT USE TIME
    # by the audio thread (_transcribe / _loop), so rebinding them on the
    # live object takes effect on the very next chunk. Public setters
    # instead of the node poking privates.
    #
    # Thread-safety: each setter rebinds ONE scalar attribute. CPython
    # attribute assignment is a single bytecode (STORE_ATTR) under the
    # GIL, and the audio thread only ever READS these, so a reader sees
    # either the old or the new value — never a torn one. No lock needed
    # (and no lock wanted: the audio loop must not block on the ROS
    # executor thread).

    def set_min_rms(self, value: float) -> float:
        """Set the energy gate (0..1 normalised RMS). Returns the old value."""
        old = float(self._min_rms)
        self._min_rms = float(value)
        return old

    def set_silence_ms(self, value: int) -> int:
        """Set the end-of-utterance silence in ms. Returns the old value."""
        old = int(self._silence_ms)
        self._silence_ms = int(value)
        return old

    def set_max_utterance_s(self, value: float) -> float:
        """Set the utterance length cap in seconds. Returns the old value."""
        old = float(self._max_utterance_s)
        self._max_utterance_s = float(value)
        return old

    # ── TTS playback windows for the word-time echo cut ──────────────
    def note_ai_speaking(self, speaking: bool) -> None:
        """Called by the node on every /speech/ai_speaking transition.

        Ref-counted: the Speak server QUEUES sentences, so two goals can
        be outstanding at once. Counting them means overlapping
        start/stop pairs collapse into a single union window rather than
        closing the window while Morty is still talking.
        """
        now = time.time()
        with self._ai_lock:
            if speaking:
                self._ai_depth += 1
                if self._ai_depth == 1:
                    self._ai_windows.append([now, None])
            else:
                self._ai_depth = max(0, self._ai_depth - 1)
                if self._ai_depth == 0:
                    for w in reversed(self._ai_windows):
                        if w[1] is None:
                            w[1] = now
                            break
            # Keep only the last minute; utterances are <= 12 s.
            cutoff = now - 60.0
            self._ai_windows = [
                w for w in self._ai_windows
                if w[1] is None or w[1] >= cutoff
            ]

    def note_ai_text(self, text: str) -> None:
        """Remember the sentence the speaker is about to play (from
        /speech/ai_said) so the word-time cut can tell Morty's echo from
        the operator talking over him (2026-09-23: the cut removed every
        word inside a playback window, including "Yes, go back home" and
        "open the gripper" said while Morty was still finishing)."""
        toks = self._echo_tokens(text)
        if not toks:
            return
        now = time.time()
        with self._ai_lock:
            self._ai_texts.append((now, toks))
            self._ai_texts = [(t, k) for t, k in self._ai_texts if now - t < 90.0]

    @staticmethod
    def _echo_tokens(text: str) -> set:
        return {w for w in re.findall(r"[a-zäöüß']+", (text or "").lower()) if len(w) >= 3}

    def _looks_like_my_echo(self, words: list) -> bool:
        """True when most of ``words`` (heard during playback) occur in what
        Morty said recently. Whisper mis-hears its own voice ("grass" for
        "grasp"), so a token also matches on its first four letters."""
        toks = self._echo_tokens(" ".join(words))
        if not toks:
            return True
        with self._ai_lock:
            said = set().union(*[k for _, k in self._ai_texts]) if self._ai_texts else set()
        if not said:
            return True     # nothing known about the playback: keep the old behaviour
        heads = {w[:4] for w in said}
        hits = sum(1 for w in toks if w in said or w[:4] in heads)
        return hits >= 0.5 * len(toks)

    # ── speaker gate ─────────────────────────────────────────────────
    def _speaker_ready(self) -> bool:
        if self._spk_encoder is not None:
            return True
        if not self._speaker_gate or self._spk_failed:
            return False
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            ref = np.load(self._morty_voice_path).astype(np.float32)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.expanduser("~/morty_ws/models/spkrec-ecapa-voxceleb"),
                run_opts={"device": dev})
            ref = np.atleast_2d(ref)                       # one row per voice (Kokoro EN, Piper DE)
            self._spk_ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
            self._spk_encoder = enc
            self._log("info", f"Speaker gate ready (ECAPA on {dev}, fingerprint {self._morty_voice_path}, "
                              f"threshold {self._speaker_thr:.2f}).")
            return True
        except Exception as exc:
            self._spk_failed = True
            self._log("warn", f"Speaker gate unavailable ({exc}); Whisper runs without it.")
            return False

    def _morty_score(self, pcm16_buffer: bytes) -> Optional[float]:
        """Cosine similarity of the buffer's voice to Morty's fingerprint,
        None when the gate is off or the clip is too short."""
        if not self._speaker_ready() or len(pcm16_buffer) < 2 * int(0.5 * self._sample_rate):
            return None
        try:
            import torch
            audio = np.frombuffer(pcm16_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            wav = torch.from_numpy(audio)
            if self._sample_rate != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(wav, self._sample_rate, 16000)
            with torch.no_grad():
                e = self._spk_encoder.encode_batch(wav[None].to(self._spk_encoder.device)).squeeze()
                e = e / (e.norm() + 1e-9)
            return float(np.max(self._spk_ref @ e.cpu().numpy()))   # best-matching voice
        except Exception as exc:
            self._log("warn", f"speaker gate scoring failed: {exc}")
            return None

    def _tts_playing(self) -> bool:
        with self._ai_lock:
            return self._ai_depth > 0

    def _barge_in_check(self, window: bytes, quiet: bool = False) -> Optional[float]:
        """Fingerprint score of the latest window (None = gate off / too
        short). The loop decides; ``quiet`` silences the Morty verdict."""
        score = self._morty_score(window)
        if score is not None and score >= self._speaker_thr and not quiet:
            self._log("info", f"[speaker] overlap so far is Morty's own voice (cos {score:.2f}); still listening")
        return score

    def set_word_time_cut(self, value: bool) -> bool:
        old = bool(self._word_time_cut)
        self._word_time_cut = bool(value)
        return old

    def set_tts_lead_s(self, value: float) -> float:
        old = float(self._tts_lead_s)
        self._tts_lead_s = float(value)
        return old

    def set_tts_tail_s(self, value: float) -> float:
        old = float(self._tts_tail_s)
        self._tts_tail_s = float(value)
        return old

    def _time_is_ai_speech(self, t_abs: float) -> bool:
        """True when absolute time ``t_abs`` falls inside a window in
        which Morty's loudspeaker was sounding.

        The window is widened by ``tts_lead_s`` at the front (the goal is
        accepted a moment before audio actually leaves the speaker, and
        the two clocks are not the same clock) and by ``tts_tail_s`` at
        the back (room reverb plus mic/driver buffering keep the echo
        alive briefly after playback ends). Both are deliberately small:
        the operator answers ~0.5 s after Morty stops, so an over-wide
        tail would start eating HIS words -- the very bug this is
        supposed to fix."""
        now = time.time()
        with self._ai_lock:
            windows = [(w[0], w[1]) for w in self._ai_windows]
        for start, end in windows:
            s = start - self._tts_lead_s
            e = (now if end is None else end) + self._tts_tail_s
            if s <= t_abs <= e:
                return True
        return False

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

    def _has_voice(self, pcm16: bytes) -> bool:
        if not pcm16:
            return False
        if getattr(self, "_silero", None) is not None:
            try:
                x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
                if x.size < 512:
                    return self._silero_speaking
                out, self._silero_state = self._silero(x, self._silero_state, self._sample_rate)
                p = float(np.asarray(out).reshape(-1)[0])
                thr = self._silero_off if self._silero_speaking else self._silero_on
                self._silero_speaking = p >= thr
                return self._silero_speaking
            except Exception as exc:
                self._log("warn", f"Silero VAD failed ({exc}); falling back to WebRTC for this chunk.")
        bytes_per_frame = self._vad_frame_size_samples * 2
        if bytes_per_frame <= 0 or len(pcm16) < bytes_per_frame:
            return False
        for offset in range(0, len(pcm16) - bytes_per_frame + 1, bytes_per_frame):
            chunk = pcm16[offset:offset + bytes_per_frame]
            try:
                if self._vad.is_speech(chunk, self._sample_rate):
                    return True
            except Exception:
                return False
        return False

    # Whisper trained on a lot of YouTube subtitles, so on near-silence /
    # low-SNR input it confidently emits closing-credits phrases. These are
    # almost always hallucinations in our command-voice context. Match on a
    # normalised form (lowercase, alnum-only) so punctuation/case doesn't
    # let one through.
    _HALLUCINATION_PHRASES = frozenset(
        s.lower() for s in (
            "the end",
            "thanks for watching",
            "thank you for watching",
            "thank you",
            "thanks",
            "bye",
            "goodbye",
            "subscribe",
            "subscribe to my channel",
            "subscribe to the channel",
            "please subscribe",
            "like and subscribe",
            "see you next time",
            "see you in the next video",
            "subtitles by the amara org community",
            "transcription by",
            "thanks for listening",
            "you",
            "hmm",
            "uh",
            "um",
            "okay",
            "ok",
        )
    )

    @classmethod
    def _looks_like_hallucination(cls, text: str) -> bool:
        import re
        norm = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
        norm = re.sub(r"\s+", " ", norm)
        return norm in cls._HALLUCINATION_PHRASES

    def _enqueue_utterance(self, pcm16_buffer: bytes,
                          utt_start_wall: float = 0.0,
                          skip_echo_cut: bool = False) -> None:
        """Hand a finished utterance to the worker thread, dropping it
        silently if the worker is still busy with the previous one.

        ``utt_start_wall`` is the wall clock (time.time()) of the FIRST
        audio sample in the buffer. Whisper reports word times relative
        to the buffer, so this is what converts them to absolute times
        comparable with the TTS playback windows."""
        if not pcm16_buffer:
            return
        try:
            self._utterance_q.put_nowait((pcm16_buffer, utt_start_wall, skip_echo_cut))
        except queue.Full:
            self._log(
                "info",
                "Whisper still busy; dropping new utterance to avoid stale-audio "
                "hallucinations. (If this happens often the model is too slow "
                "for your hardware.)",
            )

    @staticmethod
    def _post_correct(text: str) -> str:
        """Mishearings that are certain in this room: there is no grass on
        the table, both engines write it for "grasp" (bench 2026-09-23)."""
        return re.sub(r"\bgrass\b", "grasp", text, flags=re.IGNORECASE) if text else text

    def _transcribe_parakeet(self, audio: np.ndarray) -> str:
        """Parakeet path: one call, no segments, no word times."""
        try:
            _t0 = time.monotonic()
            text = str(self._pk.recognize(audio) or "").strip()
            self._log("info", f"[timing] parakeet {time.monotonic() - _t0:.2f}s for "
                              f"{audio.size / self._sample_rate:.1f}s audio")
        except Exception as exc:
            self._log("error", f"Parakeet transcribe failed: {exc}")
            return ""
        if not text:
            self._reject("nothing recognisable (parakeet returned empty)", "")
            return ""
        if _NON_LATIN_RE.search(text):
            self._reject("contains non-Latin characters", text)
            return ""
        toks = [t.strip(".,!?;:").lower() for t in text.split() if t.strip(".,!?;:")]
        if len(toks) >= 6:
            from collections import Counter
            top, cnt = Counter(toks).most_common(1)[0]
            if cnt >= 4 and cnt / len(toks) >= 0.4:
                self._reject(f"one word repeated {cnt} times", text)
                return ""
        return self._post_correct(text)

    def _transcribe(self, pcm16_buffer: bytes,
                   utt_start_wall: float = 0.0,
                   skip_echo_cut: bool = False) -> str:
        if not pcm16_buffer:
            return ""
        # PCM16 → float32 in [-1, 1]
        audio = (
            np.frombuffer(pcm16_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        )
        # RMS gate: very-quiet buffers are almost always background noise
        # that VAD let through as a false positive. Drop without calling
        # Whisper — it would just hallucinate "The End" or similar.
        if audio.size == 0:
            return ""
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < self._min_rms:
            self._log(
                "info",
                f"Skipping low-energy buffer (rms={rms:.4f} < min={self._min_rms:.4f}); "
                "likely background noise / distant speech.",
            )
            self._reject(
                f"too quiet (rms {rms:.3f} < gate {self._min_rms:.3f})", ""
            )
            return ""
        score = self._morty_score(pcm16_buffer)
        self._last_speaker_score = score      # read by the node when it publishes the transcript
        if score is not None:
            if score >= self._speaker_thr:
                self._log("info", f"[speaker] Morty's own voice (cos {score:.2f}, rms {rms:.3f}) — not transcribed")
                self._reject(f"my own voice (speaker gate cos {score:.2f})", "")
                return ""
            self._log("info", f"[speaker] human voice (cos {score:.2f}, rms {rms:.3f})")
        if self._pk is not None:
            return self._transcribe_parakeet(audio)
        try:
            _t0 = time.monotonic()
            segments, info = self._model.transcribe(
                audio,
                # "auto"/"" -> None: whisper detects the language PER
                # UTTERANCE (Majd, 2026-08-11: EN and DE must both just
                # work). Needs a multilingual model (base, not base.en).
                language=(self._language
                          if self._language and self._language != "auto"
                          else None),
                beam_size=1,           # speed > diversity for command voice
                vad_filter=False,      # we already did VAD
                condition_on_previous_text=False,
                initial_prompt=self._initial_prompt,
                # Per-word start/end times, needed to cut Morty's echo
                # out of a transcript that holds BOTH voices. Costs an
                # extra forced-alignment pass, so it is only requested
                # when the cut is actually armed.
                word_timestamps=bool(self._word_time_cut),
            )
            # Collect segments while rejecting any with a high no_speech
            # probability. faster-whisper exposes this per-segment as
            # ``s.no_speech_prob`` (0..1). Anything > 0.6 means Whisper
            # itself thought there was no real speech — those are the
            # hallucination cases.
            parts: list[str] = []
            echoed: list[str] = []
            for s in segments:
                no_speech = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
                if no_speech > 0.6:
                    self._log(
                        "info",
                        f"Dropping low-confidence Whisper segment "
                        f"(no_speech_prob={no_speech:.2f}): {s.text!r}",
                    )
                    self._reject(
                        f"Whisper unsure it was speech "
                        f"({int(no_speech * 100)}%)", s.text.strip()
                    )
                    continue
                words = getattr(s, "words", None) or []
                # skip_echo_cut: this utterance starts at a barge-in the
                # speaker fingerprint confirmed as human — every word in it
                # overlaps the playback Morty was cutting short, and all of
                # them are the operator's (2026-09-23 test: the cut deleted
                # "Now what kind of object are you seeing right now?").
                # Majd 2026-09-23: keep only what is reliable. When the
                # fingerprint judged this utterance (score is not None) it is
                # human, or it was rejected before Whisper — no word of it is
                # cut by clock or by text. The cut remains only as the
                # fallback while the speaker gate is unavailable.
                cut_ok = (self._word_time_cut and utt_start_wall > 0.0
                          and words and not skip_echo_cut and score is None)
                if not cut_ok:
                    # No word times (older faster-whisper, or the cut is
                    # disarmed): behave exactly as before.
                    parts.append(s.text.strip())
                    continue
                keep: list[str] = []
                overlap: list[str] = []      # words inside a playback window
                for w in words:
                    try:
                        w_start = float(w.start)
                        w_end = float(w.end)
                    except (TypeError, ValueError):
                        keep.append(str(getattr(w, "word", "")))
                        continue
                    # Judge on the MIDPOINT: robust when a word straddles
                    # the boundary between Morty stopping and Majd
                    # starting.
                    mid = utt_start_wall + 0.5 * (w_start + w_end)
                    if self._time_is_ai_speech(mid):
                        overlap.append(str(getattr(w, "word", "")))
                    else:
                        keep.append(str(getattr(w, "word", "")))
                if overlap and not self._looks_like_my_echo(overlap):
                    # The operator talked over Morty: these are his words.
                    self._log("info",
                              f"Word-time cut kept {len(overlap)} overlapping word(s) — "
                              f"they do not match what Morty said: {''.join(overlap).strip()!r}")
                    keep = [str(getattr(w, "word", "")) for w in words]
                    overlap = []
                echoed.extend(x.strip() for x in overlap)
                kept = "".join(keep).strip()
                if kept:
                    parts.append(kept)
            text = " ".join(p for p in parts if p).strip()
            # One timing line per utterance so "why is it slow" is
            # answerable from this log instead of guesswork (2026-08-31).
            # NB: segments is a GENERATOR — Whisper actually runs during
            # the collection loop above, so measure here, not at the
            # transcribe() call.
            audio_s = len(pcm16_buffer) / (2.0 * self._sample_rate)
            self._log(
                "info",
                f"[timing] whisper {time.monotonic() - _t0:.2f}s for "
                f"{audio_s:.1f}s audio (word_timestamps="
                f"{'on' if self._word_time_cut else 'off'})",
            )
            if echoed:
                self._log(
                    "info",
                    "Word-time echo cut: removed "
                    f"{len(echoed)} word(s) spoken while Morty's speaker "
                    f"was sounding ({' '.join(echoed)!r}); kept {text!r}",
                )
            if echoed and not text:
                # Every word landed inside a playback window: the mic
                # heard only Morty. Nothing for the operator to lose.
                self._reject("echo of my own voice (cut by word timing)", "")
                return ""
            # Script gate: reject ONLY on hard evidence — non-Latin
            # characters in the transcript (the Korean/Cyrillic word-soup
            # case). Whisper's detected-language tag is NOT trusted at
            # all: even large-v3-turbo labels "Stop the Grasp!" Icelandic
            # and "Stopp." Indonesian with high confidence, and the
            # earlier probability-based gate ate real stop/drop commands
            # (Majd, 2026-08-12 — twice).
            if text and _NON_LATIN_RE.search(text):
                self._log(
                    "info",
                    f"Dropping transcript with non-Latin script: {text!r}",
                )
                self._reject(
                    "contains non-Latin characters (hallucinated noise)",
                    text,
                )
                return ""
            # Repetition gate: hallucination storms repeat one token many
            # times ("CD-Hülle, CD-Hülle, ×8" — trial 5, 2026-08-12). Real
            # speech never does.
            if text:
                toks = [t.strip(".,!?;:").lower()
                        for t in text.split() if t.strip(".,!?;:")]
                if len(toks) >= 6:
                    from collections import Counter
                    top, cnt = Counter(toks).most_common(1)[0]
                    if cnt >= 4 and cnt / len(toks) >= 0.4:
                        self._log(
                            "info",
                            f"Dropping repetitive hallucination "
                            f"('{top}' ×{cnt}): {text!r}",
                        )
                        self._reject(
                            f"one word repeated {cnt} times (hallucinated "
                            "noise)", text)
                        return ""
            text = self._post_correct(text)
            if text and self._looks_like_hallucination(text):
                self._log(
                    "info",
                    f"Dropping known Whisper hallucination: {text!r}",
                )
                self._reject("known Whisper hallucination pattern", text)
                return ""
            return text
        except Exception as exc:
            self._log("error", f"Whisper transcribe failed: {exc}")
            return ""

    def _loop(self) -> None:
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._frames_per_buffer,
                input_device_index=self._mic_index,
            )
            stream.start_stream()
            self._log(
                "info",
                f"WhisperTranscriber listening (rate={self._sample_rate}, "
                f"mic_index={self._mic_index}, silence_ms={self._silence_ms})",
            )

            chunk_ms = int(1000 * self._frames_per_buffer / self._sample_rate)
            buffer = bytearray()
            in_speech = False
            silence_ms = 0
            speech_ms = 0
            barge_fired = False
            next_barge_ms = self._barge_in_min_ms
            skip_echo_cut = False
            barge_hits = 0
            barge_ref_rms = -1.0
            # Wall clock of the FIRST sample in `buffer`. Whisper reports
            # word times relative to the buffer, so this anchors them to
            # the same clock as the TTS playback windows.
            utt_start_wall = 0.0
            last_state = ""

            def _emit_state(s: str) -> None:
                nonlocal last_state
                if s == last_state:
                    return
                last_state = s
                if self._on_state is not None:
                    try:
                        self._on_state(s)
                    except Exception:
                        pass

            while not self._stop.is_set():
                data = stream.read(
                    self._frames_per_buffer, exception_on_overflow=False
                )
                # Live mic level + waveform for the visualiser. Cheap;
                # runs even when muted so the user can see the mic
                # itself is alive (just gated).
                if self._on_audio_level is not None:
                    try:
                        self._on_audio_level(_compute_rms_normalised(data))
                    except Exception:
                        pass
                if self._on_audio_waveform is not None:
                    try:
                        self._on_audio_waveform(_compute_peak_envelope(data))
                    except Exception:
                        pass
                if self._muted.is_set():
                    buffer.clear()
                    in_speech = False
                    silence_ms = 0
                    speech_ms = 0
                    _emit_state("muted")
                    continue

                voice = self._has_voice(data)
                _emit_state("speech" if voice or in_speech else "idle")
                if voice:
                    if not in_speech:
                        in_speech = True
                        silence_ms = 0
                        speech_ms = 0
                        buffer.clear()
                        # `data` was captured over the PRECEDING chunk_ms,
                        # so the first sample is one chunk in the past.
                        utt_start_wall = time.time() - chunk_ms / 1000.0
                        barge_fired = False
                        next_barge_ms = self._barge_in_min_ms
                        skip_echo_cut = False
                        barge_hits = 0
                        barge_ref_rms = -1.0
                    buffer.extend(data)
                    speech_ms += chunk_ms
                    silence_ms = 0
                    # Cap utterance length to avoid huge transcribe calls.
                    # Morty's own playback trips the VAD, so an utterance
                    # heard during TTS BEGINS with his echo; the operator
                    # joins later. Judge the LATEST 0.8 s every 0.3 s (12 ms
                    # each) — one check on the buffer's head saw only Morty
                    # (2026-09-23 test). On a human verdict the utterance
                    # restarts at that window: Whisper then gets the
                    # operator's words, not the story they interrupted.
                    if (self._on_barge_in is not None and not barge_fired
                            and speech_ms >= next_barge_ms and self._tts_playing()):
                        next_barge_ms = speech_ms + 300
                        win = 2 * int(1.2 * self._sample_rate)
                        window = bytes(buffer[-win:])
                        rms = _compute_rms_normalised(window)
                        if barge_ref_rms < 0.0:
                            barge_ref_rms = rms      # first window = the echo alone (or the human, if loud)
                        # A human interrupting is LOUD in a clip-on mic; Morty's
                        # echo is faint (rms ~0.006 vs speech 0.03+, 2026-09-23).
                        # Short 0.8 s windows of his echo scored 0.14-0.49 on the
                        # fingerprint, so the voice test alone stopped him at
                        # every first sentence; loudness + two consecutive
                        # human verdicts is what a real interruption looks like.
                        # Majd 2026-09-23: "no more loudness". The echo canceller
                        # (PulseAudio module-echo-cancel on the Hollyland mic and
                        # the analog speaker) removes Morty's voice from the mic;
                        # what is left is judged by the fingerprint alone: two
                        # consecutive 1.2 s windows below the barge threshold.
                        # The plain background gate (min_rms) still applies.
                        loud = rms >= self._min_rms
                        score = self._barge_in_check(window, quiet=(barge_hits > 0 or speech_ms > self._barge_in_min_ms)) if loud else None
                        barge_hits = barge_hits + 1 if (score is not None and score < self._barge_thr) else 0
                        if barge_hits >= 2:
                            self._log("info", f"[speaker] barge-in: a human is talking over Morty "
                                              f"(cos {score:.2f}, rms {rms:.3f} vs echo {barge_ref_rms:.3f}) — stopping playback")
                            try:
                                self._on_barge_in(score)
                            except Exception:
                                pass
                            barge_fired = True
                            skip_echo_cut = True
                            del buffer[:-win]
                            utt_start_wall = time.time() - 0.8
                            speech_ms = 800
                    if speech_ms / 1000.0 >= self._max_utterance_s:
                        self._enqueue_utterance(bytes(buffer), utt_start_wall, skip_echo_cut)
                        buffer.clear()
                        in_speech = False
                        silence_ms = 0
                        speech_ms = 0
                elif in_speech:
                    # In-speech but current chunk is silence — buffer for tail
                    # (a single silent chunk in the middle of a sentence is normal).
                    buffer.extend(data)
                    silence_ms += chunk_ms
                    if silence_ms >= self._silence_ms:
                        if speech_ms >= self._min_utterance_ms:
                            self._enqueue_utterance(bytes(buffer),
                                                    utt_start_wall, skip_echo_cut)
                        buffer.clear()
                        in_speech = False
                        silence_ms = 0
                        speech_ms = 0
                # Else: silence outside of speech — ignore.
        except Exception as exc:
            self._log("error", f"Whisper loop error: {exc}")
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass
