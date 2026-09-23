"""
speech_input_node.py
====================

Captures microphone audio, runs speech-to-text on it, and publishes the
transcribed user utterances on ``/speech/transcript`` for ``chat_node`` to
consume. Also publishes auxiliary signals (mic energy level, waveform,
state) for the mic visualiser GUI.

Two STT engines are selectable via the ``stt_engine`` parameter; the actual
transcription lives in :mod:`stt`:

* ``whisper`` — ``faster-whisper`` running on the GPU. Higher quality, ~1 s
  latency on short utterances. The default for the project.
* ``vosk``    — fully offline Vosk model on CPU. Lighter, lower accuracy;
  kept as a backup for machines without a GPU.

The node also handles:

* a VAD-style energy gate (``min_speech_rms``) — buffers below the threshold
  are dropped as background noise to avoid bothering the LLM with silence.
* a ``/speech/muted`` topic so the rest of the system can pause STT (e.g.
  while Morty is speaking, to reduce self-echo before the chat-side
  echo-drop kicks in).

The default Vosk model dir is ``$HOME/morty_ws/src/communication_llm/
models/vosk-model-small-en-us-0.15`` so it is not tied to a specific machine;
override with the ``vosk_model_dir`` parameter.
"""
from __future__ import annotations

import os

import json
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from .stt import VoskTranscriber, WhisperTranscriber


class SpeechInputNode(Node):
    def __init__(self) -> None:
        super().__init__("speech_input_node")

        # Vosk model dir default, resolved relative to $HOME so it is not tied to
        # a specific machine/username. Override with the 'vosk_model_dir' param.
        # (Only used when stt_engine:=vosk; the default pipeline uses Whisper.)
        # Honours $WS_DIR (same convention as grasp_pipeline/selection/params.py);
        # falls back to the historical name. Alienware ws is ~/morty_ws. (2026-08-06)
        self.declare_parameter(
            "vosk_model_dir",
            os.path.join(
                os.environ.get(
                    "WS_DIR", os.path.expanduser("~/morty_ws")
                ),
                "src/communication_llm/models/vosk-model-small-en-us-0.15",
            ),
        )
        self.declare_parameter("transcript_topic", "/speech/transcript")
        self.declare_parameter("mute_topic", "/speech/muted")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("mic_index", -1)
        self.declare_parameter(
            "mic_device_name",
            "",
            # Examples: "echocancel_mic", "ReSpeaker", "USB Audio".
            # When non-empty, takes priority over mic_index.
        )
        self.declare_parameter("min_chars", 2)
        # Layer 2: WebRTC VAD gating (only feed Vosk when speech is detected).
        self.declare_parameter("vad_enabled", False)
        self.declare_parameter("vad_aggressiveness", 1)  # 0 = least, 3 = most
        # STT engine selection. "vosk" is fast/streaming/cheap; "whisper"
        # is faster-whisper (much better accuracy, segmented utterance mode).
        self.declare_parameter("stt_engine", "vosk")
        self.declare_parameter("parakeet_model", "nemo-parakeet-tdt-0.6b-v3")   # stt_engine:=parakeet (2026-09-23)
        self.declare_parameter("vad_engine", "silero")       # "silero" (default since 2026-09-23) or "webrtc"
        self.declare_parameter("silero_onset", 0.5)
        self.declare_parameter("silero_offset", 0.35)
        # faster-whisper config (used when stt_engine=whisper)
        self.declare_parameter("whisper_model", "base.en")
        # Transcription language ("en", "de", ...). Use a MULTILINGUAL model
        # (e.g. "base", not "base.en") for anything but English. (2026-08-11)
        self.declare_parameter("whisper_language", "en")
        self.declare_parameter("whisper_device", "auto")
        self.declare_parameter("whisper_compute_type", "auto")
        # End-of-utterance silence threshold. Lower = faster response,
        # higher = safer if you pause mid-sentence to think. 500 ms was the
        # old default; 300 ms since 2026-09-18 (Majd: commands are short,
        # the wait was the longest piece after the language model).
        # Override with whisper_silence_ms.
        self.declare_parameter("whisper_silence_ms", 300)
        self.declare_parameter("whisper_max_utterance_s", 12.0)
        # ---- word-time echo cut (2026-08-28) -----------------------
        # Cut Morty's own words out of a mixed transcript by WHEN they
        # were spoken instead of guessing from text similarity. Needs
        # /speech/ai_speaking from chat_node to know the playback
        # windows. Set word_time_cut False to fall straight back to the
        # old behaviour (also skips the extra alignment pass).
        self.declare_parameter("word_time_cut", True)
        self.declare_parameter("tts_lead_s", 0.15)
        self.declare_parameter("tts_tail_s", 0.25)
        # 2026-08-31: switched from chat_node's /speech/ai_speaking to
        # the speak SERVER's /speech/tts_active. chat_node only knows
        # about its OWN goals, so the executor/intent_router STATUS
        # narrations ("Got the small bottle. Where should I drop it?")
        # had no playback window and their echo sailed past the
        # word-time cut. The speak server serialises and plays EVERY
        # sentence from every node, and stamps True at execute start /
        # False on finish — one authoritative window source.
        self.declare_parameter("ai_speaking_topic", "/speech/tts_active")
        # Speaker gate + barge-in (2026-09-23, see stt.py): Morty's own
        # voice is recognised by fingerprint and never transcribed; a human
        # talking over him stops the playback via /speech/barge_in.
        self.declare_parameter("speaker_gate", True)
        self.declare_parameter("morty_voice_path", "~/morty_ws/models/morty_voice_ecapa.npy")
        self.declare_parameter("speaker_gate_threshold", 0.35)
        self.declare_parameter("barge_in", True)
        self.declare_parameter("barge_in_min_ms", 600)
        # Minimum RMS amplitude for a buffer to be transcribed. 0..1 scale
        # (audio normalised to [-1, 1]). Higher = louder voice required.
        # 0.025 rejects whispered AI echo bleeding through the speaker
        # plus distant chatter, while leaving a normal-volume command
        # voice (typical RMS 0.04–0.10) well above the bar. Increase
        # further if AI echo is still slipping through; drop to ~0.015
        # if you have a sensitive mic in a quiet environment.
        self.declare_parameter("whisper_min_rms", 0.025)
        # Vocabulary hint that biases Whisper toward our expected words.
        # Whisper uses this as a soft prior for recognition; just listing
        # the right words once dramatically improves recognition of names
        # like "Morty" and domain terms like "vitamin tube".
        self.declare_parameter(
            "whisper_initial_prompt",
            (
                # COMPACT (2026-08-12): the old 338-token prompt exceeded
                # whisper's 224-token cap — it kept only the TAIL (the
                # German vocab), which then dominated hallucinations
                # ("CD-Hülle x8, Befehle..."). Keep this under ~200 tokens.
                "Morty is a robot assistant. Commands: pick up, grasp, "
                "grab, open the gripper, close the gripper, release, "
                "stop, cancel, go home, go to ready, look left, look "
                "right, drop right, drop left, start all, start the "
                "pipeline, stop the driver. "
                "Objects: tape dispenser, CD case, toy faucet, kitchen "
                "towel, nutmeg jar, retail box, small bottle, wooden "
                "maze puzzle, vitamin tube, vitamin C. "
                "Examples: grasp the small bottle. Pick up the vitamin "
                "tube. Drop it on the right side. Yes, open the gripper. "
                "Auf Deutsch: Greifer öffnen, schließen, loslassen, "
                "stopp, geh nach Hause, schau links, rechts, greif die "
                "kleine Flasche, das Vitamin C, den Wasserhahn, lege es "
                "rechts ab."
            ),
        )
        # Layer 3: wake-word gate. When enabled, only transcripts that
        # begin with one of `wake_words` (or appear in `always_allow`) are
        # forwarded. The wake-word prefix is stripped before publishing.
        self.declare_parameter("wake_word_enabled", False)
        self.declare_parameter("wake_words", ["robot", "hey robot"])
        self.declare_parameter("always_allow", ["stop", "cancel", "abort", "halt"])

        self._min_chars = int(self.get_parameter("min_chars").value)
        self._wake_enabled = bool(self.get_parameter("wake_word_enabled").value)
        self._wake_words: list[str] = [
            str(w).strip().lower() for w in (self.get_parameter("wake_words").value or [])
        ]
        self._always_allow: list[str] = [
            str(w).strip().lower() for w in (self.get_parameter("always_allow").value or [])
        ]

        self._pub = self.create_publisher(
            String, str(self.get_parameter("transcript_topic").value), 10
        )
        # Live mic feedback for the visualiser. /speech/level is a
        # 0..1 normalised RMS published every chunk (~10-25 Hz). /speech/state
        # is one of "idle" / "speech" / "muted". /speech/waveform is a
        # downsampled signed peak envelope (~8 floats per chunk) used to
        # render the mirrored waveform bars. All best-effort.
        self._level_pub = self.create_publisher(Float32, "/speech/level", 10)
        # Why an utterance produced NOTHING — surfaced in the GUI chat
        # ("it doesn't show me why!", Majd 2026-08-12). Payload:
        # "<reason> :: <text-if-any>".
        self._rejected_pub = self.create_publisher(
            String, "/speech/rejected", 10
        )
        self._state_pub = self.create_publisher(String, "/speech/state", 10)
        self._barge_pub = self.create_publisher(Bool, "/speech/barge_in", 10)
        # Who spoke, per transcript (2026-09-23): cos to Morty's fingerprint,
        # published just before the transcript so chat_node can skip its
        # text-overlap echo rule when the voice was already judged human.
        self._verdict_pub = self.create_publisher(String, "/speech/speaker_verdict", 10)
        self._wave_pub = self.create_publisher(
            Float32MultiArray, "/speech/waveform", 10
        )
        self._last_level: float = 0.0

        try:
            mic_device_name = str(self.get_parameter("mic_device_name").value).strip()
            engine = str(self.get_parameter("stt_engine").value).strip().lower()
            if engine in ("whisper", "parakeet"):
                self._stt = WhisperTranscriber(
                    engine=engine,
                    vad_engine=str(self.get_parameter("vad_engine").value),
                    silero_onset=float(self.get_parameter("silero_onset").value),
                    silero_offset=float(self.get_parameter("silero_offset").value),
                    parakeet_model=str(self.get_parameter("parakeet_model").value),
                    model_size=str(self.get_parameter("whisper_model").value),
                    on_transcript=self._on_transcript,
                    sample_rate=int(self.get_parameter("sample_rate").value),
                    mic_index=int(self.get_parameter("mic_index").value),
                    mic_device_name=mic_device_name or None,
                    device=str(self.get_parameter("whisper_device").value),
                    compute_type=str(self.get_parameter("whisper_compute_type").value),
                    language=str(self.get_parameter("whisper_language").value),
                    vad_aggressiveness=int(self.get_parameter("vad_aggressiveness").value),
                    silence_ms_to_finalise=int(
                        self.get_parameter("whisper_silence_ms").value
                    ),
                    max_utterance_s=float(
                        self.get_parameter("whisper_max_utterance_s").value
                    ),
                    min_rms=float(
                        self.get_parameter("whisper_min_rms").value
                    ),
                    initial_prompt=str(
                        self.get_parameter("whisper_initial_prompt").value
                    ),
                    on_audio_level=self._publish_audio_level,
                    on_state=self._publish_listener_state,
                    on_audio_waveform=self._publish_audio_waveform,
                    on_reject=self._publish_rejected,
                    word_time_cut=bool(
                        self.get_parameter("word_time_cut").value
                    ),
                    tts_lead_s=float(self.get_parameter("tts_lead_s").value),
                    tts_tail_s=float(self.get_parameter("tts_tail_s").value),
                    speaker_gate=bool(self.get_parameter("speaker_gate").value),
                    morty_voice_path=str(self.get_parameter("morty_voice_path").value),
                    speaker_gate_threshold=float(self.get_parameter("speaker_gate_threshold").value),
                    on_barge_in=(self._publish_barge_in
                                 if bool(self.get_parameter("barge_in").value) else None),
                    barge_in_min_ms=int(self.get_parameter("barge_in_min_ms").value),
                    logger=self.get_logger(),
                )
            else:
                self._stt = VoskTranscriber(
                    model_dir=str(self.get_parameter("vosk_model_dir").value),
                    on_transcript=self._on_transcript,
                    sample_rate=int(self.get_parameter("sample_rate").value),
                    mic_index=int(self.get_parameter("mic_index").value),
                    mic_device_name=mic_device_name or None,
                    vad_enabled=bool(self.get_parameter("vad_enabled").value),
                    vad_aggressiveness=int(self.get_parameter("vad_aggressiveness").value),
                    on_audio_level=self._publish_audio_level,
                    on_state=self._publish_listener_state,
                    on_audio_waveform=self._publish_audio_waveform,
                    logger=self.get_logger(),
                )
        except Exception as exc:
            self.get_logger().error(f"Vosk init failed: {exc}")
            raise

        # Lets a TTS layer (or any orchestrator) suppress recognition while
        # audio output is active to avoid self-feedback. Publish True to mute,
        # False to resume.
        # Morty's loudspeaker on/off, published by chat_node around each
        # Speak goal. Feeds the word-time echo cut in stt.py.
        self.create_subscription(
            Bool,
            str(self.get_parameter("ai_speaking_topic").value),
            self._on_ai_speaking,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("mute_topic").value),
            self._on_mute,
            10,
        )
        # The sentence behind each playback window (speak_node publishes
        # it before playing) — lets the word-time cut keep the operator's
        # words when he talks over Morty (2026-09-23).
        self.create_subscription(String, "/speech/ai_said", self._on_ai_said, 10)

        self._stt.start()
        # LIVE tuning of the four voice-filter knobs (GUI "Voice filters"
        # Apply button, 2026-08-28). Registered LAST, after every
        # declare_parameter and after the listener exists, so it can never
        # fire against a half-built node.
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.get_logger().info(
            f"speech_input_node publishing transcripts on "
            f"{self.get_parameter('transcript_topic').value}"
        )

    # ── live parameter tuning (2026-08-28) ───────────────────────────
    # Until now the four voice-filter numbers were constructor-only: the
    # GUI could pass them as launch overrides and nothing more, so tuning
    # the mic gate meant restarting the whole voice stack. Three of them
    # live on the STT listener, which READS them at use time on its own
    # audio thread, and min_chars is read per transcript here — so setting
    # the ROS parameter at runtime can take effect on the next chunk.
    #
    # name -> (python type, min, max, unit shown in the reject reason)
    _LIVE_PARAM_LIMITS = {
        "whisper_min_rms": (float, 0.0005, 1.0, "normalised rms"),
        "whisper_silence_ms": (int, 50, 10000, "ms"),
        "whisper_max_utterance_s": (float, 0.5, 120.0, "s"),
        "min_chars": (int, 0, 200, "chars"),
    }
    _LIVE_PARAM_SETTERS = {
        "whisper_min_rms": "set_min_rms",
        "whisper_silence_ms": "set_silence_ms",
        "whisper_max_utterance_s": "set_max_utterance_s",
    }

    def _on_set_parameters(self, params) -> SetParametersResult:
        """Validate, then APPLY, runtime changes to the voice filters.

        Everything is validated BEFORE anything is applied, so a rejected
        request leaves the running listener exactly as it was. Parameters
        we do not own are accepted untouched (rclpy's default behaviour
        for an un-handled name).
        """
        pending = []
        for p in params:
            limits = self._LIVE_PARAM_LIMITS.get(p.name)
            if limits is None:
                continue
            kind, low, high, unit = limits
            raw = p.value
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return SetParametersResult(
                    successful=False,
                    reason=f"{p.name} must be a number, got {raw!r}",
                )
            if kind is int and float(raw) != int(raw):
                return SetParametersResult(
                    successful=False,
                    reason=f"{p.name} must be a whole number, got {raw!r}",
                )
            value = kind(raw)
            if not (low <= value <= high):
                return SetParametersResult(
                    successful=False,
                    reason=(
                        f"{p.name}={value} is out of range "
                        f"[{low}, {high}] {unit}"
                    ),
                )
            pending.append((p.name, value))
        for name, value in pending:
            try:
                self._apply_live_param(name, value)
            except Exception as exc:  # never let a knob kill the node
                return SetParametersResult(
                    successful=False, reason=f"{name} could not be applied: {exc}"
                )
        return SetParametersResult(successful=True)

    def _apply_live_param(self, name: str, value) -> None:
        """Push ONE validated value onto the live listener / node state.

        Degrades honestly: if the listener does not exist yet, or is an
        engine without that knob (Vosk has no RMS gate / utterance cap),
        the parameter is still stored by rclpy and honoured at the next
        construction — we just say so in the log instead of crashing.
        """
        if name == "min_chars":
            old = self._min_chars
            self._min_chars = int(value)
            self.get_logger().info(
                f"[voice-filters] min_chars {old} -> {self._min_chars} (live)"
            )
            return
        listener = getattr(self, "_stt", None)
        setter = getattr(listener, self._LIVE_PARAM_SETTERS[name], None)
        if not callable(setter):
            self.get_logger().warning(
                f"[voice-filters] {name} -> {value} stored, but the live "
                f"listener ({type(listener).__name__ if listener is not None else 'not built yet'}) "
                "has no setter for it — it will be honoured at the next start."
            )
            return
        old = setter(value)
        self.get_logger().info(
            f"[voice-filters] {name} {old} -> {value} (live)"
        )

    def _publish_rejected(self, reason: str, text: str) -> None:
        try:
            self._rejected_pub.publish(
                String(data=f"{reason} :: {text}".strip())
            )
        except Exception:
            pass

    def _on_transcript(self, text: str) -> None:
        if len(text) < self._min_chars:
            if text.strip():
                self._publish_rejected(
                    f"shorter than min_chars={self._min_chars}", text
                )
            return
        self.get_logger().info(f"Heard: {text!r}")
        forwarded = self._gate_wake_word(text)
        if forwarded is None:
            self.get_logger().info(
                "Dropped (no wake word and not in always_allow)."
            )
            self._publish_rejected("no wake word", text)
            return
        score = getattr(self._stt, "_last_speaker_score", None)
        if score is not None:
            self._verdict_pub.publish(String(data=json.dumps(
                {"text": forwarded, "cos": round(float(score), 3), "human": True})))
        msg = String()
        msg.data = forwarded
        self._pub.publish(msg)

    def _gate_wake_word(self, text: str) -> str | None:
        """Return the transcript to forward, or None to drop it.

        With wake-word gating disabled, every transcript is forwarded as-is.
        With it enabled:
          - Transcripts starting with a wake word are forwarded with the
            prefix stripped.
          - Transcripts whose entire content matches an always-allow entry
            are forwarded as-is (so "stop" still works mid-AI-speech).
          - Everything else is dropped.
        """
        if not self._wake_enabled:
            return text
        norm = text.strip().lower()
        if not norm:
            return None
        # Always-allow short commands first.
        if norm in self._always_allow:
            return text
        for wake in self._wake_words:
            if not wake:
                continue
            if norm == wake:
                # Just the wake word with no command — silently consume.
                return None
            if norm.startswith(wake + " "):
                stripped = text.strip()[len(wake):].lstrip(" ,.").strip()
                if stripped:
                    return stripped
        return None

    def _on_ai_speaking(self, msg: Bool) -> None:
        """Record when Morty's speaker starts/stops sounding.

        Only the Whisper backend implements the word-time cut, so this is
        a no-op under Vosk (which returns no word timings)."""
        note = getattr(self._stt, "note_ai_speaking", None)
        if note is not None:
            note(bool(msg.data))

    def _publish_barge_in(self, score: float) -> None:
        self._barge_pub.publish(Bool(data=True))

    def _on_ai_said(self, msg: String) -> None:
        note = getattr(self._stt, "note_ai_text", None)
        if note is not None:
            note(str(msg.data or ""))

    def _on_mute(self, msg: Bool) -> None:
        self._stt.set_muted(bool(msg.data))

    def _publish_audio_level(self, level: float) -> None:
        """Forward the STT's per-chunk RMS as a Float32 on /speech/level.

        The STT calls this every audio chunk (~10-25 Hz depending on
        engine). Light smoothing avoids a jittery visualiser bar without
        adding noticeable latency.
        """
        smoothed = 0.6 * self._last_level + 0.4 * float(level)
        self._last_level = smoothed
        try:
            self._level_pub.publish(Float32(data=float(smoothed)))
        except Exception:
            pass

    def _publish_listener_state(self, state: str) -> None:
        """Forward 'idle' / 'speech' / 'muted' onto /speech/state."""
        try:
            self._state_pub.publish(String(data=str(state)))
        except Exception:
            pass

    def _publish_audio_waveform(self, samples) -> None:
        """Forward a small signed peak-envelope array to /speech/waveform.

        The visualiser appends each batch to a ring buffer and renders
        mirrored vertical bars from the centre line. ~8 samples per
        chunk gives a smooth scrolling "audio wave" look.
        """
        try:
            msg = Float32MultiArray()
            msg.data = [float(v) for v in samples]
            self._wave_pub.publish(msg)
        except Exception:
            pass

    def destroy_node(self) -> bool:
        try:
            self._stt.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SpeechInputNode()
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
