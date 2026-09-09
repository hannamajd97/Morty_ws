#!/usr/bin/env python3
"""Interactive mic + STT check for this machine.

Added 2026-08-06 while porting the code to ~/morty_ws. Use this when the mic
level bar moves but no transcript ever appears, which on this Alienware has
been a gain problem twice over:

  * gain too LOW  -> speech_input_node logs
    "Skipping low-energy buffer (rms=... < min=0.0250)" and never transcribes.
  * gain too HIGH -> the signal clips and the room noise floor by itself sits
    above the 0.025 gate, so the node fires on silence and Whisper, correctly,
    returns an empty string for noise.

Run it, press Enter, and talk. It reports the level it actually measured and
what Whisper made of it, so you can tell a gain problem from a model problem.

    python3 tools/check_mic_stt.py            # record 6s and transcribe
    python3 tools/check_mic_stt.py --level    # level meter only, no model load
"""
import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
# Must match speech_input_node's min-energy gate, else this test lies to you.
NODE_GATE = 0.0250


def measure(seconds: float) -> np.ndarray:
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def report(audio: np.ndarray) -> float:
    """Print level stats and return the loudest half-second RMS."""
    rms = float(np.sqrt((audio ** 2).mean()))
    peak = float(np.abs(audio).max())
    win = int(0.5 * SAMPLE_RATE)
    windows = [
        float(np.sqrt((audio[i:i + win] ** 2).mean()))
        for i in range(0, max(len(audio) - win, 1), win)
    ] or [rms]
    loudest = max(windows)

    print(f"  overall rms   : {rms:.4f}")
    print(f"  peak          : {peak:.4f}" + ("   <-- CLIPPING, lower the gain"
                                             if peak >= 0.99 else ""))
    print(f"  loudest 0.5s  : {loudest:.4f}  (node gate = {NODE_GATE})")
    return loudest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--level", action="store_true",
                    help="level meter only; do not load Whisper")
    args = ap.parse_args()

    print("Quiet-room baseline — stay silent for 2s...")
    time.sleep(0.3)
    floor = report(measure(2.0))
    if floor > NODE_GATE:
        print(f"  WARNING: silence alone ({floor:.4f}) is above the gate "
              f"({NODE_GATE}). Gain is too HIGH — the node will trigger on "
              f"noise. Try:  pactl set-source-volume echocancel_mic 40%")
    print()

    input(f"Press Enter, then speak for {args.seconds:.0f}s "
          f"(e.g. 'pick up the small bottle')...")
    print(">>> SPEAKING WINDOW OPEN <<<", flush=True)
    audio = measure(args.seconds)
    print(">>> done <<<")
    loudest = report(audio)

    if loudest <= NODE_GATE:
        print("\n  Speech never crossed the gate. speech_input_node would have "
              "skipped this buffer. Raise the gain a little, or speak closer "
              "to the mic.")
    else:
        print(f"\n  Speech is {loudest / NODE_GATE:.0f}x the gate — "
              f"speech_input_node would accept this buffer.")

    if args.level:
        return 0

    print("\nLoading faster-whisper base.en on CUDA (compute_type=auto — this "
          "GPU is Pascal and has no efficient fp16)...")
    from faster_whisper import WhisperModel

    model = WhisperModel("base.en", device="cuda", compute_type="auto")
    started = time.time()
    segments = list(model.transcribe(audio, language="en", beam_size=5)[0])
    elapsed = time.time() - started

    text = " ".join(s.text for s in segments).strip()
    print(f"STT latency : {elapsed:.2f}s")
    print(f"TRANSCRIPT  : {text!r}")
    if not text:
        print("  Empty transcript with a good level usually means the buffer "
              "held noise rather than speech — check that the default source "
              "is the one you are actually talking into "
              "(pactl info | grep 'Default Source').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
