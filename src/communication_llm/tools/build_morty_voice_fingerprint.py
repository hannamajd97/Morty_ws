"""Build/evaluate Morty's voice fingerprint (ECAPA-TDNN) from the MMHRI trial recordings.

Run from ~/morty_ws with the workspace sourced. Enrols Morty from half of the
trials, reports the separation on the other half (cos to the fingerprint:
Morty >= 0.50, humans <= 0.10 on 2026-09-23), and writes the recorded-room
English row of models/morty_voice_ecapa.npy. The speech node loads that file
(speaker_gate, speaker_gate_threshold 0.35)."""
import time, glob, re, sys, numpy as np, torch, torchaudio
torch.set_grad_enabled(False)
from speechbrain.inference.speaker import EncoderClassifier
t0=time.time()
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/home/deep/morty_ws/models/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
print("model loaded in %.1fs" % (time.time()-t0))
x = torch.randn(1, 32000).cuda(); enc.encode_batch(x); torch.cuda.synchronize(); t1=time.time()
for _ in range(20): e = enc.encode_batch(x)
torch.cuda.synchronize(); print("embedding: %.1f ms per 2 s clip, dim %d" % ((time.time()-t1)/20*1000, e.shape[-1]))

def emb(wav16):  # 1-D float tensor at 16 kHz
    return torch.nn.functional.normalize(enc.encode_batch(wav16[None].cuda()).squeeze(), dim=0).cpu().numpy()

segs = []   # (trial, who, embedding)
for f in sorted(glob.glob("experiments/experiments_2*/trial_*/audio.wav")):
    conv = f.replace("audio.wav", "conversation.txt")
    try: lines = open(conv).read().splitlines()
    except Exception: continue
    wav, sr = torchaudio.load(f); wav = wav.mean(0)
    wav16 = torchaudio.functional.resample(wav, sr, 16000); n = wav16.numel()
    trial = f.split("/")[-2]
    for l in lines:
        m = re.match(r"\[\s*([0-9.]+)s\] (You|Morty[^:]*): (.*)", l)
        if not m: continue
        t, who, text = float(m.group(1)), m.group(2), m.group(3)
        if who == "You":   # stamp = transcript time ≈ 1.2 s after the speech ended
            dur = max(1.0, min(6.0, len(text) / 12.0)); a, b = t - 1.2 - dur, t - 1.2
        else:              # stamp = playback start
            dur = max(1.0, min(8.0, len(text) / 13.0)); a, b = t + 0.3, t + 0.3 + dur
        a, b = int(max(0, a) * 16000), int(min(n / 16000, b) * 16000)
        if b - a < 16000: continue
        clip = wav16[a:b]
        if clip.abs().max() < 0.01: continue
        segs.append((trial, "Morty" if who.startswith("Morty") else "You", emb(clip)))
trials = sorted({s[0] for s in segs}); half = set(trials[::2])
ref = np.mean([e for tr, w, e in segs if w == "Morty" and tr in half], axis=0); ref /= np.linalg.norm(ref)
sm = [float(e @ ref) for tr, w, e in segs if w == "Morty" and tr not in half]
sy = [float(e @ ref) for tr, w, e in segs if w == "You" and tr not in half]
print(f"segments: Morty {sum(1 for s in segs if s[1]=='Morty')}, You {sum(1 for s in segs if s[1]=='You')}; enrolled from {len(half)} trials, tested on {len(trials)-len(half)}")
print(f"cos to Morty's fingerprint — Morty: min {min(sm):.2f} median {np.median(sm):.2f}  |  You: max {max(sy):.2f} median {np.median(sy):.2f}")
for thr in (0.3, 0.4, 0.5, 0.6):
    fm = sum(1 for v in sm if v < thr); fy = sum(1 for v in sy if v >= thr)
    print(f"threshold {thr}: Morty taken for human {fm}/{len(sm)}, human taken for Morty {fy}/{len(sy)}")
np.save("/home/deep/morty_ws/models/morty_voice_ecapa.npy", ref); print("fingerprint saved")
