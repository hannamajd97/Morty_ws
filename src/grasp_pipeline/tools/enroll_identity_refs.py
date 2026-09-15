#!/usr/bin/env python3
"""Build identity-check references from UNLABELLED captures of the known objects.

    python3 enroll_identity_refs.py --images /media/.../dataset_capture --out ../grasp_pipeline/models/identity_refs

For captures without label files (the dataset_capture rounds on the USB
stick, 2026-09-15): the trained YOLO marks its own objects at high
confidence, every detection above ``--min-conf`` becomes a crop, and
DINOv2 removes the crops that do not agree with the rest of their class
(a mislabelled box lands far from the class's own cluster). What remains
are photos of each known object from every side it was captured from,
under this camera and this light — the references perception/identity_check.py
compares live detections with. Runs on the CPU by default so it can be
used while the robot stack occupies the card.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder searched recursively for jpg/png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=os.path.join(HERE, "..", "yolo26s-seg_24_04.pt"))
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from ultralytics import YOLO
    import torch

    files = sorted(f for f in glob.glob(os.path.join(a.images, "**", "*.*"), recursive=True)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{len(files)} images")
    yolo = YOLO(a.weights)
    crops: dict = {}
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        r = yolo.predict(img, verbose=False, device=a.device, conf=a.min_conf)[0]
        h, w = img.shape[:2]
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            crops.setdefault(r.names[int(b.cls)], []).append(img[y1:y2, x1:x2])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} images, {sum(len(v) for v in crops.values())} crops")

    # DINOv2 consistency: keep the crops that agree with their class.
    local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    model = (torch.hub.load(local, "dinov2_vits14", source="local", verbose=False) if os.path.isdir(local)
             else torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)).to(a.device).eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed(batch):
        xs = [((cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA)
                .astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1) for c in batch]
        with torch.no_grad():
            f = model(torch.from_numpy(np.stack(xs)).to(a.device)).float()
            return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    for name, cs in sorted(crops.items()):
        step = max(1, len(cs) // (a.per_class * 2))
        cs = cs[::step]
        emb = np.vstack([embed(cs[i:i + 32]) for i in range(0, len(cs), 32)])
        centre = emb.mean(axis=0)
        centre /= np.linalg.norm(centre)
        agree = emb @ centre
        cut = max(0.3, float(np.percentile(agree, 15)))    # drop the 15 % least typical crops
        keep = [i for i in range(len(cs)) if agree[i] >= cut]
        step = max(1, len(keep) // a.per_class)
        keep = keep[::step][: a.per_class]
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        for old in glob.glob(os.path.join(out_dir, "*.png")):
            os.remove(old)
        for j, i in enumerate(keep):
            cv2.imwrite(os.path.join(out_dir, f"{j:04d}.png"), cs[i])
        kept = emb[keep]
        sim = kept @ kept.T
        np.fill_diagonal(sim, -1.0)
        print(f"{name:20s} {len(crops[name]):4d} detections -> {len(keep):3d} references | "
              f"each has a twin at >= {sim.max(axis=1).min():.2f}, centre agreement {agree[keep].min():.2f}..{agree[keep].max():.2f}")


if __name__ == "__main__":
    main()
