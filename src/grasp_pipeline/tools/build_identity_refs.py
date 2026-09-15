#!/usr/bin/env python3
"""Build the identity-check references from a YOLO training set.

    python3 build_identity_refs.py --dataset /path/to/dataset --out ../grasp_pipeline/models/identity_refs

The dataset is the usual YOLO layout: ``images/**.jpg`` with ``labels/**.txt``
(``class cx cy w h``, normalised) and a ``data.yaml`` naming the classes.
Every labelled box becomes one reference crop under ``<out>/<class>/``,
at most ``--per-class`` per class, spread evenly over the set so all sides
of an object are represented. The pipeline's identity check compares each
live detection with these crops (see perception/identity_check.py).
"""
import argparse
import glob
import os

import cv2
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--margin", type=float, default=0.05, help="box margin as a fraction of its size")
    a = ap.parse_args()
    with open(os.path.join(a.dataset, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    per_class = {n: [] for n in names}
    for img_path in sorted(glob.glob(os.path.join(a.dataset, "**", "images", "**", "*.*"), recursive=True)):
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lab_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lab_path):
            continue
        with open(lab_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = [float(v) for v in parts[1:5]]
                per_class[names[cls]].append((img_path, cx, cy, w, h))
    for name, boxes in per_class.items():
        if not boxes:
            print(f"{name}: no boxes")
            continue
        step = max(1, len(boxes) // a.per_class)
        chosen = boxes[::step][: a.per_class]
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for i, (img_path, cx, cy, w, h) in enumerate(chosen):
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            mw, mh = w * a.margin, h * a.margin
            x1, x2 = int(max(0, (cx - w / 2 - mw) * W)), int(min(W, (cx + w / 2 + mw) * W))
            y1, y2 = int(max(0, (cy - h / 2 - mh) * H)), int(min(H, (cy + h / 2 + mh) * H))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img[y1:y2, x1:x2])
            n += 1
        print(f"{name}: {n} references from {len(boxes)} boxes")


if __name__ == "__main__":
    main()
