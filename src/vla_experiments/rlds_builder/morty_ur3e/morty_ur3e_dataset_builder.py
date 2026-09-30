"""TFDS builder for the Morty UR3e demonstrations (OpenVLA / RLDS).

Drop-in for https://github.com/kpertsch/rlds_dataset_builder :
    cd <this dir> && MORTY_VLA_RAW=/path/to/raw/morty_ur3e tfds build --overwrite

Step layout follows the rlds_dataset_builder example.  Fields OpenVLA uses:
    observation/image        primary camera (fixed Astra)          uint8 (H,W,3)
    observation/wrist_image  wrist RealSense                       uint8 (H,W,3)
    observation/state        POS_EULER: xyz(3) rpy(3) pad(1) gripper(1)  float32 (8,)
    observation/joint_state  JOINT:     joints(6) pad(1) gripper(1)      float32 (8,)
    action                   EEF_POS: dxyz(3) drpy(3) gripper(1)   float32 (7,)
    language_instruction     the task text
Register in OpenVLA as
    "morty_ur3e": {"image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
                   "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
                   "state_obs_keys": ["state"], "state_encoding": StateEncoding.POS_EULER,
                   "action_encoding": ActionEncoding.EEF_POS}
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

IMG = int(os.environ.get("MORTY_VLA_IMG", "256"))


def _load_raw_helpers():
    """Import raw_dataset.py without needing the ROS package installed."""
    cand = [
        Path(os.environ.get("MORTY_VLA_PKG", "")) / "vla_experiments",
        Path.home() / "morty_ws" / "src" / "vla_experiments" / "vla_experiments",
    ]
    for c in cand:
        if (c / "raw_dataset.py").exists():
            sys.path.insert(0, str(c.parent))
            break
    from vla_experiments.raw_dataset import (Episode, derive_actions, select_episodes, state_vectors,
                                             training_segments)
    return Episode, derive_actions, select_episodes, state_vectors, training_segments


def export_options(env=os.environ) -> dict:
    """What this export takes (2026-09-30) — each rule separately:
        MORTY_VLA_OUTCOMES       comma list, default "success"
                                 (success, failure, unknown, interrupted)
        MORTY_VLA_ALLOW_QUALITY  1 = keep recorder-flagged integrity / alignment issues
        MORTY_VLA_ALLOW_REVIEW   1 = keep episodes whose labels are unverified
                                 (every legacy episode) or carry review flags
        MORTY_VLA_INCLUDE        comma list of extra segments exported WITH their
                                 instruction: hold, reset, idle, clarify, assist
    MORTY_VLA_ALL=1 (legacy) = every outcome + quality issues + review
    required, but instruction segmentation is KEPT: before 2026-09-30 it
    also passed keep_idle=True, which put a whole episode under its first
    instruction."""
    legacy = env.get("MORTY_VLA_ALL", "") == "1"
    outcomes = tuple(x.strip() for x in env.get(
        "MORTY_VLA_OUTCOMES", "success,failure,unknown,interrupted" if legacy else "success").split(",") if x.strip())
    return {"outcomes": outcomes,
            "allow_quality": legacy or env.get("MORTY_VLA_ALLOW_QUALITY", "") == "1",
            "allow_review": legacy or env.get("MORTY_VLA_ALLOW_REVIEW", "") == "1",
            "include": tuple(x.strip() for x in env.get("MORTY_VLA_INCLUDE", "").split(",") if x.strip()),
            "legacy_all": legacy}


class MortyUr3e(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(IMG, IMG, 3), dtype=np.uint8, encoding_format="jpeg",
                                                     doc="Fixed Astra RGB."),
                        "wrist_image": tfds.features.Image(shape=(IMG, IMG, 3), dtype=np.uint8, encoding_format="jpeg",
                                                           doc="Wrist RealSense RGB."),
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32,
                                                      doc="EEF xyz(3), rpy(3), pad(1), gripper(1) in base_link."),
                        "joint_state": tfds.features.Tensor(shape=(8,), dtype=np.float32,
                                                            doc="6 joints, pad, gripper."),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32,
                                                   doc="EEF delta xyz(3), delta rpy(3) in base frame, gripper target(1)."),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                    "language_embedding": tfds.features.Tensor(shape=(512,), dtype=np.float32,
                                                               doc="Universal Sentence Encoder embedding (zeros if unavailable)."),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "success": tfds.features.Scalar(dtype=np.bool_),
                }),
            }))

    def _split_generators(self, dl_manager):
        raw = Path(os.environ.get("MORTY_VLA_RAW", str(Path.home() / "morty_ws/vla_experiments/datasets/morty_ur3e"))).expanduser()
        return {"train": self._generate_examples(raw)}

    def _generate_examples(self, raw: Path):
        Episode, derive_actions, select_episodes, state_vectors, training_segments = _load_raw_helpers()
        # MORTY_VLA_RAW may be the dataset dir, an experiments root or one trial folder (see raw_dataset.py).
        opt = export_options()
        if opt["legacy_all"]:
            print("MORTY_VLA_ALL=1: every outcome, quality issue and unverified label is taken; "
                  "instruction segmentation is kept (see export_options)")
        embed = None
        try:
            import tensorflow_hub as hub
            embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        except Exception:
            pass
        for p in select_episodes(raw, opt["outcomes"], opt["allow_quality"], opt["allow_review"], log=print):
            ep = Episode(p)
            sv = state_vectors(ep)
            for k, (lang, sel) in enumerate(training_segments(ep, include=opt["include"])):
                acts = derive_actions(ep, sel)["eef_delta"]    # never across a gap or another instruction
                emb = embed([lang])[0].numpy().astype(np.float32) if embed else np.zeros(512, np.float32)
                steps = []
                last = len(sel) - 1
                for j, i in enumerate(sel):
                    i = int(i)
                    steps.append({
                        "observation": {
                            "image": ep.load_image("exterior", i, (IMG, IMG)),
                            "wrist_image": ep.load_image("wrist", i, (IMG, IMG)),
                            "state": sv["pos_euler8"][i],
                            "joint_state": sv["joint8"][i],
                        },
                        "action": acts[j],
                        "discount": 1.0,
                        "reward": float(j == last),
                        "is_first": j == 0,
                        "is_last": j == last,
                        "is_terminal": j == last,
                        "language_instruction": lang,
                        "language_embedding": emb,
                    })
                yield f"{ep.episode_id}_{k}", {"steps": steps, "episode_metadata": {"file_path": str(p), "success": ep.outcome == "success"}}
