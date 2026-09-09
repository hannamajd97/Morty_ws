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
    from vla_experiments.raw_dataset import Episode, derive_actions, list_episodes, state_vectors
    return Episode, derive_actions, list_episodes, state_vectors


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
        Episode, derive_actions, list_episodes, state_vectors = _load_raw_helpers()
        embed = None
        try:
            import tensorflow_hub as hub
            embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        except Exception:
            pass
        for p in list_episodes(raw):
            ep = Episode(p)
            acts = derive_actions(ep)["eef_delta"]
            sv = state_vectors(ep)
            lang = ep.task
            emb = embed([lang])[0].numpy().astype(np.float32) if embed else np.zeros(512, np.float32)
            steps = []
            for i in range(ep.n):
                steps.append({
                    "observation": {
                        "image": ep.load_image("exterior", i, (IMG, IMG)),
                        "wrist_image": ep.load_image("wrist", i, (IMG, IMG)),
                        "state": sv["pos_euler8"][i],
                        "joint_state": sv["joint8"][i],
                    },
                    "action": acts[i],
                    "discount": 1.0,
                    "reward": float(i == ep.n - 1),
                    "is_first": i == 0,
                    "is_last": i == ep.n - 1,
                    "is_terminal": i == ep.n - 1,
                    "language_instruction": lang,
                    "language_embedding": emb,
                })
            yield p.name, {"steps": steps, "episode_metadata": {"file_path": str(p), "success": bool(ep.meta.get("success", True))}}
