"""Convert raw Morty episodes to a LeRobot v2.1 dataset for openpi (pi0 / pi0-FAST / pi0.5).

Mirrors openpi/examples/libero/convert_libero_data_to_lerobot.py and the
openpi UR5 example (state = 6 joints + gripper, actions = 7-D joint targets;
openpi's UR5 data config converts absolute -> delta itself).

Run this on the training machine (or a venv) where `lerobot` is installed at
the commit openpi pins (python >= 3.11):

    python convert_to_lerobot.py --raw ~/morty_ws/vla_experiments/datasets/morty_ur3e \
        --repo-id <hf_user>/morty_ur3e --fps 10 --image-size 256

Output keys per frame (openpi expects exactly these in its repack transform):
    image        (H,W,3) uint8   fixed Astra view
    wrist_image  (H,W,3) uint8   wrist RealSense view
    state        (7,)  float32   joint_pos[6] + gripper[0..1]
    actions      (7,)  float32   next joint target[6] + gripper target{0,1}
    task         str             language instruction
Alternative action spaces can be selected with --action-space {joint_abs,joint_delta,eef_delta}.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from .raw_dataset import Episode, derive_actions, list_episodes, state_vectors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="raw dataset dir (contains episode_*/)")
    ap.add_argument("--repo-id", required=True, help="e.g. majd/morty_ur3e (local name if not pushing)")
    ap.add_argument("--root", default=None, help="output root (default: ~/.cache/huggingface/lerobot/<repo-id>)")
    ap.add_argument("--fps", type=int, default=None, help="default: fps from meta.json")
    ap.add_argument("--image-size", type=int, default=256, help="resize both cameras to NxN (0 = keep native)")
    ap.add_argument("--action-space", choices=["joint_abs", "joint_delta", "eef_delta"], default="joint_abs")
    ap.add_argument("--state-space", choices=["joint7", "pos_euler8"], default="joint7")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--push", action="store_true", help="push_to_hub after conversion")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("lerobot is not installed here. Install openpi (uv sync) or lerobot at the pinned commit,\n"
              "then run:  python -m vla_experiments.convert_to_lerobot ...", file=sys.stderr)
        return 1

    eps = list_episodes(Path(a.raw).expanduser(), include_failed=a.include_failed)
    if not eps:
        print("no episodes found", file=sys.stderr)
        return 1
    first = Episode(eps[0])
    fps = a.fps or int(round(first.meta["fps"]))
    size = (a.image_size, a.image_size) if a.image_size else None
    h, w = (size[1], size[0]) if size else (first.meta["cameras"]["wrist"]["height"], first.meta["cameras"]["wrist"]["width"])
    state_dim = 7 if a.state_space == "joint7" else 8

    root = Path(a.root).expanduser() if a.root else None
    if a.overwrite:
        target = root or (Path.home() / ".cache/huggingface/lerobot" / a.repo_id)
        if target.exists():
            shutil.rmtree(target)

    ds = LeRobotDataset.create(
        repo_id=a.repo_id,
        root=root,
        robot_type="ur3e_gripper_3d",
        fps=fps,
        features={
            "image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=2,
    )
    key = {"joint_abs": "joint_abs_next", "joint_delta": "joint_delta", "eef_delta": "eef_delta"}[a.action_space]
    total = 0
    for p in eps:
        ep = Episode(p)
        acts = derive_actions(ep)[key]
        st = state_vectors(ep)[a.state_space]
        for i in range(ep.n):
            ds.add_frame({
                "image": ep.load_image("exterior", i, size),
                "wrist_image": ep.load_image("wrist", i, size),
                "state": st[i].astype(np.float32),
                "actions": acts[i].astype(np.float32),
                "task": ep.task,
            })
        ds.save_episode()
        total += ep.n
        print(f"{p.name}: {ep.n} frames  task='{ep.task}'")
    print(f"done: {len(eps)} episodes, {total} frames -> {ds.root}")
    if a.push:
        ds.push_to_hub(tags=["ur3e", "morty", "vla"], private=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
