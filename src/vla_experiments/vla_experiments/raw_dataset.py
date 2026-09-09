"""Raw episode format written by episode_recorder and read by the converters.

Layout (one directory per episode, under <dataset_root>/<dataset_name>/):

    episode_000012/
        meta.json              task, fps, success, frame count, topics, intrinsics, astra TF
        steps.npz              per-step arrays (see STEP_KEYS)
        images/exterior/000000.jpg ...   fixed Astra view (JPEG, native resolution)
        images/wrist/000000.jpg ...      wrist RealSense view

Everything a VLA converter needs is here.  Actions are NOT stored; they are
derived from consecutive states by the converters (see `derive_actions`) so
the same raw data can feed joint-space (openpi UR5 style) and EEF-delta
(OpenVLA / Bridge style) targets without re-recording.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

STEP_KEYS = [
    "t",                  # (N,)   seconds since episode start (recorder clock)
    "stamp_ros",          # (N,)   ROS time of the tick, seconds
    "joint_pos",          # (N,6)  arm joint positions [rad], ARM_JOINTS order
    "joint_vel",          # (N,6)  arm joint velocities [rad/s]
    "joint_cmd",          # (N,6)  controller reference positions (NaN if unavailable)
    "tcp_pos",            # (N,3)  ee frame position in base frame [m]
    "tcp_quat",           # (N,4)  ee frame orientation, xyzw
    "tool0_pos",          # (N,3)  tool0 position in base frame [m]
    "tool0_quat",         # (N,4)
    "gripper",            # (N,)   0 = closed .. 1 = open, from the gripper joint
    "gripper_cmd",        # (N,)   last commanded gripper target (0/1), NaN if unknown
    "grasp_detected",     # (N,)   UR tool digital input (pin 16), -1 if unknown
    "img_age_exterior",   # (N,)   seconds between tick and last exterior frame
    "img_age_wrist",      # (N,)
    "joint_age",          # (N,)
]

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ----------------------------------------------------------------- math ----
def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> roll, pitch, yaw (ZYX convention, REP-103)."""
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    return matrix_to_rpy(quat_to_matrix(q))


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------- I/O ----
class Episode:
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path / "meta.json") as f:
            self.meta = json.load(f)
        with np.load(self.path / "steps.npz") as z:
            self.steps: Dict[str, np.ndarray] = {k: z[k] for k in z.files}
        self.n = int(self.steps["t"].shape[0])

    def image_path(self, cam: str, i: int) -> Path:
        return self.path / "images" / cam / f"{i:06d}.jpg"

    def load_image(self, cam: str, i: int, size: tuple | None = None) -> np.ndarray:
        import cv2
        img = cv2.imread(str(self.image_path(cam, i)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_path(cam, i))
        if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
            img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @property
    def task(self) -> str:
        return self.meta["task"]


def list_episodes(dataset_dir: Path, include_failed: bool = False) -> List[Path]:
    dataset_dir = Path(dataset_dir)
    out = []
    for p in sorted(dataset_dir.glob("episode_*")):
        if not (p / "meta.json").exists() or not (p / "steps.npz").exists():
            continue
        if not include_failed:
            with open(p / "meta.json") as f:
                if not json.load(f).get("success", True):
                    continue
        out.append(p)
    return out


# ------------------------------------------------------------- actions ----
def derive_actions(ep: Episode) -> Dict[str, np.ndarray]:
    """Derive per-step action targets from consecutive states.

    Returns
      joint_abs_next  (N,7)  next-step absolute joint positions + gripper target
      joint_delta     (N,7)  joint_{t+1} - joint_t (wrapped) + gripper target
      eef_delta       (N,7)  [dx dy dz, droll dpitch dyaw] of the ee frame between
                             t and t+1, expressed in the BASE frame, + gripper target
      gripper_target  (N,)   1.0 = open, 0.0 = closed (binary), from next-step state
    The last step repeats the previous action (zero motion) so N is preserved.

    If the controller reference (joint_cmd) was recorded it is preferred over
    the next measured state for joint_abs_next, because it is the real command.
    """
    s = ep.steps
    n = ep.n
    jp = s["joint_pos"]
    jc = s["joint_cmd"]
    have_cmd = np.isfinite(jc).all(axis=1)

    nxt_idx = np.minimum(np.arange(n) + 1, n - 1)
    joint_next = jp[nxt_idx].copy()
    joint_next[have_cmd] = jc[have_cmd]

    grip_state = s["gripper"]
    grip_cmd = s["gripper_cmd"]
    grip_next = grip_state[nxt_idx]
    use_cmd = np.isfinite(grip_cmd)
    grip_target = np.where(use_cmd, grip_cmd, grip_next)
    grip_target = (grip_target > 0.5).astype(np.float32)

    joint_abs_next = np.concatenate([joint_next, grip_target[:, None]], axis=1).astype(np.float32)
    joint_delta = np.concatenate([wrap_pi(joint_next - jp), grip_target[:, None]], axis=1).astype(np.float32)

    pos = s["tcp_pos"]
    quat = s["tcp_quat"]
    dpos = pos[nxt_idx] - pos
    drot = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        j = nxt_idx[i]
        if j == i:
            continue
        R_i = quat_to_matrix(quat[i])
        R_j = quat_to_matrix(quat[j])
        drot[i] = matrix_to_rpy(R_j @ R_i.T)
    eef_delta = np.concatenate([dpos, drot, grip_target[:, None]], axis=1).astype(np.float32)

    return {
        "joint_abs_next": joint_abs_next,
        "joint_delta": joint_delta,
        "eef_delta": eef_delta,
        "gripper_target": grip_target,
    }


def state_vectors(ep: Episode) -> Dict[str, np.ndarray]:
    """State encodings used by the converters."""
    s = ep.steps
    grip = s["gripper"][:, None].astype(np.float32)
    joint7 = np.concatenate([s["joint_pos"].astype(np.float32), grip], axis=1)          # openpi UR5 style
    rpy = np.stack([quat_to_rpy(q) for q in s["tcp_quat"]]).astype(np.float32)
    pos_euler8 = np.concatenate([s["tcp_pos"].astype(np.float32), rpy, np.zeros_like(grip), grip], axis=1)  # OpenVLA POS_EULER
    joint8 = np.concatenate([s["joint_pos"].astype(np.float32), np.zeros_like(grip), grip], axis=1)          # OpenVLA JOINT (7 + gripper)
    return {"joint7": joint7, "pos_euler8": pos_euler8, "joint8": joint8}


# ----------------------------------------------------------------- CLI ----
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: vla_dataset_info <dataset_dir> [--all]")
        return 2
    d = Path(argv[0]).expanduser()
    eps = list_episodes(d, include_failed="--all" in argv)
    total = 0
    print(f"dataset: {d}")
    for p in eps:
        ep = Episode(p)
        total += ep.n
        a = derive_actions(ep)
        moved = float(np.abs(a["joint_delta"][:, :6]).sum())
        print(f"  {p.name}: {ep.n:5d} steps  {ep.n / ep.meta['fps']:6.1f}s  "
              f"success={ep.meta.get('success')}  drops={ep.meta.get('dropped_ticks', 0)}  "
              f"|dq|={moved:.2f}  task='{ep.task}'")
    print(f"{len(eps)} episodes, {total} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
