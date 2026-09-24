"""Raw episode format written by episode_recorder and read by the converters.

Layout — one directory per episode. Since 2026-09-24 a trial-bound episode
lives INSIDE its MMHRI trial folder next to bag/ (one experiment, one
folder); a standalone episode keeps the old dataset layout:

    experiments/experiments_24_09_2026/trial_070_LA_S4/      (episode_id = trial_070_LA_S4)
        meta.json              task, instructions, outcome, events, quality, intrinsics, astra TF
        steps.npz              per-step arrays (see STEP_KEYS)
        images/exterior/000000.jpg ...   fixed Astra view (JPEG, native resolution)
        images/wrist/000000.jpg ...      wrist RealSense view
        bag/ audio.wav trial.json ...    the MMHRI record (trial.json names vla_episode)
    vla_experiments/datasets/morty_ur3e/episode_000012/       (CLI-started episodes)

`list_episodes(root)` finds both: it walks `root` for every folder holding
meta.json + steps.npz, so `--raw ~/morty_ws/experiments` exports a whole
campaign and `--raw .../datasets/morty_ur3e` the standalone ones.

Everything a VLA converter needs is here.  Actions are NOT stored; they are
derived from consecutive states by the converters (see `derive_actions`) so
the same raw data can feed joint-space (openpi UR5 style) and EEF-delta
(OpenVLA / Bridge style) targets without re-recording.

Training slices (`training_segments`): an episode may hold several
instructions ("pick up the bottle" ... "drop it on the right side" ... "go
back home"). Each step carries `instruction_idx` and a `segment` code
(idle / task / reset). The converters emit one training episode per
instruction, from the steps of segment `task` only, so conversation, waiting
and the reset after a failure never become training frames — while a hold
the task needs (waiting with the object at the hand-over) stays, because it
happens INSIDE the instruction's span.

Outcome is `success` / `failure` / `unknown` (meta "outcome"; the older
boolean "success" is kept for v1 episodes). `quality_issues(ep)` lists what
the recorder flagged (step gaps, stale poses, gripper lag ...); the
converters skip such episodes unless asked to keep them.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SEGMENT_IDLE, SEGMENT_TASK, SEGMENT_RESET = 0, 1, 2

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
    "gripper_cmd",        # (N,)   last commanded gripper target (0/1) from the adapter's command event, NaN if none yet
    "grasp_detected",     # (N,)   UR tool digital input (pin 16), -1 if unknown
    "img_age_exterior",   # (N,)   seconds between tick and last exterior frame (receive time)
    "img_age_wrist",      # (N,)
    "joint_age",          # (N,)
    # --- v2 (2026-09-24): original stamps, the driver's own TCP, command timing, labels
    "joint_stamp",        # (N,)   /joint_states header stamp of the sample [ROS s]
    "img_stamp_exterior", # (N,)   header stamp of the exterior frame (shuttle clock)
    "img_stamp_wrist",    # (N,)   header stamp of the wrist frame
    "tcp_stamp",          # (N,)   stamp of the TF used for tcp/tool0 (NaN = rejected as stale)
    "ur_tcp_pos",         # (N,3)  /tcp_pose_broadcaster/pose, expressed in the base frame (meta frames)
    "ur_tcp_quat",        # (N,4)
    "ur_tcp_stamp",       # (N,)
    "gripper_cmd_age",    # (N,)   seconds since the last gripper command event
    "instruction_idx",    # (N,)   index into meta["instructions"], -1 = none active
    "phase",              # (N,)   index into meta["phase_names"] (executor phase marks), -1 = none
    "segment",            # (N,)   SEGMENT_IDLE / SEGMENT_TASK / SEGMENT_RESET
]

V1_STEP_KEYS = STEP_KEYS[:15]


def canonical_instruction(action: str, obj=None, attrs=None, fallback: str = "") -> str:
    """The language instruction the export trains on, built from the
    resolved intent rather than the raw transcript — a misheard "grass for
    me is a bottle" still trains "pick up the bottle". Unknown intents fall
    back to the transcript."""
    attrs = attrs or {}
    colour = str(attrs.get("colour") or "").strip()
    dest = str(attrs.get("destination") or "").strip().lower()
    side = {"right": "the right side", "left": "the left side", "home": "home"}.get(dest, dest)
    name = (f"{colour} {obj}" if colour and obj else str(obj or "")).replace("_", " ").strip()
    thing = f"the {name}" if name else "the object"
    a = (action or "").lower()
    if a in ("pick", "pick_and_place"):
        return f"pick up {thing}" + (f" and put it on {side}" if dest and dest != "home" else "")
    if a in ("drop_right", "drop_left"):
        return f"drop it on the {a.split('_')[1]} side"
    if a == "drop_held":
        return "drop it"
    if a == "place":
        return f"place it on {side}" if dest else "place it"
    if a == "go_home":
        return "go back home"
    if a in ("release", "open_gripper"):
        return "open the gripper"
    if a == "close_gripper":
        return "close the gripper"
    if a == "release_home":
        return "open the gripper and go back home"
    if a == "carry_release":
        return "go back home and hand it over" if dest == "home" else f"carry it to {side} and hand it over"
    return fallback or a.replace("_", " ")

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

    @property
    def episode_id(self) -> str:
        return str(self.meta.get("episode_id") or self.path.name)

    @property
    def outcome(self) -> str:
        return _outcome(self.meta)

    @property
    def instructions(self) -> List[dict]:
        """v2: one per motion intent; v1: the single task over the whole episode."""
        ins = self.meta.get("instructions")
        if ins:
            return list(ins)
        return [{"idx": 0, "instruction": self.task, "action_intent": self.meta.get("intent", ""), "t": 0.0}]

    def segment_codes(self) -> np.ndarray:
        """SEGMENT_* per step; a v1 episode is all task."""
        seg = self.steps.get("segment")
        return seg.astype(int) if seg is not None else np.full(self.n, SEGMENT_TASK, dtype=int)

    def instruction_idx(self) -> np.ndarray:
        idx = self.steps.get("instruction_idx")
        return idx.astype(int) if idx is not None else np.zeros(self.n, dtype=int)


def _outcome(meta: dict) -> str:
    o = meta.get("outcome")
    if o in ("success", "failure", "unknown"):
        return o
    s = meta.get("success")                       # v1: bool, default True
    return "unknown" if s is None else ("success" if s else "failure")


def quality_issues(ep: "Episode") -> List[str]:
    """What the recorder flagged (v2 meta "quality"); for v1 episodes, step
    gaps computed here so an old episode is judged by the same rule."""
    q = ep.meta.get("quality")
    if isinstance(q, dict):
        return list(q.get("issues", []))
    t = ep.steps["t"]
    issues = []
    if len(t) > 1:
        dt = np.diff(t)
        if (dt > 0.5).any():
            issues.append(f"{int((dt > 0.5).sum())} step gap(s) > 0.5 s (max {dt.max():.2f} s)")
    tcp = ep.steps["tcp_pos"]
    if ep.n > 10 and np.isfinite(tcp).all() and tcp.std(axis=0).max() < 1e-4 \
            and np.abs(ep.steps["joint_pos"] - ep.steps["joint_pos"][0]).max() > 0.2:
        issues.append("ee pose frozen while the joints moved")
    return issues


def training_segments(ep: "Episode", keep_idle: bool = False) -> List[Tuple[str, np.ndarray]]:
    """[(instruction text, step indices)] — one entry per instruction, the
    steps of segment TASK that belong to it (contiguous runs are kept as
    one slice; a hold inside the instruction stays). keep_idle=True returns
    the whole episode under its first instruction, the pre-09-24 behaviour."""
    if keep_idle or "segment" not in ep.steps:
        return [(ep.task, np.arange(ep.n))]
    seg = ep.segment_codes(); idx = ep.instruction_idx()
    out = []
    for ins in ep.instructions:
        k = int(ins.get("idx", 0))
        sel = np.nonzero((seg == SEGMENT_TASK) & (idx == k))[0]
        if sel.size >= 2:
            out.append((str(ins.get("instruction") or ep.task), sel))
    return out


def find_episode_dirs(root: Path) -> List[Path]:
    """Every folder under `root` (or `root` itself) holding meta.json + steps.npz:
    dataset episode_* folders and MMHRI trial folders alike. bag/ is not descended."""
    root = Path(root).expanduser()
    if (root / "meta.json").exists() and (root / "steps.npz").exists():
        return [root]
    out = []
    for meta in root.rglob("meta.json"):
        p = meta.parent
        if "bag" in p.relative_to(root).parts or not (p / "steps.npz").exists():
            continue
        out.append(p)
    return sorted(out)


def list_episodes(dataset_dir: Path, include_failed: bool = False, include_unknown: bool = False,
                  allow_quality_issues: bool = True) -> List[Path]:
    """Episodes under `dataset_dir` filtered by outcome (success by default;
    failures and unknowns on request) and, when allow_quality_issues is
    False, without recorder-flagged quality issues."""
    out = []
    for p in find_episode_dirs(dataset_dir):
        with open(p / "meta.json") as f:
            meta = json.load(f)
        o = _outcome(meta)
        if o == "failure" and not include_failed:
            continue
        if o == "unknown" and not include_unknown:
            continue
        if not allow_quality_issues and meta.get("quality", {}).get("issues"):
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
        print("usage: vla_dataset_info <dataset dir | experiments dir | trial dir> [--all]")
        return 2
    d = Path(argv[0]).expanduser()
    all_ = "--all" in argv
    eps = list_episodes(d, include_failed=all_, include_unknown=all_)
    total = 0; kept = 0
    print(f"root: {d}")
    for p in eps:
        ep = Episode(p)
        total += ep.n
        a = derive_actions(ep)
        moved = float(np.abs(a["joint_delta"][:, :6]).sum())
        segs = training_segments(ep)
        n_train = int(sum(len(s) for _, s in segs)); kept += n_train
        issues = quality_issues(ep)
        try:
            rel = p.relative_to(d)
        except ValueError:
            rel = p
        print(f"  {rel}: {ep.n:5d} steps ({n_train} training in {len(segs)} instruction(s))  "
              f"{ep.n / ep.meta['fps']:6.1f}s  outcome={ep.outcome}  drops={ep.meta.get('dropped_ticks', 0)}  "
              f"|dq|={moved:.2f}  task='{ep.task}'")
        for text, sel in segs:
            print(f"      [{sel[0]:5d}-{sel[-1]:5d}] '{text}'")
        for i in issues:
            print(f"      ! {i}")
    print(f"{len(eps)} episodes, {total} steps, {kept} training steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
