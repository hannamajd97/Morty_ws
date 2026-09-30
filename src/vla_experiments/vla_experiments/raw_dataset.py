"""Raw episode format written by episode_recorder and read by the converters.

Layout — one directory per episode, and that directory is the trial's own
folder: since 2026-09-24 the episode is written INSIDE its trial folder next to
bag/ (one experiment, one folder), with or without MMHRI evidence (09-25);
the 15 episodes recorded before (09-18 … 09-23) were moved into their trial
folders on 2026-09-25 (undo manifest in ~/morty_ws/backups/). Only an episode
started from the CLI (vla_episode start) keeps the old dataset layout:

    experiments/experiments_24_09_2026/trial_070_LA_S4/      (episode_id = trial_070_LA_S4)
        meta.json              task, instructions, outcome, events, quality, intrinsics, astra TF
        steps.npz              per-step arrays (see STEP_KEYS)
        images/exterior/000000.jpg ...   fixed Astra view (JPEG, native resolution)
        images/wrist/000000.jpg ...      wrist RealSense view
        bag/ audio.wav trial.json ...    the MMHRI record (trial.json names vla_episode)
    vla_experiments/datasets/morty_ur3e/episode_000012/       (CLI-started episodes)

`list_episodes(root)` finds both: it walks `root` for every folder holding
meta.json + steps.npz, so `--raw ~/morty_ws/experiments` exports the whole
campaign and `--raw .../datasets/morty_ur3e` only CLI-started episodes.

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
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# One source for the codes (recording_contract.py); 0-2 are the v2 codes.
from .recording_contract import (OUTCOMES, SEGMENT_ASSIST, SEGMENT_CLARIFY, SEGMENT_HOLD,  # noqa: F401
                                 SEGMENT_IDLE, SEGMENT_NAMES, SEGMENT_RESET, SEGMENT_TASK)

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
    "gripper",            # (N,)   0 = closed .. 1 = open, from the gripper joint. NOT a live finger
                          #        position: the adapter reports 0 or 1 once an open/close has
                          #        FINISHED, 0.5-1.7 s after the command (a close on an object ends
                          #        when the fingers stall). Actions use gripper_cmd (derive_actions).
    "gripper_cmd",        # (N,)   last commanded gripper target, 1 = open / 0 = close, from the adapter's command event, NaN if none yet
    "grasp_detected",     # (N,)   UR tool digital input (pin 16), -1 if unknown.
                          #        DO NOT USE as a feature or label (2026-09-28): the pin is
                          #        HIGH whenever the fingers stall — on an object OR on each
                          #        other — so it reads 1 on every empty closed gripper (1.00 over
                          #        all of trial 085, which grasped nothing). Kept as a raw record only.
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
    "segment",            # (N,)   recording_contract SEGMENT_* (idle/task/reset; v3: hold/clarify/assist)
    # --- v3 (2026-09-30): alignment evidence and the executor's own phase
    "img_src_age_exterior",  # (N,) tick minus the frame's capture stamp (valid when the clock is comparable)
    "img_src_age_wrist",     # (N,)
    "img_repeat_exterior",   # (N,) 1 = the same frame (source stamp) as the previous step
    "img_repeat_wrist",      # (N,)
    "camera_skew",           # (N,) exterior minus wrist capture stamp [s]
    "gripper_age",           # (N,) seconds since the last gripper joint sample (apart from the arm)
    "task_phase",            # (N,) index into meta["task_phase_names"] (executor task_state), -1 = none
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
    # 2026-09-30: the router and the model say drop_left / drop_right (the
    # executor's drop_left_side ...); only "left" / "right" were mapped, so
    # pick_and_place texts read "put it on drop_left"
    _d = dest.replace("drop_", "").replace("_side", "")
    side = {"right": "the right side", "left": "the left side", "home": "home"}.get(_d, dest)
    # 2026-09-30 (trial 135: "pick up the blue vitamin c" for the blue-capped
    # Multivitamin): "vitamin_c" is the detector's class for EVERY tube, not
    # the product — it is a vitamin tube; the colour / position say which.
    if str(obj or "").strip().lower().replace(" ", "_") == "vitamin_c":
        obj = "vitamin_tube"
    name = (f"{colour} {obj}" if colour and obj else str(obj or "")).replace("_", " ").strip()
    # 2026-09-28 (Majd's trial 082: "The right most vitamin tube." was trained
    # as "pick up the vitamin tube" — with two tubes the instruction must say
    # which): the position / ordinal the intent resolved goes into the text.
    position = str(attrs.get("position") or "").strip().lower()
    try:
        ordinal = int(attrs.get("ordinal") or 0)
    except (TypeError, ValueError):
        ordinal = 0
    if name and ordinal > 0:
        nth = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(ordinal, f"{ordinal}th")
        side_from = str(attrs.get("ordinal_from") or "left").strip().lower() or "left"
        name = f"{nth} {name} from the {side_from}"
    elif name and position:
        name = f"{position} {name}"
    thing = f"the {name}" if name else "the object"
    a = (action or "").lower()
    # 2026-09-30 (recording contract v1): a compound's "then" is part of the
    # instruction on EVERY action that can carry it (trial 126:
    # open_gripper + then=release_home was trained as "open the gripper").
    then = str(attrs.get("then") or "").strip().lower()
    if dest in ("here", "in_place", "same place", "same_place") and a in ("confirm_drop", "pick_and_place", "place"):
        base = f"pick up {thing} and put it back down" if a == "pick_and_place" else "put it back down here"
        return base + (" and go back home" if then == "release_home" else "")
    if a in ("release", "open_gripper") and then == "release_home":
        return "open the gripper and go back home"
    if a == "go_home" and then in ("release", "release_home"):
        return "bring it home and hand it over"
    if a in ("drop_right", "drop_left") and then in ("release", "release_home"):
        return (f"drop it on the {a.split('_')[1]} side and hand it over"
                + (" and go back home" if then == "release_home" else ""))
    if a == "pick_and_place" and dest and dest != "home" and then in ("release", "release_home"):
        return (f"pick up {thing} and put it on {side} and hand it over"
                + (" and go back home" if then == "release_home" else ""))
    if a in ("pick", "pick_and_place"):
        return f"pick up {thing}" + (f" and put it on {side}" if dest and dest != "home" else "")
    if a in ("drop_right", "drop_left"):
        # 2026-09-30 (trial 147, S7 "Drop right position." with an empty
        # gripper): the arm only went to the drop pose — nothing was dropped
        if attrs.get("empty_gripper"):
            return f"go to the {a.split('_')[1]} side"
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
    if a.startswith("drop_") and a.endswith(("_release", "_release_home")):
        # 2026-09-28: the instant path's compounds (drop_right_release_home …),
        # worded like the same answer through the LLM (confirm_drop + then).
        side = "right" if a.startswith("drop_right") else "left"
        return f"drop it on the {side} side" + (", hand it over and go back home"
                                                  if a.endswith("_home") else " and hand it over")
    if a == "go_home_open":
        return "bring it home and hand it over"
    if a == "confirm_drop":
        # The answer to "where should I drop it?" through the LLM: destination
        # "drop_right" / "right" / "home", then "" / "release" / "release_home".
        d = dest.replace("drop_", "").replace("_side", "")
        text = {"right": "drop it on the right side", "left": "drop it on the left side",
                "home": "bring it home"}.get(d, "drop it")
        then = str(attrs.get("then") or "").strip().lower()
        if then == "release":
            text += " and hand it over"
        elif then == "release_home":
            text += ", hand it over and go back home"
        return text
    if a in ("look_left", "look_right"):
        return f"look to the {a.split('_')[1]}"
    if a == "look_up":
        return "look up"
    if a in ("go_to_ready", "go_to_start"):
        return f"go to the {a.split('_')[2]} position"
    if a == "sleep":
        return "go to the sleep position"
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
def _fill_tcp_from_ur(s: Dict[str, np.ndarray]) -> int:
    """2026-09-28: the TF ee pose is NaN where the recorder rejected a stale
    transform (7-49 steps in each trial of 2026-09-28), and pos_euler8 /
    eef_delta carried those NaNs into the export. The driver's own TCP pose
    (ur_tcp, same flange as tool0: < 0.4 mm apart at rest) has no gaps, so
    a NaN step gets ur_tcp moved by the fixed tool0 -> ee offset measured on
    the episode's valid steps. Returns how many steps were filled (0 when
    nothing is missing or ur_tcp is absent — then nothing changes)."""
    need = ("tcp_pos", "tcp_quat", "tool0_pos", "tool0_quat", "ur_tcp_pos", "ur_tcp_quat")
    if any(k not in s for k in need):
        return 0
    bad = np.isnan(s["tcp_pos"]).any(1) | np.isnan(s["tcp_quat"]).any(1)
    ok = ~bad & ~np.isnan(s["tool0_pos"]).any(1) & ~np.isnan(s["tool0_quat"]).any(1)
    fill = bad & ~np.isnan(s["ur_tcp_pos"]).any(1) & ~np.isnan(s["ur_tcp_quat"]).any(1)
    if not fill.any() or not ok.any():
        return 0

    def rot(q):   # xyzw -> 3x3
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def qmul(a, b):
        ax, ay, az, aw = a; bx, by, bz, bw = b
        return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])

    def qinv(q):
        return np.array([-q[0], -q[1], -q[2], q[3]])

    # tool0 -> ee offset in the tool0 frame (constant: a rigid gripper)
    p_off = np.median([rot(s["tool0_quat"][i]).T @ (s["tcp_pos"][i] - s["tool0_pos"][i]) for i in np.where(ok)[0]], axis=0)
    i0 = int(np.where(ok)[0][0])
    q_off = qmul(qinv(s["tool0_quat"][i0]), s["tcp_quat"][i0])
    tcp_pos, tcp_quat = s["tcp_pos"].copy(), s["tcp_quat"].copy()
    for i in np.where(fill)[0]:
        q = s["ur_tcp_quat"][i]
        tcp_pos[i] = s["ur_tcp_pos"][i] + rot(q) @ p_off
        tcp_quat[i] = qmul(q, q_off)
    s["tcp_pos"], s["tcp_quat"] = tcp_pos, tcp_quat
    return int(fill.sum())


class Episode:
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path / "meta.json") as f:
            self.meta = json.load(f)
        with np.load(self.path / "steps.npz") as z:
            self.steps: Dict[str, np.ndarray] = {k: z[k] for k in z.files}
        self.n = int(self.steps["t"].shape[0])
        self.tcp_filled = _fill_tcp_from_ur(self.steps)

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

    @property
    def label_source(self) -> str:
        return str(self.meta.get("label_source") or "legacy_intent_arrival")

    @property
    def review_required(self) -> bool:
        """True unless the labels rest on recorded execution evidence and the
        recorder found nothing to review (v3 "semantic"); every legacy
        episode is unverified."""
        sem = (self.meta.get("quality") or {}).get("semantic")
        return True if not isinstance(sem, dict) else bool(sem.get("review_required", True))

    def segment_codes(self) -> np.ndarray:
        """SEGMENT_* per step; a v1 episode is all task."""
        seg = self.steps.get("segment")
        return seg.astype(int) if seg is not None else np.full(self.n, SEGMENT_TASK, dtype=int)

    def instruction_idx(self) -> np.ndarray:
        idx = self.steps.get("instruction_idx")
        return idx.astype(int) if idx is not None else np.zeros(self.n, dtype=int)


def _outcome(meta: dict) -> str:
    o = meta.get("outcome")
    if o in OUTCOMES:
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


INCLUDABLE = {"hold": SEGMENT_HOLD, "reset": SEGMENT_RESET, "idle": SEGMENT_IDLE,
              "clarify": SEGMENT_CLARIFY, "assist": SEGMENT_ASSIST}


# ------------------------------------------------------ step alignment ----
# 2026-09-30 (the review of 30 Sep): a late frame or a stale gripper sample
# spoils ONE step, not the demonstration. Before, any such step anywhere —
# 1 of 462 in trial 146, outside every task — made the whole episode
# ineligible. Now the steps outside the tolerances are left out of training
# (training_segments) and only episode-wide timing problems (step gaps,
# dropped ticks, a frozen stream, incomparable clocks, gripper lag) or too
# many bad steps inside tasks block an episode. The recorder writes its
# limits into meta quality.alignment.limits; these are its defaults.
ALIGN_LIMITS = {"max_image_age_s": 0.5, "max_camera_skew_s": 0.35, "max_gripper_age_s": 2.0}
# the recorder's wording of the per-step checks (its alignment "issues"
# before 2026-09-30 mixed them with the episode-wide ones)
PER_STEP_ISSUE_RE = re.compile(r"^(?:exterior|wrist) image (?:received|CAPTURED) up to "
                               r"|^cameras up to .* apart|^gripper state up to .* old")
MAX_BAD_TASK_FRACTION = 0.10


def step_misaligned(steps: Dict[str, np.ndarray], alignment: Optional[dict] = None) -> np.ndarray:
    """(N,) bool — the steps outside the timing tolerances: an image received
    or (with comparable clocks) captured longer than max_image_age_s before
    its step, the two cameras' capture stamps more than max_camera_skew_s
    apart, the gripper sample older than max_gripper_age_s. A missing value
    is not a violation (missing images are counted by the integrity check)."""
    alignment = alignment or {}
    lim = {**ALIGN_LIMITS, **(alignment.get("limits") or {})}
    images = alignment.get("images") or {}
    n = len(steps["t"]) if "t" in steps else len(next(iter(steps.values())))
    bad = np.zeros(n, dtype=bool)

    def over(key, limit, absolute=False):
        v = steps.get(key)
        if v is None:
            return np.zeros(n, dtype=bool)
        v = np.asarray(v, dtype=float)
        v = np.abs(v) if absolute else v
        return np.nan_to_num(v, nan=-np.inf) > limit

    for cam in ("exterior", "wrist"):
        bad |= over(f"img_age_{cam}", lim["max_image_age_s"])
        if (images.get(cam) or {}).get("clock_comparable", True):
            bad |= over(f"img_src_age_{cam}", lim["max_image_age_s"])
    bad |= over("camera_skew", lim["max_camera_skew_s"], absolute=True)
    bad |= over("gripper_age", lim["max_gripper_age_s"])
    return bad


def alignment_verdict(steps: Dict[str, np.ndarray], segment: np.ndarray, alignment: dict) -> dict:
    """The episode's alignment verdict from its steps and the recorder's
    alignment record: {"ok", "issues" (episode-wide, block training),
    "notes", "misaligned_steps", "misaligned_task_steps"}. The per-step
    wording in ``alignment["issues"]`` (PER_STEP_ISSUE_RE) is replaced by the
    count of steps left out; more than MAX_BAD_TASK_FRACTION of the task
    steps (at least 4) is an episode-wide issue."""
    bad = step_misaligned(steps, alignment)
    task = np.asarray(segment).astype(int) == SEGMENT_TASK
    n_bad, n_task_bad, n_task = int(bad.sum()), int((bad & task).sum()), int(task.sum())
    issues = [i for i in (alignment.get("issues") or []) if not PER_STEP_ISSUE_RE.search(i)]
    notes = []
    if n_bad:
        line = (f"{n_bad} step(s) outside the timing tolerances ({n_task_bad} of {n_task} task steps)")
        if n_task_bad > max(3, MAX_BAD_TASK_FRACTION * n_task):
            issues.append(line + " — too many to leave out")
        else:
            notes.append(line + " — left out of training")
    return {"ok": not issues, "issues": issues, "notes": notes,
            "misaligned_steps": n_bad, "misaligned_task_steps": n_task_bad}


def training_segments(ep: "Episode", keep_idle: bool = False, include=(),
                      drop_misaligned: bool = True) -> List[Tuple[str, np.ndarray]]:
    """[(instruction text, step indices)] — one entry per instruction: the
    steps of segment TASK labelled with it, plus the segments named in
    ``include`` (hold / reset / clarify / assist steps attached to that
    instruction by the labeller; "idle" attaches the unlabelled steps before
    an instruction to it). Segmentation by instruction is kept under every
    option; a slice may have gaps where an excluded segment was, and
    ``derive_actions(ep, sel)`` never derives an action across such a gap.
    ``drop_misaligned`` (default) leaves out the steps outside the timing
    tolerances (step_misaligned) — they become gaps the same way.

    keep_idle=True is the pre-09-24 behaviour (the whole episode under its
    first instruction) and is kept only for an explicit legacy request —
    the exporters no longer use it (it was what MORTY_VLA_ALL did)."""
    if keep_idle or "segment" not in ep.steps:
        return [(ep.task, np.arange(ep.n))]
    unknown = set(include) - set(INCLUDABLE)
    if unknown:
        raise ValueError(f"unknown segment(s) {sorted(unknown)}; choose from {sorted(INCLUDABLE)}")
    codes = {SEGMENT_TASK} | {INCLUDABLE[k] for k in include if k != "idle"}
    seg = ep.segment_codes(); idx = ep.instruction_idx()
    if "idle" in include:
        idx = idx.copy()
        nxt = -1
        for i in range(ep.n - 1, -1, -1):          # unlabelled idle steps join the next instruction
            if idx[i] >= 0:
                nxt = idx[i]
            elif seg[i] == SEGMENT_IDLE:
                idx[i] = nxt
        codes.add(SEGMENT_IDLE)
    keep = np.ones(ep.n, dtype=bool)
    if drop_misaligned:
        keep = ~step_misaligned(ep.steps, (ep.meta.get("quality") or {}).get("alignment"))
    out = []
    for ins in ep.instructions:
        k = int(ins.get("idx", 0))
        sel = np.nonzero(np.isin(seg, list(codes)) & (idx == k) & keep)[0]
        if sel.size >= 2:
            out.append((str(ins.get("instruction") or ep.task), sel))
    return out


def contiguous_runs(ep: "Episode", sel: np.ndarray, max_gap_s: float = 0.5) -> List[np.ndarray]:
    """``sel`` split where steps are not adjacent or further apart in time
    than ``max_gap_s`` — the pieces within which actions may be derived."""
    if len(sel) == 0:
        return []
    t = ep.steps["t"]
    cut = np.nonzero((np.diff(sel) != 1) | (np.diff(t[sel]) > max_gap_s))[0] + 1
    return [r for r in np.split(sel, cut) if len(r)]


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


def select_episodes(root: Path, outcomes=("success",), allow_quality: bool = False,
                    allow_review: bool = False, log=None) -> List[Path]:
    """The episodes an export takes, each rule separate (2026-09-30; before,
    one switch — MORTY_VLA_ALL — took everything and flattened the labels):
      outcomes       which meta "outcome" values (success / failure /
                     unknown / interrupted)
      allow_quality  keep episodes with recorder-flagged integrity or
                     alignment issues
      allow_review   keep episodes whose labels are not verified by execution
                     evidence (every legacy episode) or carry review flags
    ``log(text)`` is told why each skipped episode was skipped."""
    out = []
    for p in find_episode_dirs(root):
        try:
            ep = Episode(p)
        except Exception as exc:  # noqa: BLE001 — an unreadable episode is reported, not exported
            if log:
                log(f"skipping {p.name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        if ep.outcome not in outcomes:
            if log:
                log(f"skipping {ep.episode_id}: outcome={ep.outcome}")
            continue
        q = ep.meta.get("quality") or {}
        integ = (q.get("integrity") or {}).get("issues", []) + (q.get("alignment") or {}).get("issues", []) \
            if "integrity" in q else [i for i in quality_issues(ep) if not i.startswith("review:")]
        if integ and not allow_quality:
            if log:
                log(f"skipping {ep.episode_id} (quality): " + "; ".join(integ))
            continue
        if ep.review_required and not allow_review:
            if log:
                why = (q.get("semantic") or {}).get("flags") or [f"labels: {ep.label_source} (unverified)"]
                log(f"skipping {ep.episode_id} (needs review): " + "; ".join(why[:3]))
            continue
        out.append(p)
    return out


# ------------------------------------------------------------- actions ----
def derive_actions(ep: Episode, sel: Optional[np.ndarray] = None, max_gap_s: float = 0.5) -> Dict[str, np.ndarray]:
    """Derive per-step action targets from consecutive states.

    With ``sel`` (a training slice), the arrays are those of the steps in
    ``sel`` and "the next step" is the next step OF THE SAME CONTIGUOUS RUN
    (2026-09-30): the last step of a run — before an excluded gap, another
    instruction or a time gap > max_gap_s — gets zero motion and its own
    gripper target instead of the motion to an unrelated frame.

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
    if sel is not None:
        # next index only inside the same run of the slice; run ends map to themselves
        nxt_idx = np.arange(n)
        for run in contiguous_runs(ep, np.asarray(sel, dtype=int), max_gap_s):
            nxt_idx[run[:-1]] = run[1:]
        have_cmd = have_cmd & (nxt_idx != np.arange(n))   # a run end has no command to follow
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

    out = {
        "joint_abs_next": joint_abs_next,
        "joint_delta": joint_delta,
        "eef_delta": eef_delta,
        "gripper_target": grip_target,
    }
    if sel is not None:
        sel = np.asarray(sel, dtype=int)
        out = {k: v[sel] for k, v in out.items()}
    return out


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
