"""Full-rate joint data of a trial, from its ROS bag (2026-09-28, Majd: "do
we also have the velocity and acceleration of every joint at every moment?").

    python3 -m vla_experiments.joints_full <trial dir> [...]
    (or `vla_joints <trial dir> ...` after a colcon build of vla_experiments)

The recorder's steps.npz is sampled at 10 Hz and holds joint position and
velocity only. The trial's bag holds the arm's /joint_states at the driver's
rate (about 480 Hz: position, velocity and motor current for all six joints)
and the trajectory controller's state (about 85 Hz: the COMMANDED position,
velocity and acceleration and the tracking error). This tool writes, next to
the bag:

* ``joints_full.npz`` — every arm sample: ``t`` (s on the trial's episode
  clock when a VLA episode exists, else from the first sample), ``stamp``
  (ROS s), ``pos`` / ``vel`` / ``acc`` / ``current`` (N,6, joints in
  ARM_JOINTS order: pan, lift, elbow, wrist 1-3; rad, rad/s, rad/s^2, A), and
  the controller's ``ctrl_t`` / ``ctrl_stamp`` / ``ctrl_ref_pos`` /
  ``ctrl_ref_vel`` / ``ctrl_ref_acc`` (NaN where a message carried none) /
  ``ctrl_err_pos``;
* ``joints_full.csv`` — the arm samples as plain columns, for plotting;

and, when the trial has a VLA episode, adds two per-step arrays to its
steps.npz, taken at each step's own joint sample (joint_stamp):
``joint_acc`` (N,6) and ``joint_current`` (N,6). Nothing else in steps.npz
changes; the first rewrite keeps steps.npz.orig_<date>.

Acceleration: the driver reports none, so it is the time derivative of the
measured velocity (central differences on the samples' own stamps), smoothed
by a centred 20 ms moving average. Measured on trial 083: 99 % of samples
below 1.4-2.5 rad/s^2; the largest 20 ms peaks (up to ~13 rad/s^2 on the
elbow and wrist 1) sit at the starts and stops of motions. The controller's
commanded acceleration is kept as recorded but is NOT a complete signal:
only some trajectories carry accelerations (6 % of the messages in 083).

The bag's /joint_states also carries the gripper adapter's two-joint
messages and lists the arm joints in the driver's order (lift, elbow, wrist
1-3, pan): arm messages are picked and reordered BY NAME.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .raw_dataset import ARM_JOINTS

SMOOTH_S = 0.020          # moving-average window of the acceleration
CTRL_TOPIC = "/scaled_joint_trajectory_controller/controller_state"


def _stamp(h) -> float:
    return float(h.stamp.sec) + float(h.stamp.nanosec) * 1e-9


def _ordered(names, values, order):
    names = list(names)
    if len(values) != len(names):
        return None
    try:
        return [float(values[names.index(j)]) for j in order]
    except ValueError:
        return None


def read_bag(bag_dir: Path) -> dict:
    """Arm joint samples and controller states from a trial bag."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics = [t for t in ("/joint_states", CTRL_TOPIC) if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    cls = {t: get_message(types[t]) for t in topics}
    js = {"stamp": [], "pos": [], "vel": [], "cur": []}
    cs = {"stamp": [], "ref_pos": [], "ref_vel": [], "ref_acc": [], "err_pos": []}
    nan6 = [float("nan")] * 6
    while reader.has_next():
        topic, data, _t = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        if topic == "/joint_states":
            if "shoulder_pan_joint" not in msg.name:
                continue                      # the gripper adapter's messages
            p = _ordered(msg.name, msg.position, ARM_JOINTS)
            if p is None:
                continue
            js["stamp"].append(_stamp(msg.header))
            js["pos"].append(p)
            js["vel"].append(_ordered(msg.name, msg.velocity, ARM_JOINTS) or nan6)
            js["cur"].append(_ordered(msg.name, msg.effort, ARM_JOINTS) or nan6)
        else:
            names = list(msg.joint_names)
            cs["stamp"].append(_stamp(msg.header))
            cs["ref_pos"].append(_ordered(names, msg.reference.positions, ARM_JOINTS) or nan6)
            cs["ref_vel"].append(_ordered(names, msg.reference.velocities, ARM_JOINTS) or nan6)
            cs["ref_acc"].append(_ordered(names, msg.reference.accelerations, ARM_JOINTS) or nan6)
            cs["err_pos"].append(_ordered(names, msg.error.positions, ARM_JOINTS) or nan6)
    out = {k: np.asarray(v, dtype=np.float64) for k, v in js.items()}
    out.update({"ctrl_" + k: np.asarray(v, dtype=np.float64) for k, v in cs.items()})
    if len(out["stamp"]):
        order = np.argsort(out["stamp"], kind="stable")
        keep = np.r_[True, np.diff(out["stamp"][order]) > 0]      # drop repeated stamps
        for k in ("stamp", "pos", "vel", "cur"):
            out[k] = out[k][order][keep]
    return out


def acceleration(stamp: np.ndarray, vel: np.ndarray, smooth_s: float = SMOOTH_S) -> np.ndarray:
    """d(vel)/dt on the samples' own stamps, centred moving average."""
    if len(stamp) < 3:
        return np.full_like(vel, np.nan)
    acc = np.gradient(vel, stamp, axis=0)
    dt = float(np.median(np.diff(stamp)))
    k = max(1, int(round(smooth_s / dt))) if dt > 0 else 1
    if k > 1:
        ker = np.ones(k) / k
        acc = np.stack([np.convolve(acc[:, j], ker, mode="same") for j in range(acc.shape[1])], axis=1)
    return acc


def process(trial_dir: Path, write_csv: bool = True) -> dict:
    """Write joints_full.npz/.csv and the per-step arrays. Returns a summary."""
    trial_dir = Path(trial_dir)
    bags = sorted(p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("bag") and (p / "metadata.yaml").exists())
    if not bags:
        return {"trial": trial_dir.name, "skipped": "no closed bag (bag/metadata.yaml)"}
    d = read_bag(bags[0])
    if len(d["stamp"]) < 3:
        return {"trial": trial_dir.name, "skipped": "no arm /joint_states in the bag"}
    acc = acceleration(d["stamp"], d["vel"])
    steps_p, meta_p = trial_dir / "steps.npz", trial_dir / "meta.json"
    t0 = None
    if meta_p.exists():
        try:
            t0 = float(json.loads(meta_p.read_text()).get("started_ros"))
        except Exception:
            t0 = None
    if t0 is None:
        t0 = float(d["stamp"][0])
    full = {"t": d["stamp"] - t0, "stamp": d["stamp"], "pos": d["pos"], "vel": d["vel"], "acc": acc,
            "current": d["cur"], "ctrl_t": d["ctrl_stamp"] - t0 if len(d["ctrl_stamp"]) else d["ctrl_stamp"],
            "ctrl_stamp": d["ctrl_stamp"], "ctrl_ref_pos": d["ctrl_ref_pos"], "ctrl_ref_vel": d["ctrl_ref_vel"],
            "ctrl_ref_acc": d["ctrl_ref_acc"], "ctrl_err_pos": d["ctrl_err_pos"],
            "joint_names": np.array(ARM_JOINTS), "info": np.array(
                "units: rad, rad/s, rad/s^2, A; t = s on the episode clock (meta.json started_ros) when a VLA "
                "episode exists; acc = d(vel)/dt, centred 20 ms moving average; ctrl_* = the trajectory "
                "controller's reference (commanded) values, ctrl_ref_acc NaN where a message carried none")}
    np.savez_compressed(trial_dir / "joints_full.npz", **full)
    if write_csv:
        short = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
        cols = (["t_s", "stamp_s"] + [f"pos_{j}_rad" for j in short] + [f"vel_{j}_rad_s" for j in short]
                + [f"acc_{j}_rad_s2" for j in short] + [f"current_{j}_A" for j in short])
        table = np.column_stack([full["t"], full["stamp"], full["pos"], full["vel"], full["acc"], full["current"]])
        fmt = ["%.4f", "%.6f"] + ["%.6f"] * 12 + ["%.4f"] * 6 + ["%.4f"] * 6
        np.savetxt(trial_dir / "joints_full.csv", table, delimiter=",", header=",".join(cols), comments="", fmt=fmt)
    summary = {"trial": trial_dir.name, "samples": int(len(d["stamp"])),
               "rate_hz": round(float(1.0 / np.median(np.diff(d["stamp"]))), 1),
               "ctrl_samples": int(len(d["ctrl_stamp"])),
               "ctrl_acc_filled": int(np.sum(np.any(np.abs(np.nan_to_num(d["ctrl_ref_acc"])) > 1e-9, axis=1)))
               if len(d["ctrl_stamp"]) else 0,
               "acc_p99": [round(float(v), 2) for v in np.nanpercentile(np.abs(acc), 99, axis=0)]}
    if steps_p.exists():
        with np.load(steps_p) as z:
            steps = {k: z[k] for k in z.files}
        ts = steps["joint_stamp"] if "joint_stamp" in steps else steps["stamp_ros"]
        steps["joint_acc"] = np.stack([np.interp(ts, d["stamp"], acc[:, j], left=np.nan, right=np.nan) for j in range(6)], 1)
        steps["joint_current"] = np.stack([np.interp(ts, d["stamp"], d["cur"][:, j], left=np.nan, right=np.nan) for j in range(6)], 1)
        tag = ".orig_" + _dt.date.today().strftime("%Y%m%d")
        if not (trial_dir / ("steps.npz" + tag)).exists():
            shutil.copy2(steps_p, trial_dir / ("steps.npz" + tag))
        np.savez_compressed(steps_p, **steps)
        # the per-step velocity the recorder stored vs the bag's, at the same sample
        # (only v2 steps carry the joint sample's own stamp; only inside the bag's span)
        vb = np.stack([np.interp(ts, d["stamp"], d["vel"][:, j], left=np.nan, right=np.nan) for j in range(6)], 1)
        diff = np.abs(vb - steps["joint_vel"])
        summary["step_vel_vs_bag_max"] = (round(float(np.nanmax(diff)), 4)
                                          if "joint_stamp" in steps and np.isfinite(diff).any() else None)
        summary["steps_filled"] = int(np.sum(np.all(np.isfinite(steps["joint_acc"]), axis=1)))
        summary["steps"] = int(len(ts))
    return summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    no_csv = "--no-csv" in argv
    dirs = [a for a in argv if not a.startswith("--")]
    if not dirs:
        print(__doc__)
        return 2
    for a in dirs:
        try:
            print(json.dumps(process(Path(a).expanduser(), write_csv=not no_csv)))
        except Exception as exc:  # noqa: BLE001 - one bad bag must not stop the rest
            print(json.dumps({"trial": Path(a).name, "error": f"{type(exc).__name__}: {exc}"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
