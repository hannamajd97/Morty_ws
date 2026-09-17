#!/usr/bin/env python3
"""Record the TCP speed of every arm motion: planned (controller "desired")
against executed ("actual"), from the trajectory controller's own state
stream, and save one plot + CSV per motion.

Why this exists (2026-09-17, Majd: "it still stops, and it does not feel
smooth"): whether a felt hesitation is the PATH (a speed dip at the bend),
the TIME LAW (bang-bang acceleration edges) or the CONTROLLER (executed
lagging planned) cannot be told by feel. One plot per motion tells it.

Run next to the running stack (driver up):
    python3 src/ur3e_gripper_3d_moveit_commander/tools/record_motion_profile.py
Output: experiments/motion_profiles/<HHMMSS>_<n>.png / .csv and a summary
line per motion on stdout. Ctrl-C to stop.

FK is the UR3e flange (tool0) from the standard DH table; the gripper's
offset only adds rotation-induced speed, which is not what we judge here.
"""
import csv
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from control_msgs.msg import JointTrajectoryControllerState

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# UR3e DH (d, a, alpha), Universal Robots parameter sheet.
DH = [(0.15185, 0.0, math.pi / 2), (0.0, -0.24355, 0.0), (0.0, -0.2132, 0.0),
      (0.13105, 0.0, math.pi / 2), (0.08535, 0.0, -math.pi / 2), (0.0921, 0.0, 0.0)]
MOVING = 0.02      # m/s: desired TCP speed above this = "the arm is moving"
QUIET_S = 0.4      # this long below MOVING ends a motion


def fk(q):
    T = np.eye(4)
    for (d, a, alpha), th in zip(DH, q):
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(alpha), math.sin(alpha)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0.0, sa, ca, d],
                          [0.0, 0.0, 0.0, 1.0]])
    return T[:3, 3]


class Recorder(Node):
    def __init__(self, out_dir):
        super().__init__("motion_profile_recorder")
        self.out_dir = out_dir
        self.samples = []          # (t, p_des, p_act, v_des_joint_max)
        self.last_moving_t = None
        self.n = 0
        self.order = None
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointTrajectoryControllerState,
                                 "/scaled_joint_trajectory_controller/controller_state", self.cb, qos)
        self.create_subscription(JointTrajectoryControllerState,
                                 "/scaled_joint_trajectory_controller/state", self.cb, qos)
        self.get_logger().info(f"recording motions to {out_dir} (Ctrl-C to stop)")

    def cb(self, msg):
        if self.order is None:
            try:
                self.order = [list(msg.joint_names).index(j) for j in JOINTS]
            except ValueError:
                self.get_logger().error(f"unexpected joint names: {list(msg.joint_names)}")
                return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        des = getattr(msg, "reference", None) or msg.desired
        act = getattr(msg, "feedback", None) or msg.actual
        if len(des.positions) < 6 or len(act.positions) < 6:
            return
        qd = [des.positions[i] for i in self.order]
        qa = [act.positions[i] for i in self.order]
        p_des, p_act = fk(qd), fk(qa)
        self.samples.append((t, p_des, p_act))
        if len(self.samples) < 2:
            return
        t0, pd0, _ = self.samples[-2]
        dt = t - t0
        v = float(np.linalg.norm(p_des - pd0) / dt) if dt > 0 else 0.0
        if v > MOVING:
            self.last_moving_t = t
        elif self.last_moving_t is not None and t - self.last_moving_t > QUIET_S:
            self.finish()
        if self.last_moving_t is None and len(self.samples) > 200:
            self.samples = self.samples[-200:]     # keep a short pre-roll only

    def finish(self):
        S = self.samples
        self.samples = []
        self.last_moving_t = None
        t = np.array([s[0] for s in S]); pd = np.array([s[1] for s in S]); pa = np.array([s[2] for s in S])
        dt = np.diff(t)
        ok = dt > 1e-4
        vd = np.linalg.norm(np.diff(pd, axis=0), axis=1)[ok] / dt[ok]
        va = np.linalg.norm(np.diff(pa, axis=0), axis=1)[ok] / dt[ok]
        tt = t[1:][ok] - t[0]
        mov = np.where(vd > MOVING)[0]
        if len(mov) < 5:
            return
        a, b = mov[0], mov[-1]
        tt, vd, va = tt[a:b + 1] - tt[a], vd[a:b + 1], va[a:b + 1]
        # smooth the numeric derivative a little (controller state at ~100 Hz)
        k = 5
        sm = lambda x: np.convolve(x, np.ones(k) / k, mode="same")
        vd_s, va_s = sm(vd), sm(va)
        dur = float(tt[-1])
        peak = float(vd_s.max())
        # dips: local minima of the planned speed inside the motion, below 60 % of the peak
        inner = slice(int(0.15 * len(vd_s)), int(0.85 * len(vd_s)))
        dips = []
        for i in range(inner.start + 1, inner.stop - 1):
            if vd_s[i] < vd_s[i - 1] and vd_s[i] <= vd_s[i + 1] and vd_s[i] < 0.6 * peak:
                dips.append((float(tt[i]), float(vd_s[i])))
        lag = float(np.mean(np.abs(va_s - vd_s)))
        path_len = float(np.sum(np.linalg.norm(np.diff(pd[a:b + 2], axis=0), axis=1)))
        self.n += 1
        stem = os.path.join(self.out_dir, f"{datetime.now().strftime('%H%M%S')}_{self.n}")
        with open(stem + ".csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_s", "v_planned_mps", "v_executed_mps"])
            for row in zip(tt, vd, va):
                w.writerow([f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}"])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(tt, vd_s, label="planned (controller desired)", lw=2)
            ax.plot(tt, va_s, label="executed (actual)", lw=1, alpha=0.8)
            for td, vdip in dips:
                ax.axvline(td, color="r", ls="--", alpha=0.5); ax.annotate(f"dip {vdip:.2f}", (td, vdip))
            ax.set_xlabel("s"); ax.set_ylabel("TCP speed m/s")
            ax.set_title(f"motion {self.n}: {dur:.2f} s, path {path_len*100:.0f} cm, peak {peak:.2f} m/s, "
                         f"{len(dips)} dip(s), planned-executed gap {lag*100:.1f} cm/s")
            ax.grid(alpha=0.3); ax.legend()
            fig.tight_layout(); fig.savefig(stem + ".png", dpi=110); plt.close(fig)
        except Exception as e:  # no matplotlib: the CSV is enough
            print(f"[plot skipped: {e}]", file=sys.stderr)
        dip_txt = ", ".join(f"{v:.2f} m/s at {t_:.2f} s" for t_, v in dips) or "none"
        print(f"motion {self.n}: {dur:.2f} s, path {path_len*100:.0f} cm, peak {peak:.2f} m/s, "
              f"dips: {dip_txt}, planned-vs-executed gap {lag*100:.1f} cm/s -> {stem}.png", flush=True)


def main():
    out = os.path.expanduser("~/morty_ws/experiments/motion_profiles")
    os.makedirs(out, exist_ok=True)
    rclpy.init()
    node = Recorder(out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
