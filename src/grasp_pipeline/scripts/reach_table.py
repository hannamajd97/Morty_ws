#!/usr/bin/env python3
"""Regenerate FRONT_LEVEL_REACH for grasp_pipeline/perception/reach_envelope.py.

Samples the UR3e forward kinematics (calibrated link lengths from
config/ur3e_calibration.yaml, nominal DH otherwise) with the gripper TCP
offset, keeps the grasp family the filter chain keeps (tool within 25 deg
of horizontal, approaching from the base side within 60 deg of yaw), and
prints the 99.5th-percentile TCP distance per 10-degree band of the
direction off the base z axis.  ~40 s.  Run after changing the tool offset.
"""
import numpy as np

D1, A2, A3, D4, D5, D6 = 0.15205, -0.24411, -0.21339, 0.13144, 0.08538, 0.09231
TCP_OFFSET = 0.12                                   # gripper_3d.xacro tcp_xyz z
UP_IN_BASE = np.array([-0.71, 0.71, 0.0])           # table-frame up (base z is horizontal on this mount)
ALPHA = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
A = [0, A2, A3, 0, 0, 0]
D = [D1, 0, 0, D4, D5, D6]


def fk(q):
    T = np.eye(4)
    for i in range(6):
        ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(ALPHA[i]), np.sin(ALPHA[i])
        T = T @ np.array([[ct, -st * ca, st * sa, A[i] * ct],
                          [st, ct * ca, -ct * sa, A[i] * st],
                          [0, sa, ca, D[i]], [0, 0, 0, 1]])
    return T


def main(n=800000, seed=3):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(-np.pi, np.pi, (n, 6))
    P = np.zeros((n, 3)); Z = np.zeros((n, 3))
    for i in range(n):
        T = fk(Q[i]); P[i] = (T @ np.array([0, 0, TCP_OFFSET, 1]))[:3]; Z[i] = T[:3, 2]
    r = np.linalg.norm(P, axis=1); u = P / r[:, None]
    up = UP_IN_BASE / np.linalg.norm(UP_IN_BASE)
    level = np.abs(Z @ up) < np.sin(np.radians(25))
    ang = np.degrees(np.arccos(np.abs(u[:, 2])))

    def proj(v):
        w = v - np.outer(v @ up, up)
        return w / np.maximum(np.linalg.norm(w, axis=1), 1e-9)[:, None]

    yaw = np.degrees(np.arccos(np.clip(np.sum(proj(Z) * proj(u), axis=1), -1, 1)))
    front = level & (yaw < 60)
    table = []
    for lo in range(0, 90, 10):
        m = front & (ang >= lo) & (ang < lo + 10)
        table.append((lo + 5, round(float(np.percentile(r[m], 99.5)), 3)))
    print("FRONT_LEVEL_REACH = (")
    for a_, v in table:
        print(f"    ({a_}, {v:.3f}),")
    print(")")


if __name__ == "__main__":
    main()
