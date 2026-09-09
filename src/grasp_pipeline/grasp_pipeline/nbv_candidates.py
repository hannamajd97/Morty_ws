"""Helpers for generating dynamic NBV pose candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class NBVCandidatePose:
    """A generated camera-pose candidate for next-best-view probing."""

    name: str
    camera_pose: np.ndarray  # kept for backward compat (may be identity for joint candidates)


@dataclass
class NBVJointCandidate:
    """A joint-space interpolated NBV candidate."""

    name: str
    joint_values: np.ndarray  # (N,) array of joint values
    alpha: float  # interpolation fraction (0 = seed, 1 = target)


# ──────────────────────────────────────────────────────────────────────
# Joint-space interpolation candidates (new approach)
# ──────────────────────────────────────────────────────────────────────

def parse_srdf_named_targets(srdf_path: str, group_name: str) -> Dict[str, np.ndarray]:
    """Parse SRDF file and return named targets as {name: joint_values} dict.

    Joint values are returned in the order they appear in the SRDF group_state.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(srdf_path)
    root = tree.getroot()
    targets: Dict[str, np.ndarray] = {}
    for gs in root.findall(".//group_state"):
        if gs.get("group") != group_name:
            continue
        name = gs.get("name", "")
        joints = []
        for j in gs.findall("joint"):
            joints.append(float(j.get("value", "0.0")))
        if joints:
            targets[name] = np.array(joints, dtype=np.float64)
    return targets


# Standard UR joint order used by MoveIt planning group
UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def generate_joint_interpolated_candidates(
    seed_joints: np.ndarray,
    target_configs: Dict[str, np.ndarray],
    requested_names: Iterable[str],
    alpha: float = 1.0,
) -> List[NBVJointCandidate]:
    """Generate NBV candidates by interpolating in joint space.

    Parameters
    ----------
    seed_joints : (N,) array
        Current / seed joint configuration (e.g. initgrasp).
    target_configs : dict
        Mapping from target name (e.g. "detect_left") to (N,) joint array.
    requested_names : iterable of str
        Which targets to generate candidates for.
    alpha : float
        Interpolation fraction.  1.0 = go all the way to the named target.
        0.5 = halfway between seed and target.
    """
    seed = np.asarray(seed_joints, dtype=np.float64).ravel()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    candidates: List[NBVJointCandidate] = []

    for name in requested_names:
        name = str(name).strip()
        if name not in target_configs:
            continue
        target = np.asarray(target_configs[name], dtype=np.float64).ravel()
        if target.shape != seed.shape:
            continue
        interp = seed + alpha * (target - seed)
        candidates.append(NBVJointCandidate(
            name=name,
            joint_values=interp,
            alpha=alpha,
        ))

    return candidates


# ──────────────────────────────────────────────────────────────────────
# Camera-sphere candidates (original approach, kept for reference)
# ──────────────────────────────────────────────────────────────────────

def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    k = _normalize(np.asarray(axis, dtype=np.float64).reshape(3))
    if np.linalg.norm(k) <= 1e-12 or abs(float(angle_rad)) <= 1e-12:
        return v.copy()
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def make_look_at_camera_pose(
    position: np.ndarray,
    target: np.ndarray,
    x_hint: np.ndarray,
) -> np.ndarray:
    """Build a camera optical-frame pose that looks at ``target``."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    tgt = np.asarray(target, dtype=np.float64).reshape(3)
    x_ref = _normalize(np.asarray(x_hint, dtype=np.float64).reshape(3))

    z_axis = _normalize(tgt - pos)
    if np.linalg.norm(z_axis) <= 1e-12:
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos.astype(np.float32)
        return T

    x_axis = x_ref - z_axis * float(np.dot(x_ref, z_axis))
    if np.linalg.norm(x_axis) <= 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_axis = np.cross(world_up, z_axis)
        if np.linalg.norm(x_axis) <= 1e-6:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = np.cross(world_up, z_axis)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x_axis.astype(np.float32)
    T[:3, 1] = y_axis.astype(np.float32)
    T[:3, 2] = z_axis.astype(np.float32)
    T[:3, 3] = pos.astype(np.float32)
    return T


def generate_look_at_arc_waypoints(
    base_camera_pose: np.ndarray,
    object_center: np.ndarray,
    direction: str,
    n_waypoints: int = 8,
    lateral_offset_m: float = 0.12,
    vertical_offset_m: float = 0.10,
    upward_bias_m: float = 0.01,
) -> List[np.ndarray]:
    """Generate camera poses along a great-circle arc, all looking at *object_center*.

    Returns a list of *n_waypoints* 4x4 float32 poses (camera optical frame in
    base_link).  The first waypoint is *close* to the current camera pose (but
    re-oriented to look at the centroid), and the last is the target direction.

    To convert these to EE poses, multiply each by ``T_cam_ee`` (the fixed
    transform from camera optical frame to the planning end-effector frame).
    """
    T_base_cam = np.asarray(base_camera_pose, dtype=np.float64).reshape(4, 4)
    center = np.asarray(object_center, dtype=np.float64).reshape(3)
    pos0 = T_base_cam[:3, 3].copy()
    x0 = _normalize(T_base_cam[:3, 0])
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rel0 = pos0 - center
    radius = float(np.linalg.norm(rel0))
    if radius <= 1e-6 or n_waypoints < 1:
        return []

    lateral = max(float(lateral_offset_m), 0.0)
    vertical = max(float(vertical_offset_m), 0.0)
    upward_bias = max(float(upward_bias_m), 0.0)
    lateral_angle = min(lateral / radius, np.deg2rad(25.0))
    vertical_angle = min(vertical / radius, np.deg2rad(18.0))

    # Determine the end relative position for the chosen direction
    direction = str(direction).strip()
    if direction == "detect_left":
        rel_end = _rodrigues_rotate(rel0, world_up, -lateral_angle)
    elif direction == "detect_right":
        rel_end = _rodrigues_rotate(rel0, world_up, lateral_angle)
    elif direction == "detect_higher":
        axis = np.cross(rel0, world_up)
        if np.linalg.norm(axis) <= 1e-6:
            axis = x0
        rel_end = _rodrigues_rotate(rel0, axis, -vertical_angle)
        if rel_end[2] < rel0[2]:
            rel_end = _rodrigues_rotate(rel0, axis, vertical_angle)
    else:
        return []

    rel_end = radius * _normalize(rel_end)
    pos_end = center + rel_end
    min_z = pos0[2] + (upward_bias if direction == "detect_higher" else 0.0)
    if pos_end[2] < min_z:
        pos_end = pos_end.copy()
        pos_end[2] = min_z

    # Interpolate along the great-circle arc on the viewing sphere
    # using SLERP on the relative vectors (rel0 → rel_end).
    rel0_n = _normalize(rel0)
    rel_end_n = _normalize(rel_end - center + center)  # already normalised below
    rel_end_n = _normalize(pos_end - center)
    dot = float(np.clip(np.dot(rel0_n, rel_end_n), -1.0, 1.0))
    omega = float(np.arccos(dot))  # angle between start and end on sphere

    waypoints: List[np.ndarray] = []
    for i in range(n_waypoints):
        t = (i + 1) / n_waypoints  # skip t=0 (current pose), end at t=1
        if omega > 1e-6:
            # SLERP
            s0 = float(np.sin((1.0 - t) * omega) / np.sin(omega))
            s1 = float(np.sin(t * omega) / np.sin(omega))
            rel_t = s0 * rel0_n + s1 * rel_end_n
        else:
            rel_t = (1.0 - t) * rel0_n + t * rel_end_n
        rel_t = radius * _normalize(rel_t)
        pos_t = center + rel_t
        # Enforce minimum z for "higher" direction
        if direction == "detect_higher" and pos_t[2] < min_z:
            pos_t = pos_t.copy()
            pos_t[2] = min_z
        T = make_look_at_camera_pose(pos_t, center, x0)
        waypoints.append(T)

    return waypoints


def generate_dynamic_nbv_candidates(
    requested_names: Iterable[str],
    base_camera_pose: np.ndarray,
    object_center: np.ndarray,
    lateral_offset_m: float,
    vertical_offset_m: float,
    upward_bias_m: float,
) -> List[NBVCandidatePose]:
    """Generate camera-pose candidates around the current observation pose (legacy)."""
    T_base_cam = np.asarray(base_camera_pose, dtype=np.float64).reshape(4, 4)
    center = np.asarray(object_center, dtype=np.float64).reshape(3)
    pos0 = T_base_cam[:3, 3]
    x0 = _normalize(T_base_cam[:3, 0])
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rel0 = pos0 - center
    radius = float(np.linalg.norm(rel0))
    if radius <= 1e-6:
        return []

    requested = [str(name).strip() for name in requested_names if str(name).strip()]
    lateral = max(float(lateral_offset_m), 0.0)
    vertical = max(float(vertical_offset_m), 0.0)
    upward_bias = max(float(upward_bias_m), 0.0)
    lateral_angle = min(lateral / radius, np.deg2rad(25.0))
    vertical_angle = min(vertical / radius, np.deg2rad(18.0))

    candidates: List[NBVCandidatePose] = []
    for name in requested:
        if name == "detect_left":
            rel = _rodrigues_rotate(rel0, world_up, -lateral_angle)
        elif name == "detect_right":
            rel = _rodrigues_rotate(rel0, world_up, lateral_angle)
        elif name == "detect_higher":
            axis = np.cross(rel0, world_up)
            if np.linalg.norm(axis) <= 1e-6:
                axis = x0
            rel = _rodrigues_rotate(rel0, axis, -vertical_angle)
            if rel[2] < rel0[2]:
                rel = _rodrigues_rotate(rel0, axis, vertical_angle)
        else:
            continue
        rel = radius * _normalize(rel)
        pos = center + rel
        min_z = pos0[2] + (upward_bias if name == "detect_higher" else 0.0)
        if pos[2] < min_z:
            pos = pos.copy()
            pos[2] = min_z
        T = make_look_at_camera_pose(pos, center, x0)
        candidates.append(NBVCandidatePose(name=name, camera_pose=T))
    return candidates
