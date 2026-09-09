"""
utils.py
========

This module provides helper functions for processing grasp proposals
and converting between different representations (rotation matrices,
quaternions). It also includes a simple clustering-based cleaning
function to remove outliers from a point cloud and a utility to pick
the best grasp from a list of candidates.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Tuple, List, Dict, Any
import numpy as np


def _estimate_table_height(pts: np.ndarray) -> float:
    """Estimate a support surface (table) height from a point cloud.

    Returns the 5th percentile of the Z coordinates as a robust
    estimate of the table height.
    """
    if pts.size == 0:
        return 0.0
    return float(np.percentile(pts[:, 2], 5))


def _R_t_from_matrix16(M: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation and translation from a 4x4 homogeneous matrix.

    :param M: Flattened or nested 4x4 transform matrix.
    :returns: (R, t) where R is a 3x3 rotation and t is a length-3 vector.
    """
    M = np.asarray(M, dtype=np.float32).reshape(4, 4)
    R = M[:3, :3].copy()
    t = M[:3, 3].copy()
    return R, t


def _quat_from_R(R: Iterable[Any]) -> Tuple[float, float, float, float]:
    """Convert a rotation matrix into a quaternion (x,y,z,w).

    :param R: 3x3 rotation matrix (or array-like).
    :returns: Quaternion as (x, y, z, w).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s
    return float(x), float(y), float(z), float(w)


def _clean_cropped_cloud(pts_np: np.ndarray, cols_np: np.ndarray | None,
                         eps: float = 0.02, min_samples: int = 10) -> Tuple[np.ndarray, np.ndarray | None]:
    """Remove outliers and keep the dominant cluster from a cropped cloud.

    The function performs several steps:

    1. Remove non-finite points.
    2. Filter Z values to the 5th–90th percentile range.
    3. Optionally apply DBSCAN clustering to find the dominant object and
       discard small clusters.

    :param pts_np: N x 3 array of points.
    :param cols_np: Optional N x 3 array of colors corresponding to points.
    :param eps: Epsilon distance for DBSCAN (approx metres).
    :param min_samples: Minimum samples for a core point in DBSCAN.
    :returns: (pts, cols) after cleaning. Colors array may be ``None``.
    """
    if pts_np is None or len(pts_np) == 0:
        return pts_np, cols_np

    ok = np.isfinite(pts_np).all(axis=1)
    pts = pts_np[ok]
    cols = cols_np[ok] if (cols_np is not None) else None
    if len(pts) == 0:
        return pts, cols

    # Z quantile filter to roughly remove the table and far outliers
    z = pts[:, 2]
    z_lo, z_hi = np.percentile(z, 2), np.percentile(z, 98)
    keep = (z >= z_lo) & (z <= z_hi)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]
    if len(pts) == 0:
        return pts, cols

    # DBSCAN clustering to isolate the dominant object cluster
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
        labels = db.labels_
        if labels.size and np.any(labels >= 0):
            counts = Counter(labels[labels >= 0])
            major = max(counts, key=counts.get)
            sel = labels == major
            pts = pts[sel]
            if cols is not None:
                cols = cols[sel]
    except Exception:
        # clustering errors are silently ignored
        pass

    return pts, cols


def keep_largest_cluster(
    pts_np: np.ndarray,
    cols_np: np.ndarray | None,
    eps: float = 0.02,
    min_samples: int = 12,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Keep only the largest DBSCAN cluster (drop isolated far blobs)."""
    if pts_np is None or len(pts_np) == 0:
        return pts_np, cols_np
    try:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(pts_np)
        valid = labels >= 0
        if not np.any(valid):
            return pts_np, cols_np
        counts = Counter(labels[valid])
        major = max(counts, key=counts.get)
        keep = labels == major
        pts = pts_np[keep]
        cols = cols_np[keep] if cols_np is not None else None
        return pts, cols
    except Exception:
        return pts_np, cols_np


def radius_outlier_filter(
    pts_np: np.ndarray,
    cols_np: np.ndarray | None,
    radius_m: float = 0.02,
    min_neighbors: int = 8,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Remove sparse points that do not have enough neighbors within radius."""
    if pts_np is None or len(pts_np) == 0:
        return pts_np, cols_np
    r = float(radius_m)
    k = int(min_neighbors)
    if r <= 0.0 or k <= 0:
        return pts_np, cols_np

    keep = None
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts_np)
        cnts = tree.query_ball_point(pts_np, r=r, return_length=True)
        keep = np.asarray(cnts) >= k
    except Exception:
        try:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(radius=r).fit(pts_np)
            neigh = nn.radius_neighbors(pts_np, return_distance=False)
            keep = np.asarray([len(ix) >= k for ix in neigh], dtype=bool)
        except Exception:
            return pts_np, cols_np

    if keep is None or not np.any(keep):
        return pts_np, cols_np
    pts = pts_np[keep]
    cols = cols_np[keep] if cols_np is not None else None
    return pts, cols


def pick_best_grasp(grasps: Iterable[Dict[str, Any]], pts_np: np.ndarray,
                    table_margin: float = 0.01, prefer_side: bool = True,
                    min_open: float = 0.05, max_open: float = 0.09,
                    gravity_axis: np.ndarray = np.array([0, 0, 1.0]),
                    ignore_behind: bool = True, behind_threshold: float = 0.7,
                    ignore_above: bool = True, above_threshold: float = 0.7) -> Dict[str, Any] | None:
    """Choose the best grasp from a list of candidates.

    The function filters out grasps that are too low (below the table
    height plus ``table_margin``) or whose jaw opening width is
    outside the specified range. It optionally rewards side grasps
    (approach direction more horizontal) when ``prefer_side`` is True.
    It can also filter out grasps coming from behind the camera or
    from above (flipped/inverted orientations).

    :param grasps: Iterable of grasp dicts returned by GraspGen.
    :param pts_np: The point cloud used for estimating table height.
    :param table_margin: Minimal clearance above the table height.
    :param prefer_side: If True, add a bonus to grasps approaching from the side.
    :param min_open: Minimum allowable gripper opening (metres).
    :param max_open: Maximum allowable gripper opening (metres).
    :param gravity_axis: 3-vector indicating the direction of gravity.
    :param ignore_behind: If True, filter out grasps approaching from behind (+z camera).
    :param behind_threshold: Dot product threshold to consider a grasp "from behind".
    :param ignore_above: If True, filter out grasps approaching from above (flipped).
    :param above_threshold: Dot product threshold to consider a grasp "from above".
    :returns: The selected grasp dict or ``None`` if none are acceptable.
    """
    # convert iterable to list for repeated traversal
    grasps_list = list(grasps) if grasps is not None else []
    if not grasps_list:
        return None
    z_table = _estimate_table_height(pts_np)
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    camera_z_axis = np.array([0, 0, 1.0])  # Camera's +z direction
    
    for g in grasps_list:
        try:
            M = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
            t = M[:3, 3]
            # reject grasps too low
            if t[2] < z_table + table_margin:
                continue
            # reject grasps outside opening range
            width = float(g.get("width", (min_open + max_open) / 2.0))
            if not (min_open <= width <= max_open):
                continue
            
            # BEHIND FILTER: reject grasps approaching from behind (+z camera direction)
            if ignore_behind:
                grasp_z_axis = M[:3, 2]  # Grasp approach direction (z-axis)
                grasp_z_axis = grasp_z_axis / (np.linalg.norm(grasp_z_axis) + 1e-6)
                # If dot product > threshold, grasp is coming from behind
                behind_dot = float(np.dot(grasp_z_axis, camera_z_axis))
                if behind_dot < -behind_threshold:
                    continue  # Skip this grasp
            
            # ABOVE FILTER: reject grasps approaching from above (flipped/inverted)
            if ignore_above:
                grasp_z_axis = M[:3, 2]  # Grasp approach direction (z-axis)
                grasp_z_axis = grasp_z_axis / (np.linalg.norm(grasp_z_axis) + 1e-6)
                # If dot product < -threshold, grasp is coming from above (negative z)
                above_dot = float(np.dot(grasp_z_axis, camera_z_axis))
                if above_dot < -above_threshold:
                    continue  # Skip this grasp
            
            score = float(g.get("score", 0.0))
            # reward side grasps: approach axis ~ horizontal means z-axis close to horizontal
            if prefer_side:
                z_axis = M[:3, 2]
                z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
                side_bonus = 1.0 - abs(float(np.dot(z_axis, gravity_axis)))
                score += 0.2 * side_bonus
            candidates.append((score, g))
        except Exception:
            continue
    if not candidates:
        # fallback to max score if nothing passes the filters
        return max(grasps_list, key=lambda x: x.get("score", 0.0)) if grasps_list else None
    # choose the candidate with the highest score
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
