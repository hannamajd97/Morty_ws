"""
plane_filter.py
===============

This module implements simple plane estimation and point cloud
filtering utilities used to remove points below a support surface (e.g.
a table) before running grasp generation. The intention is to avoid
generating grasps that approach an object from below the table or
other support surface, which are infeasible for many robots.

Two functions are provided:

* :func:`estimate_plane_svd` fits a plane to a point cloud using
  singular value decomposition (SVD) and orients the normal to
  point roughly upward (positive z).
* :func:`filter_points_above_plane` returns a mask of points that lie
  above the plane by at least a specified clearance.

These helpers are designed to be fast and avoid heavy dependencies.
"""

from __future__ import annotations

from typing import Tuple, Iterable
import numpy as np


def estimate_plane_svd(points: np.ndarray, max_iters: int = 3,
                       trim_quantile: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a planar surface from a set of 3D points.

    Uses a robust approach based on trimming the largest residuals
    relative to the current plane estimate and recomputing the
    principal component analysis on the remaining points. The final
    normal is oriented such that its Z component is non-negative.

    :param points: N x 3 array of 3D points.
    :param max_iters: Number of trimming iterations to perform.
    :param trim_quantile: Fraction of points to discard based on
        distance from the plane in each iteration.
    :returns: (normal, point) where normal is a unit vector pointing
        upward (n[2] >= 0) and point is a point on the plane.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError("points must be an Nx3 array with N>=3")

    # work on a copy of points for trimming
    subset = P.copy()
    for _ in range(max_iters):
        centroid = subset.mean(axis=0)
        U, S, Vt = np.linalg.svd(subset - centroid, full_matrices=False)
        n = Vt[-1]  # normal to plane
        # orient the normal to point upward (+z)
        if n[2] < 0:
            n = -n
        # compute signed distances to the plane
        d = (subset - centroid) @ n
        # trim largest residuals (both above and below)
        q = np.quantile(np.abs(d), 1 - trim_quantile)
        mask = np.abs(d) <= q
        subset = subset[mask]
        if subset.shape[0] < 3:
            break
    # recompute on final subset
    centroid = subset.mean(axis=0)
    U, S, Vt = np.linalg.svd(subset - centroid, full_matrices=False)
    n = Vt[-1]
    if n[2] < 0:
        n = -n
    n = n / (np.linalg.norm(n) + 1e-8)
    return n, centroid


def filter_points_above_plane(points: np.ndarray, normal: np.ndarray, point_on_plane: np.ndarray,
                              clearance: float = 0.01) -> np.ndarray:
    """Return a boolean mask of points lying above a plane by ``clearance``.

    :param points: N x 3 array of points to test.
    :param normal: Unit normal vector of the plane (should be oriented so
        that positive distances correspond to points above the plane).
    :param point_on_plane: A point that lies on the plane.
    :param clearance: Minimum distance above the plane (in metres) for a
        point to be considered valid.
    :returns: A boolean array of shape (N,) where ``True`` marks points
        above the plane by at least ``clearance``.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    heights = (points - point_on_plane) @ normal
    return heights > float(clearance)