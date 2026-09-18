"""
Multi-view point-cloud registration helpers.

This module provides a small registration/fusion pipeline for segmented
object clouds captured from multiple viewpoints. The intended use is:

1. Capture one or more per-view object clouds in their camera frames.
2. Transform each cloud into a fixed target frame using TF.
3. Optionally run sequential ICP to correct small TF / hand-eye errors.
4. Merge and voxel-downsample the accumulated cloud.

The implementation keeps TF alignment as the primary source of truth and
uses ICP only as a bounded local refinement step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import math
import numpy as np

import rclpy.duration
import rclpy.time
from tf2_ros import TransformException


@dataclass
class RegistrationView:
    """A single segmented object cloud observation."""

    points: np.ndarray
    colors: Optional[np.ndarray]
    frame_id: str
    stamp: object
    label: str = ""


@dataclass
class RegistrationConfig:
    """Parameters controlling TF registration and optional ICP refinement."""

    target_frame: str = "base_link"
    voxel_size_m: float = 0.003
    use_icp: bool = False
    icp_use_point_to_plane: bool = True
    icp_max_correspondence_m: float = 0.008
    icp_max_iterations: int = 40
    icp_min_fitness: float = 0.35
    icp_max_rmse_m: float = 0.006
    icp_max_translation_correction_m: float = 0.01
    icp_max_rotation_correction_deg: float = 8.0
    icp_local_map_radius_m: float = 0.12
    icp_normal_radius_m: float = 0.02
    tf_timeout_s: float = 0.4


@dataclass
class ObservationDebug:
    """Per-observation registration diagnostics."""

    index: int
    frame_id: str
    input_points: int
    tf_points: int
    fused_points: int
    icp_attempted: bool = False
    icp_applied: bool = False
    icp_fitness: Optional[float] = None
    icp_rmse_m: Optional[float] = None
    icp_translation_correction_m: Optional[float] = None
    icp_rotation_correction_deg: Optional[float] = None
    note: str = ""


@dataclass
class RegistrationResult:
    """Final fused cloud and debug information."""

    fused_points: np.ndarray
    fused_colors: Optional[np.ndarray]
    observations: List[ObservationDebug] = field(default_factory=list)


@dataclass
class ObservationGain:
    """Novelty score for a candidate observation against the current fused cloud."""

    total_voxels: int
    new_voxels: int
    gain_ratio: float


def _transform_to_matrix(transform_msg) -> np.ndarray:
    qx = float(transform_msg.transform.rotation.x)
    qy = float(transform_msg.transform.rotation.y)
    qz = float(transform_msg.transform.rotation.z)
    qw = float(transform_msg.transform.rotation.w)
    tx = float(transform_msg.transform.translation.x)
    ty = float(transform_msg.transform.translation.y)
    tz = float(transform_msg.transform.translation.z)

    # Quaternion -> rotation matrix
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    T = np.eye(4, dtype=np.float64)
    T[0, 0] = 1.0 - 2.0 * (yy + zz)
    T[0, 1] = 2.0 * (xy - wz)
    T[0, 2] = 2.0 * (xz + wy)
    T[1, 0] = 2.0 * (xy + wz)
    T[1, 1] = 1.0 - 2.0 * (xx + zz)
    T[1, 2] = 2.0 * (yz - wx)
    T[2, 0] = 2.0 * (xz - wy)
    T[2, 1] = 2.0 * (yz + wx)
    T[2, 2] = 1.0 - 2.0 * (xx + yy)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def lookup_transform_matrix(
    tf_buffer,
    target_frame: str,
    source_frame: str,
    stamp,
    timeout_s: float,
    logger=None,
) -> np.ndarray:
    if str(target_frame) == str(source_frame):
        return np.eye(4, dtype=np.float64)
    if getattr(stamp, "sec", 0) == 0 and getattr(stamp, "nanosec", 0) == 0:
        t = rclpy.time.Time()
    else:
        t = rclpy.time.Time.from_msg(stamp)
    timeout = rclpy.duration.Duration(seconds=float(timeout_s))
    # The exact-stamp lookup waits the full timeout when the buffer does not
    # reach that stamp yet (decimated /tf listener), then falls back to the
    # latest anyway: 0.5 s lost on the click path (2026-09-18). Wait 0.1 s
    # at most for the exact stamp; the fallback keeps the full timeout.
    exact_timeout = rclpy.duration.Duration(seconds=min(float(timeout_s), 0.1))
    try:
        tf_msg = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            t,
            timeout=exact_timeout,
        )
    except TransformException as exc:
        # Registration is often requested a few seconds after the first object was selected.
        # If the TF buffer no longer contains that exact timestamp, use the latest transform
        # rather than aborting the whole selection cycle.
        if logger is not None:
            logger.warn(
                f"[registration] TF lookup at exact stamp failed for {source_frame} -> {target_frame}: {exc}. "
                "Falling back to latest available transform."
            )
        tf_msg = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=timeout,
        )
    return _transform_to_matrix(tf_msg)


def transform_points(points: np.ndarray, T_target_source: np.ndarray) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64)
    out = (T_target_source[:3, :3] @ pts.T).T + T_target_source[:3, 3]
    return out.astype(np.float32, copy=False)


def transform_homogeneous(T_left: np.ndarray, T_right: np.ndarray) -> np.ndarray:
    """Compose two homogeneous transforms."""
    return (np.asarray(T_left, dtype=np.float64) @ np.asarray(T_right, dtype=np.float64)).astype(np.float32)


def _rotation_angle_deg(T: np.ndarray) -> float:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def _to_o3d_cloud(points: np.ndarray, colors: Optional[np.ndarray]):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and len(colors) == len(points):
        cols = np.asarray(colors, dtype=np.float64)
        if cols.max(initial=0.0) > 1.0:
            cols = cols / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def _from_o3d_cloud(pcd):
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = None
    if pcd.has_colors():
        c = np.asarray(pcd.colors, dtype=np.float32)
        cols = np.clip(np.round(c * 255.0), 0.0, 255.0).astype(np.uint8)
    return pts, cols


def _voxel_downsample_numpy(points: np.ndarray, colors: Optional[np.ndarray], voxel_size_m: float):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0 or voxel_size_m <= 0.0:
        return pts, None if colors is None else np.asarray(colors)

    keys = np.floor(pts / float(voxel_size_m)).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_vox = int(inverse.max()) + 1

    counts = np.bincount(inverse, minlength=n_vox).astype(np.float32).reshape(-1, 1)
    pts_sum = np.zeros((n_vox, 3), dtype=np.float64)
    np.add.at(pts_sum, inverse, pts.astype(np.float64))
    pts_ds = (pts_sum / counts).astype(np.float32)

    cols_ds = None
    if colors is not None and len(colors) == len(pts):
        cols = np.asarray(colors)
        cols_sum = np.zeros((n_vox, 3), dtype=np.float64)
        np.add.at(cols_sum, inverse, cols.astype(np.float64))
        cols_ds = np.clip(np.round(cols_sum / counts), 0.0, 255.0).astype(np.uint8)

    return pts_ds, cols_ds


def voxel_downsample(points: np.ndarray, colors: Optional[np.ndarray], voxel_size_m: float):
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32), None if colors is None else np.empty((0, 3), dtype=np.uint8)
    if voxel_size_m <= 0.0:
        return np.asarray(points, dtype=np.float32), None if colors is None else np.asarray(colors)

    try:
        pcd = _to_o3d_cloud(points, colors)
        pcd = pcd.voxel_down_sample(float(voxel_size_m))
        return _from_o3d_cloud(pcd)
    except ModuleNotFoundError:
        return _voxel_downsample_numpy(points, colors, voxel_size_m)


def compute_observation_gain(
    fused_points: np.ndarray,
    observation_points: np.ndarray,
    voxel_size_m: float,
) -> ObservationGain:
    """Estimate how much new surface an observation adds using voxel occupancy."""
    obs = np.asarray(observation_points, dtype=np.float32)
    fused = np.asarray(fused_points, dtype=np.float32)
    if len(obs) == 0:
        return ObservationGain(total_voxels=0, new_voxels=0, gain_ratio=0.0)
    voxel = max(float(voxel_size_m), 1e-6)
    obs_keys = np.unique(np.floor(obs / voxel).astype(np.int32), axis=0)
    if len(obs_keys) == 0:
        return ObservationGain(total_voxels=0, new_voxels=0, gain_ratio=0.0)
    fused_keys = np.unique(np.floor(fused / voxel).astype(np.int32), axis=0) if len(fused) else np.empty((0, 3), dtype=np.int32)
    fused_set = {tuple(row.tolist()) for row in fused_keys}
    new_voxels = sum(1 for row in obs_keys if tuple(row.tolist()) not in fused_set)
    ratio = float(new_voxels) / float(len(obs_keys)) if len(obs_keys) else 0.0
    return ObservationGain(
        total_voxels=int(len(obs_keys)),
        new_voxels=int(new_voxels),
        gain_ratio=ratio,
    )


def _estimate_normals_in_place(pcd, radius_m: float):
    import open3d as o3d

    if len(pcd.points) < 5 or radius_m <= 0.0:
        return
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius_m),
            max_nn=30,
        )
    )


def _select_local_target(points: np.ndarray, colors: Optional[np.ndarray], center: np.ndarray, radius_m: float):
    if radius_m <= 0.0 or len(points) == 0:
        return points, colors
    d = np.linalg.norm(points - center.reshape(1, 3), axis=1)
    keep = d <= float(radius_m)
    if int(np.count_nonzero(keep)) < 30:
        return points, colors
    pts = points[keep]
    cols = colors[keep] if colors is not None and len(colors) == len(points) else None
    return pts, cols


def register_and_fuse_views(
    views: List[RegistrationView],
    tf_buffer,
    config: RegistrationConfig,
    logger=None,
) -> RegistrationResult:
    """Transform each observation into the target frame and fuse them.

    TF provides the coarse alignment for each view. If enabled, ICP is
    run sequentially with identity initialization because the TF pose
    has already placed the source cloud close to the target cloud.
    """

    if not views:
        return RegistrationResult(
            fused_points=np.empty((0, 3), dtype=np.float32),
            fused_colors=None,
            observations=[],
        )

    fused_points = np.empty((0, 3), dtype=np.float32)
    fused_colors: Optional[np.ndarray] = None
    obs_debug: List[ObservationDebug] = []

    for idx, view in enumerate(views):
        dbg = ObservationDebug(
            index=idx,
            frame_id=str(view.frame_id),
            input_points=int(len(view.points)),
            tf_points=0,
            fused_points=int(len(fused_points)),
        )
        if view.points is None or len(view.points) == 0:
            dbg.note = "empty input view"
            obs_debug.append(dbg)
            continue

        T_target_source = lookup_transform_matrix(
            tf_buffer,
            config.target_frame,
            view.frame_id,
            view.stamp,
            config.tf_timeout_s,
            logger=logger,
        )
        pts_tf = transform_points(view.points, T_target_source)
        cols_tf = None if view.colors is None else np.asarray(view.colors)
        dbg.tf_points = int(len(pts_tf))

        if logger is not None and len(pts_tf) > 0:
            c = pts_tf.mean(axis=0)
            is_identity = np.allclose(T_target_source, np.eye(4), atol=1e-6)
            logger.info(
                f"[registration] view={idx} centroid_in_{config.target_frame}="
                f"({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}) "
                f"tf_is_identity={is_identity} src_frame={view.frame_id}"
            )

        if idx == 0 or len(fused_points) < 30:
            fused_points = pts_tf
            fused_colors = cols_tf
            fused_points, fused_colors = voxel_downsample(fused_points, fused_colors, config.voxel_size_m)
            dbg.fused_points = int(len(fused_points))
            dbg.note = "seed view"
            obs_debug.append(dbg)
            continue

        if logger is not None and len(fused_points) > 0 and len(pts_tf) > 0:
            fc = fused_points.mean(axis=0)
            nc = pts_tf.mean(axis=0)
            offset = float(np.linalg.norm(fc - nc))
            logger.info(
                f"[registration] view={idx} ALIGNMENT CHECK: "
                f"fused_centroid=({fc[0]:.4f}, {fc[1]:.4f}, {fc[2]:.4f}) "
                f"new_centroid=({nc[0]:.4f}, {nc[1]:.4f}, {nc[2]:.4f}) "
                f"centroid_offset={offset:.4f}m"
            )

        if config.use_icp:
            dbg.icp_attempted = True
            target_pts, target_cols = _select_local_target(
                fused_points,
                fused_colors,
                center=pts_tf.mean(axis=0).astype(np.float32),
                radius_m=float(config.icp_local_map_radius_m),
            )
            if len(target_pts) >= 30 and len(pts_tf) >= 30:
                try:
                    import open3d as o3d
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "registration_use_icp:=true requires open3d in the grasp pipeline venv "
                        f"({exc}). Install it with pip --user (morty_ws uses no venv) or disable ICP."
                    ) from exc

                src_pts, src_cols = voxel_downsample(pts_tf, cols_tf, max(config.voxel_size_m * 0.5, 0.001))
                tgt_pts, tgt_cols = voxel_downsample(target_pts, target_cols, max(config.voxel_size_m * 0.5, 0.001))
                if len(src_pts) >= 30 and len(tgt_pts) >= 30:
                    src = _to_o3d_cloud(src_pts, src_cols)
                    tgt = _to_o3d_cloud(tgt_pts, tgt_cols)

                    if config.icp_use_point_to_plane:
                        _estimate_normals_in_place(src, config.icp_normal_radius_m)
                        _estimate_normals_in_place(tgt, config.icp_normal_radius_m)
                        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    else:
                        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

                    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=int(config.icp_max_iterations)
                    )
                    result = o3d.pipelines.registration.registration_icp(
                        src,
                        tgt,
                        float(config.icp_max_correspondence_m),
                        np.eye(4, dtype=np.float64),
                        estimator,
                        criteria,
                    )

                    corr = np.asarray(result.transformation, dtype=np.float32)
                    t_corr = float(np.linalg.norm(corr[:3, 3]))
                    r_corr = float(_rotation_angle_deg(corr))
                    dbg.icp_fitness = float(result.fitness)
                    dbg.icp_rmse_m = float(result.inlier_rmse)
                    dbg.icp_translation_correction_m = t_corr
                    dbg.icp_rotation_correction_deg = r_corr

                    if (
                        float(result.fitness) >= float(config.icp_min_fitness)
                        and float(result.inlier_rmse) <= float(config.icp_max_rmse_m)
                        and t_corr <= float(config.icp_max_translation_correction_m)
                        and r_corr <= float(config.icp_max_rotation_correction_deg)
                    ):
                        pts_tf = transform_points(pts_tf, corr)
                        dbg.icp_applied = True
                        dbg.note = "tf+icp"
                    else:
                        dbg.note = "icp rejected by gates"
                else:
                    dbg.note = "icp skipped: downsampled clouds too small"
            else:
                dbg.note = "icp skipped: local overlap too small"
        else:
            dbg.note = "tf only"

        fused_points = np.vstack([fused_points, pts_tf])
        if fused_colors is not None or cols_tf is not None:
            if fused_colors is None:
                fused_colors = np.zeros((len(fused_points) - len(pts_tf), 3), dtype=np.uint8)
            if cols_tf is None:
                cols_tf = np.zeros((len(pts_tf), 3), dtype=np.uint8)
            fused_colors = np.vstack([fused_colors, cols_tf])
        fused_points, fused_colors = voxel_downsample(fused_points, fused_colors, config.voxel_size_m)
        dbg.fused_points = int(len(fused_points))
        obs_debug.append(dbg)

        if logger is not None:
            msg = (
                f"[registration] view={idx} frame={view.frame_id} pts_in={dbg.input_points} "
                f"pts_tf={dbg.tf_points} fused={dbg.fused_points} note={dbg.note}"
            )
            if dbg.icp_attempted:
                msg += (
                    f" icp_applied={dbg.icp_applied} fitness={dbg.icp_fitness} "
                    f"rmse={dbg.icp_rmse_m} dt={dbg.icp_translation_correction_m} "
                    f"dr={dbg.icp_rotation_correction_deg}"
                )
            logger.info(msg)

    return RegistrationResult(
        fused_points=fused_points,
        fused_colors=fused_colors,
        observations=obs_debug,
    )
