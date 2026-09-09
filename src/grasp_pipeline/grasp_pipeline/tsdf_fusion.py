"""
tsdf_fusion.py
==============

Multi-view TSDF (Truncated Signed Distance Field) fusion for the grasp
pipeline. Given several RGBD observations of the SAME object — each with its
camera pose (extrinsic) — this integrates them into one Open3D
ScalableTSDFVolume and extracts a single, denser, less-noisy surface point
cloud than any single view could provide.

Why TSDF rather than just concatenating per-view clouds: the TSDF volume
averages overlapping surface measurements into a signed-distance field, which
cancels per-view depth noise and fills small gaps, giving GraspGen cleaner
geometry to plan grasps on.

Used by the multi-view registration pipeline (``registration_pipeline``) when
the robot sweeps the object from multiple observation poses.

Each observation dict is expected to carry: ``depth_frame``, ``rgb_frame``,
``intrinsics`` (fx/fy/cx/cy/width/height), ``capture_transform`` (4x4
target<-source pose), a ``selected_mask`` or ``bbox`` isolating the object,
``depth_scale``, and a ``tsdf_eligible`` flag (set by the caller once a view is
judged good enough to fuse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass
class TSDFConfig:
    """Tunables for the TSDF integration. Defaults are in metres."""
    voxel_length_m: float = 0.001     # TSDF voxel size; smaller = finer but slower/heavier
    sdf_trunc_m: float = 0.004        # truncation distance of the signed-distance field
    depth_trunc_m: float = 1.2        # ignore depth beyond this range (drops far background)
    point_voxel_size_m: float = 0.0008  # voxel size for downsampling the extracted cloud
    mask_erode_px: int = 5            # shrink the object mask to drop noisy edge pixels
    depth_band_m: float = 0.04        # keep only depth within ±band of the median (drops outliers)


def _rect_mask(shape: Tuple[int, int], bbox) -> np.ndarray:
    """Fallback object mask: a filled rectangle from a bbox, clamped to image
    bounds. Used when no precise segmentation mask is available."""
    h, w = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _build_mask(obs: Dict) -> Optional[np.ndarray]:
    """Pick the best available object mask for an observation: prefer a precise
    segmentation mask, fall back to the bbox rectangle, else None."""
    depth = obs.get("depth_frame")
    if depth is None:
        return None
    mask = obs.get("selected_mask")
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape[:2] == depth.shape[:2]:
            return m
    bbox = obs.get("bbox")
    if bbox is not None:
        return _rect_mask(depth.shape[:2], bbox)
    return None


def _erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Shrink the mask inward by erode_px. Object-boundary depth pixels are the
    noisiest (they straddle object and background), so trimming the edge keeps
    only confidently-on-object depth before fusion."""
    if erode_px <= 0:
        return mask
    try:
        import cv2

        k = max(1, int(erode_px))
        kernel = np.ones((k, k), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return eroded.astype(bool)
    except Exception:
        return mask


def integrate_tsdf_surface_cloud(
    observations: Iterable[Dict],
    config: TSDFConfig,
    logger=None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fuse the eligible observations into one TSDF volume and return the
    extracted surface cloud as (points Nx3 float32, colors Nx3 uint8 or None).

    Observations that are not ``tsdf_eligible`` or are missing any of
    depth/rgb/intrinsics/extrinsic/mask are skipped (logged, not fatal), so a
    few bad views never abort the whole fusion."""
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        if logger is not None:
            logger.warn("[tsdf] open3d is not available in the grasp pipeline venv; skipping TSDF integration.")
        return np.empty((0, 3), dtype=np.float32), None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(config.voxel_length_m),
        sdf_trunc=float(config.sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    total_obs = 0
    eligible_obs = 0
    for obs in observations:
        total_obs += 1
        if not bool(obs.get("tsdf_eligible", False)):
            if logger is not None:
                reason = str(obs.get("tsdf_skip_reason", "not tsdf_eligible"))
                logger.info(f"[tsdf] skipping observation: {reason}")
            continue
        eligible_obs += 1
        depth = obs.get("depth_frame")
        rgb = obs.get("rgb_frame")
        intr = obs.get("intrinsics")
        T_target_source = obs.get("capture_transform")
        if depth is None or rgb is None or intr is None or T_target_source is None:
            if logger is not None:
                logger.warn("[tsdf] skipping eligible observation: missing depth/rgb/intrinsics/extrinsic")
            continue

        mask = _build_mask(obs)
        if mask is None:
            if logger is not None:
                logger.warn("[tsdf] skipping eligible observation: no valid mask/bbox available")
            continue

        depth_np = np.asarray(depth).copy()
        if depth_np.ndim != 2:
            if logger is not None:
                logger.warn(f"[tsdf] skipping eligible observation: depth ndim={depth_np.ndim}, expected 2")
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth_np.shape:
            if logger is not None:
                logger.warn(
                    f"[tsdf] skipping eligible observation: mask shape {mask.shape} != depth shape {depth_np.shape}"
                )
            continue
        # Zero out every depth pixel outside the (eroded) object mask so the
        # TSDF only integrates the object, not the table/background behind it.
        mask = _erode_mask(mask, int(config.mask_erode_px))
        depth_np[~mask] = 0
        if not np.any(depth_np > 0):
            if logger is not None:
                logger.warn("[tsdf] skipping eligible observation: masked depth has no valid pixels")
            continue

        depth_scale = float(obs.get("depth_scale", 1000.0))
        valid_depth = depth_np[depth_np > 0].astype(np.float32) / max(depth_scale, 1e-6)
        if valid_depth.size == 0:
            if logger is not None:
                logger.warn("[tsdf] skipping eligible observation: no positive masked depth after erosion")
            continue
        # Depth-band filter: keep only pixels whose depth is within ±band of the
        # median object depth. Removes stray foreground/background pixels the
        # mask let through (e.g. a hand edge, or table seen through a gap).
        if float(config.depth_band_m) > 0.0:
            depth_med = float(np.median(valid_depth))
            keep = np.abs((depth_np.astype(np.float32) / depth_scale) - depth_med) <= float(config.depth_band_m)
            depth_np[~keep] = 0
            if not np.any(depth_np > 0):
                if logger is not None:
                    logger.warn(
                        f"[tsdf] skipping eligible observation: depth-band filter removed all pixels "
                        f"(median={depth_med:.3f}m band={float(config.depth_band_m):.3f}m)"
                    )
                continue

        orig_dtype = str(depth_np.dtype)
        if np.issubdtype(depth_np.dtype, np.integer):
            if depth_np.dtype != np.uint16:
                depth_np = depth_np.astype(np.uint16, copy=False)
            depth_np = np.ascontiguousarray(depth_np, dtype=np.uint16)
        else:
            depth_np = np.ascontiguousarray(depth_np, dtype=np.float32)

        rgb_np = np.asarray(rgb)
        if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
            if logger is not None:
                logger.warn(f"[tsdf] skipping eligible observation: rgb shape {rgb_np.shape} is invalid")
            continue
        # ROS images arrive BGR; Open3D expects RGB, so flip the channel order.
        rgb_np = np.ascontiguousarray(rgb_np[:, :, ::-1], dtype=np.uint8)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(intr["width"]),
            int(intr["height"]),
            float(intr["fx"]),
            float(intr["fy"]),
            float(intr["cx"]),
            float(intr["cy"]),
        )
        try:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_np.astype(np.uint8)),
                o3d.geometry.Image(depth_np),
                depth_scale=depth_scale,
                depth_trunc=float(config.depth_trunc_m),
                convert_rgb_to_intensity=False,
            )
        except Exception as exc:
            if logger is not None:
                logger.warn(
                    f"[tsdf] failed to build RGBD image: depth_dtype={orig_dtype} "
                    f"converted_dtype={depth_np.dtype} depth_shape={depth_np.shape} "
                    f"depth_contig={depth_np.flags.c_contiguous} rgb_contig={rgb_np.flags.c_contiguous} "
                    f"error={exc}"
                )
            continue
        # Open3D's integrate() wants the world->camera extrinsic, but we store
        # the camera->world capture pose (target<-source), so invert it.
        extrinsic = np.linalg.inv(np.asarray(T_target_source, dtype=np.float64))
        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated += 1
        if logger is not None:
            valid_px = int(np.count_nonzero(depth_np > 0))
            logger.info(
                f"[tsdf] integrated observation {integrated}: valid_depth_px={valid_px} "
                f"depth_dtype={orig_dtype}->{depth_np.dtype} "
                f"mask_erode_px={int(config.mask_erode_px)} depth_band_m={float(config.depth_band_m):.3f}"
            )

    if integrated == 0:
        if logger is not None:
            logger.warn(
                f"[tsdf] no eligible observations were integrated "
                f"(total={total_obs}, eligible={eligible_obs})."
            )
        return np.empty((0, 3), dtype=np.float32), None

    # Pull the fused surface out of the volume, then optionally voxel-downsample
    # to a manageable, evenly-spaced cloud for GraspGen.
    pcd = volume.extract_point_cloud()
    if config.point_voxel_size_m > 0.0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(float(config.point_voxel_size_m))

    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = None
    if pcd.has_colors():
        # Open3D colors are 0..1 floats; convert back to 0..255 uint8 RGB.
        cols = np.clip(np.round(np.asarray(pcd.colors, dtype=np.float32) * 255.0), 0.0, 255.0).astype(np.uint8)
    if logger is not None:
        logger.info(
            f"[tsdf] integrated {integrated}/{eligible_obs} eligible observations "
            f"(total={total_obs}) into {len(pts)} surface points."
        )
    return pts, cols
