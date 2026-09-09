"""
crop_utils.py
================

This module defines helper functions for cropping depth and color
images to obtain a point cloud in the camera coordinate system.

The main function :func:`crop_depth_to_cloud_and_numpy` takes a
depth image and corresponding RGB image and a bounding box, then
computes a 3D point cloud in the camera optical frame. It returns
both a ``sensor_msgs/PointCloud2`` message and the raw numpy arrays
for further processing (e.g. filtering or coloring).

"""
# grasp_pipeline/crop_utils.py
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2


def crop_depth_to_cloud_and_numpy(
    depth_img: np.ndarray,
    rgb_img: np.ndarray,
    bbox: List[int],              # [x1, y1, x2, y2] in ORIGINAL image coords
    header: Header,               # should be camera_color_optical_frame
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 1000.0,  # 16UC1 in millimetres → divide by 1000
    mask_full: Optional[np.ndarray] = None,  # Optional HxW boolean mask in original image coords
) -> Tuple[Optional[PointCloud2], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Crop the aligned depth (and RGB) to bbox, back-project to a point cloud in the
    **camera optical frame** (X right, Y down, Z forward), and build a PointCloud2.

    Returns (cloud_msg, pts_np[m,3], cols_np[m,3]) or (None, None, None) if empty.
    """
    if depth_img is None or rgb_img is None:
        return None, None, None

    H, W = depth_img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Clamp ROI to image bounds
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W,     x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H,     y2))
    if x2 <= x1 or y2 <= y1:
        return None, None, None

    depth_roi = depth_img[y1:y2, x1:x2]
    rgb_roi = rgb_img[y1:y2, x1:x2, :] if rgb_img.ndim == 3 else None
    mask_roi: Optional[np.ndarray] = None
    if mask_full is not None:
        if mask_full.shape[0] != H or mask_full.shape[1] != W:
            raise ValueError(
                f"mask_full shape {mask_full.shape} must match image shape {(H, W)}"
            )
        mask_roi = mask_full[y1:y2, x1:x2].astype(bool)

    # Convert depth to METERS
    if depth_roi.dtype == np.uint16:
        scale = depth_scale if depth_scale and depth_scale > 1.0 else 1000.0
        Z = depth_roi.astype(np.float32) / float(scale)
    elif depth_roi.dtype == np.float32:
        Z = depth_roi
    else:
        raise ValueError(f"Unsupported depth dtype: {depth_roi.dtype}")

    # Build pixel grids in ORIGINAL image coordinates
    u = np.arange(x1, x2, dtype=np.float32)
    v = np.arange(y1, y2, dtype=np.float32)
    UU, VV = np.meshgrid(u, v)

    # Valid depth mask (+ optional segmentation mask)
    valid = Z > 0.0
    if mask_roi is not None:
        valid = valid & mask_roi
    if not np.any(valid):
        return None, None, None

    # Back-project in **optical frame** (X right, Y down, Z forward)
    X = (UU - cx) / fx * Z
    Y = (VV - cy) / fy * Z

    xs = X[valid].ravel()
    ys = Y[valid].ravel()
    zs = Z[valid].ravel()

    pts_np = np.column_stack([xs, ys, zs]).astype(np.float32)

    cols_np: Optional[np.ndarray] = None
    if rgb_roi is not None:
        cols_np = rgb_roi.reshape(-1, 3)[valid.reshape(-1)].astype(np.uint8)

    # Build cloud in the SAME optical frame as the header we were given
    out_header = Header()
    out_header.stamp = header.stamp
    out_header.frame_id = header.frame_id  # e.g., camera_color_optical_frame
    cloud_msg = pc2.create_cloud_xyz32(out_header, pts_np.tolist())

    return cloud_msg, pts_np, cols_np
