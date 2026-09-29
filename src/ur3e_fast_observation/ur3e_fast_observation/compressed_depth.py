"""Decode image_transport's compressedDepth format — the Astra relay's depth.

The shuttle relays the Astra depth as /astra/depth/compressed_slow
(sensor_msgs/CompressedImage, format "16UC1; compressedDepth png"). Two
readers decode it: vla_experiments' astra_cloud_local (the RViz cloud) and
grasp_pipeline's astra_pointing (the Astra as the pointing camera,
2026-09-29). It lives here, in the light shared package, because importing
grasp_pipeline loads YOLO and torch (see latest_message.py).
"""
from __future__ import annotations

import struct
from typing import Optional

import cv2
import numpy as np


def decode_compressed_depth(m) -> Optional[np.ndarray]:
    """image_transport compressedDepth: 12-byte ConfigHeader (int32 format,
    float depthQuantA, depthQuantB) + PNG. Returns float32 metres (0 = no
    depth) or None when the PNG does not decode."""
    fmt, qa, qb = struct.unpack("<iff", bytes(m.data[:12]))
    img = cv2.imdecode(np.frombuffer(bytes(m.data[12:]), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 1000.0
    if img.dtype == np.uint8 and qa != 0.0:  # inverse-depth quantised 32FC1
        d = np.zeros(img.shape, np.float32); nz = img > 0
        d[nz] = qa / (img[nz].astype(np.float32) - qb)
        return d
    return img.astype(np.float32)
