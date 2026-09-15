"""
Geometry helpers and marker utilities for the grasp selection pipeline.

This module contains small functions that perform common geometric
computations, such as axis-aligned bounding boxes, denormalising
transforms from unit cubes, checking whether points are inside
axis-aligned bounding boxes, estimating a view direction from a
point cloud and constructing rotation matrices for gripper-to-TCP
transformations. It also provides a helper for publishing RViz
markers to visualise grasp poses.

These helpers were factored out of the monolithic grasp selection
node into a separate module to improve readability and reuse. Other
parts of the pipeline, such as filtering functions and the main node
itself, import utilities from here instead of duplicating code.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from tf_transformations import quaternion_from_matrix
from rclpy.node import Node
from visualization_msgs.msg import Marker


def _simplify_object_name(label: str) -> str:
    """Map model class names to correct display names.

    The y26s_seg model was trained with inexact class names, so this
    function remaps them to the real object identities. Unknown labels
    are returned lower-cased as-is.

    Args:
        label: The original label returned by the detector.

    Returns:
        The corrected, lower-cased display name.
    """
    label_lower = label.lower().strip()

    name_mapping = {
        'this is food pink': 'pink food',
        'green pringles': 'pringles',
        'astra': 'red astra',
        'this is food': 'orange food',
        'red bull': 'lightblue redbull',
        'vitamin_c': 'vitamin_tube',
    }

    return name_mapping.get(label_lower, label_lower)


def _aabb_min_max(pts_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the axis‑aligned bounding box (AABB) of a set of 3D points.

    Args:
        pts_xyz: An (N×3) array of point coordinates.

    Returns:
        A tuple ``(min, max)`` giving the per‑axis minimum and maximum
        coordinates of the points. Both arrays have shape (3,).
    """
    return pts_xyz.min(axis=0), pts_xyz.max(axis=0)


def _denorm_T_from_unit_cube(
    T_raw: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray, column_major: bool
) -> np.ndarray:
    """Denormalise a 4×4 transform from unit‑cube coordinates to metric space.

    GraspGen returns grasp poses with translations in unit‑cube coordinates
    (i.e. values in [−1, 1]) when the ``graspgen_matrix_normalized`` flag is
    set. This helper maps such transforms back into the metric space
    defined by the axis‑aligned bounding box of the point cloud.

    Args:
        T_raw: The raw 4×4 grasp pose matrix returned by GraspGen.
        aabb_min: Minimum point of the cloud AABB.
        aabb_max: Maximum point of the cloud AABB.
        column_major: Whether ``T_raw`` is stored in column‑major order.

    Returns:
        A 4×4 matrix with translation mapped into metric space.
    """
    # Fix memory layout if needed
    T = T_raw.T.copy() if column_major else T_raw.copy()
    center = 0.5 * (aabb_min + aabb_max)
    scale = 0.5 * (aabb_max - aabb_min)
    T[:3, 3] = T[:3, 3] * scale + center
    return T


def _shift_grasp_origins(grasps, dist_m: float, column_major: bool) -> None:
    """Move every grasp origin ``dist_m`` along its own approach axis (+z),
    in place.

    GraspGen's checkpoints describe a grasp in the gripper-BASE frame of
    the gripper they were trained for; for the Franka Panda hand the
    fingertip centre lies 0.1034 m ahead of that frame along +z. Measured
    on the service (2026-09-15, synthetic tube): grasp origins sit 8.8-14.8
    cm from the object, median 10.5 cm. The executor moves ``gripper_tcp``
    (the printed gripper's fingertip centre) to the pose, so the origin
    has to be carried to the fingertips once, here, before any filter or
    consumer looks at it.
    """
    if not dist_m:
        return
    for g in grasps:
        m = np.asarray(g.get("matrix"), dtype=np.float64).reshape(4, 4)
        T = m.T.copy() if column_major else m.copy()
        T[:3, 3] += float(dist_m) * T[:3, 2]
        g["matrix"] = (T.T if column_major else T).reshape(-1).tolist()


def _inside_aabb(
    p: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray, margin: float = 1e-3
) -> bool:
    """Check whether a point lies inside (or near) an AABB.

    Args:
        p: The point to test (shape (3,)).
        aabb_min: Minimum corner of the AABB.
        aabb_max: Maximum corner of the AABB.
        margin: A small margin to expand the bounds.

    Returns:
        True if ``p`` lies inside the AABB expanded by ``margin`` in all
        directions; False otherwise.
    """
    return bool(np.all(p >= (aabb_min - margin)) and np.all(p <= (aabb_max + margin)))


def _estimate_view_dir_from_cloud(pts: np.ndarray, trim: float = 0.10) -> np.ndarray:
    """Estimate the viewing direction from a cropped point cloud.

    This function computes an approximate view direction by averaging
    normalised rays from the origin to points in the cloud. It trims
    extreme points along the Z axis before averaging to reduce the
    influence of outliers.

    Args:
        pts: An (N×3) array of point coordinates.
        trim: Fraction of points to trim at both ends of the Z range.

    Returns:
        A 3‑element unit vector giving the estimated viewing direction.
    """
    if pts.shape[0] < 10:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    z = pts[:, 2]
    lo, hi = np.quantile(z, [trim, 1.0 - trim])
    m = (z >= lo) & (z <= hi)
    rays = pts[m]
    unit = rays / (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-9)
    v = unit.mean(axis=0)
    v /= (np.linalg.norm(v) + 1e-9)
    return v.astype(np.float32)


def _make_T_offset(xyz) -> np.ndarray:
    """Pure translation, applied AFTER the grasp-to-TCP rotation, so it is
    expressed in the executed TCP frame RViz shows (x red, y green, z blue):
    the measured difference between where GraspGen's gripper and the
    printed gripper put the object (``grasp2tcp_offset_m``, 2026-09-11)."""
    T = np.eye(4, dtype=np.float32)
    v = np.asarray(list(xyz)[:3], dtype=np.float32) if xyz is not None else np.zeros(3, dtype=np.float32)
    if v.shape[0] == 3:
        T[:3, 3] = v
    return T


def _make_T_grasp_to_tcp(rot_name: str = "flipY") -> np.ndarray:
    """Return a correction matrix mapping a grasp frame to the gripper TCP.

    The returned matrix is a 4×4 rigid transform that applies only a
    rotation. It is used to adjust the orientation of a grasp pose so
    that the gripper approaches along the correct axis (e.g. flipping
    the Y axis).

    Args:
        rot_name: Name of the rotation. One of ``'flipY'``, ``'x90'``,
            ``'x-90'``, ``'z180'``, or ``'identity'``.

    Returns:
        A 4×4 rotation matrix. The bottom row is (0, 0, 0, 1).
    """
    if rot_name == "flipY":  # 180° around Y
        c, s = math.cos(math.pi), math.sin(math.pi)
        R = np.array(
            [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c],
            ],
            dtype=np.float32,
        )
    elif rot_name == "x90":  # +90° around X
        c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
        R = np.array(
            [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ],
            dtype=np.float32,
        )
    elif rot_name == "x-90":  # -90° around X
        c, s = math.cos(-math.pi / 2), math.sin(-math.pi / 2)
        R = np.array(
            [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ],
            dtype=np.float32,
        )
    elif rot_name == "z180":  # 180° around Z
        c, s = math.cos(math.pi), math.sin(math.pi)
        R = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    else:  # identity
        R = np.eye(3, dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    return T


def _publish_marker(
    node: Node, T: np.ndarray, frame_id: str, topic: str = "/graspgen/best_marker"
) -> None:
    """Publish a visualisation marker representing a grasp pose.

    This helper creates and publishes an arrow marker in RViz to
    visualise a grasp pose. It caches the publisher on the node to
    avoid creating a new publisher each time.

    Args:
        node: The ROS2 node used to create and cache the publisher.
        T: A 4×4 homogeneous transform representing the pose.
        frame_id: The frame in which the pose is expressed.
        topic: The topic on which to publish the marker.
    """
    # Reuse the publisher if available
    pub = getattr(node, "_best_marker_pub", None)
    if pub is None:
        pub = node.create_publisher(Marker, topic, 1)
        node._best_marker_pub = pub

    qx, qy, qz, qw = quaternion_from_matrix(T)
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "grasp"
    m.id = 0
    m.type = Marker.ARROW
    m.action = Marker.ADD

    m.pose.position.x = float(T[0, 3])
    m.pose.position.y = float(T[1, 3])
    m.pose.position.z = float(T[2, 3])
    m.pose.orientation.x = float(qx)
    m.pose.orientation.y = float(qy)
    m.pose.orientation.z = float(qz)
    m.pose.orientation.w = float(qw)

    # Set arrow scale; these values can be tuned for visibility
    m.scale.x = 0.10
    m.scale.y = 0.02
    m.scale.z = 0.02
    m.color.a = 1.0
    m.color.r = 0.0
    m.color.g = 1.0
    m.color.b = 0.0

    pub.publish(m)
