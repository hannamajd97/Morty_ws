"""Collision boxes for the OTHER objects on the table (2026-08-27, Majd).

Until now the planner's world held only the static furniture (table,
wall, mount_surface, flower) — verified live: the items standing ON the
table were simply absent, so a straight approach could sweep through a
vitamin tube standing next to the target and the planner never objected.

This module gives the planner one padded box per DETECTED object, with
one deliberate exception: the object being grasped gets NO box, because
a box means "do not touch" and the gripper must touch its target.

Boxes are rebuilt every pick from the same depth frame the capture used
and stale ids are removed, because objects move between picks.

A wireframe point cloud of the same boxes is published for RViz so the
operator can see exactly what the planner sees, next to the grasp.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from sensor_msgs.msg import PointCloud2, PointField
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header

from ..registration import lookup_transform_matrix

# Every id this module creates starts with this prefix. The executors
# preserve these through their pre-approach scene wipe (same mechanism
# that protects the table).
OBSTACLE_ID_PREFIX = "scene_obj_"


def _depth_in_metres(node, raw: np.ndarray) -> np.ndarray:
    """Depth in metres. frame_buffer stores a DIVISOR (1000.0 for uint16
    mm), RealSense metadata a MULTIPLIER (0.001 m/unit) — both occur."""
    d = np.asarray(raw, dtype=np.float32)
    ds = float(getattr(node.fb, "depth_scale", 0.0) or 0.0)
    if ds > 1.0:
        return d / ds
    if ds > 0.0:
        return d * ds
    return d / 1000.0


def _param(node, name: str, default: float) -> float:
    """A float parameter, tolerating a node that predates the parameter."""
    try:
        return float(node.get_parameter(name).value)
    except Exception:
        return float(default)


def _up_axis(node) -> np.ndarray:
    """The rig's UP direction in the target frame, as a unit vector.

    base +z is HORIZONTAL on this robot, so up lives in the existing
    `known_up_base` parameter (default [-0.7071, 0.7071, 0.0]). Falls back
    to [0, 0, 1] only when that parameter cannot be read.
    """
    try:
        v = np.asarray([float(x) for x in
                        node.get_parameter("known_up_base").value],
                       dtype=np.float64)
        if v.shape == (3,) and float(np.linalg.norm(v)) > 1e-6:
            return v / float(np.linalg.norm(v))
    except Exception:
        pass
    try:
        node.get_logger().debug(
            "[obstacles] known_up_base unavailable — assuming up = [0,0,1]; "
            "the vertical/horizontal split may be wrong on this rig")
    except Exception:
        pass
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _horizontal_basis(up: np.ndarray):
    """Any orthonormal pair spanning the plane perpendicular to `up`."""
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(up, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(up, ref)
    e1 = e1 / (float(np.linalg.norm(e1)) + 1e-12)
    e2 = np.cross(up, e1)
    e2 = e2 / (float(np.linalg.norm(e2)) + 1e-12)
    return e1, e2


def _box_from_bbox(node, depth_m: np.ndarray, bbox, T_base_cam: np.ndarray,
                   min_px: int, up: np.ndarray, other_bboxes=(),
                   depth_band: float = 0.06) -> Optional[Dict[str, Any]]:
    """Axis-aligned box in the TARGET frame for one detection, or None.

    Rewritten 2026-08-28 — measured live, the boxes were ~70 % too big
    (a 7 cm bottle got 0.117 x 0.119 x 0.080 m), which rejects legitimate
    grasps. Four causes, all fixed here:

    1. ROTATION INFLATION (the big one). The old code built an
       axis-aligned box in the CAMERA frame and transformed its 8
       CORNERS; wrapping a rotated box in an axis-aligned one always
       inflates it, and this camera sits at roughly -80 deg pitch, so the
       inflation was large. The sampled POINTS are now transformed into
       the target frame FIRST (one matrix multiply) and the extents are
       measured THERE. No camera-frame AABB is ever built.
    2. LOOSE DEPTH BAND. The median-depth trim kept +/- 0.12 m, wide
       enough for the table around/behind the object to join in; the band
       is now the `neighbour_obstacle_depth_band_m` parameter (0.06).
    3. STRAY PIXELS. Extents were absolute min/max, so ONE bad pixel
       stretched the box; they are now 2nd/98th percentiles.
    4. BBOX BLEED. Detection rectangles overlap, so a neighbour's pixels
       leaked into this ROI; every OTHER detection's bbox is zeroed out
       before the points are sampled.

    The depth camera still only sees each object's FRONT, so the cloud is
    a shell and the unseen rear must be filled in — but in the TARGET
    frame now, not the camera frame. These objects stand UPRIGHT on a
    table, so the model is a SQUARE FOOTPRINT: measure the two horizontal
    extents (the axes perpendicular to `up`) and the vertical one, then
    grow the smaller horizontal extent symmetrically about the footprint
    centre until it equals the larger (an object is about as deep as it
    is wide). The vertical extent is kept as measured. The returned box
    is the axis-aligned wrap of that upright prism.
    """
    H, W = depth_m.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_m[y1:y2, x1:x2]
    valid = (roi > 0.25) & (roi < 1.5)
    # BBOX BLEED: a pixel claimed by ANOTHER detection is not ours.
    for ob in (other_bboxes or ()):
        try:
            ox1, oy1, ox2, oy2 = [int(v) for v in ob]
        except Exception:
            continue
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            valid[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = False
    if int(np.count_nonzero(valid)) < min_px:
        return None

    vs, us = np.nonzero(valid)
    dd = roi[vs, us]
    # Reject depth outliers (table/background around the object) — keep
    # the front cluster only.
    med = float(np.median(dd))
    keep = np.abs(dd - med) < max(0.005, float(depth_band))
    if int(np.count_nonzero(keep)) < min_px:
        return None
    vs, us, dd = vs[keep], us[keep], dd[keep]

    us_full = us + x1
    vs_full = vs + y1
    fx, fy = float(node.fb.fx), float(node.fb.fy)
    cx, cy = float(node.fb.cx), float(node.fb.cy)
    pts_cam = np.empty((dd.shape[0], 3), dtype=np.float64)
    pts_cam[:, 0] = (us_full - cx) / fx * dd
    pts_cam[:, 1] = (vs_full - cy) / fy * dd
    pts_cam[:, 2] = dd

    # THE POINTS go to the target frame first — never a camera-frame AABB.
    R = np.asarray(T_base_cam, dtype=np.float64)[:3, :3]
    t = np.asarray(T_base_cam, dtype=np.float64)[:3, 3]
    pts = pts_cam @ R.T + t

    up = np.asarray(up, dtype=np.float64)
    e1, e2 = _horizontal_basis(up)
    lo1, hi1 = [float(v) for v in np.percentile(pts @ e1, (2.0, 98.0))]
    lo2, hi2 = [float(v) for v in np.percentile(pts @ e2, (2.0, 98.0))]
    lov, hiv = [float(v) for v in np.percentile(pts @ up, (2.0, 98.0))]

    # SQUARE FOOTPRINT: the unseen rear is as deep as the object is wide.
    side = max(hi1 - lo1, hi2 - lo2, 0.02)
    height = max(hiv - lov, 0.02)
    centre = (0.5 * (lo1 + hi1) * e1 + 0.5 * (lo2 + hi2) * e2
              + 0.5 * (lov + hiv) * up)

    # ORIENTED box (2026-08-28): this rig's base_link axes sit ~45 deg to
    # the objects' own up, so an AXIS-ALIGNED wrapper of a 7x7x12 cm
    # bottle can never be smaller than ~13x13x7 cm — pure geometry, not a
    # bug. Publishing the prism with its own orientation removes that
    # inflation entirely: dimensions are the MEASURED side/side/height and
    # the rotation [e1 e2 up] carries the tilt. (The axis-aligned wrap is
    # still returned as "aabb_size" for the RViz wireframe fallback.)
    half = 0.5 * (np.abs(e1) * side + np.abs(e2) * side + np.abs(up) * height)
    return {
        "center": centre,
        "size": np.array([side, side, height], dtype=np.float64),
        "basis": np.column_stack([e1, e2, up]),
        "aabb_size": np.maximum(2.0 * half, 0.02),
        "px": int(dd.shape[0]),
        "footprint_m": float(side),
        "height_m": float(height),
    }


def _quat_from_matrix(R: np.ndarray):
    """(x, y, z, w) from a proper orthonormal rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    n = float(np.sqrt(x * x + y * y + z * z + w * w)) or 1.0
    return x / n, y / n, z / n, w / n


def _wireframe_points(center: np.ndarray, size: np.ndarray,
                      per_edge: int = 14, basis=None) -> np.ndarray:
    """Points along the 12 edges of a box — a wireframe RViz can show."""
    h = 0.5 * size
    c = np.asarray(center, dtype=np.float32)
    signs = [-1.0, 1.0]
    pts: List[np.ndarray] = []
    t = np.linspace(0.0, 1.0, per_edge, dtype=np.float32)
    for axis in range(3):
        a, b = (axis + 1) % 3, (axis + 2) % 3
        for sa in signs:
            for sb in signs:
                p0 = np.zeros(3, dtype=np.float32)
                p0[a] = sa * h[a]; p0[b] = sb * h[b]; p0[axis] = -h[axis]
                p1 = p0.copy(); p1[axis] = h[axis]
                pts.append(p0[None, :] + t[:, None] * (p1 - p0)[None, :])
    local = np.vstack(pts)
    if basis is not None:
        # Draw the box in its OWN frame, then rotate — so the wireframe
        # in RViz matches the oriented collision object exactly.
        local = local @ np.asarray(basis, dtype=np.float32).T
    return local + c[None, :]


def _cloud_msg(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = np.asarray(points, dtype=np.float32).tobytes()
    return msg


def publish_neighbour_obstacles(node, capture_sel: Dict[str, Any],
                                depth_frame) -> int:
    """Publish one collision box per detected object EXCEPT the target.

    Returns the number of boxes published (0 when disabled or when
    nothing usable was found). Never raises — a failure here must not
    cost a pick.
    """
    try:
        if not bool(node.get_parameter("neighbour_obstacles_enabled").value):
            _clear(node)
            return 0
        if depth_frame is None or getattr(node.fb, "rgb_frame", None) is None:
            return 0

        pad = float(node.get_parameter("neighbour_obstacle_padding_m").value)
        min_px = int(node.get_parameter("neighbour_obstacle_min_px").value)
        frame = str(node.get_parameter("neighbour_obstacle_frame").value)
        # 2026-08-28 over-sized-box fix: tighter depth band + a hard cap.
        depth_band = _param(node, "neighbour_obstacle_depth_band_m", 0.06)
        max_size = _param(node, "neighbour_obstacle_max_size_m", 0.30)
        up = _up_axis(node)

        dets = list(getattr(node, "detections_cache", []) or [])
        if not dets:
            _clear(node)
            return 0

        cam_frame = None
        hdr = capture_sel.get("header")
        if hdr is not None:
            cam_frame = str(hdr.frame_id)
        if not cam_frame:
            # frame_buffer keeps the live capture headers; there is no
            # cloud_frame_id attribute (verified 2026-08-27).
            for hdr_attr in ("depth_header", "rgb_header"):
                h = getattr(node.fb, hdr_attr, None)
                if h is not None and getattr(h, "frame_id", ""):
                    cam_frame = str(h.frame_id)
                    break
        if not cam_frame:
            return 0

        stamp = hdr.stamp if hdr is not None else node.get_clock().now().to_msg()
        T_base_cam = lookup_transform_matrix(
            node.tf_buffer, frame, cam_frame, stamp, 0.5,
            logger=node.get_logger())
        if T_base_cam is None:
            node.get_logger().warn(
                "[obstacles] no TF camera->%s; neighbours not published" % frame)
            return 0

        depth_m = _depth_in_metres(node, depth_frame)
        target_bbox = [int(v) for v in capture_sel["bbox"]]

        # THE TARGET'S OWN 3D REGION (2026-08-28). Image-space IoU alone
        # was not enough: the detector reported the small bottle TWICE
        # with only partial bbox overlap, so the second copy got a
        # collision box placed ON THE TARGET and every approach was
        # rejected as "no complete straight path" (Majd's 21/21 failure).
        # Any neighbour box that intersects the target's own volume is a
        # duplicate of it and must never be published.
        target_box = _box_from_bbox(
            node, depth_m, [int(v) for v in capture_sel["bbox"]],
            T_base_cam, min_px, up, (), depth_band)
        skipped_dup = 0

        objects: List[CollisionObject] = []
        viz: List[np.ndarray] = []
        used_ids: List[str] = []
        skipped_target = 0
        skipped_big = 0

        # BBOX BLEED (2026-08-28): keep every detection's rectangle so each
        # ROI can drop the pixels that belong to its neighbours — the
        # target's rectangle included, its pixels are not ours either.
        all_bboxes: List[Optional[List[int]]] = []
        for det in dets:
            try:
                all_bboxes.append([int(v) for v in det["bbox"]])
            except Exception:
                all_bboxes.append(None)

        for idx, det in enumerate(dets):
            bbox = all_bboxes[idx]
            if bbox is None:
                continue
            # THE TARGET GETS NO BOX (it must be touchable). Matched by
            # bbox overlap rather than identity — the cache is rebuilt
            # every frame so object identity is not stable.
            if _iou(bbox, target_bbox) > 0.5:
                skipped_target += 1
                continue

            # A rectangle overlapping ours by more than half is THIS
            # object detected twice — masking with it would erase the
            # object instead of its neighbour.
            others = [b for j, b in enumerate(all_bboxes)
                      if j != idx and b is not None and _iou(bbox, b) <= 0.5]
            box = _box_from_bbox(node, depth_m, bbox, T_base_cam, min_px, up,
                                 other_bboxes=others, depth_band=depth_band)
            if box is None:
                continue
            if target_box is not None and _boxes_overlap(
                    box, target_box, basis=box["basis"]):
                skipped_dup += 1
                node.get_logger().info(
                    "[obstacles] '%s' overlaps the TARGET in 3D — "
                    "duplicate detection, no box published"
                    % str(det.get('label', 'obj')))
                continue

            size = box["size"] + 2.0 * pad
            label = str(det.get("label", "obj")).replace(" ", "_")
            # SANITY CAP (2026-08-28): a runaway box is worse than no box —
            # it silently rejects every legitimate grasp around it.
            if float(np.max(size)) > max_size:
                node.get_logger().warn(
                    "[obstacles] '%s' SKIPPED: box %.1f x %.1f x %.1f cm "
                    "exceeds neighbour_obstacle_max_size_m (%.1f cm) — "
                    "no box is safer than a runaway one"
                    % (label, size[0] * 100.0, size[1] * 100.0,
                       size[2] * 100.0, max_size * 100.0))
                skipped_big += 1
                continue
            oid = f"{OBSTACLE_ID_PREFIX}{label}_{idx}"

            co = CollisionObject()
            co.id = oid
            co.header.frame_id = frame
            co.operation = CollisionObject.ADD
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(size[0]), float(size[1]), float(size[2])]
            co.primitives.append(prim)
            pose = Pose()
            pose.position.x = float(box["center"][0])
            pose.position.y = float(box["center"][1])
            pose.position.z = float(box["center"][2])
            qx, qy, qz, qw = _quat_from_matrix(box["basis"])
            pose.orientation.x = float(qx)
            pose.orientation.y = float(qy)
            pose.orientation.z = float(qz)
            pose.orientation.w = float(qw)
            co.primitive_poses.append(pose)
            objects.append(co)
            used_ids.append(oid)
            viz.append(_wireframe_points(
                box["center"], size, basis=box["basis"]))
            node.get_logger().info(
                "[obstacles] box '%s' %.1f x %.1f x %.1f cm from %d pts "
                "(footprint %.1f cm, height %.1f cm, pad %.0f mm)"
                % (label, size[0] * 100.0, size[1] * 100.0, size[2] * 100.0,
                   int(box["px"]), box["footprint_m"] * 100.0,
                   box["height_m"] * 100.0, pad * 1000.0))

        # Remove boxes from the previous pick that no longer exist.
        for old in list(getattr(node, "_neighbour_obstacle_ids", []) or []):
            if old not in used_ids:
                rm = CollisionObject()
                rm.id = old
                rm.header.frame_id = frame
                rm.operation = CollisionObject.REMOVE
                objects.append(rm)
        node._neighbour_obstacle_ids = used_ids

        if objects:
            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects = objects
            _scene_pub(node).publish(scene)

        pts = np.vstack(viz) if viz else np.zeros((0, 3), dtype=np.float32)
        _viz_pub(node).publish(_cloud_msg(pts, frame, stamp))

        node.get_logger().info(
            "[obstacles] %d neighbour box(es) published (target skipped: %d, "
            "oversize skipped: %d, pad %.0f mm) — planner now avoids them"
            % (len(used_ids), skipped_target + skipped_dup, skipped_big,
               pad * 1000.0))
        return len(used_ids)
    except Exception as exc:  # never break a pick over this
        try:
            node.get_logger().warn(f"[obstacles] failed: {exc}")
        except Exception:
            pass
        return 0


def _boxes_overlap(a, b, basis, slack: float = 0.005) -> bool:
    """True when two SAME-ORIENTATION boxes intersect in 3D.

    Both boxes share the rig basis, so the separating-axis test reduces
    to three 1-D comparisons along e1/e2/up.
    """
    d = np.asarray(a["center"], dtype=np.float64) - np.asarray(
        b["center"], dtype=np.float64)
    for k in range(3):
        axis = np.asarray(basis, dtype=np.float64)[:, k]
        gap = abs(float(d @ axis))
        reach = 0.5 * (float(a["size"][k]) + float(b["size"][k])) + slack
        if gap > reach:
            return False          # separated on this axis -> no overlap
    return True


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


def _scene_pub(node):
    pub = getattr(node, "_neighbour_scene_pub", None)
    if pub is None:
        pub = node.create_publisher(PlanningScene, "/planning_scene", 10)
        node._neighbour_scene_pub = pub
    return pub


def _viz_pub(node):
    pub = getattr(node, "_neighbour_viz_pub", None)
    if pub is None:
        pub = node.create_publisher(
            PointCloud2, "/grasp_pipeline/neighbour_obstacles_cloud", 1)
        node._neighbour_viz_pub = pub
    return pub


def _clear(node) -> None:
    """Drop every box we published (feature switched off / nothing seen)."""
    ids = list(getattr(node, "_neighbour_obstacle_ids", []) or [])
    if not ids:
        return
    frame = str(node.get_parameter("neighbour_obstacle_frame").value)
    scene = PlanningScene()
    scene.is_diff = True
    for oid in ids:
        rm = CollisionObject()
        rm.id = oid
        rm.header.frame_id = frame
        rm.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(rm)
    _scene_pub(node).publish(scene)
    node._neighbour_obstacle_ids = []
