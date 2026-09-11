"""
Grasp candidate filtering and scoring utilities.

This module contains functions that operate on lists of grasp
candidates returned by GraspGen. Each function embodies a different
heuristic or constraint, such as discarding grasps that come from
unfavourable directions, penalising grasps far from the object's
centroid, or ensuring the gripper remains upright relative to a
support plane. These filters can be composed to incrementally prune
and reweight candidate lists before selecting the best grasp.

The functions here were previously defined inline in the monolithic
grasp selection node but have been factored out to facilitate
unit testing and reuse.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .geometry_utils import (
    _estimate_view_dir_from_cloud,
    _aabb_min_max,
)


def orientation_census(grasps, up_vec, log_info, tag: str) -> None:
    """One-line orientation summary per filter stage (Majd, 2026-08-24:
    "let us go filter by filter — that is our chance"). Classifies every
    candidate by finger-roll (lateral pinch / diagonal / vertical pinch)
    and reports the approach-tilt spread, all against gravity-up."""
    try:
        import numpy as _np
        if up_vec is None or not grasps:
            return
        lat = diag = vert = 0
        tilts = []
        for g in grasps:
            M = _np.asarray(g["matrix"], dtype=_np.float32).reshape(4, 4)
            a = -M[:3, 2]; a /= (float(_np.linalg.norm(a)) + 1e-9)
            fx = M[:3, 0]; fx /= (float(_np.linalg.norm(fx)) + 1e-9)
            x_up = abs(float(_np.dot(fx, up_vec)))
            if x_up < 0.35:
                lat += 1
            elif x_up > 0.75:
                vert += 1
            else:
                diag += 1
            tilts.append(_np.degrees(_np.arcsin(
                min(1.0, abs(float(_np.dot(a, up_vec)))))))
        tilts = _np.asarray(tilts)
        log_info(
            f"[orient] {tag}: n={len(grasps)} | fingers: "
            f"lateral={lat} diagonal={diag} VERTICAL={vert} | "
            f"tilt med={_np.median(tilts):.0f}deg max={tilts.max():.0f}deg"
        )
    except Exception:
        pass


def apply_gravity_view_filter(
    node,
    grasps,
    pts,
    *,
    plane_n,
    plane_p,
    frame_id: str,
    registration_frame_is_target: bool,
    log_info,
    log_warn,
):
    """THE single shared orientation stage — REBUILT 2026-08-24 to match
    the thesis, §4.6.14 "The Geometric Filter Cascade", on Majd's order:
    "reread the master thesis and apply their filters, simple as possible
    — but the camera is tilted, and the center of the stuff should be the
    gripper, not the camera."

    Both filter chains (GraspFilterPipeline.filter_for_utility and
    PickPipeline's inline chain) MUST call this — the logic used to be
    duplicated in both and silently diverged twice.

    Every judgment is made AT THE CANDIDATE GRIPPER TCP; "up" is the
    robot-base vertical looked up through TF; the table enters ONLY as a
    floor height. The camera is involved in nothing here.

    Thesis stages implemented in this function, in order:
      2. support-plane   — reject a TCP below the table, and a physical
         approach that ascends (comes from below).
      3. side-grasp height window — approaches within ~30° of horizontal
         must place the TCP between the table and 2× the object height.
      4. roll-upright (finger part) — the finger-opening axis must be
         within ~30° of level; a vertical finger dives into the table.
         Strict: no fallback.
      6. centroid Gaussian reweight — no rejection; score × Gaussian of
         the TCP-to-object-middle distance, σ = half the object size.

    Thesis stages implemented elsewhere:
      4. roll-upright (palm part) — _filter_roll_upright_by_plane,
         param reject_flipped_grasps (keeps the palm above the horizon).
      5. orientation-feasibility — left to the executor's per-candidate
         planning probe (thesis §4.9.2), which tests real IK instead of
         a cone approximation.
    Thesis stage 1 (view-consistency) is deliberately DROPPED: it judges
    by the CAMERA axis, and this rig's camera axis measures near-vertical
    — the orientation census proved twice (2026-08-24) that at any
    setting it exterminates every lateral-finger candidate.
    """
    import numpy as _np

    # ── "up" in the cloud frame ──────────────────────────────────────
    # PRIORITY 1 (2026-08-24, Majd's decision): the MEASURED TABLE PLANE
    # from the picture itself — "the plane normal defines up" (thesis
    # §4.6.7). Camera-tilt-proof and TF-free: the table cannot lie.
    # PRIORITY 2: world frame when the cloud is already in base_link.
    # PRIORITY 3 (fallback only): TF base-up — the old path whose wrong
    # answer made level grasps look tilted and tilted grasps look level.
    up_vec = None
    _up_from_tf = False
    _up_src = None
    # MODE SWITCH (Majd 2026-08-27: "we want to try WITHOUT the table in
    # the photo — we don't care about it anymore"). up_vector_source:
    #   "tf"       = HIS method as primary: fixed table direction through
    #                the verified camera TF, every cycle; the ring
    #                measurement is ignored for orientation AND table
    #                height (cloud-bottom is used instead).
    #   "measured" = the original ladder (measure -> memory -> tf -> prior).
    try:
        if str(node.get_parameter("up_vector_source").value
               ).strip().lower() == "tf":
            plane_n = None
            plane_p = None
    except Exception:
        pass
    try:
        if plane_n is not None and plane_p is not None:
            n_up = _np.asarray(plane_n, dtype=_np.float32).ravel()[:3].copy()
            centroid_f = _np.asarray(pts, dtype=_np.float32).mean(axis=0)
            if float(_np.dot(
                    centroid_f - _np.asarray(
                        plane_p, dtype=_np.float32).ravel()[:3],
                    n_up)) < 0.0:
                n_up = -n_up
            up_vec = (n_up / (float(_np.linalg.norm(n_up)) + 1e-9)).astype(
                _np.float32)
            _up_src = "table-plane"
        elif registration_frame_is_target or str(frame_id) in (
                "base_link", "base", "world"):
            up_vec = _np.array([0.0, 0.0, 1.0], dtype=_np.float32)
            _up_src = "world-frame"
        else:
            if frame_id:
                try:
                    # Raw tf2 API. NOTE (2026-08-21): this is equivalent
                    # to lookup_transform_matrix — both were measured and
                    # agree exactly; an earlier suspicion that the helper
                    # inverted the convention was WRONG. Kept raw so the
                    # convention is readable at the point of use.
                    import rclpy.time as _rt
                    tr = node.tf_buffer.lookup_transform(
                        str(frame_id), "base_link", _rt.Time())
                    q = tr.transform.rotation
                    x, y, z, w = q.x, q.y, q.z, q.w
                    # 2026-08-27 (Majd's design; camera calib verified in
                    # RViz + live check 12.2 deg vs measured normal):
                    # transform the KNOWN rig up-direction — the
                    # hardware-proven lift direction in base coords —
                    # through TF into the cloud frame. The old code took
                    # base +z, which is HORIZONTAL on this rig and made
                    # top-down grasps measure as "level".
                    R = _np.array([
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                         2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                         2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w),
                         1 - 2 * (x * x + y * y)],
                    ])
                    try:
                        known_up = _np.asarray(
                            [float(v) for v in
                             node.get_parameter("known_up_base").value],
                            dtype=_np.float64)
                    except Exception:
                        known_up = _np.array([-0.7071, 0.7071, 0.0])
                    v = R @ known_up
                    up_vec = (v / (float(_np.linalg.norm(v)) + 1e-9)).astype(
                        _np.float32)
                    _up_from_tf = True
                    _up_src = "TF-known-up"
                except Exception as tf_exc:
                    up_vec = None
                    log_warn(f"TF up-vector failed ({tf_exc}); trying plane")
            if up_vec is None and plane_n is not None and plane_p is not None:
                n_up = _np.asarray(plane_n, dtype=_np.float32)
                centroid_f = _np.asarray(pts, dtype=_np.float32).mean(axis=0)
                if float(_np.dot(
                        centroid_f - _np.asarray(plane_p, dtype=_np.float32),
                        n_up)) < 0.0:
                    n_up = -n_up
                up_vec = n_up / (float(_np.linalg.norm(n_up)) + 1e-9)
    except Exception as e:
        log_warn(f"up-vector estimation failed ({e}); cascade skipped")
    # RIG FACT OVERRIDE (2026-08-27): the TF fallback resolves to base +z,
    # which on this rig points HORIZONTALLY at the table — every cycle
    # that fell back (fresh session, wall-poisoned ring, no memory yet)
    # judged tilt against a sideways "up", so TOP-DOWN grasps passed the
    # level band as "level" (Majd's ~20% from-above picks). The camera
    # prior [0,-1,0] (optical up — the SAME direction the plane-memory
    # plausibility gate trusts, ~10 deg from today's measured normal) is
    # the correct last resort. Never base +z.
    if up_vec is None:
        # Last resort (TF dead too): camera up-prior. Only sane when the
        # cloud is in the camera frame.
        up_vec = _np.asarray([0.0, -1.0, 0.0], dtype=_np.float32)
        _up_src = "OPTICAL-PRIOR"
        log_warn(
            "no measurement, no memory, no TF — using the camera up-prior "
            "[0,-1,0] as the last resort.")
    elif _up_from_tf:
        log_warn(
            "table plane NOT measured this cycle and no memory — using the "
            "KNOWN rig up via TF (Majd's fallback, calib verified "
            "2026-08-27).")
    # Loud provenance: silent fallbacks hid a wrong up-vector twice
    # (2026-08-21). One line per cycle is cheap.
    if up_vec is not None:
        src = _up_src or ("TF-FALLBACK" if _up_from_tf else "plane-normal")
        log_info(
            f"up-vector [{src}] = "
            f"[{up_vec[0]:+.2f} {up_vec[1]:+.2f} {up_vec[2]:+.2f}]"
        )

    orientation_census(grasps, up_vec, log_info, "entering thesis cascade")

    if up_vec is None or not grasps:
        log_warn("no up-vector or no grasps — thesis cascade skipped")
        return grasps

    normalized = bool(node.get_parameter("graspgen_matrix_normalized").value)
    pts_a = _np.asarray(pts, dtype=_np.float32)
    aabb_min, aabb_max = pts_a.min(axis=0), pts_a.max(axis=0)
    centre = 0.5 * (aabb_min + aabb_max)
    scale = 0.5 * (aabb_max - aabb_min)
    centroid = pts_a.mean(axis=0)

    def _pose(g):
        # GraspGen raw +z points fingers->wrist; physical approach = -z.
        # Fingers open along local +x. Translations may be unit-cube
        # normalized — undo before any metric comparison (the retired
        # no-under filter forgot this AND used the +z wrist direction,
        # so its from-below test was inverted).
        M = _np.asarray(g["matrix"], dtype=_np.float32).reshape(4, 4)
        a = -M[:3, 2]
        a = a / (float(_np.linalg.norm(a)) + 1e-9)
        fx = M[:3, 0]
        fx = fx / (float(_np.linalg.norm(fx)) + 1e-9)
        t = M[:3, 3]
        t_m = (t * scale + centre) if normalized else t
        return a, fx, t_m

    heights = pts_a @ up_vec
    if plane_p is not None:
        h_table = float(_np.dot(
            _np.asarray(plane_p, dtype=_np.float32).ravel()[:3], up_vec))
        h_src = "RANSAC plane"
    else:
        h_table = float(heights.min())
        h_src = "cloud bottom"
    h_obj = max(float(heights.max()) - h_table, 0.02)

    # ── thesis stage 2: support-plane ────────────────────────────────
    try:
        min_h = float(node.get_parameter("support_min_height_m").value)
        max_up = float(node.get_parameter("support_from_below_dot").value)
        kept = []
        n_from_below = n_under = 0
        for g in grasps:
            a, _fx, t_m = _pose(g)
            if float(_np.dot(a, up_vec)) > max_up:
                n_from_below += 1      # physical approach ascends
                continue
            if float(_np.dot(t_m, up_vec)) < h_table + min_h:
                n_under += 1           # TCP below the table
                continue
            kept.append(g)
        grasps = kept
        log_info(
            f"after support-plane: {len(grasps)} "
            f"(from-below {n_from_below}, under-table {n_under}; "
            f"table at {h_table:.3f} [{h_src}])")
        if not grasps:
            log_warn("support-plane removed ALL grasps — nothing executable")
            return []
    except Exception as e:
        log_warn(f"support-plane filter error: {e}")

    # ── LEVEL band — the human coffee grasp (Majd, 2026-08-24) ───────
    # "grasp it straight from the middle, not tilted up (can't — the
    # table is there) and not tilted down (I'd crash my fingers)."
    # Reject approaches tilted more than max_level_tilt_deg from
    # horizontal, in either direction, measured against the table-plane
    # up. <= 0 disables the band (pure thesis mode: no tilt cap).
    try:
        lvl_deg = float(node.get_parameter("max_level_tilt_deg").value)
        if lvl_deg > 0.0:
            lvl_dot = float(_np.sin(_np.radians(min(89.0, lvl_deg))))
            kept = []
            for g in grasps:
                a, _fx, _t = _pose(g)
                if abs(float(_np.dot(a, up_vec))) <= lvl_dot:
                    kept.append(g)
            cut = len(grasps) - len(kept)
            grasps = kept
            log_info(
                f"after level-band (<={lvl_deg:.0f}deg): {len(grasps)} "
                f"(cut {cut})")
            if not grasps:
                log_warn(
                    "level-band removed ALL grasps — no level approach in "
                    "this candidate set")
                return []
    except Exception as e:
        log_warn(f"level-band filter error: {e}")

    # ── thesis stage 3: mid-height band on the object ────────────────
    # Grip the object's BODY: TCP height between min_frac and max_frac
    # of the object's height above the table (defaults 0.10–0.90: not
    # scraping the table, not pinching the cap).
    try:
        horiz_dot = float(node.get_parameter("side_horizontal_dot").value)
        f_lo = float(node.get_parameter("side_height_min_frac").value)
        f_hi = float(node.get_parameter("side_height_max_frac").value)
        h_lo = h_table + f_lo * h_obj
        h_hi = h_table + f_hi * h_obj
        kept = []
        n_side = n_cut = 0
        for g in grasps:
            a, _fx, t_m = _pose(g)
            if abs(float(_np.dot(a, up_vec))) >= horiz_dot:
                kept.append(g)         # steep approach: band n/a (thesis)
                continue
            n_side += 1
            if h_lo <= float(_np.dot(t_m, up_vec)) <= h_hi:
                kept.append(g)
            else:
                n_cut += 1
        grasps = kept
        log_info(
            f"after height-window: {len(grasps)} "
            f"(side {n_side}, cut {n_cut}; band "
            f"{f_lo:.0%}-{f_hi:.0%} of {h_obj * 100:.1f}cm object)")
        if not grasps:
            log_warn("height-window removed ALL grasps — nothing executable")
            return []
    except Exception as e:
        log_warn(f"height-window filter error: {e}")

    # ── thesis stage 4 (finger part): roll-upright ───────────────────
    try:
        max_fv = float(node.get_parameter("max_finger_vertical_dot").value)
        if max_fv < 1.0:
            kept = []
            for g in grasps:
                _a, fx, _t = _pose(g)
                if abs(float(_np.dot(fx, up_vec))) <= max_fv:
                    kept.append(g)
            cut = len(grasps) - len(kept)
            grasps = kept
            log_info(f"after roll-upright(fingers): {len(grasps)} (cut {cut})")
            if not grasps:
                log_warn(
                    "roll-upright removed ALL grasps — refusing a "
                    "vertical-finger fallback (one finger would dive at "
                    "the table)")
                return []
    except Exception as e:
        log_warn(f"roll-upright filter error: {e}")

    # ── palm canonicalization (2026-08-24 night) ─────────────────────
    # A parallel-jaw grasp and its 180°-about-approach twin grip the
    # object identically, but only ONE of them arrives with the gripper
    # right side up. GraspGen emits either twin arbitrarily; the old
    # reject_flipped filter had an INVERTED sign (it kept the flipped
    # twin — Majd tonight: "the gripper have to go 180 to the position");
    # and the executor's twin rewrite is disabled by standing order.
    # So canonicalize HERE: the hardware's correct side is TCP y_up < 0
    # (executor convention upright_twin_y_up_positive=false, validated
    # by Majd 2026-08-21), and grasp→tcp = flipY preserves the y axis —
    # any candidate with raw y·up > 0 is rotated 180° about its approach
    # (negate the x and y columns). No candidate is lost.
    try:
        if bool(node.get_parameter("palm_canonicalize").value) and grasps:
            n_flip = 0
            fixed = []
            for g in grasps:
                M = _np.asarray(g["matrix"], dtype=_np.float32).reshape(4, 4)
                if float(_np.dot(M[:3, 1], up_vec)) > 0.0:
                    M2 = M.copy()
                    M2[:3, 0] *= -1.0
                    M2[:3, 1] *= -1.0
                    g2 = dict(g)
                    g2["matrix"] = M2.reshape(-1).tolist()
                    fixed.append(g2)
                    n_flip += 1
                else:
                    fixed.append(g)
            grasps = fixed
            log_info(
                f"palm-canonicalize: rotated {n_flip}/{len(grasps)} twins "
                "right side up")
    except Exception as e:
        log_warn(f"palm-canonicalize error: {e}")

    # ── thesis stage 1: from-behind (camera rule, Majd 2026-08-24) ───
    # "dismiss those who are from behind — I don't care how."
    # Only valid when the cloud is in the CAMERA's own frame: the camera
    # sits at the origin, so the viewing line to the object is simply the
    # centroid direction — pure data convention, no TF. A grasp whose
    # approach opposes that line flies in from the side the camera never
    # saw -> delete. Side grasps (dot ~ 0) survive; only the hidden back
    # hemisphere dies. min_dot -0.15 tolerates noise around sideways.
    try:
        fb_min = float(node.get_parameter("from_behind_min_dot").value)
        if fb_min > -1.0 and not registration_frame_is_target and grasps:
            view_dir = centroid / (float(_np.linalg.norm(centroid)) + 1e-9)
            kept = []
            for g in grasps:
                a, _fx, _t = _pose(g)
                if float(_np.dot(a, view_dir)) >= fb_min:
                    kept.append(g)
            cut = len(grasps) - len(kept)
            grasps = kept
            log_info(f"after from-behind: {len(grasps)} (cut {cut})")
            if not grasps:
                log_warn(
                    "from-behind removed ALL grasps — every candidate "
                    "closes on the unseen back side")
                return []
    except Exception as e:
        log_warn(f"from-behind filter error: {e}")

    # ── thesis stage 6: centroid Gaussian reweight ───────────────────
    # "The center of the stuff should be the gripper, not the camera"
    # (Majd): the distance is measured from the candidate TCP position
    # to the object's middle, in the cloud frame — no camera anywhere.
    try:
        sf = float(node.get_parameter("centroid_sigma_factor").value)
        if sf > 0.0 and grasps:
            sigma = max(sf * float((aabb_max - aabb_min).max()), 0.01)
            rescored = []
            dists = []
            for g in grasps:
                _a, _fx, t_m = _pose(g)
                d = float(_np.linalg.norm(t_m - centroid))
                dists.append(d)
                g2 = dict(g)
                g2["score"] = float(g.get("score", 0.0)) * float(
                    _np.exp(-0.5 * (d / sigma) ** 2))
                rescored.append(g2)
            grasps = rescored
            log_info(
                f"centroid-reweight (sigma {sigma * 100:.1f}cm): "
                f"TCP-to-middle median {float(_np.median(dists)) * 100:.1f}cm,"
                f" best {min(dists) * 100:.1f}cm")
    except Exception as e:
        log_warn(f"centroid-reweight error: {e}")

    # ── front-of-camera preference (Majd 2026-08-26) ─────────────────
    # "grasp from the front of the camera, not the side": multiply each
    # score by exp(w * (approach·view - 1)) — a perfectly frontal
    # approach keeps its score, a pure side approach is down-weighted by
    # e^-w. No rejection, pure ordering. Bonus: the executor probes
    # candidates in score order and the side grasps are exactly the ones
    # that fail IK (~0.6 s each), so front-first ordering also cuts the
    # 20-reject probe marathon that made grasping slow.
    try:
        fw = float(node.get_parameter("front_approach_weight").value)
        if fw > 0.0 and not registration_frame_is_target and grasps:
            view_dir = centroid / (float(_np.linalg.norm(centroid)) + 1e-9)
            rescored = []
            dots = []
            for g in grasps:
                a, _fx, _t = _pose(g)
                dot = float(_np.dot(a, view_dir))
                dots.append(dot)
                g2 = dict(g)
                g2["score"] = float(g.get("score", 0.0)) * float(
                    _np.exp(fw * (min(dot, 1.0) - 1.0)))
                rescored.append(g2)
            grasps = rescored
            log_info(
                f"front-preference (w={fw:.1f}): approach·view median "
                f"{float(_np.median(dots)):.2f}, best {max(dots):.2f}")
    except Exception as e:
        log_warn(f"front-preference error: {e}")

    orientation_census(grasps, up_vec, log_info, "leaving thesis cascade")

    return grasps


def _filter_view_consistent(
    grasps: List[dict],
    pts: np.ndarray,
    *,
    min_view_dot: float = 0.3,
    require_in_front: bool = True,
    approach_axis: int = 2,
    approach_toward_camera: bool = True,
    matrices_normalized: bool = True,
    column_major: bool = False,
) -> List[dict]:
    """Filter grasps based on their approach direction relative to the view.

    Only retain grasps whose approach axis points roughly towards the
    camera (or away from it if ``approach_toward_camera`` is False) and
    whose TCP lies in front of the object's centre along the view
    direction.

    Args:
        grasps: List of grasp dicts containing a ``matrix`` field.
        pts: Point cloud of the cropped object.
        min_view_dot: Minimum dot product between approach axis and view.
        require_in_front: If True, require the TCP to be in front of the object.
        approach_axis: Column index in the rotation matrix for approach.
        approach_toward_camera: Whether positive dot implies towards the camera.
        matrices_normalized: If True, translations are in unit cube coords.
        column_major: Whether matrices are stored column‑major.

    Returns:
        A filtered list of grasps satisfying the constraints.
    """
    view_dir = _estimate_view_dir_from_cloud(pts)
    aabb_min, aabb_max = _aabb_min_max(pts)
    center = 0.5 * (aabb_min + aabb_max)
    scale = 0.5 * (aabb_max - aabb_min)
    keep: List[dict] = []
    for g in grasps:
        M_raw = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
        M = M_raw.T if column_major else M_raw
        R = M[:3, :3]
        t = M[:3, 3]
        # Orthonormalise R for safety
        U, _, Vt = np.linalg.svd(R); R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        approach = R[:, approach_axis]
        view_dot = float(np.dot(approach, view_dir))
        if not approach_toward_camera:
            view_dot = -view_dot
        if view_dot < float(min_view_dot):
            continue
        if require_in_front:
            t_m = (t * scale + center) if matrices_normalized else t
            if float(np.dot(center - t_m, view_dir)) <= 0.0:
                continue
        keep.append(g)
    return keep


def _filter_no_under_grasp(
    grasps: List[dict],
    pts: np.ndarray,
    *,
    plane_n: Optional[np.ndarray],
    plane_p: Optional[np.ndarray],
    approach_axis: int = 2,
    max_up_dot: float = 0.2,
    min_height: float = 0.005,
) -> List[dict]:
    """Remove grasps that approach the object from below the support plane.

    The plane is defined by its normal ``plane_n`` and a point on the
    plane ``plane_p``. Grasps whose approach axis points too much in
    the upward direction (towards +plane_n) or whose TCP lies below
    the plane are discarded. If ``plane_n`` or ``plane_p`` is None,
    the original list is returned.

    Args:
        grasps: List of grasp dicts.
        pts: Point cloud of the cropped object.
        plane_n: Normal vector of the fitted support plane.
        plane_p: A point on the plane.
        approach_axis: Column index for the approach axis in the rotation.
        max_up_dot: Maximum allowed dot between approach and plane normal.
        min_height: Minimum TCP height above the plane.

    Returns:
        Filtered list of grasps.
    """
    if plane_n is None or plane_p is None or len(grasps) == 0:
        return grasps

    centroid = pts.mean(axis=0).astype(np.float32)
    n = np.asarray(plane_n, dtype=np.float32)
    p0 = np.asarray(plane_p, dtype=np.float32)
    if float(np.dot(centroid - p0, n)) < 0.0:
        n = -n
    keep: List[dict] = []
    for g in grasps:
        M = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
        R, t = M[:3, :3], M[:3, 3]
        approach = R[:, approach_axis]
        if float(np.dot(approach, n)) > float(max_up_dot):
            continue
        if float(np.dot(t - p0, n)) < float(min_height):
            continue
        keep.append(g)
    return keep


def _filter_side_center_window(
    grasps: List[dict],
    pts: np.ndarray,
    *,
    plane_n: Optional[np.ndarray],
    plane_p: Optional[np.ndarray],
    approach_axis: int = 2,
    max_vertical_dot: float = 0.3,
    centroid_down_only: bool = True,
    max_down: float = 0.05,
    max_up: float = 0.0,
    matrices_normalized: bool = True,
    column_major: bool = False,
) -> List[dict]:
    """Keep only grasps that approach roughly horizontally near the object's centre.

    This filter discards top‑down approaches and retains grasps whose
    TCP height is within a window measured along the plane normal
    around the object's centroid. When ``centroid_down_only`` is True
    the height window extends only below the centroid (i.e. avoid
    approaching from above).

    Args:
        grasps: List of grasp dicts with matrices.
        pts: Point cloud.
        plane_n: Plane normal (None to disable filtering).
        plane_p: A point on the plane.
        approach_axis: Approach axis index in the rotation.
        max_vertical_dot: Maximum allowed vertical component of approach.
        centroid_down_only: If True, restrict TCP to below the centroid.
        max_down: Maximum distance below the centroid.
        max_up: Maximum distance above the centroid.
        matrices_normalized: Whether translations are in unit cube coords.
        column_major: Whether matrices are stored column‑major.

    Returns:
        Filtered list of grasps.
    """
    if not grasps or plane_n is None or plane_p is None or pts is None or len(pts) < 3:
        return grasps
    aabb_min, aabb_max = _aabb_min_max(pts)
    center = 0.5 * (aabb_min + aabb_max)
    scale = 0.5 * (aabb_max - aabb_min)
    centroid = pts.mean(axis=0).astype(np.float32)
    n = np.asarray(plane_n, dtype=np.float32)
    p0 = np.asarray(plane_p, dtype=np.float32)
    if float(np.dot(centroid - p0, n)) < 0.0:
        n = -n
    n /= (np.linalg.norm(n) + 1e-9)
    h_centroid = float(np.dot(centroid - p0, n))
    keep: List[dict] = []
    for g in grasps:
        try:
            M_raw = np.asarray(g.get("matrix"), dtype=np.float32).reshape(4, 4)
            M = M_raw.T if column_major else M_raw
            R = M[:3, :3]
            t = M[:3, 3].astype(np.float32)
            t_m = (t * scale + center) if matrices_normalized else t
            # Horizontal approach condition
            a = R[:, approach_axis]
            a /= (np.linalg.norm(a) + 1e-9)
            if abs(float(np.dot(a, n))) > float(max_vertical_dot):
                continue
            # Height window around centroid
            h_tcp = float(np.dot(t_m - p0, n))
            dh = h_centroid - h_tcp
            if centroid_down_only:
                if dh < 0.0 or dh > float(max_down):
                    continue
            else:
                if dh < float(-max_up) or dh > float(max_down):
                    continue
            keep.append(g)
        except Exception:
            continue
    return keep if keep else grasps


def _reweight_scores_by_centroid(
    grasps: List[dict],
    centroid_metric: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    *,
    matrices_normalized: bool,
    column_major: bool,
    weight: float,
) -> List[dict]:
    """Bias grasp scores towards translations near the object's centroid.

    Produces a new list of grasp dicts with updated ``score`` fields.
    The new score is ``old_score - weight * distance``, where
    ``distance`` is the Euclidean distance between the TCP and the
    centroid in metric space. A larger ``weight`` penalises grasps
    farther from the centroid more strongly.

    Args:
        grasps: List of grasp candidate dicts.
        centroid_metric: Object centroid in metric coordinates.
        aabb_min: AABB minimum for unit cube scaling.
        aabb_max: AABB maximum for unit cube scaling.
        matrices_normalized: Whether translations are in unit cube coords.
        column_major: Whether matrices are column‑major.
        weight: Penalty weight per metre distance.

    Returns:
        A new list of grasps with updated ``score`` fields.
    """
    if not grasps:
        return []
    centroid_metric = np.asarray(centroid_metric, dtype=np.float32)
    aabb_min = np.asarray(aabb_min, dtype=np.float32)
    aabb_max = np.asarray(aabb_max, dtype=np.float32)
    center = 0.5 * (aabb_min + aabb_max)
    scale = 0.5 * (aabb_max - aabb_min)
    out: List[dict] = []
    for g in grasps:
        try:
            M_raw = np.asarray(g.get("matrix"), dtype=np.float32).reshape(4, 4)
            M = M_raw.T if column_major else M_raw
            t = M[:3, 3].astype(np.float32)
            t_m = (t * scale + center) if matrices_normalized else t
            d = float(np.linalg.norm(t_m - centroid_metric))
            new_g = dict(g)
            new_g["score"] = float(new_g.get("score", 0.0)) - (weight * d)
            out.append(new_g)
        except Exception:
            out.append(g)
    return out


def _filter_roll_upright_by_plane(
    grasps: List[dict],
    plane_n: Optional[np.ndarray],
    plane_p: Optional[np.ndarray] = None,
    *,
    up_axis: int = 1,
    min_up_dot: float = 0.0,
) -> List[dict]:
    """Discard grasps whose gripper 'up' axis points below the support plane.

    The ``up_axis`` index corresponds to the local frame axis that should
    align with the plane normal. Grasps with an up vector that has a
    dot product with the plane normal smaller than ``min_up_dot`` are
    rejected. A non‑zero ``min_up_dot`` encourages more upright grasps.

    Args:
        grasps: List of grasp dicts with ``matrix`` fields.
        plane_n: Normal of the support plane.
        plane_p: A point on the plane (unused here but accepted for API symmetry).
        up_axis: Index of the local up axis (0=X, 1=Y, 2=Z).
        min_up_dot: Minimum acceptable dot between up axis and plane normal.

    Returns:
        Filtered list of grasps.
    """
    if plane_n is None or not grasps:
        return grasps
    try:
        n = np.asarray(plane_n, dtype=np.float32)
        n /= (np.linalg.norm(n) + 1e-9)
        keep: List[dict] = []
        for g in grasps:
            try:
                M = np.asarray(g.get("matrix"), dtype=np.float32).reshape(4, 4)
                R = M[:3, :3]
                # Orthonormalise R
                U, _, Vt = np.linalg.svd(R); R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                up = R[:, int(up_axis)]
                up /= (np.linalg.norm(up) + 1e-9)
                if float(np.dot(up, n)) >= float(min_up_dot):
                    keep.append(g)
            except Exception:
                keep.append(g)
        return keep if keep else grasps
    except Exception:
        return grasps


def _filter_reject_side_grasps(
    grasps: List[dict],
    *,
    plane_n: Optional[np.ndarray],
    min_vertical_dot: float = 0.7,
    approach_axis: int = 2,
    matrices_normalized: bool = True,
    column_major: bool = False,
) -> List[dict]:
    """Reject grasps that approach horizontally; keep only top‑down grasps.

    This filter checks the angle between the approach direction and the
    plane normal. A large absolute dot product indicates a vertical
    approach. Grasps whose approach direction is too horizontal are
    discarded. If ``plane_n`` is None the input list is returned.

    Args:
        grasps: List of grasp dicts.
        plane_n: Plane normal vector.
        min_vertical_dot: Minimum acceptable absolute dot product.
        approach_axis: Column index for approach direction.
        matrices_normalized: Ignored but kept for API compatibility.
        column_major: Whether matrices are column‑major.

    Returns:
        Filtered list of grasps.
    """
    if not grasps or plane_n is None:
        return grasps
    try:
        plane_n = np.array(plane_n, dtype=np.float32)
        if np.linalg.norm(plane_n) < 1e-6:
            return grasps
        plane_n = plane_n / np.linalg.norm(plane_n)
        keep: List[dict] = []
        for g in grasps:
            try:
                M = np.array(g["matrix"], dtype=np.float32).reshape(4, 4)
                if column_major:
                    M = M.T
                # Extract approach direction (negative for approach)
                approach_col = M[:3, approach_axis]
                approach_dir = -approach_col / (np.linalg.norm(approach_col) + 1e-9)
                vertical_dot = abs(float(np.dot(approach_dir, plane_n)))
                if vertical_dot >= min_vertical_dot:
                    keep.append(g)
            except Exception:
                keep.append(g)
        return keep if keep else grasps
    except Exception:
        return grasps


def _filter_top_only_plane(
    grasps: List[dict],
    *,
    plane_n: Optional[np.ndarray],
    approach_axis: int = 2,
    min_dot: float = 0.9,
) -> List[dict]:
    """Prefer top‑down grasps by aligning approach axis with the plane normal.

    If ``plane_n`` is None the input list is returned. Otherwise,
    grasps whose approach axis deviates too far from the plane normal
    (dot product below ``min_dot``) are discarded.

    Args:
        grasps: List of grasp dicts.
        plane_n: Plane normal vector.
        approach_axis: Column index for approach axis.
        min_dot: Minimum acceptable absolute dot between approach and plane normal.

    Returns:
        Filtered list of grasps.
    """
    if plane_n is None:
        return grasps
    n = plane_n / (np.linalg.norm(plane_n) + 1e-9)
    keep: List[dict] = []
    for g in grasps:
        M = np.asarray(g["matrix"], dtype=np.float32).reshape(4, 4)
        R = M[:3, :3]
        approach = R[:, approach_axis]
        approach /= (np.linalg.norm(approach) + 1e-9)
        if abs(float(np.dot(approach, n))) < float(min_dot):
            continue
        keep.append(g)
    return keep


def _filter_finger_penetration(
    grasps: List[dict],
    pts: np.ndarray,
    *,
    matrices_normalized: bool = True,
    column_major: bool = False,
    aabb_min: Optional[np.ndarray] = None,
    aabb_max: Optional[np.ndarray] = None,
    finger_length: float = 0.08,
    finger_thickness: float = 0.012,
    finger_height: float = 0.025,
    gripper_opening: float = 0.085,
    finger_z_offset: float = 0.06,
    max_points_per_finger: int = 5,
    min_points_in_closing_volume: int = 0,
) -> List[dict]:
    """Reject grasps whose fingers would penetrate the object cloud.

    For each grasp the two finger volumes (left/right) are constructed
    in the grasp's local frame and transformed to world. We count how
    many cloud points fall inside each finger's oriented bounding box.
    Grasps with too many points inside either finger are rejected
    because executing them would crush the object.

    The closing-volume check (between the fingers, where the object is
    supposed to fit) is optional: when ``min_points_in_closing_volume``
    is > 0, grasps with too few points there are also rejected because
    they would "pinch air".

    Conventions (matches the rest of this file):
      - Approach axis: rotation column 2 (+Z local) — points toward object.
      - Closing axis: rotation column 0 (X local) — between fingers.
      - Up axis: rotation column 1 (Y local).

    Geometry defaults are sized for the gripper_3d two-finger gripper
    (TCP at 0.12m forward of the wrist, finger blades ~80mm long).

    Args:
        grasps: List of grasp dicts with ``matrix`` field.
        pts: Object point cloud in *world* frame.
        matrices_normalized: When True, grasp translations are in the
            unit cube of ``[aabb_min, aabb_max]`` and need rescaling.
        column_major: Whether matrices are column-major.
        aabb_min, aabb_max: Required when ``matrices_normalized`` is True.
        finger_length: Finger length along approach axis (m).
        finger_thickness: Finger thickness along closing axis (m).
        finger_height: Finger half-height along up axis (m), full = 2x.
        gripper_opening: Distance between the two finger centerlines (m).
        finger_z_offset: Distance from grasp origin to finger root along
            the approach axis. Most generators put the grasp pose at
            the TCP, so the fingers extend from offset ``-finger_z_offset``
            to ``-finger_z_offset + finger_length`` along approach.
        max_points_per_finger: Reject if more than this many cloud points
            are inside either finger's volume.
        min_points_in_closing_volume: If > 0, reject grasps with fewer
            than this many points in the closing volume.

    Returns:
        Filtered list of grasps.
    """
    if not grasps or pts is None or len(pts) == 0:
        return grasps

    pts_w = np.asarray(pts, dtype=np.float32)

    # Pre-compute scaling for normalized matrices.
    if matrices_normalized:
        if aabb_min is None or aabb_max is None:
            return grasps
        aabb_min = np.asarray(aabb_min, dtype=np.float32)
        aabb_max = np.asarray(aabb_max, dtype=np.float32)
        center_t = 0.5 * (aabb_min + aabb_max)
        scale_t = 0.5 * (aabb_max - aabb_min)
    else:
        center_t = np.zeros(3, dtype=np.float32)
        scale_t = np.ones(3, dtype=np.float32)

    half_open = 0.5 * float(gripper_opening)
    half_thick = 0.5 * float(finger_thickness)
    half_height = float(finger_height)
    z_near = float(-finger_z_offset)
    z_far = z_near + float(finger_length)

    keep: List[dict] = []
    for g in grasps:
        try:
            M_raw = np.asarray(g.get("matrix"), dtype=np.float32).reshape(4, 4)
            M = M_raw.T if column_major else M_raw
            R = M[:3, :3]
            t = M[:3, 3].astype(np.float32)
            t_w = (t * scale_t + center_t) if matrices_normalized else t

            # Transform world points into grasp local frame.
            # p_local = R^T (p_w - t_w)
            local = (pts_w - t_w[None, :]) @ R  # numpy: (N,3) @ R == (N,3) R^T·v
            x = local[:, 0]
            y = local[:, 1]
            z = local[:, 2]

            in_z = (z >= z_near) & (z <= z_far)
            in_y = np.abs(y) <= half_height

            # Left finger: x ∈ [+half_open - half_thick, +half_open + half_thick]
            in_x_left = (x >= (half_open - half_thick)) & (x <= (half_open + half_thick))
            # Right finger: x ∈ [-half_open - half_thick, -half_open + half_thick]
            in_x_right = (x >= (-half_open - half_thick)) & (x <= (-half_open + half_thick))
            # Closing volume: between the fingers
            in_x_closing = np.abs(x) < (half_open - half_thick)

            mask_z_y = in_z & in_y
            n_left = int(np.count_nonzero(mask_z_y & in_x_left))
            n_right = int(np.count_nonzero(mask_z_y & in_x_right))

            if n_left > int(max_points_per_finger):
                continue
            if n_right > int(max_points_per_finger):
                continue

            if int(min_points_in_closing_volume) > 0:
                n_closing = int(np.count_nonzero(mask_z_y & in_x_closing))
                if n_closing < int(min_points_in_closing_volume):
                    continue

            keep.append(g)
        except Exception:
            keep.append(g)

    return keep

def _filter_reject_from_behind(
    grasps,
    pts,
    *,
    observation_point,
    matrices_normalized = True,
    column_major = False,
    aabb_min = None,
    aabb_max = None,
    approach_axis = 2,
    min_forward_dot = 0.0,
):
    """Reject grasps that approach the object from the wrong side.

    A grasp is "from behind" when its approach direction points away
    from the observation viewpoint instead of toward the object. The
    forward direction is computed once per call as the unit vector from
    ``observation_point`` to the object centroid. Grasps whose approach
    direction has a dot product below ``min_forward_dot`` with this
    forward vector are rejected.

    Conventions match the rest of this module:
      - approach axis: rotation column 2 (+Z local), pointing toward the object.

    Args:
        grasps: List of grasp dicts with ``matrix`` field.
        pts: Object point cloud (used only for the centroid).
        observation_point: 3D point representing where the robot will
            execute the grasp from (e.g. the TCP at ``initgrasp``).
        matrices_normalized: If True, grasp translation is in unit cube.
        column_major: Whether matrices are stored column-major.
        aabb_min, aabb_max: Required when ``matrices_normalized`` is True.
        approach_axis: Column index of the approach direction.
        min_forward_dot: Minimum forward dot to keep the grasp.
            0.0 = reject only strict back approaches; >0 = stricter.

    Returns:
        Filtered list of grasps.
    """
    import numpy as _np
    if not grasps or pts is None or len(pts) == 0:
        return grasps
    obs = _np.asarray(observation_point, dtype=_np.float32).ravel()
    if obs.shape[0] != 3:
        return grasps
    centroid = _np.asarray(pts, dtype=_np.float32).mean(axis=0)
    forward = centroid - obs
    fn = float(_np.linalg.norm(forward))
    if fn < 1e-6:
        return grasps
    forward = forward / fn

    keep = []
    for g in grasps:
        try:
            M_raw = _np.asarray(g.get("matrix"), dtype=_np.float32).reshape(4, 4)
            M = M_raw.T if column_major else M_raw
            R = M[:3, :3]
            approach = R[:, int(approach_axis)]
            an = float(_np.linalg.norm(approach))
            if an < 1e-6:
                continue
            approach = approach / an
            if float(_np.dot(approach, forward)) < float(min_forward_dot):
                continue
            keep.append(g)
        except Exception:
            keep.append(g)
    return keep


def _filter_gripper_width(
    grasps: List[dict],
    pts: np.ndarray,
    *,
    matrices_normalized: bool = True,
    column_major: bool = False,
    aabb_min: Optional[np.ndarray] = None,
    aabb_max: Optional[np.ndarray] = None,
    max_width: float = 0.080,
    slab_half_depth: float = 0.03,
    finger_half_height: float = 0.02,
    min_points: int = 10,
    stats: Optional[dict] = None,
) -> List[dict]:
    """Drop grasps the printed gripper cannot make: the object is wider
    than its opening along the closing axis (2026-09-11).

    GraspGen's checkpoints assume a 105 mm (Franka) or 136 mm (Robotiq)
    opening; the printed gripper opens 85 mm. A grasp whose closing line
    spans more object than that ends with the fingers on the object's
    sides. Measured per grasp in its own frame: cloud points in a slab
    around the object's centre along the approach axis (the slab sits at
    the point of the approach axis closest to the cloud centroid, so no
    assumption about where the frame origin is), within the finger height,
    and the 3rd-97th percentile span along the closing axis is the width.
    Fewer than ``min_points`` in the slab: the grasp is kept (no verdict).
    """
    if not grasps or pts is None or len(pts) == 0:
        return grasps
    pts_w = np.asarray(pts, dtype=np.float32)
    centroid = pts_w.mean(axis=0)
    if matrices_normalized:
        if aabb_min is None or aabb_max is None:
            return grasps
        aabb_min = np.asarray(aabb_min, dtype=np.float32)
        aabb_max = np.asarray(aabb_max, dtype=np.float32)
        center_t = 0.5 * (aabb_min + aabb_max)
        scale_t = 0.5 * (aabb_max - aabb_min)
    kept: List[dict] = []
    widest = 0.0
    n_cut = 0
    for g in grasps:
        try:
            M_raw = np.asarray(g.get("matrix"), dtype=np.float32).reshape(4, 4)
            M = M_raw.T if column_major else M_raw
            R = M[:3, :3]
            t_w = M[:3, 3].copy()
            if matrices_normalized:
                t_w = center_t + t_w * scale_t
            local = (pts_w - t_w[None, :]) @ R          # points in the grasp frame
            z_c = float(((centroid - t_w) @ R)[2])       # object centre along the approach
            sel = (np.abs(local[:, 2] - z_c) <= slab_half_depth) & (np.abs(local[:, 1]) <= finger_half_height)
            if int(sel.sum()) < min_points:
                kept.append(g)
                continue
            x = local[sel, 0]
            width = float(np.percentile(x, 97) - np.percentile(x, 3))
            widest = max(widest, width)
            if width > max_width:
                n_cut += 1
                continue
            kept.append(g)
        except Exception:
            kept.append(g)
    if stats is not None:
        stats["cut"] = n_cut
        stats["widest_m"] = widest
    return kept

