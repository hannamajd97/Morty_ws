"""Reach warning for detected objects, before any planning happens.

Majd, 2026-09-09: "I only find out an object is too far when the executor
has already tried 26 grasps. Make it simple: front grasps, angle from the
base, distance from the base, worst case." This module is that rule.

The limit depends on the direction (2026-09-09 measurements, sampled
forward kinematics with the calibrated UR3e link lengths and the 12 cm
tool offset): an object straight out along the base axis can be grasped
level from the front at up to ~0.84 m, one perpendicular to the base axis
only at ~0.68 m. The table below is the 99.5th percentile TCP distance per
10-degree band over 800k random joint configurations, restricted to the
grasp family the filter chain keeps: tool within 25 degrees of horizontal,
approaching from the base side (yaw within 60 degrees). Regenerate with
tools/reach_table.py if the tool offset or the arm changes.

Percent = distance / limit(angle). Under 85 % the planner still has a free
choice of orientation (both of yesterday's edge cases sat at 90-93 %):
green. 85-100 %: only the best-placed grasps exist and any neighbour on
that side becomes fatal: orange, "edge". Over 100 %: red, "too far", with
the centimetres the object must come toward the base to reach 85 %.

This is a warning label, not a planner: it ignores neighbours, the table
edge and the mount, and it judges the object's centre, not a grasp pose.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

# (angle off the base z axis in degrees, reach limit in metres)
FRONT_LEVEL_REACH = (
    (5, 0.837), (15, 0.843), (25, 0.829), (35, 0.810), (45, 0.785),
    (55, 0.760), (65, 0.731), (75, 0.705), (85, 0.675),
)
GREEN_BELOW = 0.85
RED_FROM = 1.00


def reach_limit_m(angle_deg: float) -> float:
    """Linear interpolation in the table; clamped at both ends."""
    a = min(85.0, max(5.0, float(angle_deg)))
    angs = [x for x, _ in FRONT_LEVEL_REACH]
    lims = [y for _, y in FRONT_LEVEL_REACH]
    return float(np.interp(a, angs, lims))


def reach_info(p_base: Sequence[float]) -> Dict[str, object]:
    """Reach verdict for a point in base_link.

    Returns dist_m, angle_deg (off the base z axis), limit_m, pct (0-100+),
    state ('green' | 'orange' | 'red'), closer_cm (how far toward the base
    to land at the green threshold; 0 when already green) and note (the
    short label suffix, empty when green)."""
    p = np.asarray(p_base, dtype=float)[:3]
    dist = float(np.linalg.norm(p))
    if dist < 1e-6:
        return {"dist_m": 0.0, "angle_deg": 0.0, "limit_m": reach_limit_m(0.0),
                "pct": 0.0, "state": "green", "closer_cm": 0.0, "note": ""}
    angle = math.degrees(math.acos(min(1.0, abs(p[2]) / dist)))
    limit = reach_limit_m(angle)
    frac = dist / limit
    pct = 100.0 * frac
    if frac < GREEN_BELOW:
        state, note = "green", ""
    elif frac < RED_FROM:
        state, note = "orange", f"edge {pct:.0f}% @{angle:.0f}deg"
    else:
        state, note = "red", f"too far {pct:.0f}% @{angle:.0f}deg"
    closer = max(0.0, (dist - GREEN_BELOW * limit) * 100.0) if state != "green" else 0.0
    return {"dist_m": dist, "angle_deg": angle, "limit_m": limit, "pct": pct,
            "state": state, "closer_cm": closer, "note": note}
