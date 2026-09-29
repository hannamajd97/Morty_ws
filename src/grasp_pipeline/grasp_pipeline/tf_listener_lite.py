"""Moved to ur3e_fast_observation.tf_listener_lite (2026-09-29): importing it
from here ran grasp_pipeline/__init__, which loads YOLO and torch — the VLA
recorder paid ~13 s and hundreds of MB for a /tf listener. Kept as a
re-export so existing imports keep working."""
from ur3e_fast_observation.tf_listener_lite import DecimatedTransformListener  # noqa: F401
