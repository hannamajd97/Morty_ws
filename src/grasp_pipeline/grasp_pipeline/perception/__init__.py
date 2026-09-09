"""Perception subsystem.

Owns single-view cloud extraction, multi-view tracking helpers, and the
shared filter chain that turns a YOLO selection into a clean point
cloud ready for grasp generation.
"""

from .cloud_extraction import CloudExtractor  # noqa: F401
