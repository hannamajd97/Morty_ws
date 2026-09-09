"""Grasping subsystem.

Pulled out of grasp_selection_node so the orchestrator stops carrying
GraspGen RPC plumbing, utility-score arithmetic, and EE-orientation
TF chasing.

Components:

  - GraspGenClient — issues RPCs to the GraspGen ZMQ service and
    returns a list of candidate grasps (or None on failure).
  - GraspScorer — utility math (count, diversity, mean score,
    instability), grasp_matrix_to_pose unit-cube denormalisation, and
    the early-stop policy used during multi-view sweeps.
  - GraspFilterPipeline — orchestrates the chain of geometric filters
    in grasp_filters.py and re-weighters that runs on every candidate
    set before scoring.
  - OrientationFilter — TF-aware EE-orientation consistency filter and
    score re-weighter. Holds the small quaternion helpers as well.

All components are stateless beyond a back-reference to the node so
they can read parameters and use TF / camera intrinsics on demand.
Parameter names are unchanged.
"""

from .graspgen_client import GraspGenClient  # noqa: F401
from .grasp_scorer import GraspScorer  # noqa: F401
from .filter_pipeline import GraspFilterPipeline  # noqa: F401
from .orientation import OrientationFilter  # noqa: F401
from .pick_pipeline import PickPipeline  # noqa: F401
