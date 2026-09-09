"""Motion subsystem for grasp_pipeline.

Owns every code path that drives the arm to a registration / observation
pose:

  - Fast-replay through ``ur3e_fast_observation`` (cached named poses;
    publishes on the replay topic, waits on the status topic).
  - PTP fallback through the ``observation_ptp_commander`` subprocess.
  - Joint-space targets, cartesian look-at arcs, single pose targets.

Consumers see one entry point — ``MotionDispatcher`` — and never have to
care which path was taken. This keeps ``grasp_selection_node`` agnostic
of the motion layer and lets us swap implementations later (e.g. a real
action server) without touching the orchestrator.
"""

from .fast_replay import FastReplayHandle  # noqa: F401
from .motion_dispatcher import MotionDispatcher  # noqa: F401
