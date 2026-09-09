"""Debug rendering subsystem.

Owns every code path that produces user-visible debug output:

  - The annotated debug image (YOLO detections + selection highlight +
    hand landmarks + status overlay) -> DebugRenderer.
  - RViz preview point clouds (multi-view registration preview, TSDF
    surface reconstruction) -> CloudPublisher.

Pulled out of grasp_selection_node so the orchestrator no longer carries
PointCloud2 packing helpers, TSDF config plumbing, or cv_bridge-y
overlay code. Public surface is consumed only from the orchestrator;
parameter names and topic names are unchanged.
"""

from .debug_renderer import DebugRenderer  # noqa: F401
from .cloud_publisher import CloudPublisher  # noqa: F401
