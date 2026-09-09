"""
grasp_pipeline package

YOLO-based detection, segmentation, depth-crop processing, and grasp
selection for robotic grasping. Voice/TTS lived here historically and
has been REMOVED — the communication_llm package owns speech now and
talks to this pipeline via:

  - parameter `auto_select_label` on /yolo_live_select_grasp (label
    targeting), set with rcl_interfaces SetParameters;
  - topic `/yolo/select_index` (numeric index targeting);
  - topic `/grasp_pipeline/selection_prompt` (this pipeline -> chat
    layer; user-facing prompt strings to be spoken by communication_llm).

Modules:

* ``frame_buffer`` – subscription helpers for RGB and depth images
* ``yolo_detector`` – wrapper around Ultralytics YOLO-World
* ``debug_image_publisher`` – publishes annotated debug images
* ``grasp_client`` – simple ZMQ client for GraspGen service
* ``utils`` – grasp selection helpers and geometry conversions
* ``crop_utils`` – functions to crop depth images to point clouds
* ``plane_filter`` – helpers to estimate a support plane and remove
  below-plane points prior to grasp generation
* ``grasp_selection_node`` – ROS2 node combining the above to perform
  object detection, user selection and grasp generation
"""

from .frame_buffer import FrameBuffer  # noqa: F401
from .yolo_detector import YoloWorldDetector  # noqa: F401
from .debug_image_publisher import DebugImagePublisher  # noqa: F401
from .grasp_client import send_points_to_graspgen  # noqa: F401
from .utils import pick_best_grasp, _R_t_from_matrix16, _quat_from_R  # noqa: F401
from .crop_utils import crop_depth_to_cloud_and_numpy  # noqa: F401
from .plane_filter import estimate_plane_svd, filter_points_above_plane  # noqa: F401