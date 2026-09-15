"""
frame_buffer.py
================

This module defines the :class:`FrameBuffer` class which subscribes to
RGB and depth image topics along with their corresponding
``CameraInfo`` topics. It keeps the most recent frames and camera
intrinsics handy for downstream processing. The class is designed to
operate with ROS2 and the sensor messages are converted to OpenCV
images using ``cv_bridge``.

Example::

    from grasp_pipeline.frame_buffer import FrameBuffer
    import rclpy

    rclpy.init()
    node = rclpy.create_node('example')
    fb = FrameBuffer(node,
                     color_topic="/camera/color/image_raw",
                     depth_topic="/camera/aligned_depth_to_color/image_raw",
                     color_info_topic="/camera/color/camera_info",
                     depth_info_topic="/camera/aligned_depth_to_color/camera_info")
    # Later, get latest frames:
    rgb, depth, (fx, fy, cx, cy), depth_scale, header = fb.latest()
    ...

"""

from __future__ import annotations

import collections
import time

import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from cv_bridge import CvBridge


class FrameBuffer:
    """Subscribe to RGB and depth topics and cache the latest frames.

    ``FrameBuffer`` subscribes to both color and aligned depth images along
    with their camera info topics. It stores the most recent frames
    along with camera intrinsics so that consumers can query the latest
    RGB/depth frames and intrinsics without managing subscriptions
    directly.

    :param node: The ROS2 node used for creating subscriptions.
    :param color_topic: The topic name for the RGB image.
    :param depth_topic: The topic name for the aligned depth image.
    :param color_info_topic: The topic name for the RGB camera info.
    :param depth_info_topic: The topic name for the depth camera info.
    """

    def __init__(self, node: Node,
                 color_topic: str = "/camera/camera/color/image_raw",
                 depth_topic: str = "/camera/camera/aligned_depth_to_color/image_raw",
                 color_info_topic: str = "/camera/color/camera_info",
                 depth_info_topic: str = "/camera/aligned_depth_to_color/camera_info") -> None:
        self.node = node
        self.bridge = CvBridge()

        self.rgb_frame: np.ndarray | None = None  # HxWx3 BGR
        self.depth_frame: np.ndarray | None = None  # HxW depth
        self.rgb_header = None
        self.depth_header = None
        # Ring of the most recent depth frames (2026-09-09): the capture used
        # to WAIT for 5 fresh frames after the selection (0.45 s at the
        # ~10 Hz they reach this node); with the arm still they are as good
        # taken before the selection as after it, so keep them ready.
        self.depth_ring: collections.deque = collections.deque(maxlen=8)   # (monotonic_s, frame)

        # Intrinsics defaults
        self.fx: float = 615.0
        self.fy: float = 615.0
        self.cx: float = 320.0
        self.cy: float = 240.0
        self.depth_scale: float = 1000.0  # mm → m for uint16
        self.have_color_K: bool = False
        self.have_depth_K: bool = False
        # Robot self-filter hook (2026-09-15): (shape, frame_id) -> uint8 mask
        # or None. Pixels under the mask are zeroed in every stored depth
        # frame, so the gripper never reaches a cloud, a neighbour box or a
        # distance readout. Set by the node once the filter exists.
        self.depth_mask_fn = None

        # Subscriptions
        #
        # QoS note (2026-08-06): `qos_profile_sensor_data` is BEST_EFFORT, the usual
        # choice for camera streams — but a BEST_EFFORT subscriber CANNOT match a
        # RELIABLE publisher. The realsense2_camera build on this machine publishes
        # image topics as RELIABLE + TRANSIENT_LOCAL, so this subscription never
        # connected: `ros2 topic hz` showed data (it makes its own compatible
        # subscriber) while this node's frame buffer stayed empty, `_publish_debug_frame`
        # returned early on `rgb is None`, and `/yolo/debug_image` published nothing.
        #
        # Default to RELIABLE to match that camera; override the
        # `camera_qos_reliability` parameter with "best_effort" for cameras that
        # only offer BEST_EFFORT.
        image_qos = qos_profile_sensor_data
        try:
            if not node.has_parameter("camera_qos_reliability"):
                node.declare_parameter("camera_qos_reliability", "reliable")
            _rel = str(node.get_parameter("camera_qos_reliability").value).strip().lower()
        except Exception:
            _rel = "reliable"
        if _rel == "reliable":
            image_qos = QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )

        self.sub_color = node.create_subscription(Image, color_topic, self._color_cb, image_qos)
        self.sub_depth = node.create_subscription(Image, depth_topic, self._depth_cb, image_qos)
        self.sub_cinfo = node.create_subscription(CameraInfo, color_info_topic, self._cinfo_cb, 10)
        self.sub_dinfo = node.create_subscription(CameraInfo, depth_info_topic, self._dinfo_cb, 10)

    # --- Callbacks ---
    def _color_cb(self, msg: Image) -> None:
        """Callback for RGB image messages.

        Converts the ROS image to a BGR ``numpy`` array for OpenCV and
        stores the header for later use.
        """
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.rgb_header = msg.header
        except Exception:
            # ignore conversion errors
            self.rgb_frame = None
            self.rgb_header = None

    def _depth_cb(self, msg: Image) -> None:
        """Callback for depth image messages.

        Stores the depth image as a numpy array and infers the scale.
        """
        arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        mask_fn = self.depth_mask_fn
        if mask_fn is not None:
            try:
                m = mask_fn(arr.shape[:2], str(msg.header.frame_id))
            except Exception:
                m = None
            if m is not None:
                arr = arr.copy()          # cv_bridge may hand back a read-only view
                arr[m > 0] = 0
        self.depth_frame = arr
        self.depth_header = msg.header
        self.depth_ring.append((time.monotonic(), arr))   # a new array per message: no copy needed
        # infer scale (uint16 mm → m or float32 m)
        if arr.dtype == np.uint16:
            self.depth_scale = 1000.0
        elif np.issubdtype(arr.dtype, np.floating):
            self.depth_scale = 1.0

    def _cinfo_cb(self, msg: CameraInfo) -> None:
        K = msg.k
        self.fx, self.fy, self.cx, self.cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])
        self.have_color_K = True

    def _dinfo_cb(self, msg: CameraInfo) -> None:
        K = msg.k
        self.fx, self.fy, self.cx, self.cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])
        self.have_depth_K = True

    # --- API ---
    def recent_depths(self, n: int, max_age_s: float = 0.6):
        """The last ``n`` depth frames if the newest is younger than
        ``max_age_s`` and the whole set spans less than ``max_age_s``; else
        None (caller then waits for fresh frames the old way)."""
        frames = list(self.depth_ring)[-n:]
        if len(frames) < n:
            return None
        now = time.monotonic()
        if now - frames[-1][0] > max_age_s or frames[-1][0] - frames[0][0] > max_age_s:
            return None
        return [f for _, f in frames]

    def latest(self):
        """Return the latest RGB/depth frames and intrinsics.

        :returns: (rgb_frame, depth_frame, (fx, fy, cx, cy), depth_scale, rgb_header)
        """
        return self.rgb_frame, self.depth_frame, (self.fx, self.fy, self.cx, self.cy), self.depth_scale, self.rgb_header