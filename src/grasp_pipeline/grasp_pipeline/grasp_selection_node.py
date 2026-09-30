"""
ROS2 node orchestrating object detection, user selection and grasp planning.

This module defines the :class:`YoloLiveSelectGrasp` node, which subscribes
to RGB and depth images, performs object detection with YOLO‑World,
interacts with the user via voice or pointing to select an object, crops
the corresponding point cloud, calls GraspGen to generate candidate
grasps and then filters and selects the best grasp according to a set
of heuristics. The node publishes both the selected grasp pose and
associated debug information.

The implementation here was refactored from a single large file into
multiple modules. Helper functions for geometry and grasp filtering
live in :mod:`grasp_pipeline.geometry_utils` and
:mod:`grasp_pipeline.grasp_filters` respectively. Splitting the code
improves maintainability and makes the helper functions reusable.
"""

#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
import sys
import os
import copy
import io
import json
import subprocess
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import traceback
import math

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    qos_profile_sensor_data,
)

from sensor_msgs.msg import PointCloud2, Image, PointField
from std_msgs.msg import Header, Int32, Int64, String
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from .frame_buffer import FrameBuffer
from .view_cloud import ViewCloudPublisher
from .perception.self_filter import GripperSelfFilter
from .perception.open_set import OpenSetDetector
from .perception.identity_check import IdentityCheck
from .yolo_detector import YoloWorldDetector
from .grasp_client import send_points_to_graspgen
from .crop_utils import crop_depth_to_cloud_and_numpy
from .segmentation import GrabCutConfig, build_segmenter
from .plane_filter import estimate_plane_svd, filter_points_above_plane
from .utils import _clean_cropped_cloud, pick_best_grasp, _R_t_from_matrix16
from .utils import keep_largest_cluster, radius_outlier_filter
from .debug_image_publisher import DebugImagePublisher
from .point_selector import PointOrVoiceSelector
from .astra_pointing import AstraPointingCamera
from .registration import (
    RegistrationConfig,
    RegistrationView,
    compute_observation_gain,
    lookup_transform_matrix,
    register_and_fuse_views,
    transform_points,
    voxel_downsample,
)
from .tsdf_fusion import TSDFConfig, integrate_tsdf_surface_cloud
from .motion import MotionDispatcher
from .debug import CloudPublisher, DebugRenderer
from .grasping import (
    GraspGenClient,
    GraspScorer,
    GraspFilterPipeline,
    OrientationFilter,
    PickPipeline,
)
from .grasping.precompute import GraspPrecomputer
from .perception import CloudExtractor
from .registration_pipeline import MultiViewRegistrar
from .selection import declare_all_parameters
from .nbv_candidates import (
    generate_dynamic_nbv_candidates,
    generate_joint_interpolated_candidates,
    generate_look_at_arc_waypoints,
    parse_srdf_named_targets,
)
from tf_transformations import quaternion_from_matrix
# Add these imports after the existing imports (around line 50-60)
import tf2_ros
from .tf_listener_lite import DecimatedTransformListener
import tf2_geometry_msgs
import rclpy.duration

from .geometry_utils import (
    _simplify_object_name,
    _aabb_min_max,
    _denorm_T_from_unit_cube,
    _inside_aabb,
    _estimate_view_dir_from_cloud,
    _make_T_grasp_to_tcp,
    _publish_marker,
)
from .grasp_filters import (
    _filter_view_consistent,
    _filter_no_under_grasp,
    _filter_side_center_window,
    _reweight_scores_by_centroid,
    _filter_roll_upright_by_plane,
    _filter_reject_side_grasps,
    _filter_top_only_plane,
    _filter_finger_penetration,
    _filter_reject_from_behind,
)


# Back-compat alias for any external code that imported the private class.
# The implementation has moved to grasp_pipeline.motion.fast_replay.
from .motion.fast_replay import FastReplayHandle as _FastReplayHandle  # noqa: E402,F401


def _number_duplicates(dets: List[Dict[str, Any]]) -> None:
    """Set ``display`` on every detection: the simplified label, and for
    several objects of one label "<label>", "<label> 2", ... numbered LEFT
    TO RIGHT by box centre (2026-09-24). Numbered in list order — score
    order after the dedupe — the two vitamin tubes swapped names whenever
    their scores crossed, and everything that follows an object by its name
    saw it jump: the grasp precompute threw the stored grasps away ("object
    moved during the capture") and the pick took the slow fresh path. The
    list order itself (the selection indices) is left alone."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        groups.setdefault(d["simplified_label"], []).append(d)
    for label, group in groups.items():
        group = sorted(group, key=lambda d: (0.5 * (d["bbox"][0] + d["bbox"][2]),
                                             0.5 * (d["bbox"][1] + d["bbox"][3])))
        for i, d in enumerate(group, 1):
            d["display"] = label if i == 1 else f"{label} {i}"


def _bbox_colour_name(bgr_frame, bbox) -> str:
    """Basic colour name for the central region of a detection bbox.

    Used to answer "pick the RED one" — published per detection in
    /yolo/all_detections so intent_router can disambiguate same-label
    objects by colour (Hamid-schema integration, 2026-08-20). Returns
    "" when undecidable.
    """
    try:
        import cv2
        x1, y1, x2, y2 = (int(v) for v in bbox)
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 6:
            return ""
        # Central third — avoids background bleeding into the estimate.
        patch = bgr_frame[y1 + h // 3:y2 - h // 3, x1 + w // 3:x2 - w // 3]
        if patch.size == 0:
            return ""
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        sat, val = hsv[:, 1].astype(np.float32), hsv[:, 2].astype(np.float32)
        # sat threshold 60->45 (2026-08-25, measured): in the shadowed
        # right of the scene the yellow cap sits at sat ~50-80 val ~76 —
        # at 60 it kept falling into the "gray" give-up branch.
        coloured = hsv[(sat > 45) & (val > 45)]
        # Mostly unsaturated -> judge by brightness only.
        if coloured.shape[0] < 0.2 * hsv.shape[0]:
            v_med = float(np.median(val))
            if v_med > 190:
                return "white"
            if v_med < 60:
                return "black"
            return "gray"
        h_med = float(np.median(coloured[:, 0]))  # OpenCV hue: 0..179
        if h_med < 10 or h_med >= 170:
            return "red"
        if h_med < 25:
            return "orange"
        if h_med < 35:
            return "yellow"
        if h_med < 85:
            return "green"
        if h_med < 130:
            return "blue"
        return "purple"
    except Exception:
        return ""


class YoloLiveSelectGrasp(Node):
    """YOLO detection, user selection and grasp synthesis pipeline.

    This ROS2 node subscribes to camera topics, runs YOLO‑World to
    detect objects, interacts with the user via text‑to‑speech and
    finger pointing or voice to select one of the detections, extracts
    and cleans the corresponding point cloud, calls a GraspGen service
    to generate candidate grasps and then filters and selects the
    best grasp. The selected grasp pose is published continuously as
    a :class:`geometry_msgs.msg.PoseStamped` so that other nodes can
    consume it reliably.
    """

    def __init__(self) -> None:
        super().__init__("yolo_live_select_grasp")
        # All ROS parameters declared in selection.params for clarity.
        # Names / defaults are byte-identical to the legacy inline block
        # so launch files and CLI overrides keep working unchanged.
        declare_all_parameters(self)
        # Initialise components
        classes = [c.strip() for c in self.get_parameter("classes").value.split(",") if c.strip()]
        self._allowed_classes = set(c.lower() for c in classes) if classes else set()
        # Build per-class confidence override map from the two parallel
        # string-list params. Silently skip entries where either side is
        # missing or the label is empty.
        _pcm_labels = list(
            self.get_parameter("per_class_min_conf_labels").value or []
        )
        _pcm_values = list(
            self.get_parameter("per_class_min_conf_values").value or []
        )
        per_class_min_conf: Dict[str, float] = {}
        for lbl, val in zip(_pcm_labels, _pcm_values):
            lbl_s = str(lbl).strip()
            if lbl_s:
                per_class_min_conf[lbl_s] = float(val)
        self.detector = YoloWorldDetector(
            model_path=self.get_parameter("yolo_world_model").value,
            classes=classes,
            conf=float(self.get_parameter("conf").value),
            imgsz=int(self.get_parameter("imgsz").value),
            device=self.get_parameter("device").value,
            half=bool(self.get_parameter("yolo_half").value),
            per_class_min_conf=per_class_min_conf,
            logger=self.get_logger(),
        )
        self.fb = FrameBuffer(
            self,
            color_topic=self.get_parameter("camera_topic").value,
            depth_topic=self.get_parameter("depth_topic").value,
            color_info_topic=self.get_parameter("color_info_topic").value,
            depth_info_topic=self.get_parameter("depth_info_topic").value,
        )
        # Voice/TTS removed from grasp_pipeline. communication_llm owns
        # speech and reaches the pipeline via the parameter / topic API.
        # Selector kept as an attribute for back-compat with downstream
        # code that still passes it around — it's always None.
        self.selector = None
        # Publisher for user-facing prompt strings ("Pick from: ...").
        # communication_llm subscribes and speaks them. If the topic
        # parameter is empty we publish nowhere (head-less mode).
        prompt_topic = str(
            self.get_parameter("selection_prompt_topic").value
        ).strip()
        self._prompt_pub = (
            self.create_publisher(String, prompt_topic, 10)
            if prompt_topic
            else None
        )
        # 2026-09-22: the router sets auto_select_label and waits for the
        # executor's cycle_done; when THIS node gives the label up, no cycle
        # ever starts and the router sat busy for its 90 s watchdog, dropping
        # every repeat as a duplicate. The verdict goes out here in the
        # cycle_done grammar ("failed|0|<reason>") so the router's handler
        # can take it unchanged.
        self._result_pub = self.create_publisher(String, "/grasp_pipeline/selection_result", 10)
        self.pub_detection = self.create_publisher(
            Detection2D, str(self.get_parameter("selected_detection_topic").value), 10
        )
        # Scene-awareness publisher: emits a JSON list of every current detection
        # so external nodes (chat_node, an LLM, a UI) can know what is visible
        # without running their own detector. Disable via publish_all_detections=false.
        self._publish_all_detections = bool(
            self.get_parameter("publish_all_detections").value
        )
        self.pub_all_detections = (
            self.create_publisher(
                String,
                str(self.get_parameter("all_detections_topic").value),
                10,
            )
            if self._publish_all_detections
            else None
        )
        self.pub_cloud = self.create_publisher(
            PointCloud2, str(self.get_parameter("selected_cloud_topic").value), 10
        )
        self.pub_graspgen_cloud = self.create_publisher(
            PointCloud2, self.get_parameter("graspgen_input_cloud_topic").value, 10
        )
        self.pub_fused_cloud = self.create_publisher(
            PointCloud2, str(self.get_parameter("registration_fused_cloud_topic").value), 10
        )
        self.pub_tsdf_cloud = self.create_publisher(
            PointCloud2, str(self.get_parameter("registration_tsdf_cloud_topic").value), 10
        )
        # Debug subsystem: cloud encoder/publisher (registration preview +
        # TSDF) and the debug-image renderer. Wired here so the publishers
        # they wrap exist before any registration tick runs.
        self._cloud_pub = CloudPublisher(
            self, fused_pub=self.pub_fused_cloud, tsdf_pub=self.pub_tsdf_cloud
        )
        # Colored copy of the selected crop (2026-08-07, Majd's two-window
        # request): /yolo/selected_cloud is XYZ-only for the grasp path; this
        # RGB twin feeds the result-viewer so single-view clouds show colour
        # like the TSDF does.
        self.pub_cloud_rgb = self.create_publisher(
            PointCloud2, "/yolo/selected_cloud_rgb", 10
        )
        # Published EXACTLY ONCE per cycle, at the end (2026-08-07): the cloud
        # the grasps were computed on — TSDF surface after a multiview sweep,
        # the final coloured crop in single view. The "Morty Final Result"
        # window subscribes only to this, so it never churns during a sweep.
        self.pub_final_cloud = self.create_publisher(
            PointCloud2, "/graspgen/final_cloud", 10
        )
        self.last_selected_cloud_rgb_msg = None
        self.last_tsdf_cloud_msg = None
        # ORANGE = the grasp the ROBOT actually executes (2026-08-07, Majd's
        # point 3): the C++ executor publishes its first LIN-FEASIBLE
        # candidate — which may not be GraspGen's best — on
        # /grasp_executor_lin/selected_pose. Bridge it into the viz frame so
        # the result window can draw it where the cloud is.
        self.last_viz_frame = ""
        self.pub_viz_executing = self.create_publisher(
            # Plain depth-10 QoS: best_pose_qos is defined further down in
            # __init__ and the viewer is alive during execution anyway.
            PoseArray, "/graspgen/viz/executing_pose", 10
        )
        # The camera's capture viewpoint in the viz frame — the ONLY correct
        # reference for the viewer's initial orientation (2026-08-07; all
        # hand-tuned heading presets kept facing the wrong way).
        self.pub_viz_campose = self.create_publisher(
            PoseStamped, "/graspgen/viz/camera_pose", 10
        )
        self.create_subscription(
            PoseStamped, "/grasp_executor_lin/selected_pose",
            self._on_executor_selected_pose, 10,
        )
        # The interactive LLM executor (the one the GUI actually runs)
        # publishes on its own namespaced topic — without this second
        # subscription the ORANGE executed-pick glyph silently vanished
        # when we switched executors (Majd, 2026-08-12).
        self.create_subscription(
            PoseStamped, "/grasp_executor_lin_llm/selected_pose",
            self._on_executor_selected_pose, 10,
        )
        self._debug_renderer = DebugRenderer(self)
        # Grasping subsystem: GraspGen RPC client, utility scorer, the
        # geometric filter chain, and the TF-aware EE-orientation
        # filter/reweighter. Order matters — the filter pipeline holds a
        # reference to the orientation filter.
        self._graspgen = GraspGenClient(self)
        self._scorer = GraspScorer(self)
        self._orientation_filter = OrientationFilter(self)
        self._grasp_filter = GraspFilterPipeline(self, self._orientation_filter)
        # Cloud extraction: single-view crop, per-frame filter chain, and
        # multi-view ROI tracking helpers. Reads frame buffer + selection
        # state directly from this node.
        self._cloud = CloudExtractor(self)
        # Multi-view registration: observation -> register -> fuse loop,
        # NBV candidate generation, continuous-sweep helpers. Reaches
        # back into _cloud / _motion / _grasp_filter via orchestrator
        # wrappers, so it must be created AFTER those.
        self._registrar = MultiViewRegistrar(self)
        # End-to-end pick pipeline (the legacy 580-line _publish_selected
        # method, now extracted). Reaches back into all the subsystems
        # above via the orchestrator's wrappers, so it must be created
        # last.
        self._pick = PickPipeline(self)
        # Grasp precompute cache (2026-09-11): a worker that runs the pick's
        # perception phases for every stable detection while nobody has
        # selected. Off unless `graspgen_precompute` is true; then a selection
        # with a valid stored result skips straight to publishing.
        self._precompute = GraspPrecomputer(self, self._pick)
        overlay_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.pub_selected_mask = self.create_publisher(
            Image, str(self.get_parameter("segmentation_mask_topic").value), overlay_qos
        )
        self.pub_selected_overlay = self.create_publisher(
            Image, str(self.get_parameter("segmentation_overlay_topic").value), overlay_qos
        )
        self._cv_bridge = CvBridge()
        self._segmenter = None
        self._segmenter_key = None
        best_pose_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.pub_best = self.create_publisher(PoseStamped, "/graspgen/best_pose", best_pose_qos)
        centroid_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.pub_centroid = self.create_publisher(PoseStamped, "/graspgen/centroid", centroid_qos)
        self.pub_centroid_base = self.create_publisher(PoseStamped, "/graspgen/centroid_base", centroid_qos)
        
        # TF buffer + DECIMATED listener (2026-09-08). The stock listener
        # deserialized /tf at 830 Hz on this node's executor and held the
        # interpreter lock for most of every second, slowing GraspGen 25x
        # and YOLO 30x. See tf_listener_lite.py for the measurements.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = DecimatedTransformListener(
            self.tf_buffer, self,
            max_hz=float(self.get_parameter("tf_listener_max_hz").value),
        )
        # Robot self-filter (2026-09-15): the gripper's own hulls projected
        # into the image. Detections on it are dropped in the detection
        # thread; depth under it is zeroed in the frame buffer. Off, or
        # before the URDF arrives, everything behaves exactly as before.
        self._self_filter = GripperSelfFilter(self)
        self.fb.depth_mask_fn = self._self_filter.depth_mask
        # Open-set detection + identity check (2026-09-15): the trained YOLO
        # keeps naming its ten objects; a detection that does not look like
        # its class is rejected, and everything else on the table gets a box
        # from YOLOE and a name from the naming service, once, in the
        # background. Both off = the detection thread as before.
        self._identity = IdentityCheck(self)
        self._open_set = OpenSetDetector(self)
        
        # Create additional publisher for base_link transformed pose
        self.pub_best_base = self.create_publisher(PoseStamped, "/graspgen/best_pose_base_link", best_pose_qos)
        self.pub_candidates_base = self.create_publisher(
            PoseArray, str(self.get_parameter("grasp_candidates_topic").value), best_pose_qos
        )
        # Grasp-visualisation topics (2026-08-06): every GraspGen candidate in
        # the CAMERA frame, split into filter-chain survivors and rejects, so
        # grasp_viz_viewer can colour them (green = kept, red = filtered out).
        self.pub_viz_kept = self.create_publisher(
            PoseArray, "/graspgen/viz/kept_poses", best_pose_qos
        )
        self.pub_viz_rejected = self.create_publisher(
            PoseArray, "/graspgen/viz/rejected_poses", best_pose_qos
        )
        # Best grasp in the SAME raw grasp frame as kept/rejected — the viewer
        # uses this instead of the TCP-converted /graspgen/best_pose, whose
        # flipY rotation made the blue glyph render mirrored ("from behind").
        self.pub_viz_best = self.create_publisher(
            PoseArray, "/graspgen/viz/best_pose_raw", best_pose_qos
        )
        self.pub_candidate_scores = self.create_publisher(
            Float32MultiArray, str(self.get_parameter("grasp_candidate_scores_topic").value), best_pose_qos
        )
        self.pub_selection_cycle_id = self.create_publisher(
            Int64, str(self.get_parameter("selection_cycle_id_topic").value), best_pose_qos
        )
        
        self.debug_pub = DebugImagePublisher(
            self,
            topic=str(self.get_parameter("debug_image_topic").value),
            show_window=bool(self.get_parameter("show_debug_window").value),
            window_name=str(self.get_parameter("debug_window_name").value),
            window_width=int(self.get_parameter("debug_window_width").value),
            window_height=int(self.get_parameter("debug_window_height").value),
            on_click_index=self._on_debug_window_choice,
        )
        # Click-to-select via rqt_image_view (2026-08-06): the native cv2
        # window cannot run on this machine (Wayland + executor, see
        # SYSTEM_STATE_LOG #28), so clicks come from rqt_image_view instead.
        # Enable the checkbox next to the "<topic>_mouse_left" field in
        # rqt_image_view and every left-click arrives here as a
        # geometry_msgs/Point in ORIGINAL image pixel coordinates — same
        # hit-test, same selection behaviour as clicking the old window.
        self._mouse_click_sub = self.create_subscription(
            Point,
            str(self.get_parameter("debug_image_topic").value) + "_mouse_left",
            self._on_image_view_click,
            10,
        )
        # Mouse hover from the GUI (2026-09-18): the object under the mouse
        # goes first in the precompute queue, so it is verified by the time
        # it is clicked. Hover only steers the queue; it selects nothing.
        self._hover_sub = self.create_subscription(Point, "/yolo/hover_pixel", self._on_hover_pixel, 10)
        self.last_published_pose: Optional[PoseStamped] = None
        self.last_published_pose_base: Optional[PoseStamped] = None
        self.last_centroid: Optional[PoseStamped] = None
        # Which selection cycle the cached candidates belong to (see
        # republish_last_pose gating — ghost-pick hardening 2026-08-12).
        self._candidates_cycle_id: int = -1
        self.last_centroid_base: Optional[PoseStamped] = None
        self.last_candidates_base: Optional[PoseArray] = None
        self.last_candidate_scores: Optional[Float32MultiArray] = None
        self.last_selected_detection_msg: Optional[Detection2D] = None
        self.last_selected_cloud_msg: Optional[PointCloud2] = None
        self.last_selected_mask_msg: Optional[Image] = None
        self.last_selected_overlay_msg: Optional[Image] = None
        self.last_selected_mask_np: Optional[np.ndarray] = None
        self.last_selected_debug_det: Optional[Dict[str, Any]] = None
        self.last_selected_debug_frame: Optional[np.ndarray] = None
        self.last_selected_debug_header: Optional[Header] = None
        self.last_selected_debug_source: str = ""
        self.last_selected_debug_stamp_mono: float = 0.0
        self._debug_status_text: str = ""
        # Callback groups (2026-08-06): YOLO inference in `_on_timer` takes
        # 0.9-2.7 s per frame here. With everything in the default group and a
        # SingleThreadedExecutor, that starved the debug timer completely —
        # /yolo/debug_image never published and the debug window never opened.
        # Giving the debug/republish timers their own group lets them run under
        # the MultiThreadedExecutor while inference is still busy.
        # See run_grasp_pipeline.py.
        self._debug_cb_group = MutuallyExclusiveCallbackGroup()
        # A light RealSense cloud for RViz, only while RViz listens (view_cloud.py).
        if float(self.get_parameter("view_cloud_period_s").value) > 0:
            self._view_cloud = ViewCloudPublisher(
                self, self.fb, str(self.get_parameter("view_cloud_topic").value),
                period_s=float(self.get_parameter("view_cloud_period_s").value),
                stride=int(self.get_parameter("view_cloud_stride").value),
                callback_group=self._debug_cb_group)

        self.pose_republish_timer = self.create_timer(
            0.2, self.republish_last_pose, callback_group=self._debug_cb_group
        )
        self.debug_publish_timer = self.create_timer(
            max(1.0 / max(float(self.get_parameter("debug_publish_fps").value), 1.0), 0.02),
            self._publish_debug_frame,
            callback_group=self._debug_cb_group,
        )
        self.selected_outputs_timer = self.create_timer(
            1.0 / max(float(self.get_parameter("republish_selected_rate_hz").value), 0.2),
            self._republish_selected_outputs,
            callback_group=self._debug_cb_group,
        )

        # Node state
        self.detections_cache: List[Dict[str, Any]] = []
        self.selected_idx: Optional[int] = None
        self.selection_made: bool = False
        self._infer_lock = threading.Lock()
        self._infer_running: bool = False
        self._last_infer_t: float = 0.0
        self._perf_last_log_t: float = time.time()
        self._perf_infer_count: int = 0
        self._perf_infer_ms_acc: float = 0.0
        self._selection_in_progress: bool = False
        # True while the click path waits for / runs the selected-object
        # segmentation: the detector thread then starts no new inference
        # (2026-09-17: the segmentation queued behind the detector's
        # YOLO + identity + open-set chain for 1-3 s on every click).
        self._seg_wait: bool = False
        self._selection_prompted: bool = False
        self._selection_prompt_sig: Optional[tuple] = None
        self._pointer_px = None
        self._hand_landmarks_px: Optional[List[tuple[float, float]]] = None
        self._hand_connections: Optional[List[tuple[int, int]]] = None
        self._pending_manual_choice: Optional[int] = None
        self._pending_manual_choice_stamp: float = 0.0
        # The actual clicked detection (label+bbox) — see _resolve_clicked_choice.
        self._pending_manual_choice_det: Optional[Dict[str, Any]] = None
        self._selection_cycle_start_t: float = 0.0
        # When the last pick cycle ended (executor done/failed or operator
        # abort). A queued selection older than THIS is a leftover from a
        # finished cycle and must not re-fire a pick; anything newer was
        # queued while we were genuinely waiting and is fair to consume.
        self._last_cycle_end_t: float = 0.0
        self._executor_cycle_done: bool = False
        self._executor_cycle_status: str = ""
        self._awaiting_executor_cycle_done: bool = False
        # Cycle ids are unique ACROSS pipeline restarts (2026-09-30, Majd:
        # "Pick up the small bottle." twice, nothing moved). The counter used
        # to start at 0 in every pipeline process, but the executor and the
        # router live on: after a restart the new "cycle 1" equalled the
        # cycle 1 the executor had just finished, so it skipped it as
        # "already processed", the pipeline waited for its cycle_done
        # forever and the router dropped the repeated pick as a duplicate.
        # Now each run numbers from its start second x 1000 (run 1790785538
        # -> 1790785538001, ...002): later runs always count higher, so the
        # id never repeats and never goes backwards (fewer than 1000 cycles
        # per second of pipeline lifetime — always true).
        self._cycle_id_counter: int = int(time.time()) * 1000
        self._waiting_cycle_id: int = 0

        # The camera of the running pointing session (pointing_camera,
        # 2026-09-29): the hand and the arrow are drawn on THAT camera's
        # picture — Astra pixels on the RealSense image would be nonsense.
        self._point_cam = "realsense"
        self._astra_pointing: Optional[AstraPointingCamera] = None

        def _on_hand(pts, connections):
            if self._point_cam == "astra" and self._astra_pointing is not None:
                self._astra_pointing.hand = (pts, connections)
                return
            self._hand_landmarks_px = pts
            self._hand_connections = connections

        def _on_pointer(px):
            if self._point_cam == "astra" and self._astra_pointing is not None:
                self._astra_pointing.pointer_px = px
                label = self._last_pointed_label
                self._astra_pointing.publish_overlay(
                    f"Astra pointing: {label}" if label else "Astra pointing")
                return
            self._pointer_px = px

        # Latched publisher: emits the label of the detection the user is
        # currently pointing at (or "" when no finger is over an object).
        # chat_node subscribes so the LLM can answer "what am I pointing
        # at?" or use the hover as disambiguation when the user says
        # "pick this one" with no name.
        self._pointed_object_pub = self.create_publisher(
            String, "/yolo/pointed_object", 1
        )
        self._current_names: list[str] = []
        self._last_pointed_label: str = ""

        def _on_hover(idx):
            try:
                if idx is None or not self._current_names or idx < 0 or idx >= len(self._current_names):
                    label = ""
                else:
                    label = str(self._current_names[idx])
                if label != self._last_pointed_label:
                    self._last_pointed_label = label
                    self._pointed_object_pub.publish(String(data=label))
                    self._precompute.bump(idx)
            except Exception:
                pass

        self.point_or_voice = PointOrVoiceSelector(
            vosk_selector=self.selector,
            use_mediapipe=bool(self.get_parameter("use_mediapipe_selection").value),
            min_hold_ms=int(self.get_parameter("mediapipe_min_hold_ms").value),
            steady_px=float(self.get_parameter("mediapipe_steady_px").value),
            pointer_pref_inside=bool(self.get_parameter("mediapipe_require_inside").value),
            snap_px=float(self.get_parameter("mediapipe_snap_px").value),
            ray_extension=float(self.get_parameter("mediapipe_ray_extension").value),
            min_detection_confidence=float(self.get_parameter("mediapipe_min_detection_confidence").value),
            min_tracking_confidence=float(self.get_parameter("mediapipe_min_tracking_confidence").value),
            # 3D pointing (2026-09-03): the frame buffer's aligned depth +
            # intrinsics feed the finger ray; see point_selector.py.
            pointing_mode=str(self.get_parameter("pointing_mode").value),
            cone_deg=float(self.get_parameter("mediapipe_cone_deg").value),
            hysteresis=float(self.get_parameter("mediapipe_hysteresis").value),
            use_tasks_api=bool(self.get_parameter("mediapipe_use_tasks_api").value),
            model_path=str(self.get_parameter("mediapipe_model_path").value),
            # colour + depth of the same moment (2026-09-24), from the session's camera
            depth_provider=lambda: (self._astra_pointing.sample if self._point_cam == "astra"
                                    and self._astra_pointing is not None else self.fb.paired()),
            decision=str(self.get_parameter("pointing_decision").value),
            vote_tau_s=float(self.get_parameter("pointing_vote_tau_s").value),
            vote_commit_s=float(self.get_parameter("pointing_vote_commit_s").value),
            vote_margin=float(self.get_parameter("pointing_vote_margin").value),
            sigma_deg=float(self.get_parameter("pointing_sigma_deg").value),
            evidence=str(self.get_parameter("pointing_evidence").value),
            ray_source=str(self.get_parameter("pointing_ray_source").value),
            logger=self.get_logger(),
            on_pointer=_on_pointer,
            on_hand=_on_hand,
            on_hover=_on_hover,
        )
        # Live switch for the hand pointing (2026-09-18: while Majd rearranged
        # objects with the GUI, MediaPipe read his hand as pointing and picked
        # the same object twice with no click). The GUI's Detection tab sets
        # use_mediapipe_selection; the tracker keeps running, its verdict is
        # simply not used while off.
        def _on_params(params):
            from rcl_interfaces.msg import SetParametersResult
            for prm in params:
                if prm.name == "use_mediapipe_selection":
                    on = bool(prm.value) and getattr(self.point_or_voice, "tracker", None) is not None
                    self.point_or_voice.use_mediapipe = on
                    self.get_logger().info(f"Hand pointing selection {'ON' if on else 'OFF'} (use_mediapipe_selection).")
                elif prm.name == "pointing_camera":
                    cam = str(prm.value).strip().lower()
                    if cam not in ("realsense", "astra"):
                        return SetParametersResult(successful=False,
                                                   reason="pointing_camera must be 'realsense' or 'astra'")
                    if cam == "astra":
                        self._astra()   # start buffering Astra frames before the next session
                    self.get_logger().info(f"Pointing camera: {cam} (the next pointing session watches it).")
            return SetParametersResult(successful=True)
        self.add_on_set_parameters_callback(_on_params)
        if str(self.get_parameter("pointing_camera").value).strip().lower() == "astra":
            self._astra()
        self.get_logger().info(
            f"Selection input: mediapipe_enabled={self.point_or_voice.use_mediapipe}, "
            f"pointing_mode={self.point_or_voice.pointing_mode}, "
            f"pointing_decision={self.point_or_voice.decision}, evidence={self.point_or_voice.evidence_mode}, "
            f"ray_source={self.point_or_voice.ray_source}, "
            f"pointing_camera={self.get_parameter('pointing_camera').value}, "
            f"require_selection={bool(self.get_parameter('require_selection').value)}, "
            f"attempt_timeout_s={float(self.get_parameter('selection_attempt_timeout_s').value):.1f}, "
            f"manual_wait_s={float(self.get_parameter('selection_manual_wait_s').value):.1f}"
        )

        # ------------------------------------------------------------------
        # Motion dispatcher: owns fast-replay state + every motion entry
        # point used by the registration / observation paths. Created here
        # so it sees the parameters declared above.
        # ------------------------------------------------------------------
        self._motion = MotionDispatcher(self)

        self.create_subscription(
            Int32,
            str(self.get_parameter("manual_select_topic").value),
            self._on_manual_select_idx,
            10,
        )
        # Label+bbox selection from intent_router spatial picks. Carries the
        # detection so we can re-resolve the index against the CURRENT frame
        # (same as window clicks) — raw indices go stale when YOLO reorders.
        self.create_subscription(
            String,
            "/yolo/select_detection",
            self._on_manual_select_det,
            10,
        )
        # Operator stop for an IN-FLIGHT pick cycle. The executor abort only
        # covers the execution phase; a "cancel" during segmentation/GraspGen
        # used to be ignored — the pipeline finished the cycle and handed
        # the executor fresh candidates that CLEARED its abort (trial 1001,
        # Majd 2026-08-12). Checked at PickPipeline phase boundaries.
        self._pick_abort = False
        self.create_subscription(
            String,
            "/grasp_pipeline/abort",
            self._on_pick_abort,
            10,
        )
        # 2026-09-28 (Majd): pointing confirmation — see _confirm_pointed_choice.
        # The answer has its own callback group so a busy default group
        # cannot hold a "yes" back.
        self._point_prop_id = 0
        self._point_answer = None       # (proposal id, "yes" | "no") from intent_router
        self._point_rejected = None     # {"label", "bbox", "t"} after a NO / timeout
        self._point_prop_pub = self.create_publisher(String, "/grasp_pipeline/point_proposal", 10)
        self.create_subscription(
            String,
            "/grasp_pipeline/point_answer",
            self._on_point_answer,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_subscription(
            String,
            str(self.get_parameter("executor_cycle_done_topic").value),
            self._on_executor_cycle_done,
            10,
        )
        self._registration_prepositioned_for_selection = False
        self._registration_init_move_failed_this_cycle = False
        self.get_logger().info(
            "Detector config: "
            f"model='{self.get_parameter('yolo_world_model').value}', "
            f"device={self.get_parameter('device').value}, "
            f"imgsz={self.get_parameter('imgsz').value}, "
            f"half={self.get_parameter('yolo_half').value}, "
            f"conf={self.get_parameter('conf').value}"
        )
        self.get_logger().info(
            "Detector allowed classes: "
            f"{classes if classes else 'all model classes'}"
        )
        self.get_logger().info(
            "Manual selection enabled on topic "
            f"'{self.get_parameter('manual_select_topic').value}' (std_msgs/Int32 index)."
        )
        self.get_logger().info(
            "Perception topics: "
            f"detection='{self.get_parameter('selected_detection_topic').value}', "
            f"cloud='{self.get_parameter('selected_cloud_topic').value}', "
            f"debug='{self.get_parameter('debug_image_topic').value}', "
            f"mask='{self.get_parameter('segmentation_mask_topic').value}', "
            f"mask_overlay='{self.get_parameter('segmentation_overlay_topic').value}'"
        )
        
        # Timers and threads
        # Own group so the slow YOLO callback cannot block the debug timers
        # (see the callback-group note above).
        self._infer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            max(float(self.get_parameter("infer_timer_period_s").value), 0.01),
            self._on_timer,
            callback_group=self._infer_cb_group,
        )
        # Event-driven selection wake (2026-08-21, Majd: "4 s to react to
        # a click is insane — it should select directly"): every sleep in
        # the selection thread is interruptible; a click/voice selection
        # sets the event and the loop reacts within milliseconds instead
        # of finishing its nap.
        self._selection_wake = threading.Event()
        threading.Thread(target=self._selection_thread, daemon=True).start()

        # GraspGen preload (2026-08-06): the first grasp cycle used to pay
        # ~24 s of one-time cost (importing grasp_gen + loading both
        # checkpoints onto the GPU) right when the user was waiting for their
        # first pick. Warm it in the background at startup instead — the
        # in-process sampler is a module-level singleton in grasp_client, so
        # the real cycle then reuses it instantly. Skipped in ZMQ mode (the
        # external service owns the model there).
        self.declare_parameter("graspgen_preload", True)
        if (
            bool(self.get_parameter("graspgen_preload").value)
            and os.environ.get("GRASPGEN_MODE", "inprocess").strip().lower() != "zmq"
        ):
            threading.Thread(target=self._graspgen_warmup, daemon=True).start()
        # Segmenter pre-warm (2026-08-21, Majd: "why does segmentation
        # need 4.7 s?!" — it didn't: the .pt load + CUDA warmup happened
        # lazily INSIDE the first pick). Build it and run one dummy
        # inference now, in the background, exactly like the GraspGen
        # preload above. Steady-state segmentation is then ~0.2-0.5 s.
        threading.Thread(target=self._segmenter_warmup, daemon=True).start()

    def _graspgen_warmup(self) -> None:
        try:
            t0 = time.time()
            self.get_logger().info("[graspgen-preload] loading model in background...")
            from grasp_pipeline.grasp_client import send_points_to_graspgen

            pts = np.random.uniform(-0.03, 0.03, size=(512, 3)).astype(np.float32)
            pts[:, 2] += 0.4
            # min_grasps=1 / max_tries=1: this is a warm-up, not a real query —
            # without them the default min_grasps=40 forced retry iterations.
            resp = send_points_to_graspgen(pts, num_grasps=32, min_grasps=1, max_tries=1)
            # send_points_to_graspgen swallows in-process errors into
            # {"ok": False}, so check what actually happened instead of
            # unconditionally claiming the model is warm (2026-08-27: the old
            # unconditional log line could lie on failure, and it printed the
            # moment the warmup FINISHED — if the user clicked earlier, the
            # first pick waited on the in-flight build and shared it; the
            # checkpoint-load lines it saw were the preload's own, not a
            # second load).
            from grasp_pipeline.grasp_client import sampler_is_warm
            if not sampler_is_warm():
                err = resp.get("error") if isinstance(resp, dict) else resp
                raise RuntimeError(f"model did not load: {err}")
            if isinstance(resp, dict) and resp.get("ok", False):
                self.get_logger().info(
                    f"[graspgen-preload] model warm in {time.time() - t0:.1f}s — "
                    "the first selection will skip the load."
                )
            else:
                err = resp.get("error") if isinstance(resp, dict) else resp
                self.get_logger().warn(
                    f"[graspgen-preload] model loaded+cached in {time.time() - t0:.1f}s "
                    f"but the warm-up inference failed ({err}) — the first selection "
                    "will reuse the model (no reload) but may pay CUDA kernel warmup."
                )
        except Exception as e:
            self.get_logger().warn(
                f"[graspgen-preload] failed ({e}) — first selection will load the model instead."
            )

    def _segmenter_warmup(self) -> None:
        """Load the segmentation model + warm the CUDA path at startup so
        the first pick doesn't pay for it (measured 4.7 s in-cycle)."""
        try:
            import time as _t
            t0 = _t.monotonic()
            seg = self._get_segmenter()
            if seg is None:
                return
            import numpy as _n
            dummy = _n.zeros((240, 320, 3), dtype=_n.uint8)
            try:
                seg.segment(dummy, [80, 60, 240, 180])
            except Exception:
                pass
            self.get_logger().info(
                f"[seg-preload] segmenter warm in {_t.monotonic() - t0:.1f}s "
                "— the first selection will skip the load."
            )
        except Exception as exc:
            self.get_logger().warn(f"[seg-preload] warmup failed: {exc}")

    def _get_segmenter(self):
        if not bool(self.get_parameter("segmentation_enabled").value):
            return None

        backend = str(self.get_parameter("segmentation_backend").value)
        sam_model_path = str(self.get_parameter("sam_model_path").value)
        device = str(self.get_parameter("device").value)
        cfg = GrabCutConfig(
            iters=int(self.get_parameter("grabcut_iters").value),
            margin_px=int(self.get_parameter("grabcut_margin_px").value),
            close_kernel=int(self.get_parameter("grabcut_close_kernel").value),
            open_kernel=int(self.get_parameter("grabcut_open_kernel").value),
        )
        key = (backend.strip().lower(), sam_model_path, device, cfg.iters, cfg.margin_px, cfg.close_kernel, cfg.open_kernel)
        if self._segmenter is not None and self._segmenter_key == key:
            return self._segmenter

        try:
            self._segmenter = build_segmenter(
                backend,
                sam_model_path=sam_model_path if sam_model_path else None,
                device=device,
                grabcut_cfg=cfg,
                imgsz=int(self.get_parameter("imgsz").value),
                half=bool(self.get_parameter("yolo_half").value),
            )
            self._segmenter_key = key
            self.get_logger().info(
                f"Segmentation enabled (backend={backend}, model='{sam_model_path or 'none'}')."
            )
        except Exception as e:
            self.get_logger().warn(f"Segmentation disabled (failed to init backend={backend}): {e}")
            self._segmenter = None
            self._segmenter_key = None
        return self._segmenter

    def _build_selection_dict(self, det: Dict[str, Any], header) -> Dict[str, Any]:
        return {
            "label": det.get("simplified_label", det["label"]),
            "score": float(det["score"]),
            "bbox": [int(v) for v in det["bbox"]],
            "header": header,
        }

    def _detect_current_selection(self, label: str) -> Optional[Dict[str, Any]]:
        rgb = self.fb.rgb_frame
        header = self.fb.rgb_header
        if rgb is None or header is None:
            return None
        try:
            dets = self.detector.predict(rgb)
            dets = self._self_filter.filter_detections(dets, rgb.shape[:2], str(header.frame_id))
        except Exception as e:
            self.get_logger().warn(f"Registration refresh detection failed: {e}")
            return None
        if self._allowed_classes:
            dets = [d for d in dets if d["label"].lower() in self._allowed_classes]
        if not dets:
            return None

        best = None
        wanted = str(label).strip().lower()
        for d in dets:
            d2 = d.copy()
            d2["simplified_label"] = _simplify_object_name(d["label"])
            if str(d2["simplified_label"]).strip().lower() != wanted:
                continue
            if best is None or float(d2["score"]) > float(best["score"]):
                best = d2
        if best is None:
            return None
        return self._build_selection_dict(best, header)

    def _publish_detection_message(self, sel: Dict[str, Any]) -> Detection2D:
        det_msg = Detection2D()
        det_msg.header = sel["header"]
        x1, y1, x2, y2 = sel["bbox"]
        bbox_msg = BoundingBox2D()
        if hasattr(bbox_msg.center, "x"):
            bbox_msg.center.x = float((x1 + x2) / 2.0)
            bbox_msg.center.y = float((y1 + y2) / 2.0)
            if hasattr(bbox_msg.center, "theta"):
                bbox_msg.center.theta = 0.0
        else:
            bbox_msg.center.position.x = float((x1 + x2) / 2.0)
            bbox_msg.center.position.y = float((y1 + y2) / 2.0)
        bbox_msg.size_x = float(x2 - x1)
        bbox_msg.size_y = float(y2 - y1)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(sel.get("simplified_label", sel["label"]))
        hyp.hypothesis.score = float(sel["score"])
        det_msg.bbox = bbox_msg
        det_msg.results.append(hyp)
        self.pub_detection.publish(det_msg)
        self.last_selected_detection_msg = copy.deepcopy(det_msg)
        self.last_selected_debug_det = {
            "label": str(sel["label"]),
            "display": str(sel.get("simplified_label", sel["label"])),
            "score": float(sel["score"]),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        }
        return det_msg

    def _compute_selected_mask(
        self,
        sel: Dict[str, Any],
        *,
        publish_outputs: bool,
        keep_state: bool = True,
    ) -> Optional[np.ndarray]:
        selected_mask = None
        segmenter = self._get_segmenter()
        if segmenter is None or self.fb.rgb_frame is None:
            if keep_state:
                self.last_selected_mask_np = None
            return None
        try:
            bbox = [int(v) for v in sel["bbox"]]
            self.get_logger().info(
                f"[seg] starting selected-object segmentation bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
            )
            t_seg0 = time.time()
            self._seg_wait = True
            try:
                with self._infer_lock:
                    t_lock = time.time()
                    selected_mask = segmenter.segment(self.fb.rgb_frame, bbox)
            finally:
                self._seg_wait = False
            t_end = time.time()
            seg_ms = (t_end - t_seg0) * 1000.0
            self.get_logger().info(
                f"[seg] selected-object segmentation finished in {seg_ms:.1f} ms "
                f"(waited {(t_lock - t_seg0) * 1000.0:.0f} ms for the detector's lock, "
                f"model {(t_end - t_lock) * 1000.0:.0f} ms)")
            if selected_mask is None:
                if keep_state:
                    self.last_selected_mask_np = None
                return None
            dilate_px = int(self.get_parameter("segmentation_mask_dilate_px").value)
            if dilate_px > 0:
                try:
                    import cv2

                    k = max(1, 2 * dilate_px + 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    selected_mask = cv2.dilate(
                        selected_mask.astype(np.uint8), kernel, iterations=1
                    ).astype(bool)
                except Exception as e:
                    self.get_logger().warn(f"Segmentation mask dilation failed: {e}")
            area = int(np.count_nonzero(selected_mask))
            min_area = int(self.get_parameter("segmentation_min_area_px").value)
            if area < min_area:
                self.get_logger().warn(
                    f"Segmentation mask too small (area={area}px < {min_area}px); falling back to bbox crop."
                )
                if keep_state:
                    self.last_selected_mask_np = None
                return None
            self._publish_selected_mask_outputs(
                sel,
                selected_mask,
                publish_outputs=publish_outputs,
                mask_label="fresh-seg",
                keep_state=keep_state,
            )
        except Exception as e:
            self.get_logger().warn(f"Segmentation failed; falling back to bbox crop: {e}")
            if keep_state:
                self.last_selected_mask_np = None
            return None
        return selected_mask

    def _publish_selected_mask_outputs(
        self,
        sel: Dict[str, Any],
        selected_mask: np.ndarray,
        *,
        publish_outputs: bool,
        mask_label: str = "seg",
        keep_state: bool = True,
    ) -> None:
        # keep_state=False (2026-09-11): a background precompute segments an
        # object nobody selected — the overlay must not flash its mask.
        if keep_state:
            self.last_selected_mask_np = np.asarray(selected_mask, dtype=bool).copy()
            if self.fb.rgb_frame is not None:
                self.last_selected_debug_frame = self.fb.rgb_frame.copy()
                self.last_selected_debug_header = copy.deepcopy(sel.get("header", self.fb.rgb_header))
                self.last_selected_debug_source = str(mask_label)
                self.last_selected_debug_stamp_mono = time.monotonic()
        if not publish_outputs:
            return
        try:
            import cv2

            debug_frame = self.last_selected_debug_frame if self.last_selected_debug_frame is not None else self.fb.rgb_frame
            if debug_frame is None:
                return
            mask_img = (selected_mask.astype(np.uint8) * 255)
            msg = self._cv_bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
            msg.header = sel["header"]
            self.pub_selected_mask.publish(msg)
            self.last_selected_mask_msg = copy.deepcopy(msg)

            overlay = debug_frame.copy()
            overlay[selected_mask] = (0, 0, 255)
            vis = cv2.addWeighted(overlay, 0.55, debug_frame, 0.45, 0.0)
            cnts, _ = cv2.findContours(
                selected_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
            overlay_msg = self._cv_bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            overlay_msg.header = sel["header"]
            self.pub_selected_overlay.publish(overlay_msg)
            self.last_selected_overlay_msg = copy.deepcopy(overlay_msg)
            if bool(self.get_parameter("publish_debug_image").value):
                self.debug_pub.publish(
                    sel["header"],
                    debug_frame,
                    [{"label": sel["label"], "score": float(sel["score"]), "bbox": sel["bbox"]}],
                    selected_idx=0,
                    pointer_px=self._pointer_px,
                    hand_landmarks_px=self._hand_landmarks_px,
                    hand_connections=self._hand_connections,
                    selected_mask=selected_mask,
                    status_text=f"debug: {mask_label} on captured frame",
                    render_window=False,
                )
        except Exception as e:
            self.get_logger().warn(f"Failed publishing segmentation mask image: {e}")

    # ------------------------------------------------------------------
    # Cloud-extraction delegation (real impl in perception.CloudExtractor).
    # Wrappers preserve the legacy method names so internal call sites
    # in registration / sweep / dynamic-NBV code keep working.
    # ------------------------------------------------------------------

    def _extract_filtered_object_cloud(self, sel, *, publish_outputs, background=False):
        return self._cloud._extract_filtered_object_cloud(
            sel, publish_outputs=publish_outputs, background=background)

    def _filter_object_cloud(self, pts_np, cols_np, *, apply_depth_clamp=True,
                             lightweight=False, skip_cluster_filter=False,
                             skip_radius_outlier=False):
        return self._cloud._filter_object_cloud(
            pts_np, cols_np,
            apply_depth_clamp=apply_depth_clamp,
            lightweight=lightweight,
            skip_cluster_filter=skip_cluster_filter,
            skip_radius_outlier=skip_radius_outlier,
        )

    def _make_tracking_roi(self, obs_tf, reg_cfg):
        return self._cloud._make_tracking_roi(obs_tf, reg_cfg)

    def _mask_points_near_reference(self, points, ref_points, max_distance):
        return self._cloud._mask_points_near_reference(points, ref_points, max_distance)

    def _roi_fallback_is_plausible(self, obs_tf, tracked_roi):
        return self._cloud._roi_fallback_is_plausible(obs_tf, tracked_roi)

    def _tracked_followup_is_plausible(self, obs_tf, tracked_roi):
        return self._cloud._tracked_followup_is_plausible(obs_tf, tracked_roi)

    def _extract_followup_object_cloud_roi(self, tracked_roi, reg_cfg):
        return self._cloud._extract_followup_object_cloud_roi(tracked_roi, reg_cfg)

    def _project_tracking_roi_to_bbox(self, tracked_roi, reg_cfg, *, use_latest_tf=False):
        return self._cloud._project_tracking_roi_to_bbox(
            tracked_roi, reg_cfg, use_latest_tf=use_latest_tf
        )

    def _project_reference_points_to_mask(self, tracked_roi, reg_cfg):
        return self._cloud._project_reference_points_to_mask(tracked_roi, reg_cfg)

    def _extract_object_cloud_with_mask(self, sel, mask, *, debug_label):
        return self._cloud._extract_object_cloud_with_mask(sel, mask, debug_label=debug_label)

    def _build_followup_tracked_selection_and_mask(self, sel, tracked_roi, reg_cfg, bbox):
        return self._cloud._build_followup_tracked_selection_and_mask(
            sel, tracked_roi, reg_cfg, bbox
        )

    def _publish_followup_tracking_preview(self, sel, tracked_roi, reg_cfg):
        return self._cloud._publish_followup_tracking_preview(sel, tracked_roi, reg_cfg)

    def _extract_followup_object_cloud_tracked(self, sel, tracked_roi, reg_cfg):
        return self._cloud._extract_followup_object_cloud_tracked(sel, tracked_roi, reg_cfg)

    @staticmethod
    def _bbox_iou_xyxy(a, b):
        from .perception.cloud_extraction import CloudExtractor as _CE
        return _CE._bbox_iou_xyxy(a, b)

    def _get_active_image_size(self):
        return self._cloud._get_active_image_size()

    def _evaluate_continuous_bbox_framing(self, bbox, tracked_roi):
        return self._cloud._evaluate_continuous_bbox_framing(bbox, tracked_roi)

    # ------------------------------------------------------------------
    # Multi-view registration delegation. Real impl lives in
    # registration_pipeline.MultiViewRegistrar. Wrappers below preserve
    # method names so the orchestrator interface is unchanged.
    # ------------------------------------------------------------------

    def _transform_observation_to_target_frame(
        self,
        obs: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        *,
        note_prefix: str,
    ) -> Dict[str, Any]:
        return self._registrar._transform_observation_to_target_frame(
            obs, reg_cfg, note_prefix=note_prefix
        )

    def _reload_fast_replay_cache(self) -> None:
        self._motion.reload_fast_replay_cache()

    def _on_fast_replay_status(self, msg: String) -> None:
        # Subscription is owned by the dispatcher now; this wrapper is
        # only kept for back-compat with anything that referenced the
        # method by name.
        self._motion._on_fast_replay_status(msg)

    def _cancel_fast_replay(self, label: str) -> None:
        self._motion.cancel_fast_replay(label)

    def _try_fast_replay(self, target_name: str) -> Optional[bool]:
        return self._motion.try_fast_replay(target_name)

    def _move_to_registration_target(self, target_name: str) -> bool:
        return self._motion.move_to_registration_target(target_name)

    def _append_observation_commander_limits(self, cmd: List[str]) -> None:
        # Kept for back-compat; real impl lives on the dispatcher.
        self._motion._append_observation_commander_limits(cmd)

    def _move_to_joint_target(self, target_name: str, joint_values: np.ndarray) -> bool:
        return self._motion.move_to_joint_target(target_name, joint_values)

    def _start_fast_replay_process(
        self,
        target_name: str,
        joint_values: np.ndarray,
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
    ):
        return self._motion.start_fast_replay_process(
            target_name,
            joint_values,
            velocity_scaling_override=velocity_scaling_override,
            timeout_override=timeout_override,
        )

    def _start_joint_target_process(
        self,
        target_name: str,
        joint_values: np.ndarray,
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
    ):
        return self._motion.start_joint_target_process(
            target_name,
            joint_values,
            velocity_scaling_override=velocity_scaling_override,
            timeout_override=timeout_override,
        )

    def _start_cartesian_lookat_process(
        self,
        target_name: str,
        camera_pose: np.ndarray,
        object_centroid: np.ndarray,
        direction: str,
        reg_cfg: "RegistrationConfig",
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
        n_waypoints: int = 8,
    ):
        return self._motion.start_cartesian_lookat_process(
            target_name,
            camera_pose,
            object_centroid,
            direction,
            reg_cfg,
            velocity_scaling_override=velocity_scaling_override,
            timeout_override=timeout_override,
            n_waypoints=n_waypoints,
        )

    def _generate_lookat_ee_waypoints(
        self,
        camera_pose: np.ndarray,
        object_centroid: np.ndarray,
        direction: str,
        reg_cfg: "RegistrationConfig",
        *,
        n_waypoints: int = 8,
    ):
        return self._motion._generate_lookat_ee_waypoints(
            camera_pose,
            object_centroid,
            direction,
            reg_cfg,
            n_waypoints=n_waypoints,
        )

    def _start_pose_target_process(
        self,
        target_name: str,
        pose_matrix: np.ndarray,
        reference_frame: str,
        *,
        velocity_scaling_override: Optional[float] = None,
        timeout_override: Optional[float] = None,
        planner_id_override: Optional[str] = None,
    ):
        return self._motion.start_pose_target_process(
            target_name,
            pose_matrix,
            reference_frame,
            velocity_scaling_override=velocity_scaling_override,
            timeout_override=timeout_override,
            planner_id_override=planner_id_override,
        )

    def _wait_for_motion_process(
        self,
        target_name: str,
        proc,
        timeout_s: float,
        *,
        log_prefix: str,
        settle_s: float = 0.0,
    ) -> bool:
        return self._motion.wait_for_motion_process(
            target_name,
            proc,
            timeout_s,
            log_prefix=log_prefix,
            settle_s=settle_s,
        )

    def _move_to_registration_pose(
        self, target_name: str, pose_matrix: np.ndarray, reference_frame: str
    ) -> bool:
        return self._motion.move_to_registration_pose(
            target_name, pose_matrix, reference_frame
        )

    def _generate_dynamic_registration_candidates(
        self,
        first_obs_tf: Dict[str, Any],
        tracked_roi: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        requested_names: List[str],
    ) -> List[Dict[str, Any]]:
        return self._registrar._generate_dynamic_registration_candidates(
            first_obs_tf, tracked_roi, reg_cfg, requested_names
        )

    def _partition_dynamic_joint_candidates(
        self,
        candidate_specs: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self._registrar._partition_dynamic_joint_candidates(candidate_specs)

    def _generate_joint_interpolated_candidates(
        self,
        requested_names: List[str],
        reg_cfg: RegistrationConfig,
    ) -> List[Dict[str, Any]]:
        return self._registrar._generate_joint_interpolated_candidates(
            requested_names, reg_cfg
        )

    def _generate_single_joint_candidate_spec(
        self,
        target_name: str,
        reg_cfg: RegistrationConfig,
        *,
        alpha_override: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        return self._registrar._generate_single_joint_candidate_spec(
            target_name, reg_cfg, alpha_override=alpha_override
        )

    def _collect_continuous_joint_sweep_observations(self, *args, **kwargs):
        return self._registrar._collect_continuous_joint_sweep_observations(
            *args, **kwargs
        )

    def _choose_continuous_direction_without_probe(self, *args, **kwargs):
        return self._registrar._choose_continuous_direction_without_probe(
            *args, **kwargs
        )

    def _generate_camera_sphere_candidates(self, *args, **kwargs):
        return self._registrar._generate_camera_sphere_candidates(*args, **kwargs)

    @staticmethod
    def _pose_delta(T_ref: np.ndarray, T_cur: np.ndarray) -> tuple[float, float]:
        return MotionDispatcher.pose_delta(T_ref, T_cur)

    def _move_to_registration_candidate(self, candidate: Dict[str, Any]) -> bool:
        return self._motion.move_to_registration_candidate(candidate)

    def _dynamic_candidate_achieved_is_distinct(
        self,
        candidate: Dict[str, Any],
        first_obs_tf: Dict[str, Any],
        reg_cfg: RegistrationConfig,
    ) -> bool:
        return self._motion.dynamic_candidate_achieved_is_distinct(
            candidate, first_obs_tf, reg_cfg
        )

    # ------------------------------------------------------------------
    # Debug-cloud delegation (real impl in debug.CloudPublisher).
    # ------------------------------------------------------------------

    @staticmethod
    def _colors_to_rgb_float(colors: np.ndarray) -> np.ndarray:
        return CloudPublisher.colors_to_rgb_float(colors)

    def _create_cloud_xyzrgb(
        self,
        header: Header,
        points: np.ndarray,
        colors: Optional[np.ndarray],
    ) -> PointCloud2:
        return self._cloud_pub.create_cloud_xyzrgb(header, points, colors)

    def _publish_registration_preview_cloud(
        self, views: List[RegistrationView], reg_cfg: RegistrationConfig
    ) -> None:
        self._cloud_pub.publish_registration_preview(views, reg_cfg)

    def _build_tsdf_config(self) -> TSDFConfig:
        return self._cloud_pub.build_tsdf_config()

    def _publish_tsdf_cloud(
        self,
        accepted_obs_meta: List[Dict[str, Any]],
        reg_cfg: RegistrationConfig,
    ):
        return self._cloud_pub.publish_tsdf_cloud(accepted_obs_meta, reg_cfg)

    def _publish_incremental_tsdf_cloud(
        self,
        seed_obs_meta: Optional[Dict[str, Any]],
        accepted_obs_meta: List[Dict[str, Any]],
        reg_cfg: RegistrationConfig,
    ):
        return self._cloud_pub.publish_incremental_tsdf_cloud(
            seed_obs_meta, accepted_obs_meta, reg_cfg
        )

    def _capture_registration_observation(
        self,
        sel: Dict[str, Any],
        reg_cfg: RegistrationConfig,
        tracked_roi: Dict[str, Any],
        *,
        refresh: bool,
        use_roi_fallback: bool,
        attempt_label: str,
    ) -> Optional[Dict[str, Any]]:
        return self._registrar._capture_registration_observation(
            sel, reg_cfg, tracked_roi,
            refresh=refresh,
            use_roi_fallback=use_roi_fallback,
            attempt_label=attempt_label,
        )

    def _finalize_registered_views(self, *args, **kwargs):
        return self._registrar._finalize_registered_views(*args, **kwargs)

    def _grasp_matrix_to_pose(
        self,
        grasp: Dict[str, Any],
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> np.ndarray:
        return self._scorer.grasp_matrix_to_pose(grasp, aabb_min, aabb_max)

    def _compute_angular_diversity_deg(
        self,
        grasps: List[Dict[str, Any]],
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> float:
        return self._scorer.angular_diversity_deg(grasps, aabb_min, aabb_max)

    def _compute_grasp_utility(
        self,
        grasps: List[Dict[str, Any]],
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        instability_penalty: float = 0.0,
    ) -> Dict[str, float]:
        return self._scorer.compute_utility(
            grasps, aabb_min, aabb_max, instability_penalty=instability_penalty
        )

    def _estimate_registration_instability_penalty(
        self,
        accepted_obs_meta: List[Dict[str, Any]],
    ) -> float:
        return GraspScorer.instability_penalty(accepted_obs_meta)

    def _request_graspgen_candidates(
        self,
        pts_send: np.ndarray,
        cols_send: Optional[np.ndarray],
        *,
        log_prefix: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        return self._graspgen.request_candidates(
            pts_send, cols_send, log_prefix=log_prefix
        )

    def _filter_grasps_for_utility(
        self,
        grasps: List[Dict[str, Any]],
        pts_filter: np.ndarray,
        pts_metric: np.ndarray,
        plane_n: Optional[np.ndarray],
        plane_p: Optional[np.ndarray],
        registration_frame_is_target: bool,
        frame_id: str,
        *,
        log_prefix: Optional[str],
    ) -> List[Dict[str, Any]]:
        return self._grasp_filter.filter_for_utility(
            grasps,
            pts_filter,
            pts_metric,
            plane_n,
            plane_p,
            registration_frame_is_target,
            frame_id,
            log_prefix=log_prefix,
        )

    def _evaluate_registration_grasp_utility(
        self,
        sel: Dict[str, Any],
        first_obs: Dict[str, Any],
        views: List[RegistrationView],
        accepted_obs_meta: List[Dict[str, Any]],
        reg_cfg: RegistrationConfig,
        min_pts: int,
        *,
        context: str,
        log_details: bool = True,
    ) -> Optional[Dict[str, float]]:
        # Orchestrator-level method: still calls _filter_object_cloud
        # (cloud extraction module — Step 4 will move it). For now we
        # keep this body inline so the multi-view sweep keeps producing
        # identical utility numbers.
        if not bool(
            self.get_parameter("registration_grasp_utility_logging_enabled").value
        ):
            return None
        if not views:
            return None

        if len(views) == 1:
            pts_np = np.asarray(first_obs["points"], dtype=np.float32)
            cols_np = (
                None if first_obs["colors"] is None
                else np.asarray(first_obs["colors"])
            )
            frame_id = str(first_obs["header"].frame_id)
            registration_frame_is_target = False
            plane_n = first_obs.get("plane_n")
            plane_p = first_obs.get("plane_p")
        else:
            result = register_and_fuse_views(
                views, self.tf_buffer, reg_cfg, logger=self.get_logger()
            )
            if result.fused_points is None or len(result.fused_points) < min_pts:
                if log_details:
                    self.get_logger().warn(
                        f"[grasp-utility] {context}: fused cloud too small for passive utility evaluation."
                    )
                return None
            pts_np = np.asarray(result.fused_points, dtype=np.float32)
            cols_np = (
                None if result.fused_colors is None
                else np.asarray(result.fused_colors)
            )
            frame_id = str(reg_cfg.target_frame)
            registration_frame_is_target = True
            plane_n = None
            plane_p = None
            lightweight = bool(
                self.get_parameter("pregrasp_filter_lightweight_during_sweep").value
            )
            fused_filtered = self._filter_object_cloud(
                pts_np, cols_np, apply_depth_clamp=False, lightweight=lightweight
            )
            if fused_filtered is not None:
                pts_np, cols_np, plane_n, plane_p = fused_filtered

        if pts_np is None or pts_np.shape[0] < min_pts:
            if log_details:
                self.get_logger().warn(
                    f"[grasp-utility] {context}: not enough points after cloud preparation (<{min_pts})."
                )
            return None

        max_graspgen_points = int(self.get_parameter("max_graspgen_points").value)
        if max_graspgen_points > 0 and pts_np.shape[0] > max_graspgen_points:
            idx = np.random.choice(pts_np.shape[0], max_graspgen_points, replace=False)
            pts_send = pts_np[idx]
            cols_send = cols_np[idx] if cols_np is not None else None
        else:
            pts_send = pts_np
            cols_send = cols_np

        raw_grasps = self._graspgen.request_candidates(
            pts_send,
            cols_send,
            log_prefix=(f"[grasp-utility] {context}:" if log_details else None),
        )
        if raw_grasps is None:
            return None
        filtered_grasps = self._grasp_filter.filter_for_utility(
            raw_grasps,
            pts_np,
            pts_send,
            plane_n,
            plane_p,
            registration_frame_is_target,
            frame_id,
            log_prefix=(f"[grasp-utility/filter/{context}]" if log_details else None),
        )
        aabb_min_pick, aabb_max_pick = _aabb_min_max(pts_send)
        instability_penalty = GraspScorer.instability_penalty(accepted_obs_meta)
        utility = self._scorer.compute_utility(
            filtered_grasps,
            aabb_min_pick,
            aabb_max_pick,
            instability_penalty=instability_penalty,
        )
        if log_details:
            self.get_logger().info(
                f"[grasp-utility] {context}: U={utility['total']:.3f} "
                f"(count={int(utility['surviving_count'])}, "
                f"diversity_deg={utility['diversity_deg']:.2f}, "
                f"mean_score={utility['mean_score']:.3f}, "
                f"instability={utility['instability_penalty']:.3f})"
            )
        return utility

    def _update_registration_utility_stop_state(
        self,
        *,
        context: str,
        view_count: int,
        prev_utility: Optional[Dict[str, float]],
        utility_now: Optional[Dict[str, float]],
        small_gain_streak: int,
    ) -> tuple[bool, int, Optional[float]]:
        return self._scorer.update_stop_state(
            context=context,
            view_count=view_count,
            prev_utility=prev_utility,
            utility_now=utility_now,
            small_gain_streak=small_gain_streak,
        )

    def _maybe_register_multi_view_cloud(
        self,
        sel: Dict[str, Any],
        first_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._registrar._maybe_register_multi_view_cloud(sel, first_obs)

    def _refresh_selection_for_capture(self, sel: Dict[str, Any]) -> Dict[str, Any]:
        refresh = bool(self.get_parameter("registration_refresh_detection").value)
        if not refresh:
            return sel
        # Fresh-scene shortcut (2026-08-21 speed pass): the selection was
        # made against a detection younger than 1 s — re-running YOLO for
        # the same frame costs ~1.5 s and changes nothing. Robot has not
        # moved between selection and capture in the single-view flow.
        if (time.time() - float(getattr(self, "_last_infer_t", 0.0))) < 1.0:
            self.get_logger().info(
                "[registration] detection is fresh (<1s) — skipping re-detect."
            )
            return sel
        maybe_sel = self._detect_current_selection(sel["label"])
        if maybe_sel is None:
            self.get_logger().warn(
                f"[registration] could not refresh selected label '{sel['label']}' before capture; using previous bbox."
            )
            return sel
        return maybe_sel
        
    def _pose_from_matrix16(self, matrix16, frame_id, stamp):
        """Convert a 4×4 matrix into a :class:`PoseStamped` message."""
        M = np.asarray(matrix16, dtype=np.float32).reshape(4, 4)
        self.get_logger().info(f"[debug] raw matrix:\n{M}")
        t = M[:3, 3]
        qx, qy, qz, qw = quaternion_from_matrix(M)
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        if getattr(stamp, "sec", 0) == 0 and getattr(stamp, "nanosec", 0) == 0:
            stamp = self.get_clock().now().to_msg()
        ps.header.stamp = stamp
        ps.pose.position.x = float(t[0])
        ps.pose.position.y = float(t[1])
        ps.pose.position.z = float(t[2])
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        self.get_logger().info(
            f"[debug] PoseStamped t=({ps.pose.position.x:.3f},{ps.pose.position.y:.3f},{ps.pose.position.z:.3f}) "
            f"q=({ps.pose.orientation.x:.3f},{ps.pose.orientation.y:.3f},{ps.pose.orientation.z:.3f},{ps.pose.orientation.w:.3f})"
        )
        return ps

    # ------------------------------------------------------------------
    # Orientation delegation (real impl in grasping.OrientationFilter).
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        return OrientationFilter.quat_normalize(q)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        return OrientationFilter.quat_multiply(q1, q2)

    @staticmethod
    def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
        return OrientationFilter.quat_angle_deg(q1, q2)

    def _filter_by_ee_orientation_consistency(
        self,
        grasps: List[Dict[str, Any]],
        frame_id: str,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> List[Dict[str, Any]]:
        return self._orientation_filter.filter_by_ee_orientation_consistency(
            grasps, frame_id, aabb_min, aabb_max
        )

    def _reweight_scores_by_ee_orientation(
        self,
        grasps: List[Dict[str, Any]],
        frame_id: str,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> List[Dict[str, Any]]:
        return self._orientation_filter.reweight_scores_by_ee_orientation(
            grasps, frame_id, aabb_min, aabb_max
        )

    def republish_last_pose(self) -> None:
        """Continuously republish the last selected pose with an updated timestamp."""
        current_time = self.get_clock().now().to_msg()
        
        # Republish camera frame pose
        if self.last_published_pose is not None:
            current_pose = PoseStamped()
            current_pose.header.frame_id = self.last_published_pose.header.frame_id
            current_pose.header.stamp = current_time
            current_pose.pose = self.last_published_pose.pose
            self.pub_best.publish(current_pose)
        
        # Republish base_link frame pose
        if self.last_published_pose_base is not None:
            current_pose_base = PoseStamped()
            current_pose_base.header.frame_id = self.last_published_pose_base.header.frame_id
            current_pose_base.header.stamp = current_time
            current_pose_base.pose = self.last_published_pose_base.pose
            self.pub_best_base.publish(current_pose_base)

        # Republish centroid in camera frame
        if self.last_centroid is not None:
            current_centroid = PoseStamped()
            current_centroid.header.frame_id = self.last_centroid.header.frame_id
            current_centroid.header.stamp = current_time
            current_centroid.pose = self.last_centroid.pose
            self.pub_centroid.publish(current_centroid)

        # Republish centroid in base frame
        if self.last_centroid_base is not None:
            current_centroid_base = PoseStamped()
            current_centroid_base.header.frame_id = self.last_centroid_base.header.frame_id
            current_centroid_base.header.stamp = current_time
            current_centroid_base.pose = self.last_centroid_base.pose
            self.pub_centroid_base.publish(current_centroid_base)

        # Republish candidate grasp poses (base frame) — but ONLY while they
        # still belong to the active cycle. Restamping a previous cycle's
        # candidates makes the executor drive to the previous object
        # (ghost pick #2, Majd 2026-08-12).
        if (self.last_candidates_base is not None
                and getattr(self, "_candidates_cycle_id", -1)
                == int(getattr(self, "_waiting_cycle_id", -1))):
            current_candidates = copy.deepcopy(self.last_candidates_base)
            current_candidates.header.stamp = current_time
            self.pub_candidates_base.publish(current_candidates)
            if self.last_candidate_scores is not None:
                self.pub_candidate_scores.publish(copy.deepcopy(self.last_candidate_scores))

    def _on_timer(self) -> None:
        """Periodic callback to run object detection and publish debug images."""
        if self._infer_running or self.selection_made:
            return
        if (
            bool(self.get_parameter("suspend_inference_while_selecting").value)
            and self._selection_in_progress
        ):
            return
        if self._seg_wait:
            # The click's segmentation is waiting for the lock: do not queue
            # another 0.4-0.8 s detector chain in front of it.
            return
        rgb, depth, (fx, fy, cx, cy), depth_scale, header = self.fb.latest()
        if rgb is None:
            return
        if time.time() - self._last_infer_t < float(self.get_parameter("min_infer_interval_s").value):
            return

        def run() -> None:
            with self._infer_lock:
                self._infer_running = True
                try:
                    t_infer0 = time.time()
                    dets = self.detector.predict(rgb)
                    infer_ms = (time.time() - t_infer0) * 1000.0
                    dets = self._self_filter.filter_detections(
                        dets, rgb.shape[:2], str(header.frame_id) if header is not None else "")
                    if self._allowed_classes:
                        dets = [d for d in dets if d["label"].lower() in self._allowed_classes]
                    for d in dets:
                        d["source"] = "ys"          # trained YOLO ("yolo standard")
                    if self._seg_wait:
                        # A click is waiting for this lock: stop after the
                        # fast half (YOLO) and skip the identity + open-set
                        # stages for this one round (2026-09-17: the click's
                        # 50 ms segmentation waited up to 0.8 s here).
                        pass
                    else:
                        dets, rejected = self._identity.check(rgb, dets)
                        dets = dets + self._open_set.detect(
                            rgb, depth, header, dets,
                            [list(r["bbox"]) + [float(r.get("score", 0.5))] for r in rejected])
                    self._perf_infer_count += 1
                    self._perf_infer_ms_acc += infer_ms
                    # Annotate every detection with its simplified label.
                    # Special policy: for most classes, collapse to the
                    # single highest-scoring detection per class — YOLO
                    # tends to over-fire low-confidence ghosts of the
                    # same physical object (e.g. two small_bottle boxes
                    # at slightly different positions). For the
                    # labels in MULTI_INSTANCE_LABELS we keep ALL
                    # detections so genuine multiples (two vitamin
                    # tubes on the table) survive.
                    MULTI_INSTANCE_LABELS = {"vitamin_tube", "vitamin_c"}
                    by_label_best: Dict[str, Dict[str, Any]] = {}
                    multi_instance: List[Dict[str, Any]] = []
                    for d in dets:
                        d_copy = d.copy()
                        simple_name = _simplify_object_name(d["label"])
                        d_copy["simplified_label"] = simple_name
                        # Named unknown objects are all kept: two "white box"
                        # on the table are two objects, not a ghost.
                        if simple_name.lower() in MULTI_INSTANCE_LABELS or d_copy.get("source") in ("q8b", "id"):
                            multi_instance.append(d_copy)
                        else:
                            prev = by_label_best.get(simple_name)
                            if prev is None or d_copy["score"] > prev["score"]:
                                by_label_best[simple_name] = d_copy
                    simplified_dets: List[Dict[str, Any]] = (
                        list(by_label_best.values()) + multi_instance
                    )
                    if bool(self.get_parameter("dedupe_overlaps").value) and len(simplified_dets) > 1:
                        def _iou_xyxy(b1, b2) -> float:
                            x11, y11, x12, y12 = [float(v) for v in b1]
                            x21, y21, x22, y22 = [float(v) for v in b2]
                            ix1, iy1 = max(x11, x21), max(y11, y21)
                            ix2, iy2 = min(x12, x22), min(y12, y22)
                            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                            inter = iw * ih
                            if inter <= 0.0:
                                return 0.0
                            a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
                            a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
                            den = (a1 + a2 - inter)
                            return (inter / den) if den > 1e-6 else 0.0

                        thr = float(self.get_parameter("dedupe_iou_threshold").value)
                        kept: List[Dict[str, Any]] = []
                        for d in sorted(simplified_dets, key=lambda x: float(x["score"]), reverse=True):
                            if any(_iou_xyxy(d["bbox"], k["bbox"]) >= thr for k in kept):
                                continue
                            kept.append(d)
                        simplified_dets = kept
                    _number_duplicates(simplified_dets)
                    _cap_frame = self.fb.rgb_frame
                    for d in simplified_dets:
                        simple_label = d["simplified_label"]
                        # Cap colour attached to the CACHE itself
                        # (2026-08-25): the debug overlay draws from
                        # detections_cache, and computing cap_colour only
                        # in the /yolo/all_detections payload left the
                        # overlay blind ("I can't see a shit" — Majd).
                        if simple_label == "vitamin_tube" and _cap_frame is not None:
                            try:
                                b = d["bbox"]
                                d["cap_colour"] = _bbox_colour_name(
                                    _cap_frame,
                                    [b[0], b[1], b[2],
                                     b[1] + max(8, min(26, int((b[3] - b[1]) * 0.12)))],
                                )
                            except Exception:
                                d["cap_colour"] = ""
                    self.detections_cache = simplified_dets
                    # Base distance + reach verdict on the fresh dicts, so the
                    # JSON below and the overlay agree (2026-09-09).
                    try:
                        self._debug_renderer._annotate_distances(self, simplified_dets)
                    except Exception:
                        pass
                    self._precompute.observe(simplified_dets)
                    self._last_infer_t = time.time()
                    if self.pub_all_detections is not None:
                        try:
                            frame_for_colour = self.fb.rgb_frame
                            payload = [
                                {
                                    "label": str(d.get("simplified_label", "")),
                                    "display": str(d.get("display", "")),
                                    "score": float(d.get("score", 0.0)),
                                    "bbox": [float(v) for v in d.get("bbox", [])],
                                    "reach_pct": d.get("reach_pct"),
                                    "reach_state": d.get("reach_state"),
                                    "reach_angle_deg": d.get("reach_angle_deg"),
                                    "reach_closer_cm": d.get("reach_closer_cm"),
                                    "source": str(d.get("source", "ys")),
                                    "colour": (
                                        _bbox_colour_name(
                                            frame_for_colour, d.get("bbox", [])
                                        )
                                        if frame_for_colour is not None
                                        else ""
                                    ),
                                    # Cap colour: vitamin tubes only (Majd,
                                    # 2026-08-21 — the three tubes differ ONLY
                                    # by cap: C=yellow, B12=red, Multi=blue).
                                    # Top 20% of the bbox.
                                    "cap_colour": (
                                        _bbox_colour_name(
                                            frame_for_colour,
                                            [
                                                d["bbox"][0],
                                                d["bbox"][1],
                                                d["bbox"][2],
                                                d["bbox"][1]
                                                + max(8, min(26, int((d["bbox"][3] - d["bbox"][1]) * 0.12))),
                                            ],
                                        )
                                        if frame_for_colour is not None
                                        and str(d.get("simplified_label", "")) == "vitamin_tube"
                                        else ""
                                    ),
                                }
                                for d in simplified_dets
                            ]
                            msg = String()
                            msg.data = json.dumps(payload)
                            self.pub_all_detections.publish(msg)
                        except Exception as exc:
                            self.get_logger().warn(
                                f"Failed to publish all_detections: {exc}"
                            )
                    if bool(self.get_parameter("perf_log").value):
                        now = time.time()
                        period = max(float(self.get_parameter("perf_log_period_s").value), 0.5)
                        dt = now - self._perf_last_log_t
                        if dt >= period and self._perf_infer_count > 0:
                            infer_fps = self._perf_infer_count / dt
                            mean_ms = self._perf_infer_ms_acc / self._perf_infer_count
                            self.get_logger().info(
                                f"[perf] infer_fps={infer_fps:.2f}, infer_ms_avg={mean_ms:.1f}, "
                                f"last_dets={len(simplified_dets)}, detector_device={self.detector.device}"
                            )
                            self._perf_last_log_t = now
                            self._perf_infer_count = 0
                            self._perf_infer_ms_acc = 0.0
                except Exception as e:
                    self.get_logger().error(f"YOLO inference failed: {e}\n{traceback.format_exc()}")
                finally:
                    self._infer_running = False
        threading.Thread(target=run, daemon=True).start()

    def _publish_debug_frame(self) -> None:
        """Publish debug image at a fixed rate independent of detector throughput.

        Real implementation lives in debug.DebugRenderer; this wrapper is
        kept so the timer wiring (``self.create_timer(..., self._publish_debug_frame)``)
        and any external references continue to work."""
        self._debug_renderer.publish()

    def _republish_selected_outputs(self) -> None:
        """Republish selected detection/masks (lightweight) so consumers can latch onto stable outputs.

        Note: The point cloud is intentionally NOT republished here — it is large
        and already published once on selection.  Continuous deep-copy at 5 Hz was
        the main cause of RViz2 lag.
        """
        if not bool(self.get_parameter("republish_selected_outputs").value):
            return
        if not self.selection_made:
            return
        stamp = self.get_clock().now().to_msg()
        try:
            if self.last_selected_detection_msg is not None:
                msg = copy.deepcopy(self.last_selected_detection_msg)
                msg.header.stamp = stamp
                self.pub_detection.publish(msg)
            if self.last_selected_mask_msg is not None:
                msg = copy.deepcopy(self.last_selected_mask_msg)
                msg.header.stamp = stamp
                self.pub_selected_mask.publish(msg)
            if self.last_selected_overlay_msg is not None:
                msg = copy.deepcopy(self.last_selected_overlay_msg)
                msg.header.stamp = stamp
                self.pub_selected_overlay.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Republish selected outputs failed: {e}")

    def _publish_selected(self, sel: Dict[str, Any]) -> bool:
        """End-to-end pick pipeline. Real implementation lives in
        grasping.PickPipeline; this wrapper preserves the legacy method
        name so internal callers (the selection thread) keep working."""
        return self._pick.run(sel)

    def _on_executor_selected_pose(self, msg: PoseStamped) -> None:
        """Bridge the executor's ACTUAL pick into the viz frame for ORANGE."""
        try:
            target = self.last_viz_frame or msg.header.frame_id
            pose = msg.pose
            if target and target != msg.header.frame_id:
                tf = self.tf_buffer.lookup_transform(
                    target, msg.header.frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5),
                )
                pose = tf2_geometry_msgs.do_transform_pose(msg.pose, tf)
            pa = PoseArray()
            pa.header.frame_id = target
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.poses.append(pose)
            self.pub_viz_executing.publish(pa)
            self.get_logger().info(
                f"[viz] executor's ACTUAL pick bridged to viz frame '{target}' (ORANGE)."
            )
        except Exception as e:
            self.get_logger().warn(f"executing-pose bridge failed: {e}")

    def _on_manual_select_idx(self, msg: Int32) -> None:
        """Queue a manual selection index published from another ROS process."""
        self._pending_manual_choice = int(msg.data)
        self._pending_manual_choice_stamp = time.time()
        self._pending_choice_source = "voice"
        self._selection_wake.set()
        # CLI selection is an explicit index — make sure no stale clicked-det
        # from an earlier window click hijacks the resolution.
        self._pending_manual_choice_det = None
        self.get_logger().info(f"Queued manual selection index: {self._pending_manual_choice}")

    def _on_pick_abort(self, msg: String) -> None:
        self._pick_abort = True
        self._last_cycle_end_t = time.time()
        self.get_logger().warn(
            "Operator abort received — the in-flight pick cycle will end "
            "at the next phase boundary."
        )

    def _on_point_answer(self, msg: String) -> None:
        """intent_router's answer to a pointing proposal: "yes|<id>" / "no|<id>"."""
        word, _, pid = str(msg.data or "").strip().lower().partition("|")
        try:
            pid_i = int(pid)
        except ValueError:
            self.get_logger().warn(f"Bad point answer {msg.data!r}")
            return
        if word in ("yes", "no"):
            self._point_answer = (pid_i, word)
            self._selection_wake.set()

    # ── pointing camera (2026-09-29) ─────────────────────────────────
    def _astra(self) -> AstraPointingCamera:
        """The Astra pointing camera, created on first use: while the
        RealSense is the pointing camera nothing of the Astra is received."""
        if self._astra_pointing is None:
            self._astra_pointing = AstraPointingCamera(self, self.tf_buffer, logger=self.get_logger())
        return self._astra_pointing

    def _choose_by_pointing(self, dets, names) -> Optional[int]:
        """One pointing session in the camera pointing_camera names. Only
        that camera is watched (Majd: "when 1 is selected then it should
        only be focused on it"); switching the camera ends the session so
        the next one starts in the new camera at once."""
        cam = "astra" if str(self.get_parameter("pointing_camera").value).strip().lower() == "astra" else "realsense"

        def cancel() -> bool:
            return (self._pending_manual_choice is not None or self._pick_abort
                    or self._voice_label_pending()
                    or str(self.get_parameter("pointing_camera").value).strip().lower() != cam)

        common = dict(names=names,
                      listen_seconds=float(self.get_parameter("listen_seconds").value),
                      overall_timeout_s=float(self.get_parameter("selection_attempt_timeout_s").value),
                      cancel=cancel)
        self._point_cam = cam
        if cam == "realsense":
            return self.point_or_voice.choose(bgr_frame=self.fb.rgb_frame, dets=dets,
                                              frame_provider=lambda: self.fb.rgb_frame, **common)
        astra = self._astra()
        if not astra.alive():
            self.get_logger().warn(
                "[point] the pointing camera is the Astra, but no Astra frames arrive — pointing waits for "
                "them (shuttle camera: astra restart script; or choose the RealSense in Advanced -> Detection)",
                throttle_duration_sec=30.0)
        rs = self.fb.paired()
        if rs is None or rs[1] is None:
            rs = self.fb.latest()
        return self.point_or_voice.choose(bgr_frame=None, dets=astra.begin(dets, rs),
                                          frame_provider=astra.next_frame, **common)

    def _confirm_pointed_choice(self, dets, idx: int, header) -> str:
        """2026-09-28 (Majd: "the pointing just starts the grasp immediately —
        ask 'you pointed at this object, should I grasp it?'"): hold a
        POINTED choice until the operator confirms it.

        The proposal goes out on /grasp_pipeline/point_proposal (JSON: id,
        state "open", label, display, index, bbox, stamp) and the object is
        highlighted in the debug image; intent_router asks the question —
        worded for its which-one clarification and for an object other than
        the one asked for — and answers on /grasp_pipeline/point_answer. The
        verdict: "yes"; "no"; "superseded" (a spoken pick or a click arrived
        and is taken next); "abort" (stop / reset); "timeout"
        (point_confirm_timeout_s); "ignored" (the object was just rejected —
        nothing is asked). Every verdict but "ignored" is published as the
        proposal's closing state, so the router closes its question.

        GraspGen's background job keeps running meanwhile, so a yes usually
        finds the grasps ready."""
        det = dets[idx]
        label = str(det.get("simplified_label", det["label"]))
        bbox = [int(v) for v in det["bbox"]]
        cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        rej = self._point_rejected
        if rej is not None:
            hold = float(self.get_parameter("point_reject_hold_s").value)
            rx, ry = 0.5 * (rej["bbox"][0] + rej["bbox"][2]), 0.5 * (rej["bbox"][1] + rej["bbox"][3])
            if (rej["label"] == label and time.time() - rej["t"] < hold
                    and ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 < 80.0):
                self.get_logger().info(
                    f"[point] {label} was just rejected — not asking again for "
                    f"{hold - (time.time() - rej['t']):.1f} s.")
                return "ignored"
        self._point_prop_id += 1
        pid = self._point_prop_id
        self._point_answer = None
        self._point_prop_pub.publish(String(data=json.dumps({
            "id": pid, "state": "open", "label": label,
            "display": str(det.get("display", det["label"])), "index": int(idx),
            "bbox": bbox, "stamp": round(time.time(), 3)})))
        self.get_logger().info(f"[point] proposal {pid}: {label} — waiting for the operator's yes/no.")
        try:
            if bool(self.get_parameter("publish_debug_image").value) and self.fb.rgb_frame is not None:
                self.debug_pub.publish(header, self.fb.rgb_frame, dets, selected_idx=int(idx),
                                       pointer_px=self._pointer_px,
                                       hand_landmarks_px=self._hand_landmarks_px,
                                       hand_connections=self._hand_connections,
                                       render_window=False)
        except Exception as exc:
            self.get_logger().debug(f"proposal highlight failed: {exc}")
        timeout_s = float(self.get_parameter("point_confirm_timeout_s").value)
        t0 = time.time()
        verdict = ""
        while rclpy.ok() and not verdict:
            ans = self._point_answer
            if ans is not None and ans[0] == pid:
                verdict = ans[1]
            elif self._pick_abort:
                verdict = "abort"
            elif self._pending_manual_choice is not None or self._voice_label_pending():
                verdict = "superseded"
            elif timeout_s > 0.0 and time.time() - t0 >= timeout_s:
                verdict = "timeout"
            else:
                self._selection_nap(0.05)
        verdict = verdict or "abort"
        self._point_prop_pub.publish(String(data=json.dumps({
            "id": pid, "state": verdict, "label": label, "stamp": round(time.time(), 3)})))
        self.get_logger().info(f"[point] proposal {pid} ({label}): {verdict} after {time.time() - t0:.1f} s.")
        if verdict in ("no", "timeout"):
            self._point_rejected = {"label": label, "bbox": bbox, "t": time.time()}
        return verdict

    def _on_manual_select_det(self, msg: String) -> None:
        """Queue a selection carrying label+bbox (from intent_router spatial
        picks). Stored like a window click so _resolve_clicked_choice maps it
        onto the live detection list at consumption time."""
        try:
            payload = json.loads(msg.data)
            index = int(payload.get("index", -1))
            label = str(payload.get("label", ""))
            bbox = [float(v) for v in (payload.get("bbox") or [])]
        except (ValueError, TypeError, json.JSONDecodeError):
            self.get_logger().warn(
                f"Bad /yolo/select_detection payload: {msg.data!r}"
            )
            return
        self._pending_manual_choice = index
        self._pending_manual_choice_stamp = time.time()
        self._pending_choice_source = "voice"
        self._pending_manual_choice_det = (
            {"label": label, "bbox": bbox} if label and len(bbox) == 4 else None
        )
        # Wake the selection loop NOW — the idx and window-click handlers
        # both do this; without it the pick waits out the nap and the next
        # pass resets _selection_cycle_start_t past our stamp = "stale"
        # (2026-08-26: voice qualified picks silently eaten).
        self._selection_wake.set()
        self.get_logger().info(
            f"Queued spatial selection: index={index}, label='{label}' "
            "(will re-resolve against live detections)."
        )

    def _on_debug_window_choice(self, idx: int) -> None:
        """Queue selection index from the OpenCV debug window mouse click."""
        if not bool(self.get_parameter("enable_debug_window_click_select").value):
            return
        self._pending_manual_choice = int(idx)
        self._pending_manual_choice_stamp = time.time()
        self._pending_choice_source = "click"
        # Also keep the clicked detection itself (label + bbox): the index is
        # re-resolved at consumption time because YOLO reorders detections
        # between frames (2026-08-07 wrong-object bug).
        self._pending_manual_choice_det = getattr(
            self.debug_pub, "last_clicked_det", None
        )
        self._selection_wake.set()
        self.get_logger().info(f"Queued window-click selection index: {self._pending_manual_choice}")

    @staticmethod
    def _same_object(a, b, tol_px: float = 40.0) -> bool:
        """Same label and bbox centre within tol_px: the retried object."""
        try:
            if str(a.get("label", "")).strip().lower() != str(b.get("label", "")).strip().lower():
                return False
            ax1, ay1, ax2, ay2 = [float(v) for v in a.get("bbox", [0, 0, 0, 0])]
            bx1, by1, bx2, by2 = [float(v) for v in b.get("bbox", [0, 0, 0, 0])]
            return abs(0.5 * (ax1 + ax2) - 0.5 * (bx1 + bx2)) < tol_px and abs(0.5 * (ay1 + ay2) - 0.5 * (by1 + by2)) < tol_px
        except Exception:
            return False

    def _resolve_clicked_choice(self, dets, queued_idx: int) -> int:
        """Map a queued click onto the CURRENT detection list.

        Prefers the detection with the same label whose bbox centre is
        nearest the clicked one; falls back to the raw index when no clicked
        det was recorded (e.g. CLI /yolo/select_index) or nothing matches."""
        det = getattr(self, "_pending_manual_choice_det", None)
        self._pending_manual_choice_det = None
        if not det or not dets:
            return queued_idx
        try:
            # Router dets carry the simplified label; window-click dets carry
            # the raw one. Match against every naming a pipeline det has.
            label = str(det.get("label", "")).strip().lower()

            def _names_of(d) -> set:
                return {
                    str(d.get(k, "")).strip().lower()
                    for k in ("label", "simplified_label", "display")
                } - {""}

            x1, y1, x2, y2 = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            best_i, best_d, n_label = None, 1e9, 0
            for i, d in enumerate(dets):
                if label not in _names_of(d):
                    continue
                n_label += 1
                bx1, by1, bx2, by2 = [float(v) for v in d.get("bbox", [0, 0, 0, 0])]
                dist = ((0.5 * (bx1 + bx2) - cx) ** 2 + (0.5 * (by1 + by2) - cy) ** 2) ** 0.5
                if dist < best_d:
                    best_i, best_d = i, dist
            # Distance gates (2026-08-25): even a UNIQUE same-label match
            # must be near the requested bbox — the old "unique wins
            # outright" rule grabbed the Vitamin C 400px away when the
            # requested blue Multi flickered out of the detection list
            # (nutmeg_jar misdetection frames), narrating one tube and
            # physically picking another. Same class ≠ same object.
            gate = 250.0 if n_label == 1 else 150.0
            if best_i is not None and best_d < gate:
                if best_i != queued_idx:
                    self.get_logger().info(
                        f"[click] re-resolved selection: index {queued_idx} -> {best_i} "
                        f"('{label}', center moved {best_d:.0f}px) — detections reordered."
                    )
                return best_i
            if best_i is not None:
                self.get_logger().warn(
                    f"[click] REFUSING selection: nearest '{label}' is "
                    f"{best_d:.0f}px from the requested one (gate {gate:.0f}px)"
                    " — the requested object is not currently detected. "
                    "Not grabbing a different object of the same class."
                )
                return -1
            self.get_logger().warn(
                f"[click] REFUSING selection: no '{label}' detection visible "
                "right now — the requested object vanished from the frame."
            )
            return -1
        except Exception:
            pass
        return queued_idx

    def _on_hover_pixel(self, msg: Point) -> None:
        try:
            dets = list(getattr(self.debug_pub, "_last_dets", []) or [])
            idx = None
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                if min(x1, x2) <= int(msg.x) <= max(x1, x2) and min(y1, y2) <= int(msg.y) <= max(y1, y2):
                    idx = i
                    break
            if idx is not None and idx != getattr(self, "_last_hover_idx", None):
                self._last_hover_idx = idx
                self._precompute.bump(idx)
        except Exception:
            pass

    def _on_image_view_click(self, msg: Point) -> None:
        """Left-click from rqt_image_view ("publish click location" checkbox).

        The Point carries pixel coordinates in the ORIGINAL image, so no
        rescaling is needed — hit-test directly against the last detections
        and reuse the same selection path as the native window click.
        (2026-08-06)"""
        hit = self.debug_pub.click_at(int(msg.x), int(msg.y))
        if not hit:
            self.get_logger().info(
                f"Image-view click at ({int(msg.x)},{int(msg.y)}) hit no detection box."
            )

    def _voice_label_pending(self) -> bool:
        """True while the router has set auto_select_label and no pass has
        consumed it yet. 2026-09-23 (12.4 s from "tape dispenser" to the
        first motion): a voice label is read only at the top of a loop
        pass, so one that arrived during the pointing chooser's wait, the
        retry nap or a background GraspGen job waited them all out — 7 s
        of the 12. A label now cancels the chooser's wait, skips the nap
        and interrupts an uncommitted background job, exactly as a click
        does through _pending_manual_choice."""
        try:
            return bool(str(self.get_parameter("auto_select_label").value).strip())
        except Exception:
            return False

    def _on_executor_cycle_done(self, msg: String) -> None:
        """Receive commander completion notification ('done' or 'failed')."""
        # Selections queued before this moment belong to the cycle that
        # just ended — the stale guard uses this stamp (2026-08-26).
        self._last_cycle_end_t = time.time()
        if not self._awaiting_executor_cycle_done:
            # Ignore stale status from previous runs/cycles.
            return
        if self._executor_cycle_done:
            # Ignore duplicate status updates for the same armed wait.
            return
        raw = str(msg.data).strip()
        status = raw
        cycle_id = None
        if "|" in raw:
            left, right = raw.split("|", 1)
            left = left.strip()
            right = right.strip()
            if left in ("done", "failed"):
                status = left
                try:
                    # Payload may be "<cycle_id>" or "<cycle_id>|<reason>"
                    # (failure reason added 2026-08-11).
                    cycle_id = int(right.split("|", 1)[0].strip())
                except ValueError:
                    cycle_id = None
        if status not in ("done", "failed"):
            self.get_logger().warn(f"Ignoring malformed executor cycle status: '{raw}'")
            return
        require_id = bool(self.get_parameter("require_matching_cycle_id").value)
        if cycle_id is None and require_id:
            self.get_logger().warn(
                f"Ignoring executor status without cycle id while require_matching_cycle_id=true: '{raw}'"
            )
            return
        if cycle_id is not None and cycle_id != self._waiting_cycle_id:
            self.get_logger().info(
                f"Ignoring executor status for different cycle_id={cycle_id} (waiting for {self._waiting_cycle_id})."
            )
            return
        self._executor_cycle_status = status
        # Third field, when present: the executor's reason for a failure
        # ("failed|<cycle>|<reason>"); read by the retry logic below.
        parts = raw.split("|")
        self._executor_cycle_reason = parts[2].strip() if len(parts) > 2 else ""
        self._executor_cycle_done = True
        if cycle_id is not None:
            self.get_logger().info(
                f"Received executor cycle status: {self._executor_cycle_status} (cycle_id={cycle_id})"
            )
        else:
            self.get_logger().info(f"Received executor cycle status: {self._executor_cycle_status}")

    def _speak_if_enabled(self, text: str) -> None:
        """Log internal prompt strings only — they are NOT sent out for
        TTS. The LLM (communication_llm.chat_node) owns scene
        narration entirely: it subscribes to /yolo/all_detections and
        speaks via its proactive scene narration or on-demand 'what do
        you see?' replies. The pipeline writing a redundant 'I see N
        objects…' line just creates double narration with no benefit.

        We keep the method name + signature so existing internal call
        sites work unchanged. Topic publication is also skipped so no
        downstream subscriber can accidentally re-introduce the
        double-narrate bug."""
        if not text:
            return
        self.get_logger().info(f"[prompt] {text}")

    def _selection_nap(self, seconds: float) -> None:
        """Sleep that a pending click cancels instantly."""
        if self._selection_wake.wait(timeout=float(seconds)):
            self._selection_wake.clear()

    def _selection_thread(self) -> None:
        """Background thread handling the user's object selection via voice or pointing."""
        while rclpy.ok():
            try:
                if not self.selection_made and self.detections_cache:
                    if (
                        bool(self.get_parameter("registration_enabled").value)
                        and bool(self.get_parameter("registration_move_to_initial_target").value)
                        and bool(self.get_parameter("registration_preposition_before_selection").value)
                        and not self._registration_prepositioned_for_selection
                    ):
                        init_target = str(self.get_parameter("registration_initial_named_target").value).strip()
                        if init_target:
                            moved = self._move_to_registration_target(init_target)
                            if moved:
                                self._registration_init_move_failed_this_cycle = False
                                self.get_logger().info(
                                    f"[registration] prepositioned to '{init_target}' before selection."
                                )
                            else:
                                self._registration_init_move_failed_this_cycle = True
                                self.get_logger().warn(
                                    f"[registration] preselection move to '{init_target}' failed; continuing at current pose."
                                )
                        self._registration_prepositioned_for_selection = True
                        self.detections_cache = []
                        self._selection_prompted = False
                        self._selection_prompt_sig = None
                        self._selection_in_progress = False
                        self._selection_nap(0.2)
                        continue
                    self._selection_in_progress = True
                    self._selection_cycle_start_t = time.time()
                    self._pick_abort = False  # stale stop must not kill this cycle
                    dets = list(self.detections_cache)
                    header = self.fb.rgb_header
                    if not dets or header is None:
                        self._selection_in_progress = False
                        self._selection_nap(0.1)
                        continue
                    names = [d.get("display", d["label"]) for d in dets]
                    # Snapshot for the hover publisher so on_hover(idx)
                    # can map idx → label without holding a reference to
                    # the local dets list.
                    self._current_names = list(names)
                    sig = tuple(names)
                    if self._selection_prompt_sig != sig:
                        self._selection_prompt_sig = sig
                        self._selection_prompted = False
                    # Auto-select path (no TTS / no stdin)
                    choice = None
                    auto_label = str(self.get_parameter("auto_select_label").value).strip().lower()
                    auto_idx = int(self.get_parameter("auto_select_index").value)
                    noninteractive = bool(self.get_parameter("auto_select_noninteractive").value)
                    if self._pending_manual_choice is not None:
                        queued_idx = int(self._pending_manual_choice)
                        queued_stamp = float(self._pending_manual_choice_stamp)
                        self._pending_manual_choice = None
                        if (queued_stamp < self._last_cycle_end_t
                                or (time.time() - queued_stamp) > 120.0):
                            self._pending_manual_choice_det = None
                            self.get_logger().info(
                                f"Ignoring stale queued selection index {queued_idx} from previous cycle."
                            )
                        else:
                            queued_idx = self._resolve_clicked_choice(dets, queued_idx)
                            if 0 <= queued_idx < len(names):
                                choice = queued_idx
                                self._choice_source_this = getattr(self, "_pending_choice_source", None) or "voice"
                                self.get_logger().info(
                                    f"Using manual topic selection index {choice}: {names[choice]}"
                                )
                            else:
                                self.get_logger().warn(
                                    f"Manual selection index {queued_idx} out of range [0, {len(names)-1}]."
                                )

                    if auto_label:
                        # RETRY CAP (2026-08-26, the small-bottle loop:
                        # gated/failing cycles re-fired the same label
                        # forever). 2026-09-22: the counter counted loop
                        # PASSES, not started cycles, and no completed
                        # cycle ever reset it — the small bottle picked at
                        # 19:47 had spent the budget, and the next request
                        # died after three frames in which it was simply
                        # not detected. Now a START is counted where a
                        # cycle starts (3 allowed), a MISS where the label
                        # is not among the detections (5 allowed), a
                        # completed cycle resets both, and the give-up is
                        # reported to the router (selection_result).
                        if auto_label != getattr(self, "_auto_label_last", ""):
                            self._auto_label_last = auto_label
                            self._auto_label_starts = 0
                            self._auto_label_missing = 0
                        found = None
                        for i, n in enumerate(names):
                            if str(n).strip().lower() == auto_label:
                                found = i
                                break
                        give_up = ""
                        spoken_label = auto_label.replace('_', ' ')
                        if found is None:
                            self._auto_label_missing = getattr(self, "_auto_label_missing", 0) + 1
                            self.get_logger().warn(
                                f"auto_select_label='{auto_label}' not found in {names} "
                                f"({self._auto_label_missing}/5); falling back to interactive.")
                            if self._auto_label_missing >= 5:
                                give_up = f"I can't see the {spoken_label} right now."
                        elif getattr(self, "_auto_label_starts", 0) >= 3:
                            give_up = (f"I could not grasp the {spoken_label} after "
                                       "several tries — it may be out of my reach. "
                                       "Pick something else or move it closer.")
                        if give_up:
                            self.get_logger().warn(
                                f"auto_select_label='{auto_label}': {give_up} — clearing "
                                "the label and stopping the retry loop.")
                            try:
                                from rclpy.parameter import Parameter as _P
                                self.set_parameters([_P(
                                    "auto_select_label",
                                    _P.Type.STRING, "")])
                            except Exception:
                                pass
                            self._auto_label_last = ""
                            self._auto_label_starts = 0
                            self._auto_label_missing = 0
                            try:
                                self._result_pub.publish(String(data=f"failed|0|{give_up}"))
                            except Exception:
                                pass
                            auto_label = ""
                        elif found is not None:
                            self._auto_label_starts = getattr(self, "_auto_label_starts", 0) + 1
                            self._auto_label_missing = 0
                            choice = found
                            # 2026-09-28: a spoken pick — it was logged "(by
                            # pointing)" (the default below), and a pointed
                            # choice is now asked to be confirmed.
                            self._choice_source_this = "voice"

                    if choice is None and auto_idx >= 0 and auto_idx < len(names):
                        choice = auto_idx
                        self._choice_source_this = "voice"

                    if choice is None and noninteractive and not sys.stdin.isatty():
                        choice = 0
                        self.get_logger().info("stdin is not a TTY; auto-selecting index 0.")
                    # User-facing prompt. Voice/stdin paths in this node
                    # are gone — point with your finger, click in the
                    # debug window, or use voice via communication_llm.
                    if len(names) == 1:
                        spoken = (
                            f"I see one object: a {names[0]}. "
                            "Point at it, click it in the window, or "
                            "tell me to pick it."
                        )
                    else:
                        listed = ", ".join(
                            f"a {n}" for n in names
                        )
                        spoken = (
                            f"I see {len(names)} objects: {listed}. "
                            "Point at one, click it in the window, or "
                            "tell me which to pick."
                        )
                    if choice is None:
                        prompt_once = bool(self.get_parameter("selection_prompt_once").value)
                        if (not prompt_once) or (not self._selection_prompted):
                            self._speak_if_enabled(spoken)
                            self._selection_prompted = True
                        self.get_logger().info(
                            "Waiting for selection (mediapipe pointing, debug-window click, or voice via communication_llm)..."
                        )
                        choice = self._choose_by_pointing(dets, names)
                    if choice is None:
                        if not bool(self.get_parameter("selection_prompt_once").value):
                            self._speak_if_enabled("No selection captured.")
                        self.get_logger().info(
                            f"Waiting for selection on '{self.get_parameter('manual_select_topic').value}' "
                            f"(communication_llm voice or debug-window click) or via mediapipe pointing "
                            f"(indices 0-{len(names)-1})."
                        )
                        timeout_s = float(self.get_parameter("selection_manual_wait_s").value)
                        t0 = time.time()
                        while (
                            choice is None
                            and rclpy.ok()
                            and (timeout_s < 0.0 or (time.time() - t0) < timeout_s)
                        ):
                            if self._pending_manual_choice is not None:
                                queued_idx = int(self._pending_manual_choice)
                                queued_stamp = float(self._pending_manual_choice_stamp)
                                self._pending_manual_choice = None
                                if (queued_stamp < self._last_cycle_end_t
                                        or (time.time() - queued_stamp) > 120.0):
                                    self._pending_manual_choice_det = None
                                    self.get_logger().info(
                                        f"Ignoring stale queued selection index {queued_idx} from previous cycle."
                                    )
                                else:
                                    queued_idx = self._resolve_clicked_choice(dets, queued_idx)
                                    if 0 <= queued_idx < len(names):
                                        choice = queued_idx
                                        self._choice_source_this = getattr(self, "_pending_choice_source", None) or "voice"
                                        self.get_logger().info(
                                            f"Using manual topic selection index {choice}: {names[choice]}"
                                        )
                                    else:
                                        self.get_logger().warn(
                                            f"Manual selection index {queued_idx} out of range [0, {len(names)-1}]."
                                        )
                            if choice is None:
                                # Keep mediapipe pointing active while
                                # waiting so a finger raised over an object
                                # still selects, alongside topic-based
                                # selections from communication_llm or
                                # the debug window click.
                                choice = self._choose_by_pointing(dets, names)
                            self._selection_nap(0.1)
                    if choice is None:
                        if not bool(self.get_parameter("selection_prompt_once").value):
                            self._speak_if_enabled("No valid selection was made.")
                        if bool(self.get_parameter("require_selection").value):
                            self.get_logger().info("No selection yet; retrying until a valid selection is made.")
                            self._selection_in_progress = False
                            if not self._voice_label_pending():
                                self._selection_nap(float(self.get_parameter("selection_retry_sleep_s").value))
                            continue
                        if not bool(self.get_parameter("repeat").value):
                            self.get_logger().info("Single-shot mode: stopping.")
                            self._selection_in_progress = False
                            self.timer.cancel()
                            break
                        else:
                            self.get_logger().info("Ready for next attempt.")
                            self.detections_cache = []
                            self._selection_in_progress = False
                            self._selection_nap(1.0)
                            continue
                    # 2026-09-28 (Majd): a POINTED choice (no voice / click
                    # source) waits for the operator's yes before the cycle
                    # starts — see _confirm_pointed_choice.
                    if (getattr(self, "_choice_source_this", None) is None
                            and bool(self.get_parameter("confirm_pointing").value)
                            and 0 <= int(choice) < len(dets)):
                        verdict = self._confirm_pointed_choice(dets, int(choice), header)
                        # 2026-09-30: the confirmed proposal names this pick in
                        # the selection record (recording contract v1)
                        self._selected_proposal_id = self._point_prop_id if verdict == "yes" else None
                        if verdict != "yes":
                            self._selection_in_progress = False
                            if verdict in ("no", "timeout", "ignored"):
                                self._selection_nap(0.3)
                            continue
                    self.selection_made = True
                    self.selected_idx = int(choice)
                    self._selection_prompted = False
                    self._selection_prompt_sig = None
                    self._speak_if_enabled(f"You chose {names[self.selected_idx]}.")
                    self._last_selection_source = getattr(self, "_choice_source_this", None) or "pointing"
                    self._choice_source_this = None
                    self.get_logger().info(f"Selected index {self.selected_idx}: {names[self.selected_idx]} (by {self._last_selection_source})")
                    if bool(self.get_parameter("publish_debug_image").value) and self.fb.rgb_frame is not None:
                        self.debug_pub.publish(
                            header,
                            self.fb.rgb_frame,
                            dets,
                            selected_idx=self.selected_idx,
                            pointer_px=self._pointer_px,
                            hand_landmarks_px=self._hand_landmarks_px,
                            hand_connections=self._hand_connections,
                            render_window=False,
                        )
                    if 0 <= self.selected_idx < len(dets):
                        sel_det = dets[self.selected_idx]
                        sel = {
                            "label": sel_det.get("simplified_label", sel_det["label"]),
                            "score": float(sel_det["score"]),
                            "bbox": [int(v) for v in sel_det["bbox"]],
                            "header": header,
                            # 2026-09-30 (recording contract v1): which instance —
                            # its index is meaningful only with this detection
                            # set and image (stamp, frame) — and, for a pointed
                            # pick, the confirmed proposal
                            "index": int(self.selected_idx), "n_detections": len(dets),
                            "display": str(sel_det.get("display", sel_det["label"])),
                            "proposal_id": (getattr(self, "_selected_proposal_id", None)
                                            if self._last_selection_source == "pointing" else None),
                        }
                        self._selected_proposal_id = None
                        self.last_selected_debug_det = {
                            "label": sel["label"],
                            "display": sel["label"],
                            "score": float(sel["score"]),
                            "bbox": list(sel["bbox"]),
                        }
                        # Tell the router the chosen label BEFORE the new
                        # cycle_id goes out. The full Detection2D follows
                        # ~1.5 s later (after registration), but the router
                        # narrates "Starting grasp on the X" the moment it
                        # sees the cycle_id — without this early minimal
                        # message it names the PREVIOUS pick.
                        try:
                            # Kill the keep-alive republish of the PREVIOUS
                            # selection FIRST: it restamps the old object's
                            # Detection2D at 5 Hz and overwrote the fresh
                            # label in the router mid-narration (measured
                            # 2026-08-20: maze click -> 'small_bottle'
                            # republished 200 ms later -> wrong name).
                            self.last_selected_detection_msg = None
                            early = Detection2D()
                            early.header = header
                            hyp = ObjectHypothesisWithPose()
                            hyp.hypothesis.class_id = str(sel["label"])
                            hyp.hypothesis.score = float(sel["score"])
                            early.results.append(hyp)
                            self.pub_detection.publish(early)
                        except Exception as exc:
                            self.get_logger().warn(
                                f"Early label publish failed: {exc}"
                            )
                        self._cycle_id_counter += 1
                        self._waiting_cycle_id = int(self._cycle_id_counter)
                        # Kill the keep-alive republish of the PREVIOUS
                        # cycle's grasps BEFORE announcing the new cycle_id.
                        # The 5 Hz republish timer restamps old candidates
                        # as fresh; the executor books anything arriving
                        # after the cycle_id bump under the NEW cycle and
                        # drives to the previous object's position (ghost
                        # pick #2, Majd 2026-08-12). PickPipeline.run()
                        # clears candidates too, but only AFTER this point
                        # — the gap was enough for the 50 Hz dispatcher.
                        self.last_candidates_base = None
                        self.last_candidate_scores = None
                        self.last_published_pose = None
                        self.last_published_pose_base = None
                        self.last_centroid = None
                        self.last_centroid_base = None
                        cycle_msg = Int64()
                        cycle_msg.data = self._waiting_cycle_id
                        self.pub_selection_cycle_id.publish(cycle_msg)
                        self.get_logger().info(
                            f"Published selection cycle_id={self._waiting_cycle_id} on "
                            f"'{self.get_parameter('selection_cycle_id_topic').value}'."
                        )
                        # Do not arm cycle-done waiting yet; publishing/filtering can take time,
                        # and stale completion messages must be ignored during that phase.
                        self._awaiting_executor_cycle_done = False
                        grasp_published = self._publish_selected(sel)
                    if not bool(self.get_parameter("repeat").value):
                        self.get_logger().info("Single-shot mode: stopping. Set 'repeat' to true to loop.")
                        self._selection_in_progress = False
                        self.timer.cancel()
                        break
                    else:
                        if (
                            grasp_published
                            and bool(self.get_parameter("wait_for_executor_cycle_done").value)
                        ):
                            # Arm waiting only after current cycle's grasp command has been published.
                            self._executor_cycle_done = False
                            self._executor_cycle_status = ""
                            self._awaiting_executor_cycle_done = True
                            timeout_s = float(self.get_parameter("executor_cycle_done_timeout_s").value)
                            t0 = time.time()
                            self.get_logger().info(
                                f"Waiting for executor completion on "
                                f"'{self.get_parameter('executor_cycle_done_topic').value}' "
                                f"(cycle_id={self._waiting_cycle_id})..."
                            )
                            while rclpy.ok():
                                if self._pick_abort:
                                    self.get_logger().warn(
                                        "Operator abort while waiting for "
                                        "executor completion — resetting the "
                                        "cycle (trial 1002: a dead executor "
                                        "left this wait stuck forever)."
                                    )
                                    break
                                if self._executor_cycle_done:
                                    status = self._executor_cycle_status or "done"
                                    if status != "failed":
                                        # A completed cycle restores the label's retry budget.
                                        self._auto_label_starts = 0
                                        self._auto_label_missing = 0
                                    if status == "failed":
                                        self.get_logger().warn(
                                            "Executor reported failure. Robot should have recovered to initgrasp via fallback. "
                                            "Restarting detection for next attempt."
                                        )
                                        # Only THIS object's stored grasps are
                                        # wrong; the others stay valid (the arm
                                        # is back home). The retry queues behind
                                        # new objects, five attempts at most.
                                        # Not when the executor refused to start
                                        # because it still holds the previous
                                        # object: the grasps were never tried.
                                        if "still holding" in str(getattr(self, "_executor_cycle_reason", "")):
                                            self.get_logger().warn(
                                                "Executor still holds the previous object — choose where to "
                                                "drop it (Drop left / Drop right / Release) before the next pick.")
                                        else:
                                            self._precompute.note_pick_failed(sel)
                                            # One automatic second try (Majd 2026-09-18:
                                            # "it should keep trying to execute, not
                                            # reset"): when no candidate could be planned,
                                            # the same object is queued again after the
                                            # reset with a fresh grasp set. Once per click:
                                            # a cycle that already was the retry ends here.
                                            reason = str(getattr(self, "_executor_cycle_reason", ""))
                                            mark = getattr(self, "_auto_retry_mark", None)
                                            was_retry = bool(mark) and self._same_object(mark, sel)
                                            self._auto_retry_mark = None
                                            # Only when NOTHING moved: no candidate could be
                                            # planned. A pick that moved and failed (empty
                                            # gripper, object pushed) is not retried by itself
                                            # (Majd 2026-09-18: "it grasps the same object twice").
                                            planning_only = ("tried " in reason or "reachable" in reason
                                                             or "no LIN-feasible" in reason.lower()
                                                             or "candidate" in reason.lower())
                                            # GUI clicks only (Majd 2026-09-18: a pointed or
                                            # spoken pick must not try twice by itself).
                                            if (bool(self.get_parameter("auto_retry_once").value)
                                                    and not was_retry and "stopped" not in reason
                                                    and planning_only
                                                    and getattr(self, "_last_selection_source", "") == "click"):
                                                self._auto_retry_sel = {"label": sel.get("label"),
                                                                        "bbox": list(sel.get("bbox", []))}
                                        # SELF-clear the auto-pick label on failure.
                                        # intent_router normally clears it at
                                        # cycle end, but when the voice stack is
                                        # dead the label survives and the pipeline
                                        # re-picks the same object FOREVER —
                                        # unkillable even by E-stop (Majd,
                                        # 2026-08-21, cycles 1..4 nonstop).
                                        try:
                                            if str(self.get_parameter("auto_select_label").value).strip():
                                                self.set_parameters([
                                                    rclpy.parameter.Parameter(
                                                        "auto_select_label",
                                                        rclpy.Parameter.Type.STRING,
                                                        "",
                                                    )
                                                ])
                                                self.get_logger().warn(
                                                    "auto_select_label cleared after failed cycle — "
                                                    "say/click the pick again to retry."
                                                )
                                        except Exception as exc:
                                            self.get_logger().warn(
                                                f"auto_select_label self-clear failed: {exc}"
                                            )
                                    else:
                                        self.get_logger().info(
                                            f"Executor cycle complete: {status}."
                                        )
                                    break
                                if timeout_s > 0.0 and (time.time() - t0) >= timeout_s:
                                    self.get_logger().warn(
                                        f"Still waiting for executor completion (> {timeout_s:.1f}s). "
                                        "Will not reset selection until 'done' or 'failed' is received."
                                    )
                                    # Extend the watchdog window but keep waiting for an explicit status.
                                    t0 = time.time()
                                time.sleep(0.1)
                        if not grasp_published:
                            self.get_logger().warn(
                                "No straight centre grasp was published; skipping executor wait "
                                "and returning to object selection."
                            )
                        self._awaiting_executor_cycle_done = False
                        self.get_logger().info("Resetting for next cycle...")
                        # A hard 3.0 s sleep sat here (Majd 2026-09-17: "I have to
                        # wait 1-2 s until it resets"); a short settle is enough for
                        # the overlay and the cache to see the table without the arm.
                        time.sleep(float(self.get_parameter("post_cycle_settle_s").value))
                        self.detections_cache = []
                        self.selection_made = False
                        self.selected_idx = None
                        self._registration_prepositioned_for_selection = False
                        self._registration_init_move_failed_this_cycle = False
                        self.last_selected_mask_np = None
                        self.last_selected_debug_det = None
                        self.last_selected_debug_frame = None
                        self.last_selected_debug_header = None
                        self.last_selected_debug_source = ""
                        self.last_selected_debug_stamp_mono = 0.0
                        self._selection_prompted = False
                        self._selection_prompt_sig = None
                        retry = getattr(self, "_auto_retry_sel", None)
                        self._auto_retry_sel = None
                        # A click made AFTER the cycle ended wins over the
                        # automatic retry (2026-09-18 12:25: the retry overwrote
                        # his click); a click made during the cycle is stale and
                        # is cleared as before.
                        click_fresh = (self._pending_manual_choice is not None
                                       and float(getattr(self, "_pending_manual_choice_stamp", 0.0))
                                       >= float(self._last_cycle_end_t))
                        if not click_fresh:
                            self._pending_manual_choice = None
                        if retry and click_fresh:
                            self.get_logger().info("[retry] dropped: the operator clicked something else")
                            retry = None
                        if retry:
                            self._pending_manual_choice_det = dict(retry)
                            self._pending_choice_source = "click"
                            self._pending_manual_choice = 0
                            self._pending_manual_choice_stamp = max(time.time(), float(self._last_cycle_end_t) + 0.01)
                            self._auto_retry_mark = dict(retry)
                            self._selection_wake.set()
                            self.get_logger().info(
                                f"[retry] '{retry.get('label')}' is queued once more with a fresh grasp set "
                                "(no candidate could be planned).")
                    self._selection_in_progress = False
                else:
                    time.sleep(0.1)
            except Exception as e:
                self._selection_in_progress = False
                self.get_logger().error(f"Selection thread error: {e}\n{traceback.format_exc()}")
                # A cycle that died mid-way must not leave selection_made
                # latched: nothing else clears it, so the node sat with a
                # dead cycle and Reset had nothing to reset (2026-09-11).
                self._awaiting_executor_cycle_done = False
                self.selection_made = False
                self.selected_idx = None
                self.detections_cache = []
                self._selection_prompted = False
                self._selection_prompt_sig = None
                self._pending_manual_choice = None
                self._last_cycle_end_t = time.time()
                time.sleep(1.0)


def main(args: Optional[List[str]] = None) -> None:
    """Entry point for ``ros2 run``."""
    rclpy.init(args=args)
    node = YoloLiveSelectGrasp()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
