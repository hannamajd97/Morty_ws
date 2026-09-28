"""All ROS parameter declarations for YoloLiveSelectGrasp.

Pulled out of ``__init__`` so the orchestrator stops being dominated by
~250 ``declare_parameter`` calls. Parameter names, types, and defaults
are byte-identical to the legacy inline declarations — every existing
launch file / CLI override / callsite that reads
``node.get_parameter(name).value`` keeps working unchanged.

The single public entry point is :func:`declare_all_parameters`. Call
it once near the top of the orchestrator's ``__init__`` and then read
back via ``self.get_parameter(...)`` exactly as before.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


def declare_all_parameters(node: "Node") -> None:
    """Declare every ROS parameter consumed by YoloLiveSelectGrasp.

    Parameter names, default values and types match the original
    inline declarations exactly. Call this once from
    ``YoloLiveSelectGrasp.__init__`` after ``super().__init__(...)``.
    """
    # Declare parameters
    # Resolve the workspace dir from $HOME (fall back to the user's expanded
    # home, never a hardcoded machine path). Override with WS_DIR if needed.
    home = os.environ.get("HOME", os.path.expanduser("~"))
    # Same fallback as run_grasp_pipeline.launch.py.
    ws_dir = os.environ.get("WS_DIR", os.path.join(home, "morty_ws"))
    node.declare_parameter("classes", "")
    node.declare_parameter(
        "yolo_world_model",
        "yolov8x.pt",
    )
    node.declare_parameter("conf", 0.25)
    # Per-class confidence floors. Pass as two parallel string-list
    # params; class i uses min_conf i. Empty list = no overrides.
    # Example launch override:
    #   per_class_min_conf_labels:='["vitamin_c"]'
    #   per_class_min_conf_values:='[0.02]'
    # Useful when ONE class is poorly trained and only emits weak
    # detections, while the global `conf` should stay safe for the
    # others.
    # 2026-08-25 (Majd: "he detects it as nutmeg jar?"): the Multi-
    # Mineral tube's orange label keeps flipping to nutmeg_jar at
    # 0.60-0.71 confidence. A real nutmeg jar scores 0.84+, so requiring
    # 0.8 kills the misdetection without losing the real jar.
    node.declare_parameter("per_class_min_conf_labels", ["nutmeg_jar"])
    node.declare_parameter("per_class_min_conf_values", [0.8])
    node.declare_parameter("dedupe_overlaps", True)
    node.declare_parameter("dedupe_iou_threshold", 0.65)
    node.declare_parameter("imgsz", 320)
    node.declare_parameter("device", "cuda")
    node.declare_parameter("yolo_half", True)
    node.declare_parameter("camera_topic", "/camera/camera/color/image_raw")
    node.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
    node.declare_parameter("color_info_topic", "/camera/camera/color/camera_info")
    node.declare_parameter("depth_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
    # Voice prompts/recognition removed from grasp_pipeline. The
    # communication_llm package owns speech now and reaches the
    # pipeline via /yolo_live_select_grasp/set_parameters (label) or
    # /yolo/select_index (numeric). Mediapipe finger-pointing and
    # mouse click in the debug window remain as offline fallbacks.
    node.declare_parameter("listen_seconds", 8.0)  # kept for back-compat
    # Topic where this node publishes user-facing prompt strings (e.g.
    # "Pick from: kitchen_towel, retail_box."). communication_llm
    # subscribes and routes them to its TTS. Set to "" to disable.
    node.declare_parameter(
        "selection_prompt_topic", "/grasp_pipeline/selection_prompt"
    )
    node.declare_parameter("graspgen_endpoint", "tcp://127.0.0.1:5555")
    node.declare_parameter("grasp_threshold", 0.4)
    # 1000 (was 2000): caps the cloud sent to GraspGen for faster inference
    # while retaining enough geometry for the current tabletop objects.
    # Live-tunable in the GUI Filters tab.
    node.declare_parameter("max_graspgen_points", 1000)
    # Grasp count knobs (were hardcoded 1500 / -1): 1274 surviving grasps
    # flooded the filter chain (~10 s) and executor probing. 500 sampled,
    # best 200 kept by confidence — downstream work drops ~6×
    # (Majd 2026-08-12, "it takes too long").
    # 500 -> 250 (Majd, 2026-08-21 speed pass: "let us try").
    node.declare_parameter("graspgen_num_grasps", 250)
    # Applied CLIENT-side (free numpy sort in pick_pipeline) — the server's
    # own topk path costs 5-7 s, so the RPC always sends -1. 200 keeps the
    # filter chain at ~3 s (399 grasps took 9.6 s, measured 2026-08-12).
    # -1 = NO top-k cap (Majd, 2026-08-21: "leave all the 204, no need
    # to take only the first 100").
    node.declare_parameter("graspgen_topk_num_grasps", -1)
    node.declare_parameter("publish_debug_image", True)
    node.declare_parameter("show_debug_window", True)
    node.declare_parameter("debug_window_name", "Grasp Debug")
    node.declare_parameter("debug_window_width", 600)
    node.declare_parameter("debug_window_height", 550)
    node.declare_parameter("enable_debug_window_click_select", True)
    node.declare_parameter("debug_publish_fps", 10.0)
    node.declare_parameter("republish_selected_outputs", True)
    node.declare_parameter("republish_selected_rate_hz", 2.0)
    node.declare_parameter("infer_timer_period_s", 0.05)
    node.declare_parameter("min_infer_interval_s", 0.0)
    node.declare_parameter("perf_log", True)
    node.declare_parameter("perf_log_period_s", 2.0)
    # Additive scene-awareness exports — let chat_node know what is visible.
    node.declare_parameter("publish_all_detections", True)
    node.declare_parameter("all_detections_topic", "/yolo/all_detections")
    node.declare_parameter("repeat", True)
    node.declare_parameter("wait_for_executor_cycle_done", True)
    node.declare_parameter("executor_cycle_done_topic", "/grasp_executor_lin/cycle_done")
    node.declare_parameter("executor_cycle_done_timeout_s", -1.0)
    node.declare_parameter("selection_cycle_id_topic", "/graspgen/selection_cycle_id")
    node.declare_parameter("require_matching_cycle_id", True)
    node.declare_parameter("fallback_random", False)
    node.declare_parameter("fx", 615.0)
    node.declare_parameter("fy", 615.0)
    node.declare_parameter("cx", 320.0)
    node.declare_parameter("cy", 240.0)
    node.declare_parameter("depth_scale", 1000.0)
    # False since 2026-09-15: the service returns METRES (its grasp_server
    # adds the cloud centre back); "normalised" treatment multiplied them by
    # the object's half-size and put every grasp 2-4 cm from the AABB
    # centre, differently for every object position. The origin is then
    # carried from GraspGen's gripper-base frame to the fingertips
    # (Franka Panda hand: 0.1034 m along the approach), see
    # geometry_utils._shift_grasp_origins. True restores the old behaviour.
    # Back to True on 2026-09-16: the metric path went live with the morning
    # restart before its pause test, and two picks closed on air. False +
    # the pause test is the plan; until then the old behaviour.
    node.declare_parameter("graspgen_matrix_normalized", True)
    node.declare_parameter("graspgen_origin_to_fingertip_m", 0.1034)
    # The checkpoint the GraspGen service runs (2026-09-24), set by the GUI
    # at pipeline start and after a model switch, together with the matching
    # graspgen_origin_to_fingertip_m. Part of the precompute's parameter
    # hash, so a switch drops the grasps stored from the other model; the
    # pick warns when the service answers as another gripper.
    node.declare_parameter("graspgen_model", "")
    node.declare_parameter("graspgen_column_major", False)
    node.declare_parameter("grasp2tcp_rot", "flipY")
    node.declare_parameter("prefer_side_grasps", False)
    node.declare_parameter("forbid_top_grasps", False)
    # View-cone half-width for the view-consistent filter (dot of the
    # physical approach with the camera view line). Was hardcoded 0.3.
    # BACK to 0.3 = 72deg (2026-08-21): 60deg made grasps MORE tilted -
    # the camera looks UP, so a narrower cone keeps TILTED approaches.
    # 0.3 -> 0.0 (2026-08-24, PROVEN by the orientation census): at 0.3
    # this cone RAISED the tilt median 22deg->37deg and killed 18 of 29
    # lateral-finger grasps — the camera's sight line is tilted, so
    # "camera-aligned" selected FOR tilt. 0.0 keeps the whole front
    # hemisphere (still rejects the unseen back) and leaves straightness
    # to the gravity gates, which measure TRUE vertical.
    # -1.0 = BYPASSED (2026-08-24, final census verdict): even at 0.0 this
    # camera-axis filter killed ALL 13 lateral grasps (97->10) because its
    # "forward" is near-vertical in this rig. Its legitimate job — do not
    # grasp the unseen back — is now done correctly by reject-from-behind
    # anchored at the REAL gripper TCP (Majd's frame-of-reference idea).
    node.declare_parameter("view_consistent_min_dot", -1.0)
    # ═══ THESIS §4.6.14 CASCADE (2026-08-24 teardown) ═══════════════════
    # Majd: "reread the master thesis and apply their filters, simple as
    # possible — the camera is tilted, and the center of the stuff should
    # be the gripper TCP, not the camera." The WHOLE orientation story is
    # now: these five knobs + max_finger_vertical_dot (roll-upright,
    # finger part) + reject_flipped_grasps/min_up_dot (palm part), all
    # evaluated in grasp_filters.apply_gravity_view_filter. Everything
    # marked RETIRED below is no longer read by the cascade.
    # Stage 2 (support-plane): minimum TCP height above the table, and
    # the from-below cut — physical approach ascending steeper than
    # asin(dot) is rejected (0.2 ≈ 11.5°; raise to ~0.4 if the census
    # shows the low camera pose starves the set).
    node.declare_parameter("support_min_height_m", 0.005)
    node.declare_parameter("support_from_below_dot", 0.2)
    # Stage 3 (side-grasp height window): |approach·up| below
    # side_horizontal_dot (0.5 = within 30° of horizontal) makes a grasp
    # a SIDE grasp; its TCP must then sit below table + factor × object
    # height. Steeper grasps pass through untouched (thesis: the window
    # "applies only to grasps with a near-horizontal approach").
    node.declare_parameter("side_horizontal_dot", 0.5)
    node.declare_parameter("side_height_factor", 2.0)  # RETIRED by fracs
    # Mid-height band (Majd's coffee grasp, 2026-08-24): side-grasp TCP
    # must sit between these fractions of the object's height above the
    # table — grip the body, not the cap, not the table edge.
    node.declare_parameter("side_height_min_frac", 0.10)
    node.declare_parameter("side_height_max_frac", 0.90)
    # LEVEL band (coffee grasp): approach within this many degrees of
    # horizontal, both directions, measured against the MEASURED table
    # plane. <= 0 disables (pure thesis mode: no tilt cap).
    node.declare_parameter("max_level_tilt_deg", 25.0)  # Majd: 25 -> 45 -> 35 -> 25 (2026-08-26)
    # Front-of-camera preference: score ×= exp(w·(approach·view − 1)).
    # 0 disables. Majd 2026-08-26: "grasp from the front of the camera".
    node.declare_parameter("front_approach_weight", 1.5)
    # The rig's TRUE up-direction in base coords — hardware-proven by the
    # fast-lift experiments (2026-08-26). Used as the TF fallback when the
    # ring plane can't be measured and memory is empty (Majd's design).
    node.declare_parameter("known_up_base", [-0.7071, 0.7071, 0.0])
    # "tf" = Majd's fixed-table-through-TF as the ONLY up source (2026-08-27
    # trial); "measured" = the original measure->memory->tf->prior ladder.
    # Default back to "measured" (2026-08-27 decision): the colour-based
    # table measurement below is now the PRIMARY source; "tf" remains the
    # A/B switch and the ladder fallback.
    node.declare_parameter("up_vector_source", "measured")
    # ── COLOUR table measurement (2026-08-27): the tabletop is uniform
    # saturated YELLOW, so measure the support plane from the table's own
    # colour (PRIMARY); the bbox-ring measurement stays the FALLBACK.
    # HSV bounds use OpenCV ranges — H in 0..179 (yellow ≈ 20..45), S/V
    # minimums in 0..255 reject grey/dark lookalikes.
    node.declare_parameter("table_from_colour", True)
    node.declare_parameter("table_colour_h_lo", 20)
    node.declare_parameter("table_colour_h_hi", 45)
    node.declare_parameter("table_colour_s_min", 80)
    node.declare_parameter("table_colour_v_min", 60)
    # ── NEIGHBOUR OBSTACLES (2026-08-27): one padded collision BOX per
    # detected object EXCEPT the grasp target, so MoveIt stops routing the
    # gripper through the other items standing on the table.
    # 2026-09-28 (Majd): OFF by default — the planner sees only the static
    # furniture unless the Motion tab's "Avoid other objects" turns it on.
    node.declare_parameter("neighbour_obstacles_enabled", False)
    node.declare_parameter("neighbour_obstacle_padding_m", 0.001)
    node.declare_parameter("neighbour_obstacle_min_px", 60)
    node.declare_parameter("neighbour_obstacle_frame", "base_link")
    # 2026-08-28 over-sized-box fix (measured: a 7 cm bottle got a
    # 0.117 x 0.119 x 0.080 m box, ~70 % too big). Extents are now taken
    # in the TARGET frame from percentile-trimmed points, so the boxes are
    # tight and padding drops 0.01 -> 0.005 m. depth_band = half-width of
    # the median-depth keep window (was a hard-coded 0.12 m, wide enough
    # for the table to join the object). max_size = if ANY final box
    # dimension exceeds this the object is SKIPPED — a runaway box rejects
    # every legitimate grasp near it.
    node.declare_parameter("neighbour_obstacle_depth_band_m", 0.06)
    node.declare_parameter("neighbour_obstacle_max_size_m", 0.30)
    # Thesis stage 1 from-behind, camera-frame rule: reject approaches
    # opposing the camera->object viewing line beyond this dot. -0.15
    # keeps sideways grasps (dot ~ 0) and kills the unseen back
    # hemisphere; <= -1.0 disables.
    node.declare_parameter("from_behind_min_dot", -0.15)
    # Palm canonicalization (2026-08-24 night): rotate any candidate
    # whose palm axis points UP by 180° about its approach axis — the
    # twin grips identically and arrives right side up. Replaces the
    # old reject_flipped filter, whose sign was INVERTED (it kept the
    # flipped twin — "the gripper have to go 180 to the position").
    node.declare_parameter("palm_canonicalize", True)
    # Stage 6 (centroid Gaussian reweight): σ = factor × object bbox max
    # extent ("half the object's bounding-box size"); <= 0 disables.
    node.declare_parameter("centroid_sigma_factor", 0.5)
    # ═══════════════════════════════════════════════════════════════════
    # Approach downwardness (dot with gravity-down) above which a grasp
    # counts as a TOP grasp and bypasses the view cone entirely.
    # 2.0 = DISABLED (dot can never exceed 1.0). Majd, 2026-08-20 night:
    # keep top-down grasps dismissed — the fingers crash into the table
    # on low objects. Set to ~0.5 to re-enable top grasps.
    node.declare_parameter("top_grasp_exempt_dot", 2.0)
    # Approach upwardness (dot with gravity-up) above which a grasp is
    # rejected as coming FROM BELOW. 2.0 = DISABLED (default).
    # WHY DISABLED: this camera observes the objects from BELOW their
    # height (measured 2026-08-21: camera at z=0.29, objects at z=0.62),
    # so every camera-facing grasp is mildly "upward" and this filter
    # deleted 53-71 of ~80 valid candidates — the exact grasps the old
    # working workspace executed. Enable (~0.4) only from a high,
    # downward-looking observation pose.
    node.declare_parameter("reject_from_below_max_up_dot", 2.0)
    # Keep only near-HORIZONTAL approaches (within
    # max_approach_tilt_deg of the gravity-horizontal plane). An optional
    # center_band_m can also impose a mid-height gate; <= 0 disables that
    # gate, leaving centre distance as a score preference only.
    # SOFT preference (always on): penalise scores by how far the
    # approach tilts away from horizontal, so straight-into-the-middle
    # grasps rank FIRST and the executor probes them first. 0 = off.
    # penalty = horizontal_score_weight * |approach . gravity_up|
    # (0 deg tilt -> 0, 30 deg -> 0.5w, 45 deg -> 0.71w, 90 deg -> 1.0w)
    # RETIRED 2026-08-24 (thesis teardown): the soft horizontal
    # preference, the 20°/5° tilt band, the mid-height band and the
    # TCP-frame off-axis reweight are all debugging-era inventions that
    # are NOT in the thesis cascade and are no longer read. The 20° band
    # was proven (orientation census) to kill the natural 25–45° lateral
    # grabs. Declarations kept only so old launch files/GUI don't crash.
    node.declare_parameter("horizontal_score_weight", 0.0)
    # LATERAL-PINCH rule (Majd, 2026-08-24: "the fingers going down
    # through the object from the middle always crash with the table").
    # |dot(finger-opening axis, gravity-up)| above this = the fingers
    # pinch top-and-bottom (one finger dives toward the table) -> reject.
    # 0.5 = fingers within ~30deg of horizontal. 2.0 disables.
    node.declare_parameter("max_finger_vertical_dot", 0.5)
    # TCP-frame centering (Majd, 2026-08-24: "center must be judged in
    # the GRIPPER TCP frame, at the fingers — not the camera"): score
    # penalty for the object's centroid sitting OFF the candidate
    # gripper's closing line (offset measured inside the candidate's own
    # frame, so the camera's tilt is irrelevant). 0 disables.
    node.declare_parameter("tcp_center_weight", 1.5)
    node.declare_parameter("require_horizontal_approach", True)
    node.declare_parameter("max_approach_tilt_deg", 20.0)
    # A symmetric 20 degree cone also admitted approaches that descended
    # 3.4 cm over the executor's 10 cm pre-grasp stroke.  Keep the useful
    # upward allowance, but make motion toward the object nearly level.
    node.declare_parameter("max_downward_approach_tilt_deg", 5.0)
    node.declare_parameter("center_band_m", 0.0)
    node.declare_parameter("max_vertical_dot", 0.9)
    node.declare_parameter("max_down_from_centroid", 0.04)
    node.declare_parameter("max_up_from_centroid", 0.04)
    # OFF 2026-08-24 (thesis teardown): the SUBTRACTIVE centroid penalty
    # could push scores negative; replaced by the multiplicative Gaussian
    # centroid_sigma_factor stage inside the shared cascade.
    node.declare_parameter("center_distance_weight", 0.0)
    node.declare_parameter("ignore_behind_grasps", True)
    node.declare_parameter("behind_grasp_threshold", 0.2)
    node.declare_parameter("ignore_above_grasps", True)
    node.declare_parameter("above_grasp_threshold", 0.7)
    # OFF 2026-08-24 night: sign was INVERTED (raw/TCP +y pointing UP is
    # the FLIPPED side per the executor's validated convention, and this
    # filter KEPT y·up >= min_up_dot) — it selected upside-down palms
    # and its keep-fallback hid that for weeks. Superseded by the
    # lossless palm_canonicalize stage in the shared cascade.
    node.declare_parameter("reject_flipped_grasps", False)
    node.declare_parameter("up_axis_index", 1)
    node.declare_parameter("min_up_dot", 0.1)
    # OFF 2026-08-24 (thesis teardown): this filter KEEPS ONLY
    # near-vertical approaches (|approach·plane_n| >= 0.7 ≈ tilted 44°+
    # from horizontal) — the exact opposite of the wanted side grasps.
    # Its keep-if-nonempty fallback masked it on all-lateral cycles, but
    # whenever a few steep candidates existed it deleted every lateral
    # one — a major hidden cause of "it always picks the tilted grasp".
    node.declare_parameter("reject_side_grasps", False)
    node.declare_parameter("min_vertical_dot", 0.7)
    node.declare_parameter("target_frame", "base_link")
    # Default disabled — replaced by the geometric "reject-from-behind"
    # filter, which is independent of the live TCP orientation and so
    # works correctly during multi-view sweeps. Set to True to opt
    # back into the original TCP-orientation cone filter.
    node.declare_parameter("orientation_consistency_filter", False)
    node.declare_parameter("max_orientation_delta_deg", 60.0)
    node.declare_parameter("ee_link_for_orientation_filter", "gripper_tcp")
    # OFF 2026-08-24 (thesis teardown): preferring candidates aligned
    # with the CURRENT wrist orientation is not in the thesis cascade —
    # the executor's distance-sort + planning probe covers reachability.
    node.declare_parameter("orientation_score_reweight", False)
    node.declare_parameter("orientation_score_weight", 1.0)
    node.declare_parameter("orientation_score_ref_deg", 45.0)
    # Robot self-filter (2026-09-15): the gripper's own collision hulls,
    # projected through TF into the image, are removed from detections and
    # zeroed in every depth frame. See perception/self_filter.py.
    node.declare_parameter("post_cycle_settle_s", 0.3)
    # One automatic second try of the clicked object with a fresh grasp set when no candidate could be planned (2026-09-18).
    # 2026-09-25 (Majd): OFF by default — after Morty said "I couldn't find a way
    # to reach the small bottle" the pipeline tried again on its own (cycle 8);
    # a failed pick now ends there and is retried only when he asks.
    node.declare_parameter("auto_retry_once", False)
    # 2026-09-28 (Majd): a POINTED choice is confirmed before the pick starts
    # — intent_router asks "You pointed at the X. Should I grasp it?" (or
    # "Do you mean this vitamin tube?" after its which-one question) and the
    # cycle begins on a yes. Clicks and spoken picks are not asked. See
    # _confirm_pointed_choice. Timeout 0 = wait for the answer forever; a NO
    # (or a timeout) ignores the same object for point_reject_hold_s, so the
    # hand still over it does not propose it again at once.
    node.declare_parameter("confirm_pointing", True)
    node.declare_parameter("point_confirm_timeout_s", 30.0)
    node.declare_parameter("point_reject_hold_s", 6.0)
    node.declare_parameter("self_filter_enabled", True)
    node.declare_parameter("self_filter_links",
                           ["gripper_base", "gripper_finger_left", "gripper_finger_right"])
    node.declare_parameter("self_filter_pad_px", 16)
    node.declare_parameter("self_filter_drop_overlap", 0.3)
    node.declare_parameter("self_filter_draw_overlay", True)
    # Open-set detection (2026-09-15): YOLOE proposes a box for every object,
    # the trained YOLO keeps its own, the rest are tracked and named once by
    # the naming service (Qwen3-VL-8B behind llama-server, started by the
    # GUI). See perception/open_set.py for the measurements behind the
    # defaults (m model at 416 px: 26 ms; 224 px crop + 6 tokens: 0.15 s).
    node.declare_parameter("open_set_enabled", True)
    node.declare_parameter("open_set_weights", "")
    node.declare_parameter("open_set_imgsz", 416)
    node.declare_parameter("open_set_conf", 0.10)
    # Every 2nd detection frame (measured 2026-09-15): inside the loaded
    # pipeline process YOLOE costs ~140 ms like the trained YOLO itself, not
    # the 26 ms of a bare process; every 2nd frame keeps the detection loop
    # near its old rate while a new object still gets its box within 0.5 s.
    node.declare_parameter("open_set_every_n_frames", 2)
    node.declare_parameter("open_set_min_box_px", 20)
    node.declare_parameter("open_set_max_box_frac", 0.25)
    node.declare_parameter("open_set_max_depth_m", 1.0)
    node.declare_parameter("open_set_stable_s", 0.2)
    node.declare_parameter("open_set_forget_s", 1.0)
    node.declare_parameter("open_set_naming_url", "http://127.0.0.1:8090/v1/chat/completions")
    node.declare_parameter("open_set_naming_timeout_s", 10.0)
    node.declare_parameter("open_set_crop_px", 224)
    node.declare_parameter("open_set_name_tokens", 6)
    # Identity check (2026-09-15): a YOLO detection must look like the
    # reference crops of its class (DINOv2 similarity) or it is handed to
    # the open-set path. Inactive until models/identity_refs exists.
    node.declare_parameter("identity_check_enabled", True)
    node.declare_parameter("identity_ref_dir", "")
    # 0.6 sits in the measured gap (2026-09-15, references from the stick's
    # captures): correct detections 0.76-0.88, wrong ones 0.35-0.52.
    node.declare_parameter("identity_threshold", 0.6)
    node.declare_parameter("min_depth_m", 0.0)
    node.declare_parameter("max_depth_m", 0.6)
    node.declare_parameter("apply_depth_clamp", True)
    node.declare_parameter("apply_largest_cluster_filter", True)
    node.declare_parameter("cluster_eps_m", 0.02)
    node.declare_parameter("cluster_min_samples", 12)
    node.declare_parameter("apply_radius_outlier_filter", True)
    node.declare_parameter("radius_filter_m", 0.02)
    node.declare_parameter("radius_min_neighbors", 30)
    node.declare_parameter("max_grasp_centroid_dist_m", 0.15)
    node.declare_parameter("apply_cloud_cleaning", False)
    node.declare_parameter("apply_plane_above_filter", False)
    node.declare_parameter("plane_clearance_m", 0.01)
    node.declare_parameter("min_points_for_graspgen", 100)
    node.declare_parameter("log_pregrasp_filter_counts", True)
    # Finger penetration filter — rejects grasps whose fingers would
    # crush the object. Geometry defaults match the gripper_3d
    # two-finger gripper.
    # OFF 2026-08-24 (thesis teardown): not part of the §4.6.14 cascade,
    # and the measured 12 s CPU hotspot of the whole chain. The close-
    # stall detection + slip watchdog catch a bad pinch at execution.
    node.declare_parameter("apply_finger_penetration_filter", False)
    # Gripper width and calibration offset (2026-09-11): GraspGen's models
    # assume a 105-136 mm opening, the printed gripper opens 85 mm.
    node.declare_parameter("apply_gripper_width_filter", True)
    node.declare_parameter("gripper_max_object_width_m", 0.080)
    # Translation in the executed TCP frame, metres — the axes RViz draws on
    # /grasp_executor_lin_llm/selected_pose (x red = closing direction,
    # y green, z blue = out of the fingers), applied after the grasp-to-TCP
    # rotation: the measured difference between where the model's gripper
    # and ours put the object. "Move it 5 cm along -x" = [-0.05, 0, 0].
    # Measured by Majd on 2026-09-11 (RViz, small bottle): +2 cm along the
    # closing direction puts the object between the fingers.
    node.declare_parameter("grasp2tcp_offset_m", [0.0, 0.0, 0.0])   # 0 since 2026-09-17 (Majd); the two-line approach removed the finger hits
    node.declare_parameter("finger_penetration_finger_length_m", 0.08)
    node.declare_parameter("finger_penetration_finger_thickness_m", 0.012)
    node.declare_parameter("finger_penetration_finger_height_m", 0.025)
    node.declare_parameter("finger_penetration_gripper_opening_m", 0.085)
    node.declare_parameter("finger_penetration_finger_z_offset_m", 0.06)
    node.declare_parameter("finger_penetration_max_points_per_finger", 5)
    node.declare_parameter("finger_penetration_min_points_in_closing", 0)
    # Reject-from-behind filter — keeps grasps approaching from above
    # or from the side (toward the centroid), rejects grasps that
    # would approach from behind the object (away from the observation
    # viewpoint).
    # OFF 2026-08-24 (thesis teardown): with thesis stage 1 (the camera
    # view-consistency cone) dropped for this rig, its replacement is
    # dropped too — Majd approved judging ONLY at the gripper TCP.
    node.declare_parameter("apply_reject_from_behind_filter", False)
    node.declare_parameter("reject_from_behind_observation_height_m", 0.30)
    node.declare_parameter("reject_from_behind_min_forward_dot", 0.0)
    # Performance: voxel pre-downsample before heavy outlier filters
    # (set <=0 to disable). 0 by default to keep full resolution; the
    # lightweight per-frame filter path skips the expensive filters
    # entirely so this only matters for the post-fusion filter.
    node.declare_parameter("pregrasp_filter_voxel_pre_downsample_m", 0.003)
    # When True, the per-frame utility evaluation skips cluster +
    # radius outlier filters and runs only the cheap IQR / median
    # distance trim. The full filter still runs once at the end on
    # the fused cloud.
    node.declare_parameter("pregrasp_filter_lightweight_during_sweep", True)
    # Segmentation (optional): refine bbox into a pixel mask before building the pointcloud
    node.declare_parameter("segmentation_enabled", True)
    # backends: 'grabcut', 'ultralytics_sam' (SAM checkpoint), 'yolo_seg' (YOLO segmentation .pt)
    node.declare_parameter("segmentation_backend", "grabcut")
    node.declare_parameter("sam_model_path", "")
    node.declare_parameter("segmentation_min_area_px", 500)
    node.declare_parameter("segmentation_mask_dilate_px", 4)
    node.declare_parameter("segmentation_mask_topic", "/yolo/selected_mask")
    node.declare_parameter("segmentation_overlay_topic", "/yolo/selected_mask_overlay")
    node.declare_parameter("grabcut_iters", 5)
    node.declare_parameter("grabcut_margin_px", 5)
    node.declare_parameter("grabcut_close_kernel", 7)
    node.declare_parameter("grabcut_open_kernel", 3)
    # Non-interactive selection (useful when launched without stdin)
    node.declare_parameter("auto_select_index", -1)
    node.declare_parameter("auto_select_label", "")
    node.declare_parameter("auto_select_noninteractive", False)
    node.declare_parameter("manual_select_topic", "/yolo/select_index")
    node.declare_parameter("selected_detection_topic", "/yolo/selected_detection")
    node.declare_parameter("selected_cloud_topic", "/yolo/selected_cloud")
    node.declare_parameter("debug_image_topic", "/yolo/debug_image")
    node.declare_parameter("use_mediapipe_selection", True)
    # /tf arrives at ~830 Hz from the UR driver; the node only ever asks for
    # the LATEST transform, so 50 samples/s per publisher stream is plenty
    # and costs a few % of a core instead of a whole one (tf_listener_lite).
    node.declare_parameter("tf_listener_max_hz", 50.0)
    # Grasp precompute cache (2026-09-11, Hamid/Majd): GraspGen for every
    # detected object while nobody has selected, so a selection publishes
    # stored candidates at once. Off until tested on the robot; see
    # grasping/precompute.py for the validity rules.
    node.declare_parameter("graspgen_precompute", False)
    node.declare_parameter("precompute_stable_s", 0.2)      # seen this long before it is queued
    node.declare_parameter("precompute_max_objects", 8)
    node.declare_parameter("precompute_box_shift_px", 8.0)  # box centre moved more -> recompute
    node.declare_parameter("precompute_size_tol", 0.15)     # box size changed more (ratio) -> recompute
    node.declare_parameter("precompute_depth_tol_m", 0.01)  # object median depth moved more -> recompute
    node.declare_parameter("precompute_pose_tol_m", 0.002)  # arm (tool0) moved more -> all invalid
    node.declare_parameter("precompute_pose_tol_deg", 0.5)
    node.declare_parameter("precompute_request_timeout_s", 5.0)
    node.declare_parameter("precompute_max_retries", 5)     # failed attempts per object before giving up
    node.declare_parameter("precompute_lin_precheck", True)  # hand stored sets to the executor's background LIN check
    # Hold raised 250 -> 400 ms (2026-08-07, Majd's choice): at 250 ms a
    # pointing ray sweeping across a bbox committed selections the user
    # never meant.
    node.declare_parameter("mediapipe_min_hold_ms", 400)
    node.declare_parameter("mediapipe_steady_px", 10.0)
    node.declare_parameter("mediapipe_require_inside", False)
    # Max px from a bbox edge at which an outside pointer may still snap
    # to that box (only used when mediapipe_require_inside is false).
    node.declare_parameter("mediapipe_snap_px", 80.0)
    node.declare_parameter("mediapipe_ray_extension", 1.5)
    node.declare_parameter("mediapipe_min_detection_confidence", 0.60)
    node.declare_parameter("mediapipe_min_tracking_confidence", 0.60)
    # 3D pointing (2026-09-03, Majd: "give me the beautiful result"):
    #   pointing_mode "3d" = finger ray knuckle→tip deprojected with the
    #   aligned depth, scored by angle to each box's 3D centre (Hamid's
    #   geometry; Kavousian 2026: 93 %/20 participants on a D435i);
    #   "2d" = the previous picture-plane line. 3d falls back to 2d on any
    #   frame without fingertip depth. Flip to "2d" to A/B the old behaviour.
    node.declare_parameter("pointing_mode", "3d")
    # Half-angle of the acceptance cone around the finger ray (deg).
    node.declare_parameter("mediapipe_cone_deg", 20.0)
    # Score multiplier for the object already hovered (<1 = sticky) so two
    # neighbours don't flicker while the dwell clock runs.
    node.declare_parameter("mediapipe_hysteresis", 0.8)
    # Tasks HandLandmarker in VIDEO mode (tracks between frames) when the
    # model file exists; False forces the legacy mp.solutions API.
    node.declare_parameter("mediapipe_use_tasks_api", True)
    # "" = the hand_landmarker.task shipped inside the package.
    node.declare_parameter("mediapipe_model_path", "")
    # Commit rule and ray source (2026-09-24, see point_selector.py):
    # "vote" = evidence over time (default), "dwell" = the old steady hold;
    # sigma = the finger ray's angular noise; ray source "model" = MediaPipe's
    # 3-D hand for the direction + palm depth — off until a recording
    # (pointing_bench_record / pointing_bench_eval) shows it is better.
    node.declare_parameter("pointing_decision", "vote")
    node.declare_parameter("pointing_vote_tau_s", 0.8)
    node.declare_parameter("pointing_vote_commit_s", 0.45)
    node.declare_parameter("pointing_vote_margin", 2.0)
    node.declare_parameter("pointing_sigma_deg", 7.0)
    # "arrow" (2026-09-25): each 3-D frame votes for the object the drawn
    # arrow ends on; "angle": spread by the angle off the ray (sigma above).
    node.declare_parameter("pointing_evidence", "arrow")
    node.declare_parameter("pointing_ray_source", "depth")
    # Default changed True -> False (2026-08-06): with True, YOLO stops the
    # moment the selection prompt starts, so the detection boxes in the debug
    # image FREEZE exactly when the user is trying to click one. Keeping
    # inference running lets the boxes track the scene while selecting.
    node.declare_parameter("suspend_inference_while_selecting", False)
    node.declare_parameter("require_selection", True)
    node.declare_parameter("selection_prompt_once", True)
    node.declare_parameter("selection_attempt_timeout_s", 10.0)
    node.declare_parameter("selection_manual_wait_s", -1.0)
    node.declare_parameter("selection_retry_sleep_s", 0.5)
    # Publish the exact cloud sent to GraspGen
    node.declare_parameter("publish_graspgen_input_cloud", True)
    node.declare_parameter("graspgen_input_cloud_topic", "/graspgen/input_cloud")
    node.declare_parameter("publish_grasp_candidates", True)
    node.declare_parameter("grasp_candidates_top_k", 50)
    # MoveIt's tabletop collision slab tops out at world/base z=0.600 m.
    # Keep 15 mm TCP clearance and reject low candidates before execution.
    # world and base_link share z in this robot model. <= 0 disables.
    # 0.615 -> 0.595 (2026-08-24): real object-middle grasps measure z
    # 0.60-0.64, so a 0.615 floor vetoed ENTIRE healthy cycles ("refusing
    # execution to protect the table" -> instant re-selection, Majd:
    # "I can't execute, it just restarts"). 0.595 = the collision slab's
    # top: anything above it is above the physical tabletop.
    # DISABLED 2026-08-24 night: this floor judges height through the
    # base-frame TF conversion — the exact path shown to be unreliable —
    # and it vetoed the first cycle's ONLY good survivor (level 2°,
    # lateral, FRONT) with "ALL candidates below world z=0.595". The
    # table is now protected in the cloud frame by the MEASURED ring
    # plane (support-plane stage), which does this job correctly.
    # Set > 0 to re-enable the old base-z floor.
    node.declare_parameter("candidate_min_world_z", 0.0)
    node.declare_parameter("grasp_candidates_topic", "/graspgen/candidate_poses_base_link")
    node.declare_parameter("grasp_candidate_scores_topic", "/graspgen/candidate_scores")
    node.declare_parameter("registration_enabled", True)
    node.declare_parameter("registration_num_observations", 3)
    node.declare_parameter("registration_observation_interval_s", 1.0)
    node.declare_parameter("registration_refresh_detection", True)
    node.declare_parameter("registration_min_points", 80)
    node.declare_parameter("registration_target_frame", "base_link")
    node.declare_parameter("registration_voxel_size_m", 0.0015)
    node.declare_parameter("registration_max_extra_attempts", 4)
    node.declare_parameter("registration_min_camera_translation_m", 0.03)
    node.declare_parameter("registration_min_camera_rotation_deg", 6.0)
    node.declare_parameter("registration_max_bbox_iou_same_view", 0.92)
    node.declare_parameter("registration_followup_use_tracked_segmentation", True)
    node.declare_parameter("registration_tracked_followup_max_point_ratio", 1.45)
    node.declare_parameter("registration_tracked_followup_max_extent_ratio", 1.35)
    node.declare_parameter("registration_tracked_followup_max_centroid_offset_m", 0.03)
    node.declare_parameter("registration_tracked_followup_max_bbox_area_ratio", 1.8)
    node.declare_parameter("registration_followup_use_roi_fallback", True)
    node.declare_parameter("registration_followup_use_projected_mask_prior", True)
    node.declare_parameter("registration_projected_mask_dilate_px", 10)
    node.declare_parameter("registration_projected_mask_min_overlap_ratio", 0.15)
    node.declare_parameter("registration_roi_margin_m", 0.04)
    node.declare_parameter("registration_roi_min_half_extent_m", 0.03)
    node.declare_parameter("registration_roi_fallback_max_point_ratio", 2.0)
    node.declare_parameter("registration_roi_fallback_max_extent_ratio", 1.8)
    node.declare_parameter("registration_roi_fallback_reference_voxel_m", 0.003)
    node.declare_parameter("registration_roi_fallback_max_surface_distance_m", 0.035)
    node.declare_parameter("registration_nbv_enabled", True)
    node.declare_parameter("registration_nbv_gain_voxel_size_m", 0.003)
    node.declare_parameter("registration_nbv_min_gain_ratio", 0.08)
    node.declare_parameter("registration_nbv_min_new_voxels", 150)
    node.declare_parameter("registration_dynamic_nbv_enabled", True)
    node.declare_parameter("registration_dynamic_nbv_use_joint_interpolation", True)
    node.declare_parameter("registration_dynamic_nbv_joint_alpha", 1.0)
    node.declare_parameter("registration_dynamic_nbv_joint_alpha_values", "0.6,0.8,1.0")
    node.declare_parameter("registration_dynamic_nbv_probe_alpha", 1.0)
    node.declare_parameter("registration_dynamic_nbv_refine_top_k", 2)
    node.declare_parameter("registration_dynamic_nbv_drop_nonwinning_coarse", True)
    node.declare_parameter("registration_continuous_observation_enabled", True)
    node.declare_parameter("registration_continuous_observation_velocity_scaling", 0.009)
    node.declare_parameter("registration_continuous_observation_min_camera_translation_m", 0.004)
    node.declare_parameter("registration_continuous_observation_min_camera_rotation_deg", 0.8)
    node.declare_parameter("registration_continuous_observation_poll_period_s", 0.15)
    node.declare_parameter("registration_continuous_observation_max_frames", 20)
    node.declare_parameter("registration_continuous_observation_sweep_all_directions", True)
    node.declare_parameter("registration_continuous_observation_sweep_all_alpha", 1.0)
    node.declare_parameter("registration_continuous_observation_sweep_all_return_to_seed", True)
    node.declare_parameter("registration_continuous_observation_step_alpha", 0.2)
    node.declare_parameter("registration_continuous_observation_max_consecutive_rejections", 10)
    node.declare_parameter("registration_continuous_observation_bbox_border_margin_px", 24)
    node.declare_parameter("registration_continuous_observation_min_edge_bbox_area_ratio", 0.35)
    node.declare_parameter("registration_continuous_observation_min_edge_bbox_height_ratio", 0.45)
    node.declare_parameter("registration_continuous_observation_reject_bottom_edge", True)
    node.declare_parameter("registration_continuous_observation_higher_low_penalty", 0.18)
    node.declare_parameter("registration_continuous_observation_higher_bottom_edge_penalty", 0.12)
    node.declare_parameter("registration_continuous_observation_force_direction", "")
    node.declare_parameter("registration_continuous_observation_center_bias_max", 0.04)
    node.declare_parameter("registration_continuous_observation_depth_slope_weight", 0.08)
    node.declare_parameter("registration_continuous_observation_depth_slope_deadband", 0.08)
    # Occlusion-aware NBV (Fix A + Fix B):
    # Fix A — when computing image-space direction cues (bbox center,
    # img_offset), use the YOLO segmentation mask centroid instead of
    # the raw bbox center, so an occluder sharing the bbox doesn't
    # pollute the signal.
    node.declare_parameter("registration_continuous_observation_use_mask_centroid", True)
    # Fix B — detect a YOLO box that overlaps the target's bbox by at
    # least this IoU; treat it as an occluder and bias the direction
    # AWAY from its centroid, so the sweep "looks around" the obstacle.
    node.declare_parameter("registration_continuous_observation_occluder_iou_threshold", 0.10)
    node.declare_parameter("registration_continuous_observation_occluder_bonus_max", 0.30)
    node.declare_parameter("registration_continuous_observation_occluder_min_offset_px", 30.0)
    node.declare_parameter("registration_dynamic_nbv_srdf_path", "")
    node.declare_parameter("registration_dynamic_nbv_planning_group", "ur3e_manipulator")
    node.declare_parameter("registration_dynamic_nbv_seed_target", "home")
    node.declare_parameter("registration_dynamic_nbv_lateral_offset_m", 0.12)
    node.declare_parameter("registration_dynamic_nbv_vertical_offset_m", 0.10)
    node.declare_parameter("registration_dynamic_nbv_upward_bias_m", 0.01)
    node.declare_parameter("registration_dynamic_nbv_min_translation_m", 0.03)
    node.declare_parameter("registration_dynamic_nbv_min_rotation_deg", 5.0)
    node.declare_parameter("registration_dynamic_nbv_min_ee_z_m", 0.38)
    node.declare_parameter("registration_dynamic_nbv_max_ee_translation_m", 0.08)
    node.declare_parameter("registration_dynamic_nbv_max_ee_downward_delta_m", 0.025)
    node.declare_parameter("registration_dynamic_nbv_min_relative_ee_z_delta_m", -0.025)
    node.declare_parameter("registration_dynamic_nbv_max_camera_distance_delta_m", 0.03)
    node.declare_parameter("registration_observation_ee_link", "gripper_tcp")
    node.declare_parameter("registration_move_to_initial_target", True)
    node.declare_parameter("registration_preposition_before_selection", True)
    node.declare_parameter("registration_initial_named_target", "home")
    node.declare_parameter("registration_observation_named_targets", "detect_left,detect_right,detect_higher")
    node.declare_parameter("registration_motion_settle_s", 0.8)
    node.declare_parameter("registration_motion_timeout_s", 120.0)
    node.declare_parameter("registration_motion_velocity_scaling", 0.1)
    node.declare_parameter("registration_motion_acceleration_scaling", 0.1)
    node.declare_parameter("registration_observation_planning_time_s", 1.0)
    node.declare_parameter("registration_observation_num_planning_attempts", 1)
    node.declare_parameter("registration_observation_allow_fallbacks", False)
    node.declare_parameter(
        "registration_observation_commander_executable",
        os.path.join(
            ws_dir,
            "install",
            "lib",
            "ur3e_gripper_3d_moveit_commander",
            "observation_ptp_commander",
        ),
    )
    # Fast replay (ur3e_fast_observation) — used for taught named SRDF
    # targets that are present in the cache YAML. Falls back to the PTP
    # commander automatically if the pose is missing or replay fails.
    node.declare_parameter("registration_use_fast_observation_moves", True)
    node.declare_parameter(
        "registration_fast_observation_poses_file",
        os.path.join(ws_dir, "observation_poses.yaml"),
    )
    node.declare_parameter(
        "registration_fast_observation_replay_topic",
        "/ur3e_fast_observation_replay/replay",
    )
    node.declare_parameter(
        "registration_fast_observation_status_topic",
        "/ur3e_fast_observation_replay/status",
    )
    node.declare_parameter("registration_fast_observation_timeout_s", 8.0)
    node.declare_parameter("registration_use_icp", False)
    node.declare_parameter("registration_icp_point_to_plane", True)
    node.declare_parameter("registration_icp_max_correspondence_m", 0.008)
    node.declare_parameter("registration_icp_max_iterations", 40)
    node.declare_parameter("registration_icp_min_fitness", 0.35)
    node.declare_parameter("registration_icp_max_rmse_m", 0.006)
    node.declare_parameter("registration_icp_max_translation_correction_m", 0.01)
    node.declare_parameter("registration_icp_max_rotation_correction_deg", 8.0)
    node.declare_parameter("registration_icp_local_map_radius_m", 0.12)
    node.declare_parameter("registration_icp_normal_radius_m", 0.02)
    node.declare_parameter("registration_fused_cloud_topic", "/graspgen/fused_cloud")
    node.declare_parameter("registration_tsdf_enabled", True)
    node.declare_parameter("registration_tsdf_cloud_topic", "/graspgen/tsdf_cloud")
    node.declare_parameter("registration_tsdf_voxel_length_m", 0.001)
    node.declare_parameter("registration_tsdf_sdf_trunc_m", 0.004)
    node.declare_parameter("registration_tsdf_depth_trunc_m", 1.2)
    node.declare_parameter("registration_tsdf_point_voxel_size_m", 0.0008)
    node.declare_parameter("registration_tsdf_mask_erode_px", 5)
    node.declare_parameter("registration_tsdf_depth_band_m", 0.04)
    node.declare_parameter("registration_tsdf_endpoint_only", True)
    node.declare_parameter("registration_grasp_utility_logging_enabled", True)
    node.declare_parameter("registration_grasp_utility_weight_count", 1.0)
    node.declare_parameter("registration_grasp_utility_weight_diversity", 1.0)
    node.declare_parameter("registration_grasp_utility_weight_score", 1.0)
    node.declare_parameter("registration_grasp_utility_weight_instability", 1.0)
    node.declare_parameter("registration_grasp_utility_stop_enabled", True)
    node.declare_parameter("registration_grasp_utility_stop_min_views_after_seed", 2)
    node.declare_parameter("registration_grasp_utility_stop_min_delta_u", 0.05)
    node.declare_parameter("registration_grasp_utility_stop_patience", 2)
    node.declare_parameter("registration_grasp_utility_stop_large_drop_delta_u", -0.3)
    node.declare_parameter("registration_grasp_utility_stop_during_motion", False)
