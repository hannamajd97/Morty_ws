"""
run_grasp_pipeline.launch.py
============================

Launches the perception + grasping side of the system: the YOLO live-select
node, the grasp selection node, and (optionally) the named-pose replay node.
This is what you start when you want the robot to see, segment, sweep
(multi-view), and request grasps from GraspGen — without the chat/voice
layer (that lives in ``communication_llm``).

Defaults to MULTI-VIEW mode (``registration_enabled:=true``); set it to
``false`` for single-view picks. See the project's ``2_Launches_and_First_Run``
doc for the full set of overridable arguments (there are many — YOLO model,
camera topics, GraspGen endpoint, segmentation backend, MediaPipe selection,
filter thresholds, registration / NBV / TSDF toggles, etc.).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    home = EnvironmentVariable("HOME")

    # Workspace root: honour $WS_DIR (same convention as selection/params.py),
    # fall back to ~/morty_ws. The GUI exports WS_DIR for everything it
    # spawns, so models/ and caches always resolve inside this workspace.
    ws_dir = os.environ.get(
        "WS_DIR", os.path.join(os.path.expanduser("~"), "morty_ws")
    )

    # Defaults match the common CLI you use; override via launch args as needed.
    # yolo26s-seg_24_04.pt is the 9-class segmentation model: tape_dispenser,
    # cd_case, toy_faucet, kitchen_towel, nutmeg_jar, retail_box, small_bottle,
    # wooden_maze_puzzle, vitamin_c. (The 24_04 retrain; older checkpoints removed.)
    default_model = os.path.join(
        ws_dir, "src", "grasp_pipeline", "yolo26s-seg_24_04.pt"
    )
    default_seg_model = default_model

    declare_args = [
        DeclareLaunchArgument("yolo_world_model", default_value=default_model),
        DeclareLaunchArgument(
            "classes",
            default_value=(
                "tape_dispenser, cd_case, toy_faucet, kitchen_towel, "
                "nutmeg_jar, retail_box, small_bottle, "
                "wooden_maze_puzzle, vitamin_c"
            ),
        ),
        DeclareLaunchArgument("conf", default_value="0.05"),
        # Per-class confidence overrides. Each entry in labels uses the
        # matching value index. Empty default disables overrides.
        # 2026-08-25: nutmeg_jar needs 0.8+ (Multi-Mineral tube kept
        # misdetecting as nutmeg_jar at 0.60-0.71; a real jar scores
        # 0.84+). Kept in sync with params.py — launch defaults
        # silently override it otherwise.
        DeclareLaunchArgument("per_class_min_conf_labels", default_value="['nutmeg_jar']"),
        DeclareLaunchArgument("per_class_min_conf_values", default_value="[0.8]"),
        DeclareLaunchArgument("imgsz", default_value="640"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("yolo_half", default_value="false"),
        DeclareLaunchArgument("camera_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("color_info_topic", default_value="/camera/camera/color/camera_info"),
        DeclareLaunchArgument("depth_info_topic", default_value="/camera/camera/aligned_depth_to_color/camera_info"),
        DeclareLaunchArgument(
            "selection_prompt_topic",
            default_value="/grasp_pipeline/selection_prompt",
            description="Topic the pipeline publishes user-facing prompt "
            "strings on. communication_llm subscribes and speaks them.",
        ),
        DeclareLaunchArgument("graspgen_endpoint", default_value="tcp://127.0.0.1:5555"),
        # Keep the launch override aligned with the node default. 1000 points
        # is the current speed/geometry trade-off for tabletop objects.
        DeclareLaunchArgument("max_graspgen_points", default_value="1000"),
        DeclareLaunchArgument("publish_debug_image", default_value="true"),
        # Default false on the Alienware (2026-08-06): the in-node cv2 window
        # cannot work here (SYSTEM_STATE_LOG #21/#28); the Qt click viewer
        # replaces it. Pass show_debug_window:=true only on a machine where
        # cv2 HighGUI actually displays.
        DeclareLaunchArgument("show_debug_window", default_value="false"),
        DeclareLaunchArgument("debug_window_name", default_value="Grasp Debug"),
        DeclareLaunchArgument("debug_window_width", default_value="600"),
        DeclareLaunchArgument("debug_window_height", default_value="550"),
        DeclareLaunchArgument("enable_debug_window_click_select", default_value="true"),
        DeclareLaunchArgument("debug_publish_fps", default_value="12.0"),
        DeclareLaunchArgument("republish_selected_outputs", default_value="true"),
        DeclareLaunchArgument("republish_selected_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("infer_timer_period_s", default_value="0.05"),
        DeclareLaunchArgument("min_infer_interval_s", default_value="0.0"),
        DeclareLaunchArgument("perf_log", default_value="true"),
        DeclareLaunchArgument("perf_log_period_s", default_value="2.0"),
        DeclareLaunchArgument("use_mediapipe_selection", default_value="true"),
        # 250 -> 400 (2026-08-07, Majd): 250 ms let a pointing ray sweeping
        # across a bbox commit a selection the user never meant.
        DeclareLaunchArgument("mediapipe_min_hold_ms", default_value="400"),
        DeclareLaunchArgument("mediapipe_steady_px", default_value="10.0"),
        DeclareLaunchArgument("mediapipe_require_inside", default_value="true"),
        DeclareLaunchArgument("mediapipe_min_detection_confidence", default_value="0.60"),
        DeclareLaunchArgument("mediapipe_min_tracking_confidence", default_value="0.60"),
        # 3D pointing (2026-09-03) — "3d" finger ray + angle to 3D centres,
        # "2d" the previous picture-plane line. See point_selector.py.
        DeclareLaunchArgument("pointing_mode", default_value="3d"),
        DeclareLaunchArgument("mediapipe_cone_deg", default_value="20.0"),
        DeclareLaunchArgument("mediapipe_hysteresis", default_value="0.8"),
        DeclareLaunchArgument("mediapipe_use_tasks_api", default_value="true"),
        # false so detection boxes keep refreshing while waiting for the click
        # (true froze them exactly during selection — 2026-08-06)
        DeclareLaunchArgument("suspend_inference_while_selecting", default_value="false"),
        DeclareLaunchArgument("apply_depth_clamp", default_value="true"),
        DeclareLaunchArgument("apply_largest_cluster_filter", default_value="true"),
        DeclareLaunchArgument("cluster_eps_m", default_value="0.030"),
        DeclareLaunchArgument("cluster_min_samples", default_value="20"),
        DeclareLaunchArgument("apply_radius_outlier_filter", default_value="true"),
        DeclareLaunchArgument("radius_filter_m", default_value="0.05"),
        DeclareLaunchArgument("radius_min_neighbors", default_value="50"),
        DeclareLaunchArgument("apply_cloud_cleaning", default_value="false"),
        DeclareLaunchArgument("apply_plane_above_filter", default_value="false"),
        DeclareLaunchArgument("plane_clearance_m", default_value="0.01"),
        # Compatibility only (2026-08-24 thesis teardown): a GUI already
        # running may still pass these retired knobs on its next restart.
        # They are NOT forwarded to the node — params.py defaults rule.
        # (Launch-arg defaults silently overriding params.py bit us 3x.)
        DeclareLaunchArgument("horizontal_score_weight", default_value="0.0"),
        DeclareLaunchArgument("require_horizontal_approach", default_value="true"),
        DeclareLaunchArgument("max_approach_tilt_deg", default_value="20.0"),
        DeclareLaunchArgument("max_downward_approach_tilt_deg", default_value="5.0"),
        DeclareLaunchArgument("max_gripper_roll_deg", default_value="0.0"),
        DeclareLaunchArgument("center_band_m", default_value="0.0"),
        DeclareLaunchArgument("min_points_for_graspgen", default_value="100"),
        DeclareLaunchArgument("log_pregrasp_filter_counts", default_value="true"),
        DeclareLaunchArgument("require_selection", default_value="true"),
        DeclareLaunchArgument("selection_prompt_once", default_value="true"),
        DeclareLaunchArgument("selection_attempt_timeout_s", default_value="5.0"),
        DeclareLaunchArgument("selection_manual_wait_s", default_value="-1.0"),
        DeclareLaunchArgument("selection_retry_sleep_s", default_value="0.5"),
        DeclareLaunchArgument("repeat", default_value="true"),
        DeclareLaunchArgument("wait_for_executor_cycle_done", default_value="true"),
        DeclareLaunchArgument("executor_cycle_done_topic", default_value="/grasp_executor_lin/cycle_done"),
        DeclareLaunchArgument("executor_cycle_done_timeout_s", default_value="-1.0"),
        DeclareLaunchArgument("selection_cycle_id_topic", default_value="/graspgen/selection_cycle_id"),
        # Segmentation (GrabCut by default). Use segmentation_backend:=yolo_seg and
        # sam_model_path:=/path/to/your_yolo_seg.pt for a trained YOLO segmentation model.
        DeclareLaunchArgument("segmentation_enabled", default_value="true"),
        DeclareLaunchArgument("segmentation_backend", default_value="yolo_seg"),
        DeclareLaunchArgument("sam_model_path", default_value=default_seg_model),
        DeclareLaunchArgument("segmentation_min_area_px", default_value="500"),
        DeclareLaunchArgument("grabcut_iters", default_value="5"),
        DeclareLaunchArgument("grabcut_margin_px", default_value="5"),
        DeclareLaunchArgument("grabcut_close_kernel", default_value="7"),
        DeclareLaunchArgument("grabcut_open_kernel", default_value="3"),
        DeclareLaunchArgument("auto_select_index", default_value="-1"),
        DeclareLaunchArgument("auto_select_label", default_value=""),
        DeclareLaunchArgument("auto_select_noninteractive", default_value="false"),
        DeclareLaunchArgument("manual_select_topic", default_value="/yolo/select_index"),
        DeclareLaunchArgument("selected_detection_topic", default_value="/yolo/selected_detection"),
        DeclareLaunchArgument("debug_image_topic", default_value="/yolo/debug_image"),
        # Auto-open rqt_image_view on the debug topic (the working replacement
        # for the cv2 window on this machine — see note at the bottom).
        DeclareLaunchArgument("debug_image_view", default_value="false"),
        # Live 3D grasp visualiser: object cloud + GraspGen candidates
        # (green = kept by the filter chain, red = filtered out, blue = best).
        DeclareLaunchArgument("grasp_viz", default_value="false"),
        # Window A of the two-window layout (2026-08-07): the FRAMES window.
        # Multiview: the accumulated coloured views (/graspgen/fused_cloud)
        # growing live during the sweep. Single view: the one captured
        # coloured frame (published to the same topic).
        DeclareLaunchArgument("frames_viz", default_value="false"),
        # Item 4 (2026-08-07): temporal multi-frame single-view capture —
        # median-fuse N consecutive depth frames while the arm is still.
        DeclareLaunchArgument("single_view_multiframe", default_value="true"),
        DeclareLaunchArgument("single_view_frames", default_value="5"),
        # Grasp precompute cache (2026-09-11): GraspGen for every stable
        # detection while nobody has selected. Off by default; the GUI has no
        # argument slot, so MORTY_GRASP_PRECOMPUTE=true in the environment
        # turns it on for a GUI-started pipeline. Also settable at run time:
        #   ros2 param set /yolo_live_select_grasp graspgen_precompute true
        DeclareLaunchArgument("graspgen_precompute",
                              default_value=os.environ.get("MORTY_GRASP_PRECOMPUTE", "false")),
        DeclareLaunchArgument("precompute_stable_s", default_value="0.2"),
        DeclareLaunchArgument("segmentation_mask_topic", default_value="/yolo/selected_mask"),
        DeclareLaunchArgument("segmentation_overlay_topic", default_value="/yolo/selected_mask_overlay"),
        # Default false for the current single-view experiments (2026-08-06).
        # Pass registration_enabled:=true to get the multi-view sweep back
        # (moves the arm through the observation poses and fuses views).
        DeclareLaunchArgument("registration_enabled", default_value="false"),
        DeclareLaunchArgument("registration_num_observations", default_value="3"),
        DeclareLaunchArgument("registration_observation_interval_s", default_value="1.0"),
        DeclareLaunchArgument("registration_refresh_detection", default_value="true"),
        DeclareLaunchArgument("registration_followup_use_tracked_segmentation", default_value="true"),
        DeclareLaunchArgument("registration_min_points", default_value="80"),
        DeclareLaunchArgument("registration_target_frame", default_value="base_link"),
        DeclareLaunchArgument("registration_voxel_size_m", default_value="0.0015"),
        DeclareLaunchArgument("registration_use_icp", default_value="false"),
        DeclareLaunchArgument("registration_followup_use_roi_fallback", default_value="true"),
        DeclareLaunchArgument("registration_roi_margin_m", default_value="0.04"),
        DeclareLaunchArgument("registration_roi_min_half_extent_m", default_value="0.03"),
        DeclareLaunchArgument("registration_roi_fallback_max_point_ratio", default_value="2.0"),
        DeclareLaunchArgument("registration_roi_fallback_max_extent_ratio", default_value="1.8"),
        DeclareLaunchArgument("registration_roi_fallback_reference_voxel_m", default_value="0.003"),
        DeclareLaunchArgument("registration_roi_fallback_max_surface_distance_m", default_value="0.035"),
        DeclareLaunchArgument("registration_nbv_enabled", default_value="true"),
        DeclareLaunchArgument("registration_nbv_gain_voxel_size_m", default_value="0.003"),
        DeclareLaunchArgument("registration_nbv_min_gain_ratio", default_value="0.08"),
        DeclareLaunchArgument("registration_nbv_min_new_voxels", default_value="150"),
        DeclareLaunchArgument("registration_dynamic_nbv_enabled", default_value="true"),
        DeclareLaunchArgument("registration_dynamic_nbv_use_joint_interpolation", default_value="true"),
        DeclareLaunchArgument("registration_dynamic_nbv_joint_alpha", default_value="1.0"),
        DeclareLaunchArgument("registration_dynamic_nbv_joint_alpha_values", default_value="0.6,0.8,1.0"),
        DeclareLaunchArgument("registration_dynamic_nbv_probe_alpha", default_value="1.0"),
        DeclareLaunchArgument("registration_dynamic_nbv_refine_top_k", default_value="2"),
        DeclareLaunchArgument("registration_dynamic_nbv_drop_nonwinning_coarse", default_value="true"),
        DeclareLaunchArgument("registration_continuous_observation_enabled", default_value="true"),
        DeclareLaunchArgument("registration_continuous_observation_velocity_scaling", default_value="0.009"),
        DeclareLaunchArgument("registration_continuous_observation_min_camera_translation_m", default_value="0.004"),
        DeclareLaunchArgument("registration_continuous_observation_min_camera_rotation_deg", default_value="0.8"),
        DeclareLaunchArgument("registration_continuous_observation_poll_period_s", default_value="0.15"),
        DeclareLaunchArgument("registration_continuous_observation_max_frames", default_value="20"),
        # Default false (2026-08-07, Majd): a sweep visits ONE direction —
        # force_direction if given, else the framing-aware selector's choice.
        # Pass :=true to restore the old visit-all-three behaviour.
        DeclareLaunchArgument("registration_continuous_observation_sweep_all_directions", default_value="false"),
        DeclareLaunchArgument("registration_continuous_observation_sweep_all_alpha", default_value="1.0"),
        DeclareLaunchArgument("registration_continuous_observation_sweep_all_return_to_seed", default_value="true"),
        DeclareLaunchArgument("registration_continuous_observation_step_alpha", default_value="0.2"),
        DeclareLaunchArgument("registration_continuous_observation_max_consecutive_rejections", default_value="10"),
        DeclareLaunchArgument("registration_continuous_observation_bbox_border_margin_px", default_value="24"),
        DeclareLaunchArgument("registration_continuous_observation_min_edge_bbox_area_ratio", default_value="0.35"),
        DeclareLaunchArgument("registration_continuous_observation_min_edge_bbox_height_ratio", default_value="0.45"),
        DeclareLaunchArgument("registration_continuous_observation_reject_bottom_edge", default_value="true"),
        DeclareLaunchArgument("registration_continuous_observation_higher_low_penalty", default_value="0.18"),
        DeclareLaunchArgument("registration_continuous_observation_higher_bottom_edge_penalty", default_value="0.12"),
        DeclareLaunchArgument("registration_continuous_observation_force_direction", default_value=""),
        DeclareLaunchArgument("registration_continuous_observation_center_bias_max", default_value="0.04"),
        DeclareLaunchArgument("registration_continuous_observation_depth_slope_weight", default_value="0.08"),
        DeclareLaunchArgument("registration_continuous_observation_depth_slope_deadband", default_value="0.08"),
        DeclareLaunchArgument("registration_dynamic_nbv_srdf_path",
            default_value=os.path.join(
                ws_dir, "src", "ur3e_gripper_3d_moveit_config",
                "config", "ur3e_with_gripper_3d.srdf",
            )),
        DeclareLaunchArgument("registration_dynamic_nbv_planning_group", default_value="ur3e_manipulator"),
        DeclareLaunchArgument("registration_dynamic_nbv_seed_target", default_value="home"),
        DeclareLaunchArgument("registration_dynamic_nbv_lateral_offset_m", default_value="0.12"),
        DeclareLaunchArgument("registration_dynamic_nbv_vertical_offset_m", default_value="0.10"),
        DeclareLaunchArgument("registration_dynamic_nbv_upward_bias_m", default_value="0.01"),
        DeclareLaunchArgument("registration_dynamic_nbv_min_translation_m", default_value="0.03"),
        DeclareLaunchArgument("registration_dynamic_nbv_min_rotation_deg", default_value="5.0"),
        DeclareLaunchArgument("registration_dynamic_nbv_min_ee_z_m", default_value="0.38"),
        DeclareLaunchArgument("registration_dynamic_nbv_max_ee_translation_m", default_value="0.08"),
        DeclareLaunchArgument("registration_dynamic_nbv_max_ee_downward_delta_m", default_value="0.025"),
        DeclareLaunchArgument("registration_dynamic_nbv_min_relative_ee_z_delta_m", default_value="-0.025"),
        DeclareLaunchArgument("registration_dynamic_nbv_max_camera_distance_delta_m", default_value="0.03"),
        DeclareLaunchArgument("registration_tracked_followup_max_point_ratio", default_value="1.45"),
        DeclareLaunchArgument("registration_tracked_followup_max_extent_ratio", default_value="1.35"),
        DeclareLaunchArgument("registration_tracked_followup_max_centroid_offset_m", default_value="0.03"),
        DeclareLaunchArgument("registration_tracked_followup_max_bbox_area_ratio", default_value="1.8"),
        DeclareLaunchArgument("registration_observation_ee_link", default_value="gripper_tcp"),
        DeclareLaunchArgument("registration_move_to_initial_target", default_value="true"),
        DeclareLaunchArgument("registration_preposition_before_selection", default_value="true"),
        DeclareLaunchArgument("registration_initial_named_target", default_value="home"),
        DeclareLaunchArgument("registration_observation_named_targets", default_value="detect_left,detect_right,detect_higher"),
        DeclareLaunchArgument("registration_motion_settle_s", default_value="0.8"),
        DeclareLaunchArgument("registration_motion_timeout_s", default_value="120.0"),
        DeclareLaunchArgument("registration_motion_velocity_scaling", default_value="0.1"),
        DeclareLaunchArgument("registration_motion_acceleration_scaling", default_value="0.1"),
        DeclareLaunchArgument("registration_observation_planning_time_s", default_value="1.0"),
        DeclareLaunchArgument("registration_observation_num_planning_attempts", default_value="1"),
        DeclareLaunchArgument("registration_observation_allow_fallbacks", default_value="false"),
        DeclareLaunchArgument(
            "registration_observation_commander_executable",
            default_value=os.path.join(
                ws_dir, "install", "lib", "ur3e_gripper_3d_moveit_commander",
                "observation_ptp_commander",
            ),
        ),
        # ur3e_fast_observation bridge: cache YAML + replay node integration.
        DeclareLaunchArgument("registration_use_fast_observation_moves", default_value="true"),
        # When voice_robot.launch.py is also running it ALREADY spawns
        # ur3e_fast_observation_replay. Two instances with the same node
        # name fight over the controller and one cancels the other. Set
        # `with_replay:=false` here when launching the two together. When
        # running grasp_pipeline standalone (no voice), leave it true so
        # this launch brings up its own.
        DeclareLaunchArgument("with_replay", default_value="true"),
        DeclareLaunchArgument(
            "registration_fast_observation_poses_file",
            # $WS_DIR-relative so it is not tied to a specific machine/username.
            default_value=os.path.join(ws_dir, "observation_poses.yaml"),
        ),
        DeclareLaunchArgument(
            "registration_fast_observation_replay_topic",
            default_value="/ur3e_fast_observation_replay/replay",
        ),
        DeclareLaunchArgument(
            "registration_fast_observation_status_topic",
            default_value="/ur3e_fast_observation_replay/status",
        ),
        DeclareLaunchArgument("registration_fast_observation_timeout_s", default_value="8.0"),
        DeclareLaunchArgument("registration_fast_observation_move_duration_s", default_value="2.0"),
        DeclareLaunchArgument(
            "registration_fast_observation_controller_action_ns",
            default_value="/scaled_joint_trajectory_controller",
        ),
        DeclareLaunchArgument("registration_icp_point_to_plane", default_value="true"),
        DeclareLaunchArgument("registration_icp_max_correspondence_m", default_value="0.008"),
        DeclareLaunchArgument("registration_icp_max_iterations", default_value="40"),
        DeclareLaunchArgument("registration_icp_min_fitness", default_value="0.35"),
        DeclareLaunchArgument("registration_icp_max_rmse_m", default_value="0.006"),
        DeclareLaunchArgument("registration_icp_max_translation_correction_m", default_value="0.01"),
        DeclareLaunchArgument("registration_icp_max_rotation_correction_deg", default_value="8.0"),
        DeclareLaunchArgument("registration_icp_local_map_radius_m", default_value="0.12"),
        DeclareLaunchArgument("registration_icp_normal_radius_m", default_value="0.02"),
        DeclareLaunchArgument("registration_fused_cloud_topic", default_value="/graspgen/fused_cloud"),
        DeclareLaunchArgument("registration_tsdf_enabled", default_value="true"),
        DeclareLaunchArgument("registration_tsdf_cloud_topic", default_value="/graspgen/tsdf_cloud"),
        # Finer TSDF (2026-08-07, Majd): 1.0 -> 0.7 mm voxels (~3x volume
        # density) for a smoother surface; truncation kept at ~4x voxel.
        DeclareLaunchArgument("registration_tsdf_voxel_length_m", default_value="0.0007"),
        DeclareLaunchArgument("registration_tsdf_sdf_trunc_m", default_value="0.003"),
        DeclareLaunchArgument("registration_tsdf_depth_trunc_m", default_value="1.2"),
        DeclareLaunchArgument("registration_tsdf_point_voxel_size_m", default_value="0.0006"),
        DeclareLaunchArgument("registration_tsdf_mask_erode_px", default_value="5"),
        DeclareLaunchArgument("registration_tsdf_depth_band_m", default_value="0.04"),
        DeclareLaunchArgument("registration_tsdf_endpoint_only", default_value="true"),
        DeclareLaunchArgument("registration_grasp_utility_logging_enabled", default_value="true"),
        DeclareLaunchArgument("registration_grasp_utility_weight_count", default_value="1.0"),
        DeclareLaunchArgument("registration_grasp_utility_weight_diversity", default_value="1.0"),
        DeclareLaunchArgument("registration_grasp_utility_weight_score", default_value="1.0"),
        DeclareLaunchArgument("registration_grasp_utility_weight_instability", default_value="1.0"),
        DeclareLaunchArgument("registration_grasp_utility_stop_enabled", default_value="true"),
        DeclareLaunchArgument("registration_grasp_utility_stop_min_views_after_seed", default_value="2"),
        DeclareLaunchArgument("registration_grasp_utility_stop_min_delta_u", default_value="0.05"),
        DeclareLaunchArgument("registration_grasp_utility_stop_patience", default_value="2"),
        DeclareLaunchArgument("registration_grasp_utility_stop_large_drop_delta_u", default_value="-0.3"),
        DeclareLaunchArgument("registration_grasp_utility_stop_during_motion", default_value="false"),
        # Default false (2026-08-06): only useful with registration_enabled
        # (TSDF multi-view); with single view it waits forever for a cloud
        # that never comes while burning a full CPU core (#25).
        DeclareLaunchArgument("enable_registration_viewer", default_value="false"),
        DeclareLaunchArgument("registration_viewer_topic", default_value="/graspgen/tsdf_cloud"),
        DeclareLaunchArgument("registration_viewer_point_size", default_value="3.0"),
        DeclareLaunchArgument("registration_viewer_stride", default_value="1"),
        DeclareLaunchArgument("registration_viewer_voxel_size_m", default_value="0.0"),
        DeclareLaunchArgument("registration_viewer_max_points", default_value="200000"),
        DeclareLaunchArgument("registration_viewer_update_hz", default_value="20.0"),
        DeclareLaunchArgument("registration_viewer_use_rgb", default_value="true"),
        DeclareLaunchArgument("registration_viewer_show_axes", default_value="false"),
        DeclareLaunchArgument("registration_viewer_width", default_value="800"),
        DeclareLaunchArgument("registration_viewer_height", default_value="520"),
        DeclareLaunchArgument("registration_viewer_status_log_period_s", default_value="2.0"),
        DeclareLaunchArgument("registration_viewer_status_text_flip", default_value="true"),
        # MoveIt static table obstacle (one box, published once, never updated).
        # Observation viewpoints are hand-taught and the grasp executor clears
        # world obstacles before each LIN, so the only collision check that
        # matters at runtime is "don't drive below the table".
        DeclareLaunchArgument("enable_scene_obstacles", default_value="false"),
        DeclareLaunchArgument("selected_cloud_topic", default_value="/yolo/selected_cloud"),
        DeclareLaunchArgument("planning_scene_topic", default_value="/planning_scene"),
        DeclareLaunchArgument("table_frame", default_value="base_link"),
        DeclareLaunchArgument("table_center_x", default_value="0.0"),
        DeclareLaunchArgument("table_center_y", default_value="-0.30"),
        DeclareLaunchArgument("table_center_z", default_value="0.48"),
        DeclareLaunchArgument("table_size_x", default_value="1.20"),
        DeclareLaunchArgument("table_size_y", default_value="1.00"),
        DeclareLaunchArgument("table_size_z", default_value="0.10"),
    ]

    node = Node(
        package="grasp_pipeline",
        executable="run_grasp_pipeline_venv",
        name="yolo_live_select_grasp",
        output="screen",
        parameters=[
            {"yolo_world_model": LaunchConfiguration("yolo_world_model")},
            {"classes": LaunchConfiguration("classes")},
            {"conf": ParameterValue(LaunchConfiguration("conf"), value_type=float)},
            {"per_class_min_conf_labels": LaunchConfiguration("per_class_min_conf_labels")},
            {"per_class_min_conf_values": LaunchConfiguration("per_class_min_conf_values")},
            {"imgsz": ParameterValue(LaunchConfiguration("imgsz"), value_type=int)},
            {"device": LaunchConfiguration("device")},
            {"yolo_half": ParameterValue(LaunchConfiguration("yolo_half"), value_type=bool)},
            {"camera_topic": LaunchConfiguration("camera_topic")},
            {"depth_topic": LaunchConfiguration("depth_topic")},
            {"color_info_topic": LaunchConfiguration("color_info_topic")},
            {"depth_info_topic": LaunchConfiguration("depth_info_topic")},
            {"selection_prompt_topic": LaunchConfiguration("selection_prompt_topic")},
            {"graspgen_endpoint": LaunchConfiguration("graspgen_endpoint")},
            {"max_graspgen_points": ParameterValue(LaunchConfiguration("max_graspgen_points"), value_type=int)},
            {"publish_debug_image": ParameterValue(LaunchConfiguration("publish_debug_image"), value_type=bool)},
            {"show_debug_window": ParameterValue(LaunchConfiguration("show_debug_window"), value_type=bool)},
            {"debug_window_name": LaunchConfiguration("debug_window_name")},
            {"debug_window_width": ParameterValue(LaunchConfiguration("debug_window_width"), value_type=int)},
            {"debug_window_height": ParameterValue(LaunchConfiguration("debug_window_height"), value_type=int)},
            {"enable_debug_window_click_select": ParameterValue(LaunchConfiguration("enable_debug_window_click_select"), value_type=bool)},
            {"debug_publish_fps": ParameterValue(LaunchConfiguration("debug_publish_fps"), value_type=float)},
            {"republish_selected_outputs": ParameterValue(LaunchConfiguration("republish_selected_outputs"), value_type=bool)},
            {"republish_selected_rate_hz": ParameterValue(LaunchConfiguration("republish_selected_rate_hz"), value_type=float)},
            {"infer_timer_period_s": ParameterValue(LaunchConfiguration("infer_timer_period_s"), value_type=float)},
            {"min_infer_interval_s": ParameterValue(LaunchConfiguration("min_infer_interval_s"), value_type=float)},
            {"perf_log": ParameterValue(LaunchConfiguration("perf_log"), value_type=bool)},
            {"perf_log_period_s": ParameterValue(LaunchConfiguration("perf_log_period_s"), value_type=float)},
            {"use_mediapipe_selection": ParameterValue(LaunchConfiguration("use_mediapipe_selection"), value_type=bool)},
            {"mediapipe_min_hold_ms": ParameterValue(LaunchConfiguration("mediapipe_min_hold_ms"), value_type=int)},
            {"mediapipe_steady_px": ParameterValue(LaunchConfiguration("mediapipe_steady_px"), value_type=float)},
            {"mediapipe_require_inside": ParameterValue(LaunchConfiguration("mediapipe_require_inside"), value_type=bool)},
            {"mediapipe_min_detection_confidence": ParameterValue(LaunchConfiguration("mediapipe_min_detection_confidence"), value_type=float)},
            {"mediapipe_min_tracking_confidence": ParameterValue(LaunchConfiguration("mediapipe_min_tracking_confidence"), value_type=float)},
            {"pointing_mode": ParameterValue(LaunchConfiguration("pointing_mode"), value_type=str)},
            {"mediapipe_cone_deg": ParameterValue(LaunchConfiguration("mediapipe_cone_deg"), value_type=float)},
            {"mediapipe_hysteresis": ParameterValue(LaunchConfiguration("mediapipe_hysteresis"), value_type=float)},
            {"mediapipe_use_tasks_api": ParameterValue(LaunchConfiguration("mediapipe_use_tasks_api"), value_type=bool)},
            {"suspend_inference_while_selecting": ParameterValue(LaunchConfiguration("suspend_inference_while_selecting"), value_type=bool)},
            {"apply_depth_clamp": ParameterValue(LaunchConfiguration("apply_depth_clamp"), value_type=bool)},
            {"apply_largest_cluster_filter": ParameterValue(LaunchConfiguration("apply_largest_cluster_filter"), value_type=bool)},
            {"cluster_eps_m": ParameterValue(LaunchConfiguration("cluster_eps_m"), value_type=float)},
            {"cluster_min_samples": ParameterValue(LaunchConfiguration("cluster_min_samples"), value_type=int)},
            {"apply_radius_outlier_filter": ParameterValue(LaunchConfiguration("apply_radius_outlier_filter"), value_type=bool)},
            {"radius_filter_m": ParameterValue(LaunchConfiguration("radius_filter_m"), value_type=float)},
            {"radius_min_neighbors": ParameterValue(LaunchConfiguration("radius_min_neighbors"), value_type=int)},
            {"apply_cloud_cleaning": ParameterValue(LaunchConfiguration("apply_cloud_cleaning"), value_type=bool)},
            {"apply_plane_above_filter": ParameterValue(LaunchConfiguration("apply_plane_above_filter"), value_type=bool)},
            {"plane_clearance_m": ParameterValue(LaunchConfiguration("plane_clearance_m"), value_type=float)},
            # 2026-08-24 thesis teardown: the retired orientation knobs
            # (horizontal_score_weight, require_horizontal_approach,
            # max_approach_tilt_deg, max_downward_approach_tilt_deg,
            # center_band_m) are no longer forwarded — params.py rules.
            {"min_points_for_graspgen": ParameterValue(LaunchConfiguration("min_points_for_graspgen"), value_type=int)},
            {"log_pregrasp_filter_counts": ParameterValue(LaunchConfiguration("log_pregrasp_filter_counts"), value_type=bool)},
            {"require_selection": ParameterValue(LaunchConfiguration("require_selection"), value_type=bool)},
            {"selection_prompt_once": ParameterValue(LaunchConfiguration("selection_prompt_once"), value_type=bool)},
            {"selection_attempt_timeout_s": ParameterValue(LaunchConfiguration("selection_attempt_timeout_s"), value_type=float)},
            {"selection_manual_wait_s": ParameterValue(LaunchConfiguration("selection_manual_wait_s"), value_type=float)},
            {"selection_retry_sleep_s": ParameterValue(LaunchConfiguration("selection_retry_sleep_s"), value_type=float)},
            {"repeat": ParameterValue(LaunchConfiguration("repeat"), value_type=bool)},
            {"wait_for_executor_cycle_done": ParameterValue(LaunchConfiguration("wait_for_executor_cycle_done"), value_type=bool)},
            {"executor_cycle_done_topic": LaunchConfiguration("executor_cycle_done_topic")},
            {"executor_cycle_done_timeout_s": ParameterValue(LaunchConfiguration("executor_cycle_done_timeout_s"), value_type=float)},
            {"selection_cycle_id_topic": LaunchConfiguration("selection_cycle_id_topic")},
            {"segmentation_enabled": ParameterValue(LaunchConfiguration("segmentation_enabled"), value_type=bool)},
            {"segmentation_backend": LaunchConfiguration("segmentation_backend")},
            {"sam_model_path": LaunchConfiguration("sam_model_path")},
            {"segmentation_min_area_px": ParameterValue(LaunchConfiguration("segmentation_min_area_px"), value_type=int)},
            {"grabcut_iters": ParameterValue(LaunchConfiguration("grabcut_iters"), value_type=int)},
            {"grabcut_margin_px": ParameterValue(LaunchConfiguration("grabcut_margin_px"), value_type=int)},
            {"grabcut_close_kernel": ParameterValue(LaunchConfiguration("grabcut_close_kernel"), value_type=int)},
            {"grabcut_open_kernel": ParameterValue(LaunchConfiguration("grabcut_open_kernel"), value_type=int)},
            {"auto_select_index": ParameterValue(LaunchConfiguration("auto_select_index"), value_type=int)},
            {"auto_select_label": LaunchConfiguration("auto_select_label")},
            {"auto_select_noninteractive": ParameterValue(LaunchConfiguration("auto_select_noninteractive"), value_type=bool)},
            {"manual_select_topic": LaunchConfiguration("manual_select_topic")},
            {"selected_detection_topic": LaunchConfiguration("selected_detection_topic")},
            {"selected_cloud_topic": LaunchConfiguration("selected_cloud_topic")},
            {"debug_image_topic": LaunchConfiguration("debug_image_topic")},
            {"segmentation_mask_topic": LaunchConfiguration("segmentation_mask_topic")},
            {"segmentation_overlay_topic": LaunchConfiguration("segmentation_overlay_topic")},
            {"registration_enabled": ParameterValue(LaunchConfiguration("registration_enabled"), value_type=bool)},
            {"single_view_multiframe": ParameterValue(LaunchConfiguration("single_view_multiframe"), value_type=bool)},
            {"single_view_frames": ParameterValue(LaunchConfiguration("single_view_frames"), value_type=int)},
            {"graspgen_precompute": ParameterValue(LaunchConfiguration("graspgen_precompute"), value_type=bool)},
            {"precompute_stable_s": ParameterValue(LaunchConfiguration("precompute_stable_s"), value_type=float)},
            {"registration_num_observations": ParameterValue(LaunchConfiguration("registration_num_observations"), value_type=int)},
            {"registration_observation_interval_s": ParameterValue(LaunchConfiguration("registration_observation_interval_s"), value_type=float)},
            {"registration_refresh_detection": ParameterValue(LaunchConfiguration("registration_refresh_detection"), value_type=bool)},
            {"registration_followup_use_tracked_segmentation": ParameterValue(LaunchConfiguration("registration_followup_use_tracked_segmentation"), value_type=bool)},
            {"registration_tracked_followup_max_point_ratio": ParameterValue(LaunchConfiguration("registration_tracked_followup_max_point_ratio"), value_type=float)},
            {"registration_tracked_followup_max_extent_ratio": ParameterValue(LaunchConfiguration("registration_tracked_followup_max_extent_ratio"), value_type=float)},
            {"registration_tracked_followup_max_centroid_offset_m": ParameterValue(LaunchConfiguration("registration_tracked_followup_max_centroid_offset_m"), value_type=float)},
            {"registration_tracked_followup_max_bbox_area_ratio": ParameterValue(LaunchConfiguration("registration_tracked_followup_max_bbox_area_ratio"), value_type=float)},
            {"registration_min_points": ParameterValue(LaunchConfiguration("registration_min_points"), value_type=int)},
            {"registration_target_frame": LaunchConfiguration("registration_target_frame")},
            {"registration_voxel_size_m": ParameterValue(LaunchConfiguration("registration_voxel_size_m"), value_type=float)},
            {"registration_use_icp": ParameterValue(LaunchConfiguration("registration_use_icp"), value_type=bool)},
            {"registration_followup_use_roi_fallback": ParameterValue(LaunchConfiguration("registration_followup_use_roi_fallback"), value_type=bool)},
            {"registration_roi_margin_m": ParameterValue(LaunchConfiguration("registration_roi_margin_m"), value_type=float)},
            {"registration_roi_min_half_extent_m": ParameterValue(LaunchConfiguration("registration_roi_min_half_extent_m"), value_type=float)},
            {"registration_roi_fallback_max_point_ratio": ParameterValue(LaunchConfiguration("registration_roi_fallback_max_point_ratio"), value_type=float)},
            {"registration_roi_fallback_max_extent_ratio": ParameterValue(LaunchConfiguration("registration_roi_fallback_max_extent_ratio"), value_type=float)},
            {"registration_roi_fallback_reference_voxel_m": ParameterValue(LaunchConfiguration("registration_roi_fallback_reference_voxel_m"), value_type=float)},
            {"registration_roi_fallback_max_surface_distance_m": ParameterValue(LaunchConfiguration("registration_roi_fallback_max_surface_distance_m"), value_type=float)},
            {"registration_nbv_enabled": ParameterValue(LaunchConfiguration("registration_nbv_enabled"), value_type=bool)},
            {"registration_nbv_gain_voxel_size_m": ParameterValue(LaunchConfiguration("registration_nbv_gain_voxel_size_m"), value_type=float)},
            {"registration_nbv_min_gain_ratio": ParameterValue(LaunchConfiguration("registration_nbv_min_gain_ratio"), value_type=float)},
            {"registration_nbv_min_new_voxels": ParameterValue(LaunchConfiguration("registration_nbv_min_new_voxels"), value_type=int)},
            {"registration_dynamic_nbv_enabled": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_enabled"), value_type=bool)},
            {"registration_dynamic_nbv_use_joint_interpolation": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_use_joint_interpolation"), value_type=bool)},
            {"registration_dynamic_nbv_joint_alpha": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_joint_alpha"), value_type=float)},
            {"registration_dynamic_nbv_joint_alpha_values": LaunchConfiguration("registration_dynamic_nbv_joint_alpha_values")},
            {"registration_dynamic_nbv_probe_alpha": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_probe_alpha"), value_type=float)},
            {"registration_dynamic_nbv_refine_top_k": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_refine_top_k"), value_type=int)},
            {"registration_dynamic_nbv_drop_nonwinning_coarse": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_drop_nonwinning_coarse"), value_type=bool)},
            {"registration_continuous_observation_enabled": ParameterValue(LaunchConfiguration("registration_continuous_observation_enabled"), value_type=bool)},
            {"registration_continuous_observation_velocity_scaling": ParameterValue(LaunchConfiguration("registration_continuous_observation_velocity_scaling"), value_type=float)},
            {"registration_continuous_observation_min_camera_translation_m": ParameterValue(LaunchConfiguration("registration_continuous_observation_min_camera_translation_m"), value_type=float)},
            {"registration_continuous_observation_min_camera_rotation_deg": ParameterValue(LaunchConfiguration("registration_continuous_observation_min_camera_rotation_deg"), value_type=float)},
            {"registration_continuous_observation_poll_period_s": ParameterValue(LaunchConfiguration("registration_continuous_observation_poll_period_s"), value_type=float)},
            {"registration_continuous_observation_max_frames": ParameterValue(LaunchConfiguration("registration_continuous_observation_max_frames"), value_type=int)},
            {"registration_continuous_observation_sweep_all_directions": ParameterValue(LaunchConfiguration("registration_continuous_observation_sweep_all_directions"), value_type=bool)},
            {"registration_continuous_observation_sweep_all_alpha": ParameterValue(LaunchConfiguration("registration_continuous_observation_sweep_all_alpha"), value_type=float)},
            {"registration_continuous_observation_sweep_all_return_to_seed": ParameterValue(LaunchConfiguration("registration_continuous_observation_sweep_all_return_to_seed"), value_type=bool)},
            {"registration_continuous_observation_step_alpha": ParameterValue(LaunchConfiguration("registration_continuous_observation_step_alpha"), value_type=float)},
            {"registration_continuous_observation_max_consecutive_rejections": ParameterValue(LaunchConfiguration("registration_continuous_observation_max_consecutive_rejections"), value_type=int)},
            {"registration_continuous_observation_bbox_border_margin_px": ParameterValue(LaunchConfiguration("registration_continuous_observation_bbox_border_margin_px"), value_type=int)},
            {"registration_continuous_observation_min_edge_bbox_area_ratio": ParameterValue(LaunchConfiguration("registration_continuous_observation_min_edge_bbox_area_ratio"), value_type=float)},
            {"registration_continuous_observation_min_edge_bbox_height_ratio": ParameterValue(LaunchConfiguration("registration_continuous_observation_min_edge_bbox_height_ratio"), value_type=float)},
            {"registration_continuous_observation_reject_bottom_edge": ParameterValue(LaunchConfiguration("registration_continuous_observation_reject_bottom_edge"), value_type=bool)},
            {"registration_continuous_observation_higher_low_penalty": ParameterValue(LaunchConfiguration("registration_continuous_observation_higher_low_penalty"), value_type=float)},
            {"registration_continuous_observation_higher_bottom_edge_penalty": ParameterValue(LaunchConfiguration("registration_continuous_observation_higher_bottom_edge_penalty"), value_type=float)},
            {"registration_continuous_observation_force_direction": LaunchConfiguration("registration_continuous_observation_force_direction")},
            {"registration_continuous_observation_center_bias_max": ParameterValue(LaunchConfiguration("registration_continuous_observation_center_bias_max"), value_type=float)},
            {"registration_continuous_observation_depth_slope_weight": ParameterValue(LaunchConfiguration("registration_continuous_observation_depth_slope_weight"), value_type=float)},
            {"registration_continuous_observation_depth_slope_deadband": ParameterValue(LaunchConfiguration("registration_continuous_observation_depth_slope_deadband"), value_type=float)},
            {"registration_dynamic_nbv_srdf_path": LaunchConfiguration("registration_dynamic_nbv_srdf_path")},
            {"registration_dynamic_nbv_planning_group": LaunchConfiguration("registration_dynamic_nbv_planning_group")},
            {"registration_dynamic_nbv_seed_target": LaunchConfiguration("registration_dynamic_nbv_seed_target")},
            {"registration_dynamic_nbv_lateral_offset_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_lateral_offset_m"), value_type=float)},
            {"registration_dynamic_nbv_vertical_offset_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_vertical_offset_m"), value_type=float)},
            {"registration_dynamic_nbv_upward_bias_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_upward_bias_m"), value_type=float)},
            {"registration_dynamic_nbv_min_translation_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_min_translation_m"), value_type=float)},
            {"registration_dynamic_nbv_min_rotation_deg": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_min_rotation_deg"), value_type=float)},
            {"registration_dynamic_nbv_min_ee_z_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_min_ee_z_m"), value_type=float)},
            {"registration_dynamic_nbv_max_ee_translation_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_max_ee_translation_m"), value_type=float)},
            {"registration_dynamic_nbv_max_ee_downward_delta_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_max_ee_downward_delta_m"), value_type=float)},
            {"registration_dynamic_nbv_min_relative_ee_z_delta_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_min_relative_ee_z_delta_m"), value_type=float)},
            {"registration_dynamic_nbv_max_camera_distance_delta_m": ParameterValue(LaunchConfiguration("registration_dynamic_nbv_max_camera_distance_delta_m"), value_type=float)},
            {"registration_observation_ee_link": LaunchConfiguration("registration_observation_ee_link")},
            {"registration_move_to_initial_target": ParameterValue(LaunchConfiguration("registration_move_to_initial_target"), value_type=bool)},
            {"registration_preposition_before_selection": ParameterValue(LaunchConfiguration("registration_preposition_before_selection"), value_type=bool)},
            {"registration_initial_named_target": LaunchConfiguration("registration_initial_named_target")},
            {"registration_observation_named_targets": LaunchConfiguration("registration_observation_named_targets")},
            {"registration_motion_settle_s": ParameterValue(LaunchConfiguration("registration_motion_settle_s"), value_type=float)},
            {"registration_motion_timeout_s": ParameterValue(LaunchConfiguration("registration_motion_timeout_s"), value_type=float)},
            {"registration_motion_velocity_scaling": ParameterValue(LaunchConfiguration("registration_motion_velocity_scaling"), value_type=float)},
            {"registration_motion_acceleration_scaling": ParameterValue(LaunchConfiguration("registration_motion_acceleration_scaling"), value_type=float)},
            {"registration_observation_planning_time_s": ParameterValue(LaunchConfiguration("registration_observation_planning_time_s"), value_type=float)},
            {"registration_observation_num_planning_attempts": ParameterValue(LaunchConfiguration("registration_observation_num_planning_attempts"), value_type=int)},
            {"registration_observation_allow_fallbacks": ParameterValue(LaunchConfiguration("registration_observation_allow_fallbacks"), value_type=bool)},
            {"registration_observation_commander_executable": LaunchConfiguration("registration_observation_commander_executable")},
            {"registration_use_fast_observation_moves": ParameterValue(LaunchConfiguration("registration_use_fast_observation_moves"), value_type=bool)},
            {"registration_fast_observation_poses_file": LaunchConfiguration("registration_fast_observation_poses_file")},
            {"registration_fast_observation_replay_topic": LaunchConfiguration("registration_fast_observation_replay_topic")},
            {"registration_fast_observation_status_topic": LaunchConfiguration("registration_fast_observation_status_topic")},
            {"registration_fast_observation_timeout_s": ParameterValue(LaunchConfiguration("registration_fast_observation_timeout_s"), value_type=float)},
            {"registration_icp_point_to_plane": ParameterValue(LaunchConfiguration("registration_icp_point_to_plane"), value_type=bool)},
            {"registration_icp_max_correspondence_m": ParameterValue(LaunchConfiguration("registration_icp_max_correspondence_m"), value_type=float)},
            {"registration_icp_max_iterations": ParameterValue(LaunchConfiguration("registration_icp_max_iterations"), value_type=int)},
            {"registration_icp_min_fitness": ParameterValue(LaunchConfiguration("registration_icp_min_fitness"), value_type=float)},
            {"registration_icp_max_rmse_m": ParameterValue(LaunchConfiguration("registration_icp_max_rmse_m"), value_type=float)},
            {"registration_icp_max_translation_correction_m": ParameterValue(LaunchConfiguration("registration_icp_max_translation_correction_m"), value_type=float)},
            {"registration_icp_max_rotation_correction_deg": ParameterValue(LaunchConfiguration("registration_icp_max_rotation_correction_deg"), value_type=float)},
            {"registration_icp_local_map_radius_m": ParameterValue(LaunchConfiguration("registration_icp_local_map_radius_m"), value_type=float)},
            {"registration_icp_normal_radius_m": ParameterValue(LaunchConfiguration("registration_icp_normal_radius_m"), value_type=float)},
            {"registration_fused_cloud_topic": LaunchConfiguration("registration_fused_cloud_topic")},
            {"registration_tsdf_enabled": ParameterValue(LaunchConfiguration("registration_tsdf_enabled"), value_type=bool)},
            {"registration_tsdf_cloud_topic": LaunchConfiguration("registration_tsdf_cloud_topic")},
            {"registration_tsdf_voxel_length_m": ParameterValue(LaunchConfiguration("registration_tsdf_voxel_length_m"), value_type=float)},
            {"registration_tsdf_sdf_trunc_m": ParameterValue(LaunchConfiguration("registration_tsdf_sdf_trunc_m"), value_type=float)},
            {"registration_tsdf_depth_trunc_m": ParameterValue(LaunchConfiguration("registration_tsdf_depth_trunc_m"), value_type=float)},
            {"registration_tsdf_point_voxel_size_m": ParameterValue(LaunchConfiguration("registration_tsdf_point_voxel_size_m"), value_type=float)},
            {"registration_tsdf_mask_erode_px": ParameterValue(LaunchConfiguration("registration_tsdf_mask_erode_px"), value_type=int)},
            {"registration_tsdf_depth_band_m": ParameterValue(LaunchConfiguration("registration_tsdf_depth_band_m"), value_type=float)},
            {"registration_tsdf_endpoint_only": ParameterValue(LaunchConfiguration("registration_tsdf_endpoint_only"), value_type=bool)},
            {"registration_grasp_utility_logging_enabled": ParameterValue(LaunchConfiguration("registration_grasp_utility_logging_enabled"), value_type=bool)},
            {"registration_grasp_utility_weight_count": ParameterValue(LaunchConfiguration("registration_grasp_utility_weight_count"), value_type=float)},
            {"registration_grasp_utility_weight_diversity": ParameterValue(LaunchConfiguration("registration_grasp_utility_weight_diversity"), value_type=float)},
            {"registration_grasp_utility_weight_score": ParameterValue(LaunchConfiguration("registration_grasp_utility_weight_score"), value_type=float)},
            {"registration_grasp_utility_weight_instability": ParameterValue(LaunchConfiguration("registration_grasp_utility_weight_instability"), value_type=float)},
            {"registration_grasp_utility_stop_enabled": ParameterValue(LaunchConfiguration("registration_grasp_utility_stop_enabled"), value_type=bool)},
            {"registration_grasp_utility_stop_min_views_after_seed": ParameterValue(LaunchConfiguration("registration_grasp_utility_stop_min_views_after_seed"), value_type=int)},
            {"registration_grasp_utility_stop_min_delta_u": ParameterValue(LaunchConfiguration("registration_grasp_utility_stop_min_delta_u"), value_type=float)},
            {"registration_grasp_utility_stop_patience": ParameterValue(LaunchConfiguration("registration_grasp_utility_stop_patience"), value_type=int)},
            {"registration_grasp_utility_stop_large_drop_delta_u": ParameterValue(LaunchConfiguration("registration_grasp_utility_stop_large_drop_delta_u"), value_type=float)},
            {"registration_grasp_utility_stop_during_motion": ParameterValue(LaunchConfiguration("registration_grasp_utility_stop_during_motion"), value_type=bool)},
        ],
    )

    scene_obstacles_node = Node(
        package="grasp_pipeline",
        executable="scene_obstacle_publisher",
        name="scene_obstacle_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_scene_obstacles")),
        parameters=[
            {"planning_scene_topic": LaunchConfiguration("planning_scene_topic")},
            {"table_frame": LaunchConfiguration("table_frame")},
            {"table_center_x": ParameterValue(LaunchConfiguration("table_center_x"), value_type=float)},
            {"table_center_y": ParameterValue(LaunchConfiguration("table_center_y"), value_type=float)},
            {"table_center_z": ParameterValue(LaunchConfiguration("table_center_z"), value_type=float)},
            {"table_size_x": ParameterValue(LaunchConfiguration("table_size_x"), value_type=float)},
            {"table_size_y": ParameterValue(LaunchConfiguration("table_size_y"), value_type=float)},
            {"table_size_z": ParameterValue(LaunchConfiguration("table_size_z"), value_type=float)},
        ],
    )

    registration_viewer_node = Node(
        package="grasp_pipeline",
        executable="open3d_cloud_viewer",
        name="open3d_cloud_viewer",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_registration_viewer")),
        additional_env={
            "WAYLAND_DISPLAY": "",
            "XDG_SESSION_TYPE": "x11",
        },
        parameters=[{
            "topic": LaunchConfiguration("registration_viewer_topic"),
            "point_size": ParameterValue(LaunchConfiguration("registration_viewer_point_size"), value_type=float),
            "stride": ParameterValue(LaunchConfiguration("registration_viewer_stride"), value_type=int),
            "voxel_size_m": ParameterValue(LaunchConfiguration("registration_viewer_voxel_size_m"), value_type=float),
            "max_points": ParameterValue(LaunchConfiguration("registration_viewer_max_points"), value_type=int),
            "update_hz": ParameterValue(LaunchConfiguration("registration_viewer_update_hz"), value_type=float),
            "use_rgb": ParameterValue(LaunchConfiguration("registration_viewer_use_rgb"), value_type=bool),
            "show_axes": ParameterValue(LaunchConfiguration("registration_viewer_show_axes"), value_type=bool),
            "width": ParameterValue(LaunchConfiguration("registration_viewer_width"), value_type=int),
            "height": ParameterValue(LaunchConfiguration("registration_viewer_height"), value_type=int),
            "status_log_period_s": ParameterValue(
                LaunchConfiguration("registration_viewer_status_log_period_s"),
                value_type=float,
            ),
            "status_text_flip": ParameterValue(
                LaunchConfiguration("registration_viewer_status_text_flip"),
                value_type=bool,
            ),
        }],
    )

    fast_observation_replay_node = Node(
        package="ur3e_fast_observation",
        executable="replay_node",
        name="ur3e_fast_observation_replay",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "poses_file": LaunchConfiguration("registration_fast_observation_poses_file"),
            "controller_action_ns": LaunchConfiguration(
                "registration_fast_observation_controller_action_ns"
            ),
            "move_duration_s": ParameterValue(
                LaunchConfiguration("registration_fast_observation_move_duration_s"),
                value_type=float,
            ),
            "default_pose_name": "",
        }],
        condition=IfCondition(LaunchConfiguration("with_replay")),
    )

    # Auto-open an image viewer on the debug topic (2026-08-06). The in-process
    # cv2 window (`show_debug_window`) cannot work on this machine: under
    # Wayland it needs the executor's main thread and stalls the node (see
    # SYSTEM_STATE_LOG #21/#27/#28). rqt_image_view is a separate process with
    # its own GUI loop, so it shows the same annotated image without touching
    # the pipeline. Disable with debug_image_view:=false.
    # click_select_viewer replaces rqt_image_view here (2026-08-06): it is the
    # same separate-process idea, but with click-to-select built in — a left
    # click publishes the pixel on <topic>_mouse_left, which the pipeline node
    # subscribes to. No rqt checkbox to enable, nothing extra to configure.
    debug_image_view_node = Node(
        package="grasp_pipeline",
        executable="click_select_viewer",
        name="click_select_viewer",
        parameters=[{"image_topic": LaunchConfiguration("debug_image_topic")}],
        condition=IfCondition(LaunchConfiguration("debug_image_view")),
    )

    # Majd's final 3-window spec (2026-08-07 13:25) — both 3D windows are the
    # same executable in different modes:
    #
    # "Morty Frames": every captured coloured frame, live, NO grasps.
    frames_viz_node = Node(
        package="grasp_pipeline",
        executable="grasp_viz_viewer",
        name="morty_frames_viewer",
        parameters=[{
            "cloud_topic": "/yolo/selected_cloud_rgb",
            "show_grasps": False,
            "subscribe_tsdf": False,
            "window_title": "Morty Frames - captured frame (colored)",
        }],
        condition=IfCondition(LaunchConfiguration("frames_viz")),
    )

    # "Morty Final Result": the latest 3D construction (TSDF after a
    # multiview sweep, the final frame in single view) + grasps
    # (green kept / red rejected / blue best / ORANGE while executing).
    grasp_viz_node = Node(
        package="grasp_pipeline",
        executable="grasp_viz_viewer",
        name="morty_result_viewer",
        parameters=[{
            # Only the once-per-cycle final cloud — never churns mid-sweep.
            "cloud_topic": "/graspgen/final_cloud",
            "show_grasps": True,
            "subscribe_tsdf": False,
            "window_title": "Morty Final Result | G/R/B toggles | ORANGE = executing",
        }],
        condition=IfCondition(LaunchConfiguration("grasp_viz")),
    )

    return LaunchDescription(
        declare_args
        + [
            node,
            scene_obstacles_node,
            # registration_viewer_node intentionally NOT launched anymore
            # (2026-08-07): it was the redundant 4th window ("Open3D - TSDF
            # reconstructed cloud"); "Morty Final Result" replaced it. The
            # enable_registration_viewer arg is now a no-op kept for
            # compatibility with old command lines.
            fast_observation_replay_node,
            debug_image_view_node,
            frames_viz_node,
            grasp_viz_node,
        ]
    )
