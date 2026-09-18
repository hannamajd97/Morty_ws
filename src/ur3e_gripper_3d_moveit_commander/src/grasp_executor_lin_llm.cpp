#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <atomic>
#include <array>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <deque>
#include <functional>
#include <sstream>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/string.hpp>
#include <rcl_interfaces/msg/log.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <future>
#include <moveit_msgs/action/execute_trajectory.hpp>
#include <rclcpp/parameter_client.hpp>
#include <control_msgs/control_msgs/action/gripper_command.hpp>
#include <control_msgs/control_msgs/action/follow_joint_trajectory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <morty_interfaces/action/move_to_named_pose.hpp>
#include <ur_msgs/msg/io_states.hpp>
#include <ur_msgs/srv/set_speed_slider_fraction.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/utils/moveit_error_code.h>
#include <moveit_msgs/msg/contact_information.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <moveit_msgs/srv/get_motion_sequence.hpp>
#include <moveit_msgs/msg/motion_sequence_request.hpp>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

// Pure comparison predicates for the probe-plan reuse cache. Kept in a
// ROS-free header so the standalone tolerance check compiles the very
// same code that ships (see test/plan_reuse_tolerance_check.cpp).
#include "plan_reuse_tolerance.hpp"

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

template <typename T>
T get_or_declare_param(
  const rclcpp::Node::SharedPtr & node, const std::string & name, const T & default_value)
{
  if (node->has_parameter(name)) {
    return node->get_parameter(name).get_value<T>();
  }
  return node->declare_parameter<T>(name, default_value);
}

void copy_moveit_params_from_move_group(
  const rclcpp::Node::SharedPtr & node,
  const std::string & move_group_node,
  const std::string & planning_group)
{
  const auto kinematics_solver_param =
    "robot_description_kinematics." + planning_group + ".kinematics_solver";

  if (node->has_parameter("robot_description") && node->has_parameter("robot_description_semantic") &&
      node->has_parameter(kinematics_solver_param)) {
    return;
  }

  auto params_client = std::make_shared<rclcpp::AsyncParametersClient>(node, move_group_node);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!params_client->service_is_ready()) {
    if (std::chrono::steady_clock::now() > deadline) {
      break;
    }
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  if (!params_client->service_is_ready()) {
    RCLCPP_WARN(
      node->get_logger(),
      "MoveGroup param service is not available on '%s'. Start MoveIt first.",
      move_group_node.c_str());
    return;
  }

  std::vector<std::string> names;
  try {
    auto list_future = params_client->list_parameters(
      {
        "robot_description",
        "robot_description_semantic",
        "robot_description_kinematics",
        "robot_description_planning",
      },
      10);
    while (rclcpp::spin_some(node),
           list_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    names = list_future.get().names;
  } catch (const std::exception & e) {
    RCLCPP_WARN(node->get_logger(), "Failed to list params from '%s': %s", move_group_node.c_str(), e.what());
    return;
  }

  if (names.empty()) {
    RCLCPP_WARN(node->get_logger(), "No MoveIt params found on '%s'.", move_group_node.c_str());
    return;
  }

  std::vector<rclcpp::Parameter> values;
  try {
    auto future = params_client->get_parameters(names);
    while (rclcpp::spin_some(node), future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    values = future.get();
  } catch (const std::exception & e) {
    RCLCPP_WARN(node->get_logger(), "Failed to read params from '%s': %s", move_group_node.c_str(), e.what());
    return;
  }

  for (const auto & p : values) {
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
      continue;
    }
    if (!node->has_parameter(p.get_name())) {
      node->declare_parameter(p.get_name(), p.get_parameter_value());
    }
  }
}

geometry_msgs::msg::PoseStamped compute_pregrasp_pose(
  const geometry_msgs::msg::PoseStamped & grasp_pose,
  double pregrasp_distance)
{
  // Pre-grasp is offset along the local -Z axis (approach direction)
  tf2::Transform grasp_tf;
  tf2::fromMsg(grasp_pose.pose, grasp_tf);

  tf2::Transform offset;
  offset.setIdentity();
  offset.setOrigin(tf2::Vector3(0.0, 0.0, -pregrasp_distance));

  tf2::Transform pregrasp_tf = grasp_tf * offset;

  geometry_msgs::msg::PoseStamped pregrasp_pose;
  pregrasp_pose.header = grasp_pose.header;
  pregrasp_pose.pose.position.x = pregrasp_tf.getOrigin().x();
  pregrasp_pose.pose.position.y = pregrasp_tf.getOrigin().y();
  pregrasp_pose.pose.position.z = pregrasp_tf.getOrigin().z();
  tf2::Quaternion q = pregrasp_tf.getRotation();
  pregrasp_pose.pose.orientation.x = q.x();
  pregrasp_pose.pose.orientation.y = q.y();
  pregrasp_pose.pose.orientation.z = q.z();
  pregrasp_pose.pose.orientation.w = q.w();
  return pregrasp_pose;
}

class GraspExecutorLin
{
public:
  GraspExecutorLin(
    const rclcpp::Node::SharedPtr & node,
    moveit::planning_interface::MoveGroupInterface & arm_group,
    tf2_ros::Buffer & tf_buffer)
  : node_(node), arm_group_(arm_group), tf_buffer_(tf_buffer)
  {
    execute_ = get_or_declare_param<bool>(node_, "execute", false);
    pregrasp_distance_ = get_or_declare_param<double>(node_, "pregrasp_distance", 0.03);
    // ── Pre-grasp staging (2026-08-21, Majd: "why no valid path when I
    // can SEE a valid path?"). Until now the approach was ONE Pilz LIN
    // straight from wherever the arm was (home) all the way to the grasp
    // — a ~0.5 m straight line ending at the edge of the envelope. Near
    // full stretch that forces the elbow past its speed limit, so most
    // perfectly reachable grasps were rejected as "no valid path".
    // Now: free-form PTP to a stand-off pose behind the grasp, then a
    // SHORT LIN for the final approach only. `compute_pregrasp_pose`
    // already existed for this but was never called (dead code).
    // Default FALSE (Majd, 2026-08-21: "no 2 steps, LIN is enough —
    // focus on speed"). Set true to enable the PTP-standoff + short-LIN
    // staging from #181.
    use_pregrasp_staging_ = get_or_declare_param<bool>(
      node_, "use_pregrasp_staging", false);
    // "ptp" = the 2026-09-11 stand-off (PTP to the pre-grasp, then LIN);
    // "lin" = two straight lines (Majd 2026-09-16: the V of the fingers
    // swept into the object on the direct line): LIN to a point
    // pregrasp_approach_distance behind the grasp along the gripper's z,
    // then LIN the last stretch along z. Both lines are planned in the
    // probe / background check and reused at the click.
    staging_mode_ = get_or_declare_param<std::string>(node_, "staging_mode", "ptp");
    // "blend": the two lines planned as ONE Pilz sequence with a blend
    // radius at the pre-grasp corner, so the arm rounds it instead of
    // stopping (Majd 2026-09-16: "how can we go smoother than the 2 lins").
    // Needs move_group's MoveGroupSequenceService; without it the executor
    // falls back to "lin" (two separate lines).
    blend_radius_m_ = get_or_declare_param<double>(node_, "pregrasp_blend_radius_m", 0.02);
    // "cartesian" (2026-09-16, Majd: "it must feel like one line, like a
    // human hand"): the Cartesian path home -> pre-grasp -> grasp is
    // computed as ONE path and time-parameterised once (time-optimal, within
    // the joint limits), so there is no seam at the corner — the arm slows
    // there only as much as its joints require. Corner rounding in joint
    // space is bounded by cartesian_path_tolerance_rad.
    cartesian_path_tolerance_rad_ = get_or_declare_param<double>(node_, "cartesian_path_tolerance_rad", 0.05);
    // The approach is a CURVE, not two lines (2026-09-16, Majd: "like a
    // human hand"): a quadratic Bezier from the current TCP to the grasp
    // whose control point lies cartesian_curve_lead_m behind the grasp
    // along the gripper's z, so the curve arrives along the fingers' axis
    // with no corner; the last stretch is straight by construction.
    cartesian_curve_lead_m_ = get_or_declare_param<double>(node_, "cartesian_curve_lead_m", 0.18);   // tangent length of the bend
    // Close on the way in (2026-09-16, Majd: "it stops before it grasps"):
    // the close pulse is sent close_lead_s before the motion ends, so the
    // fingers meet the object as the arm stops instead of 1.1 s later. The
    // fingers only get narrower than a 3 cm object ~0.6 s after the pulse
    // begins (+0.25 s adapter delay), so 0.4 s is safe. 0 = close after
    // arrival, as before.
    close_lead_s_ = get_or_declare_param<double>(node_, "close_lead_s", 0.4);
    // Sensor switches (Majd 2026-09-17): check_grasp_sensor = the finger
    // pin decides "object inside" after the close (off: assumed holding, no
    // no-grasp recovery, no "still holding" refusal, no fingers-apart
    // check); check_slip = the slip pin is watched after the grasp.
    check_grasp_sensor_ = get_or_declare_param<bool>(node_, "check_grasp_sensor", true);
    check_slip_ = get_or_declare_param<bool>(node_, "check_slip", true);
    // Open at the CLICK (2026-09-17): the pipeline announces a selection on
    // selection_cycle_id ~2.4 s before the candidates arrive; sending the
    // open then removes the "waiting for the fingers" pause on a cache
    // miss. Only a selection triggers it, never a background check.
    cartesian_curve_samples_ = get_or_declare_param<int>(node_, "cartesian_curve_samples", 40);   // over the bend; the slide at the same spacing
    cartesian_eef_step_m_ = get_or_declare_param<double>(node_, "cartesian_eef_step_m", 0.015);
    cartesian_max_joint_step_rad_ = get_or_declare_param<double>(node_, "cartesian_max_joint_step_rad", 0.8);
    // 2026-09-18 12:30: 150 deg refused the far objects — legitimate curves
    // there turn the elbow 180-240 deg and the base 165-170 deg — while the
    // real flips read 368-478 deg. 250 deg separates the two in every case
    // seen; the motion-time cap below is the second net.
    cartesian_max_joint_travel_rad_ = get_or_declare_param<double>(node_, "cartesian_max_joint_travel_rad", 4.36);   // 250 deg per joint over the whole curve
    cartesian_max_motion_s_ = get_or_declare_param<double>(node_, "cartesian_max_motion_s", 4.0);   // a curve timed longer than this at full scaling is a wrong configuration, not an approach   // 2026-09-18: 0.35 refused 12 curves for 0.36-0.43 rad steps in the bend (the wrist turning, not a flip); a real flip is >= 1 rad   // 2026-09-17: halves the joint solves + collision checks; the robot interpolates between points anyway
    pregrasp_approach_distance_ = get_or_declare_param<double>(
      node_, "pregrasp_approach_distance", 0.10);
    target_frame_ = get_or_declare_param<std::string>(node_, "target_frame", "world");
    tf_timeout_ = get_or_declare_param<double>(node_, "tf_timeout", 2.0);
    planning_time_ = get_or_declare_param<double>(node_, "planning_time", 10.0);
    // Tight per-candidate planning budget. The LIN planner can spend 20-30s
    // on borderline poses before reporting INVALID_MOTION_PLAN; a hard cap
    // here cuts loop time without affecting fast rejections or successes.
    lin_feasibility_planning_time_ = get_or_declare_param<double>(
      node_, "lin_feasibility_planning_time", 1.5);
    lin_feasibility_attempts_ = std::max(
      1, get_or_declare_param<int>(node_, "lin_feasibility_attempts", 1));
    // Default FALSE since 2026-08-20 (Majd: "grasp from the center"):
    // probe in the pipeline's score order — the centroid-reweighted
    // ranking — instead of nearest-to-gripper-first. Set true to
    // restore the old distance ordering.
    sort_candidates_by_reachability_ = get_or_declare_param<bool>(
      node_, "sort_candidates_by_reachability", false);
    // Preserve the pipeline's published TCP orientation by default.  The
    // wrist-twin rewrite is opt-in because changing q to q*Rz(pi) here makes
    // the executed pose disagree with the pose that the pipeline filtered,
    // scored, and visualised.
    // ⚠ MAJD'S STANDING ORDER (2026-08-21): keep FALSE. The twin
    // rewrite was the thing FLIPPING the gripper on real picks —
    // disabling it fixed the flips. Do not re-enable without asking
    // him explicitly; the pipeline's ee-orientation score reweight is
    // the anti-flip mechanism that actually behaves.
    enable_wrist_twin_rewrite_ = get_or_declare_param<bool>(
      node_, "enable_wrist_twin_rewrite", false);
    // Twin selection rule (2026-08-21, Majd: "it flips the whole gripper").
    // The two 180-degree twins grip identically, so we are free to pick.
    // TRUE  = pick the twin whose gripper "up" axis points UP in base
    //         (never hanging upside down), closeness only as tiebreak.
    // FALSE = old rule, purely closest to the current wrist.
    prefer_upright_twin_ = get_or_declare_param<bool>(
      node_, "prefer_upright_twin", true);
    // Below this |y_axis_z| the twins are equally upright (fingers in a
    // vertical plane) — then minimise wrist rotation instead.
    upright_twin_dead_zone_ = get_or_declare_param<double>(
      node_, "upright_twin_dead_zone", 0.15);
    // FLAT grasps (palm axis horizontal, fingers closing vertically):
    // both twins are equally upright, so the rule above cannot decide
    // and shortest-rotation picked a random-feeling side — THE flip
    // Majd kept seeing (2026-08-21, "grasp 2/46 ... angled, flat").
    // Deterministic tie-break: put the SAME finger on top every time
    // (sign of the finger-opening axis' z). If the hardware's natural
    // side is the other one, set flat_twin_x_up_positive:=false — one
    // switch, consistent either way.
    flat_twin_x_up_positive_ = get_or_declare_param<bool>(
      node_, "flat_twin_x_up_positive", true);
    // Which sign of y_up is the gripper's NATURAL side. Default FALSE
    // (= prefer y_up<0): Majd's probe pane showed every executed
    // candidate tagged 'upright' (y_up>0) and he consistently reported
    // those as FLIPPED (2026-08-21) — so this hardware's correct side
    // is the other one. If grasps now come consistently wrong the other
    // way, set true — one switch, never a debugging session.
    upright_twin_y_up_positive_ = get_or_declare_param<bool>(
      node_, "upright_twin_y_up_positive", false);
    num_attempts_ = get_or_declare_param<int>(node_, "num_attempts", 10);
    // Manual-button mode (Majd 2026-08-26: "just go and grasp it,
    // nothing more — I do the rest manually"): stop right after the
    // gripper closes on the object. No lift, no transport, no drop,
    // no reset — the arm HOLDS at the grasp pose and the cycle ends
    // successfully. The GUI's EXECUTE button launches with this true;
    // the voice executor keeps the full interactive sequence.
    stop_after_grasp_ = get_or_declare_param<bool>(
      node_, "stop_after_grasp", false);
    one_shot_ = get_or_declare_param<bool>(node_, "one_shot", true);
    velocity_scaling_ = get_or_declare_param<double>(node_, "velocity_scaling", 0.5);
    acceleration_scaling_ = get_or_declare_param<double>(node_, "acceleration_scaling", 0.5);
    // 0.5 s since 2026-09-16 (was 5.0: Majd "how long does it take before
    // opening?" — five seconds of standing at the drop). Just a settle.
    drop_hold_s_ = get_or_declare_param<double>(node_, "drop_hold_s", 0.5);
    // After the release the arm STAYS for this long before going home (Majd
    // 2026-09-16: "it should wait 2 s until I took the object, before it
    // hits me"). The voice flow's home-confirm has its own 3 s.
    post_release_wait_s_ = get_or_declare_param<double>(node_, "post_release_wait_s", 2.0);
    drop_target_name_ = get_or_declare_param<std::string>(node_, "drop_target_name", "drop_right_side");
    reset_target_name_ = get_or_declare_param<std::string>(node_, "reset_target_name", "home");
    // Fast-replay short-circuit: when enabled, named-target moves
    // (reset_target, drop_target) bypass MoveIt PTP and are sent straight
    // to ur3e_fast_observation_replay, which forwards joint values to
    // /scaled_joint_trajectory_controller. Big latency win because we
    // skip planning + collision check for poses we trust by name.
    use_fast_replay_for_named_targets_ = get_or_declare_param<bool>(
      node_, "use_fast_replay_for_named_targets", true);
    fast_replay_topic_ = get_or_declare_param<std::string>(
      node_, "fast_replay_topic", "/ur3e_fast_observation_replay/replay");
    fast_replay_status_topic_ = get_or_declare_param<std::string>(
      node_, "fast_replay_status_topic", "/ur3e_fast_observation_replay/status");
    fast_replay_timeout_s_ = get_or_declare_param<double>(
      node_, "fast_replay_timeout_s", 8.0);
    fast_replay_duration_s_ = get_or_declare_param<double>(
      // Medium tier. This TOPIC fallback (used only when
      // /move_to_named_pose is unavailable) was still 0.8 s = violent;
      // history 2.0 -> 1.5 -> 1.0 -> 0.8 -> 2.7 (2026-08-28).
      node_, "fast_replay_duration_s", 2.7);
    // When true, route the fast path through the /move_to_named_pose action
    // client (which does a collision pre-check against the planning scene)
    // instead of publishing directly to the replay topic. False reverts to
    // the older raw-publish behaviour for benchmarking.
    use_named_pose_action_ = get_or_declare_param<bool>(
      node_, "use_named_pose_action_for_fast_replay", true);
    named_pose_action_name_ = get_or_declare_param<std::string>(
      node_, "named_pose_action_name", "/move_to_named_pose");
    gripper_open_name_ = get_or_declare_param<std::string>(node_, "gripper_open_name", "open");
    gripper_close_name_ = get_or_declare_param<std::string>(node_, "gripper_close_name", "close");
    gripper_open_position_ = get_or_declare_param<double>(node_, "gripper_open_position", 0.5);
    gripper_close_position_ = get_or_declare_param<double>(node_, "gripper_close_position", 0.0);
    gripper_action_name_ = get_or_declare_param<std::string>(
      node_, "gripper_action_name", "/gripper_controller/gripper_cmd");
    gripper_action_timeout_s_ = get_or_declare_param<double>(node_, "gripper_action_timeout_s", 5.0);
    // The approach may never end before the fingers are apart (Majd
    // 2026-09-16): open pulse + finger travel, plus a margin. A faster plan
    // is stretched to this duration, see stretch_plan_to_min_duration.
    // 1.2 s (measured 2026-09-16): with the close pulse 1.03 s after the open
    // pulse began the fingers were still opening (Majd saw them touch the
    // object); at 1.29 s the close stalled on the object after 0.73 s.
    gripper_open_time_s_ = get_or_declare_param<double>(node_, "gripper_open_time_s", 1.0);   // 2026-09-17 evening, 8 picks: a close sent 0.80 s after the open grasped nothing twice, >= 0.93 s held six times; the adapter's 0.6 s "done" is not the fingers
    // Margin 0.5 s (2026-09-16): the adapter starts the pulse ~0.25 s after
    // the goal is sent (measured 741.43 -> 741.69), and the fingers need a
    // little settle. So the arm arrives no earlier than 1.2 s after the send.
    open_before_arrival_margin_s_ = get_or_declare_param<double>(node_, "open_before_arrival_margin_s", 0.1);
    // One duration for every approach (Majd 2026-09-17 night: "make them all
    // one speed, not mega fast, not very slow, 2.5 s"): whatever planner won
    // and at whatever scaling, a plan shorter than this is stretched to it.
    // Longer plans stay as planned (they cannot be compressed past limits).
    // It also ends the fast-arrival grasps that hit the object: the fingers'
    // 1.0 s is always inside it.
    approach_duration_s_ = get_or_declare_param<double>(node_, "approach_duration_s", 2.5);
    gripper_adapter_node_name_ = get_or_declare_param<std::string>(
      node_, "gripper_adapter_node_name", "/gripper_controller/gripper_command_adapter");
    override_gripper_pulse_from_executor_ =
      get_or_declare_param<bool>(node_, "override_gripper_pulse_from_executor", false);
    executor_pulse_hold_s_open_ =
      get_or_declare_param<double>(node_, "executor_pulse_hold_s_open", -1.0);
    executor_pulse_hold_s_close_ =
      get_or_declare_param<double>(node_, "executor_pulse_hold_s_close", -1.0);
    lift_height_m_ = get_or_declare_param<double>(node_, "lift_height_m", 0.07);
    lift_table_yaw_deg_ = get_or_declare_param<double>(node_, "lift_table_yaw_deg", 45.0);
    lift_table_frame_ = get_or_declare_param<std::string>(node_, "lift_table_frame", "");
    // After the diagonal table-frame lift (3a), insert a safety retreat:
    //   3a-up:   straight +Z in base_link to clear the object/table by post_lift_up_m
    //   3a-back: horizontal in base_link toward the robot base (XY → 0)
    //            by post_lift_back_m, so the subsequent PTP to home doesn't
    //            swing the arm through the table.
    // Set either param to 0 to skip that sub-step.
    post_lift_up_m_ = get_or_declare_param<double>(node_, "post_lift_up_m", 0.05);
    post_lift_back_m_ = get_or_declare_param<double>(node_, "post_lift_back_m", 0.10);
    // When true, skip Step 3b (PTP to reset_target_name between lift and
    // drop). The 3a/3a-up/3a-back retreats are SUPPOSED to move the arm
    // to a safe waypoint, but they're soft-failures — if Pilz LIN
    // refuses (singularity, IK gap), we log and continue. With
    // skip_reset_before_drop=true AND both retreats silently failed,
    // Step 4 plans a PTP straight from the diagonal-lift pose to the
    // drop pose, which can swing the arm through the table.
    // Default OFF so the proven safe sequence (lift → retreats →
    // home → drop) runs even if the retreats silently fail. Set to
    // true only when you trust the retreats are reliable on your setup.
    // Default flipped OFF->ON 2026-08-26 (Majd, twice: "why does he go
    // home first?"): transport goes DIRECTLY lift->drop; the planning
    // scene's table slab guards the swing-through risk, and Step 4
    // falls back to the planner if the direct move fails.
    // FALSE = always stop at home between lift and drop (Majd 2026-08-26,
    // final word: "lift up, go back home, then drop position"). Both legs
    // ride the fast collision-checked named-pose channel, so the home
    // stop costs ~2 s and is geometrically always clear of the table.
    skip_reset_before_drop_ = get_or_declare_param<bool>(
      node_, "skip_reset_before_drop", false);
    // BLENDED transport (2026-08-28). The canonical flow stays
    // lift -> HOME -> drop, but instead of two MoveToNamedPose goals
    // (each paying ~2 s of fixed overhead and each decelerating the arm
    // to a full stop at home) we send ONE goal with via_poses=[home].
    // The named-pose server turns that into a single dense, S-curve
    // timed trajectory, so the arm flows through home without stopping.
    // Measured baseline for the two-goal version: 3.7 s (home leg) +
    // 3.9 s (drop leg) = 7.6 s. Falls back to the two-step sequence
    // automatically if the blended goal is refused for any reason.
    blend_via_home_ = get_or_declare_param<bool>(
      node_, "blend_via_home", true);
    // Speed tier for the blended path. The server charges it once per
    // segment, so the default 0.3 (normal, 2.7 s) x 2 segments = 5.4 s
    // of motion, paid with ONE ~2 s overhead instead of two -> ~7.4 s
    // door to door against the measured 7.6 s two-goal baseline: the
    // blend now buys smoothness rather than time.
    //
    // Measured on the real taught poses (lift -> home ->
    // drop_right_side), worst joint as a fraction of its UR3e limit:
    //   0.50 tier, 3.2 s total -> 76%   (elbow 2.38 rad/s)
    //   0.40 tier, 4.0 s total -> 61%   (gentler than today's drop leg)
    //   0.30 tier, 5.4 s total -> 45%
    // For reference, today's UNBLENDED home->drop at the 0.5 tier peaks
    // the elbow at 111% of its limit — that is the overspeed Majd heard
    // as noise, and why the drop leg was pinned to 0.3. Blending fixes it
    // structurally: time is allotted by arc length, so the long
    // home->drop swing gets more of the clock than the short lift->home
    // one instead of an equal share.
    //
    // MEDIUM by default (Majd 2026-08-28, second pass: "make it medium
    // as before and keep it smooth"): 0.30 = the NORMAL tier, 2.7 s per
    // segment, 5.4 s of motion for lift -> home -> drop, worst joint at
    // 45% of its limit. Raise to 0.4 only if the researcher asks for
    // more pace. NEVER above 0.50 — the server clamps it, and the arm
    // protective-stops beyond it.
    blend_velocity_scaling_ = get_or_declare_param<double>(
      node_, "blend_velocity_scaling", 0.3);
    // Interactive drop mode. When true, the executor closes the gripper
    // on the object and then PAUSES at the grasp pose, waiting for a
    // String message on resume_topic carrying the destination
    // (drop_right / drop_left / home / stop). The lift + retreat + drop
    // happen only after the destination arrives. When false, the executor
    // uses drop_target_name as before — for one-shot 'pick X and place
    // on the right' commands.
    wait_for_destination_ = get_or_declare_param<bool>(
      node_, "wait_for_destination", true);
    get_or_declare_param<double>(node_, "pause_before_close_s", 0.0);
    get_or_declare_param<double>(node_, "grasp_stall_max_s", 0.0);
    lin_slow_retry_factor_ = get_or_declare_param<double>(node_, "lin_slow_retry_factor", 0.5);
    resume_topic_ = get_or_declare_param<std::string>(
      node_, "resume_topic", "/grasp_executor_lin_llm/resume");
    // 120 -> 1800 (2026-08-25, Majd: "even when I don't say where it
    // should drop, it should WAIT — why is there a timeout?"): holding
    // is safe, so the interactive pauses now wait up to 30 minutes for
    // the operator instead of giving up after 2.
    resume_timeout_s_ = get_or_declare_param<double>(
      node_, "resume_timeout_s", 1800.0);
    // Second interactive pause: AFTER arrival at drop_target, BEFORE
    // opening the gripper. Lets the user say "open" / "release" / "give
    // it to me" before Morty drops the object. Same resume topic with
    // payload 'release' (open) or 'stop' (skip release, go home with
    // gripper closed).
    wait_for_release_ = get_or_declare_param<bool>(
      node_, "wait_for_release", true);
    // ── Slip slow-down watchdog ───────────────────────────────────────
    // When TI1 goes HIGH (slip detected by the gripper's sensor) the
    // watchdog drops the UR speed slider to `slip_slow_slider_` — a
    // very slow but still-moving crawl, NOT a full stop. The hardware
    // self-tightening continues to act while the arm creeps; when TI1
    // returns LOW the slider is restored to nominal. A short String is
    // also published on /executor/slip_event so intent_router can have
    // Morty narrate "Slip detected, slowing down." / "Grip restored".
    // Disable with slip_watchdog_enabled:=false to fall back to "ignore
    // TI1, run at nominal speed".
    slip_watchdog_enabled_ = get_or_declare_param<bool>(
      node_, "slip_watchdog_enabled", true);
    // Speed slider value (0..1) applied while a slip is active. 0.05 =
    // ~5% of nominal velocity — visibly slow but unmistakably moving.
    // Tune up if too cautious or down if the arm still moves too fast
    // to keep the object in grasp.
    slip_slow_slider_ = get_or_declare_param<double>(
      node_, "slip_slow_slider", 0.05);
    slip_event_topic_ = get_or_declare_param<std::string>(
      node_, "slip_event_topic", "/executor/slip_event");
    // Tool digital inputs on UR e-series. The UR ros2_control driver
    // exposes TI0 / TI1 on /io_and_status_controller/io_states at
    // digital_in_states pin INDICES 16 and 17 (the full layout is
    // 0-7 standard DI, 8-15 configurable DI, 16-17 tool DI). The
    // previous defaults of 18/19 were wrong — those indices never get
    // published, so the sensors were never read at all.
    // TI0 (pin 16): GRASP-DETECTED sensor — HIGH when holding object.
    //   The "no-grasp" condition is the INVERSE (pin LOW after close).
    // TI1 (pin 17): SLIP sensor — HIGH when slip detected.
    grasp_pin_ = get_or_declare_param<int>(node_, "di_grasp_pin", 16);
    slip_pin_ = get_or_declare_param<int>(node_, "di_slip_pin", 17);
    io_states_topic_ = get_or_declare_param<std::string>(
      node_, "io_states_topic", "/io_and_status_controller/io_states");
    speed_slider_srv_ = get_or_declare_param<std::string>(
      node_, "speed_slider_srv",
      "/io_and_status_controller/set_speed_slider");
    nominal_speed_slider_ = get_or_declare_param<double>(
      node_, "nominal_speed_slider", 1.0);
    // After Step 3 (close gripper), check di_no_grasp. If HIGH, abort
    // the lift+drop sequence cleanly — go up + back + home — and
    // surface the failure so intent_router can narrate "I'm not
    // grasping anything, returning home."
    check_no_grasp_after_close_ = get_or_declare_param<bool>(
      node_, "check_no_grasp_after_close", true);
    // Vitamin-tube special mode. Toggled by intent_router via a
    // SetParameter call ONLY for vitamin_c picks. When true, the
    // executor uses a shorter first-close pulse, SKIPS the TI0 check
    // after that first close, runs the normal lift+retreat sequence,
    // then during the move to drop_target it slows the arm at ~20% of
    // the path and fires a SECOND, harder close pulse. The TI0 check
    // re-enables after that re-close. When false (default), behaviour
    // is identical to before — every other object uses the default
    // path. Designed to be removable later: deleting the
    // `if (vitamin_mode_active_)` branches restores the original flow.
    vitamin_mode_active_ = get_or_declare_param<bool>(
      node_, "vitamin_mode_active", false);
    vitamin_first_close_s_ = get_or_declare_param<double>(
      node_, "vitamin_first_close_s", 1.1);
    vitamin_retighten_close_s_ = get_or_declare_param<double>(
      node_, "vitamin_retighten_close_s", 1.1);
    vitamin_retighten_at_fraction_ = get_or_declare_param<double>(
      node_, "vitamin_retighten_at_fraction", 0.20);
    vitamin_retighten_speed_scaling_ = get_or_declare_param<double>(
      node_, "vitamin_retighten_speed_scaling", 0.05);
    use_current_orientation_if_identity_ =
      get_or_declare_param<bool>(node_, "use_current_orientation_if_identity", true);
    use_candidate_pose_array_ = get_or_declare_param<bool>(node_, "use_candidate_pose_array", true);
    get_or_declare_param<double>(node_, "candidate_search_timeout_s", 60.0);
    max_candidate_tries_ = std::max(1, get_or_declare_param<int>(node_, "max_candidate_tries", 10));
    candidate_try_all_batches_ =
      get_or_declare_param<bool>(node_, "candidate_try_all_batches", true);
    // Rejection autopsy budget (Majd, 2026-08-28: "if there is a
    // collision then I need to know the exact collision and with
    // what"). Naming the obstacle costs one IK solve plus one
    // /check_state_validity round-trip per rejected candidate, so
    // only the first N rejections of a cycle get the full report;
    // the rest fall back to the bare error-code message.
    collision_detail_max_reports_ = std::max(
      0, get_or_declare_param<int>(node_, "collision_detail_max_reports", 5));
    // Points checked along the straight approach when the goal itself is
    // clear (0 = off). All are queried in parallel: one round trip.
    approach_autopsy_samples_ = std::max(
      0, get_or_declare_param<int>(node_, "approach_autopsy_samples", 4));
    // Selection-to-motion budget (2026-09-09, measured 2.8 s):
    //  - the quick-IK pre-filter only ORDERS candidates, so a false negative
    //    costs nothing but position; a failing candidate burns the whole
    //    timeout, and 12 failures x 50 ms was 0.7 s per pick. 20 ms is enough
    //    to find the ones that solve instantly.
    //  - the "clear stray perception objects" step before the approach has
    //    never found anything since the pipeline stopped publishing the
    //    target box (0 hits in every log), yet it is a blocking planning-scene
    //    service call that cost 0.05-0.9 s. Off by default.
    prefilter_ik_timeout_s_ = std::max(
      0.005, get_or_declare_param<double>(node_, "prefilter_ik_timeout_s", 0.02));
    clear_stray_scene_objects_ =
      get_or_declare_param<bool>(node_, "clear_stray_scene_objects", false);
    // Created ONCE: a client per candidate would be slow and would
    // leak service endpoints for the life of the node.
    sequence_client_ = node_->create_client<moveit_msgs::srv::GetMotionSequence>("/plan_sequence_path");
    state_validity_client_ =
      node_->create_client<moveit_msgs::srv::GetStateValidity>(
        "/check_state_validity");

    // ── Probe-plan reuse (2026-08-28) ────────────────────────────────
    // MEASURED: a Pilz LIN plan that SUCCEEDS costs 6.4-12.2 s of wall
    // clock (median 11.0 s over 3 picks); a plan that FAILS returns in
    // milliseconds. So the winning candidate's trajectory is the single
    // most expensive artefact in the whole pick — and the executor used
    // to prove the pose planable in the candidate probe, throw the
    // trajectory away, and ask move_group for the identical plan again
    // in Step 2. That duplicate cost 6.36 + 11.93 + 11.06 = 29.4 s
    // across three picks (~9.8 s each). Keep the probe's trajectory and
    // execute THAT — but only when it is still provably safe.
    reuse_probe_plan_ = get_or_declare_param<bool>(node_, "reuse_probe_plan", true);
    // Pose identity. Step 2 is handed the very same PoseStamped the
    // probe planned for, so these are "is it literally the same pose"
    // guards, not "is it close enough": 0.1 mm and ~0.057 deg. A pose
    // 1 mm or 1 deg away is a DIFFERENT grasp and must be replanned.
    cached_plan_pos_tol_m_ = get_or_declare_param<double>(
      node_, "cached_plan_pos_tol_m", 1e-4);
    cached_plan_ori_tol_rad_ = get_or_declare_param<double>(
      node_, "cached_plan_ori_tol_rad", 1e-3);
    // Start-state identity — THE safety number. 0.01 rad is MoveIt's own
    // trajectory_execution/allowed_start_tolerance default, i.e. the
    // exact threshold at which move_group itself refuses to start a
    // trajectory. Matching it means we never hand the controller
    // something move_group would have rejected on our behalf.
    cached_plan_start_joint_tol_rad_ = get_or_declare_param<double>(
      node_, "cached_plan_start_joint_tol_rad", 0.01);
    // Staleness bound. A matching start state does NOT prove the scene
    // is unchanged, so an old plan is dropped however well it matches.
    cached_plan_max_age_s_ = get_or_declare_param<double>(
      node_, "cached_plan_max_age_s", 30.0);
    // Waypoints re-checked against /check_state_validity before reuse.
    // The endpoint is ALWAYS one of them.
    cached_plan_validate_samples_ = std::max(
      1, get_or_declare_param<int>(node_, "cached_plan_validate_samples", 5));
    cached_plan_validate_timeout_s_ = get_or_declare_param<double>(
      node_, "cached_plan_validate_timeout_s", 0.5);

    arm_group_.setPlanningTime(planning_time_);
    arm_group_.setNumPlanningAttempts(num_attempts_);
    arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
    arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
    gripper_action_client_ =
      rclcpp_action::create_client<control_msgs::action::GripperCommand>(node_, gripper_action_name_);

    // Fast replay: publisher for label/JSON requests + status subscriber.
    fast_replay_pub_ = node_->create_publisher<std_msgs::msg::String>(
      fast_replay_topic_, 10);
    fast_replay_status_sub_ = node_->create_subscription<std_msgs::msg::String>(
      fast_replay_status_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(fast_replay_mutex_);
        fast_replay_last_status_ = msg->data;
        fast_replay_status_cv_.notify_all();
      });

    // Own ear on /joint_states for the planner-free lift. SensorDataQoS
    // (best-effort) matches both reliable and best-effort publishers.
    joint_states_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        {
          std::lock_guard<std::mutex> lk(joint_state_mutex_);
          last_joint_state_ = *msg;
        }
        joint_state_seq_.fetch_add(1);
      });

    // Action client for collision-checked named-pose moves. The action
    // server (move_to_named_pose_server) interpolates the joint trajectory
    // and queries MoveIt's /check_state_validity for each waypoint,
    // refusing the move if the path would crash through the planning
    // scene's table / wall / mount / etc.
    named_pose_action_client_ =
      rclcpp_action::create_client<morty_interfaces::action::MoveToNamedPose>(
        node_, named_pose_action_name_);

    apply_gripper_adapter_overrides_if_enabled();

    // Resume subscriber for the interactive drop mode. The handler is
    // tiny — it just records the latest destination string and signals
    // the waiting sequence thread via a condition variable.
    resume_sub_ = node_->create_subscription<std_msgs::msg::String>(
      resume_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(resume_mutex_);
        resume_destination_ = msg->data;
        resume_received_ = true;
        resume_cv_.notify_all();
      });

    // Any message on the abort topic requests an immediate stop of the
    // current pick cycle (payload ignored).
    abort_sub_ = node_->create_subscription<std_msgs::msg::String>(
      "/grasp_executor_lin_llm/abort", 10,
      [this](const std_msgs::msg::String::SharedPtr) { request_abort(); });

    // Lifecycle status for the interactive drop. intent_router subscribes
    // to this and triggers chat_node's "Where should I drop it?" prompt
    // exactly when the pause begins — i.e. AFTER the arm has moved to
    // the grasp pose and AFTER the gripper has closed on the object.
    // Without this signal the LLM would have no way to know "the pause
    // is now" and the prompt would never be spoken at the right time.
    interactive_status_pub_ = node_->create_publisher<std_msgs::msg::String>(
      "/grasp_executor_lin_llm/interactive_status", 10);
    // Per-candidate probe status for the GUI ("Execute 2: fail because ...",
    // Majd 2026-08-12). One String per tried candidate.
    candidate_status_pub_ = node_->create_publisher<std_msgs::msg::String>(
      "/grasp_executor_lin_llm/candidate_status", 10);

    // Subscribe to UR digital inputs. The UR driver publishes IOStates at
    // its native rate (~125-500 Hz). Each callback just snaps the pin
    // states into atomic flags so the slip-watchdog thread can react
    // without any locking.
    io_states_sub_ = node_->create_subscription<ur_msgs::msg::IOStates>(
      io_states_topic_, 10,
      [this](const ur_msgs::msg::IOStates::SharedPtr msg) {
        bool slip = false;
        bool grasp = false;
        for (const auto & s : msg->digital_in_states) {
          if (static_cast<int>(s.pin) == slip_pin_) slip = s.state;
          if (static_cast<int>(s.pin) == grasp_pin_) grasp = s.state;
        }
        slip_active_.store(slip);
        if (grasp && !grasp_detected_.load()) {
          grasp_rise_t_ = std::chrono::steady_clock::now();   // 0 -> 1 edge
        }
        grasp_detected_.store(grasp);
      });
    speed_slider_client_ =
      node_->create_client<ur_msgs::srv::SetSpeedSliderFraction>(
        speed_slider_srv_);

    // Publisher used by the slip watchdog to broadcast 'slip_start' /
    // 'slip_clear' events. intent_router subscribes to this topic and
    // narrates the slip via Morty's TTS.
    slip_event_pub_ = node_->create_publisher<std_msgs::msg::String>(
      slip_event_topic_, 10);

    // Watchdog thread: poll the slip flag at high rate; on LOW→HIGH
    // drop the speed slider to slip_slow_slider_ (a slow crawl, not a
    // full stop — the arm continues so the gripper hardware can keep
    // self-tightening while the load stays under control). On HIGH→LOW
    // restore the nominal speed slider. Broadcast the transition on
    // /executor/slip_event for downstream narration.
    if (slip_watchdog_enabled_) {
      std::thread([this]() {
        rclcpp::Rate rate(100);  // 100 Hz watchdog poll
        bool last_slip = false;
        while (rclcpp::ok()) {
          // Unarmed ⇒ treated as no-slip; disarming mid-slip restores
          // nominal speed through the normal HIGH→LOW transition.
          bool slip = slip_monitor_armed_.load() && slip_active_.load();
          if (slip != last_slip) {
            std_msgs::msg::String evt;
            if (slip) {
              RCLCPP_WARN(node_->get_logger(),
                "SLIP detected on DI pin %d → slowing motion "
                "(speed_slider=%.2f).",
                slip_pin_, slip_slow_slider_);
              set_speed_slider(slip_slow_slider_);
              slip_paused_.store(true);
              evt.data = "slip_start";
            } else {
              RCLCPP_INFO(node_->get_logger(),
                "Slip cleared → restoring nominal speed "
                "(speed_slider=%.2f).",
                nominal_speed_slider_);
              set_speed_slider(nominal_speed_slider_);
              slip_paused_.store(false);
              evt.data = "slip_clear";
            }
            if (slip_event_pub_) {
              slip_event_pub_->publish(evt);
            }
            last_slip = slip;
          }
          rate.sleep();
        }
      }).detach();
    }

    // Ensure the gripper action timeout covers the longest pulse + IO overhead.
    if (override_gripper_pulse_from_executor_) {
      double max_pulse = std::max(executor_pulse_hold_s_open_, executor_pulse_hold_s_close_);
      double min_timeout = max_pulse + 2.0;  // pulse + IO round-trips + headroom
      if (gripper_action_timeout_s_ < min_timeout) {
        RCLCPP_INFO(
          node_->get_logger(),
          "Auto-adjusting gripper_action_timeout_s from %.1fs to %.1fs "
          "(max pulse=%.1fs + 2.0s headroom).",
          gripper_action_timeout_s_, min_timeout, max_pulse);
        gripper_action_timeout_s_ = min_timeout;
      }
    }
  }

  bool is_one_shot() const { return one_shot_; }

  // ── Background LIN check of a stored candidate set (2026-09-11) ─────
  // The pipeline's precompute cache sends the candidate set of every
  // object it has grasps for, while the arm waits at home. We run the
  // very same feasibility search as a real pick — first batch, same
  // start state, same scene — but with no gripper motion, no
  // candidate_status lines to the GUI and no failure reason left behind,
  // and it ends between two candidates the moment a real selection
  // arrives. The verdict (first feasible index + its verified plan +
  // the arm joints at probe time) is kept; a later selection whose
  // candidates match by content, with the arm still where it was,
  // executes without probing.
  struct PrecheckRecord {
    int64_t key = 0;
    std::string planner;        // approach the background check verified
    std::vector<geometry_msgs::msg::PoseStamped> poses;
    int feasible_idx = -1;      // index into poses, -1 = none of the batch
    size_t n_tried = 0;
    bool has_plan = false;
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    geometry_msgs::msg::PoseStamped plan_pose;
    double plan_wall_s = 0.0;
    bool has_final_plan = false;  // staging_mode lin: the last stretch pre-grasp -> grasp
    moveit::planning_interface::MoveGroupInterface::Plan final_plan;
    std::vector<double> joints;   // arm joints at probe time, jmg order
    std::chrono::steady_clock::time_point t_done{};
    std::string reason;           // the search's own verdict when nothing was feasible
  };

  // Returns false when the check did not finish (interrupted by a
  // selection, operator abort, or no joint state); true otherwise, with
  // the verdict in *rec.
  bool run_precheck(
    const std::vector<geometry_msgs::msg::PoseStamped> & cands, int64_t key,
    std::function<bool()> interrupt, PrecheckRecord * rec)
  {
    rec->key = key; rec->poses = cands; rec->feasible_idx = -1;
    rec->n_tried = 0; rec->has_plan = false;
    if (!arm_joints_now(&rec->joints)) { return false; }
    const std::string saved_reason = last_failure_reason_;
    precheck_mode_ = true; precheck_interrupted_ = false;
    precheck_interrupt_ = std::move(interrupt); precheck_tried_ = 0;
    geometry_msgs::msg::PoseStamped sel;
    bool ok = false;
    try { ok = choose_first_lin_feasible(cands, &sel); } catch (...) { ok = false; }
    precheck_mode_ = false; precheck_interrupt_ = nullptr;
    rec->reason = ok ? "" : last_failure_reason_;
    rec->planner = ok ? last_planner_ : "";
    last_failure_reason_ = saved_reason;
    skip_next_precheck_ = false;
    rec->n_tried = precheck_tried_;
    if (precheck_interrupted_ || abort_requested_.load()) {
      invalidate_cached_lin_plan("the background check was interrupted");
      return false;
    }
    if (ok) {
      for (size_t i = 0; i < cands.size(); ++i) {
        if (same_pose(cands[i].pose, sel.pose, 0.0, 0.0)) { rec->feasible_idx = static_cast<int>(i); break; }
      }
      if (cached_lin_plan_valid_) {
        rec->has_plan = true; rec->plan = cached_lin_plan_;
        rec->has_final_plan = cached_final_plan_valid_;
        if (cached_final_plan_valid_) { rec->final_plan = cached_final_plan_; }
        rec->plan_pose = cached_lin_plan_pose_; rec->plan_wall_s = cached_lin_plan_wall_s_;
      }
      invalidate_cached_lin_plan("kept in the background-check record instead");
    }
    rec->t_done = std::chrono::steady_clock::now();
    return true;
  }

  void remember_precheck(PrecheckRecord rec)
  {
    // One record per candidate set: a recomputed object replaces its old one.
    prechecks_.erase(
      std::remove_if(prechecks_.begin(), prechecks_.end(),
        [&](const PrecheckRecord & r) { return same_set(r.poses, rec.poses); }),
      prechecks_.end());
    prechecks_.push_front(std::move(rec));
    while (prechecks_.size() > precheck_max_sets_) { prechecks_.pop_back(); }
  }

  const PrecheckRecord * match_precheck(
    const std::vector<geometry_msgs::msg::PoseStamped> & cands) const
  {
    for (const auto & r : prechecks_) {
      if (same_set(r.poses, cands)) { return &r; }
    }
    return nullptr;
  }

  // The stored verdict holds only from the same arm pose: a plan is a
  // joint-space command and the probe's start state was those joints.
  bool precheck_usable(const PrecheckRecord & rec, std::string * why)
  {
    std::vector<double> now;
    if (!arm_joints_now(&now)) { if (why) { *why = "no fresh joint state"; } return false; }
    if (now.size() != rec.joints.size()) { if (why) { *why = "joint layout changed"; } return false; }
    double worst = 0.0;
    for (size_t i = 0; i < now.size(); ++i) { worst = std::max(worst, std::fabs(now[i] - rec.joints[i])); }
    if (worst > cached_plan_start_joint_tol_rad_) {
      if (why) {
        char b[96]; std::snprintf(b, sizeof(b), "arm moved %.3f rad since the check", worst);
        *why = b;
      }
      return false;
    }
    return true;
  }

  // Put the record's verified plan back where Step 2 looks for it and
  // skip the duplicate pre-check, exactly as after a live probe.
  void apply_precheck(const PrecheckRecord & rec)
  {
    if (rec.has_plan) { cache_lin_plan(rec.plan_pose, rec.plan, rec.plan_wall_s); }
    cached_final_plan_valid_ = rec.has_final_plan;
    if (rec.has_final_plan) { cached_final_plan_ = rec.final_plan; }
    skip_next_precheck_ = true;
  }

  double precheck_age_s(const PrecheckRecord & rec) const
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - rec.t_done).count();
  }

  // The open goes out WITH the first arm motion of a pick and never before
  // it (Majd 2026-09-17: "don't open the gripper if there is no real
  // movement"). Every branch of the approach calls this right before its
  // first execute, so a plan that fails never parts the fingers. The
  // adapter reports the open done ~0.6 s after the send (0.25 s to the
  // pulse, 0.20 s pulse, IO round trip) — gripper_open_time_s.
  void send_open_with_motion(const char * where)
  {
    if (open_already_sent()) { return; }   // a retry inside the same pick
    begin_gripper_open_async();
    RCLCPP_INFO(node_->get_logger(), "%s: open sent with the motion.", where);
  }
  bool open_already_sent() const { return open_pending_ || open_done_; }

  // A background-verified pick skips the probe, so nothing reached the GUI's
  // console and its STOP button never armed (it arms on candidate_status).
  void announce_precheck_hit(int idx, size_t n, const geometry_msgs::msg::PoseStamped & pose,
                             const std::string & planner = "")
  {
    try {
      std_msgs::msg::String cs;
      cs.data = "grasp " + std::to_string(idx + 1) + "/" + std::to_string(n) +
        ": FEASIBLE (verified in the background via the " + (planner.empty() ? std::string("curve") : planner) +
        ") — executing this one";
      candidate_status_pub_->publish(cs);
    } catch (...) {}
    (void)pose;
  }
  std::atomic<bool> held_object_{false};
  std::chrono::steady_clock::time_point grasp_rise_t_{};
  std::chrono::steady_clock::time_point close_cmd_t_{};
  bool grasp_verdict_holding_ = false;
  // "Holding" = the grasp pin says so AND the gripper is closed. The pin
  // alone would block every pick if it ever stuck HIGH with the fingers
  // open; the gripper joint on /joint_states (0.0 closed, ~0.23 open)
  // settles it. Unknown joint -> trust the pin.
  // "Holding" (Majd 2026-09-11 18:45): only an object THIS executor closed
  // on and has not released since. Pin 16 also goes HIGH when the empty
  // fingers stall on each other, so outside a pick the pin means nothing.
  // Set after a close whose check said holding, cleared by every gripper
  // open the executor sends; a GUI open shows as an open gripper joint.
  bool holding_object()
  {
    if (!check_grasp_sensor_) { return false; }   // without the sensor there is nothing to refuse on
    if (!held_object_.load()) { return false; }
    if (!grasp_detected_.load()) { return false; }
    sensor_msgs::msg::JointState js;
    { std::lock_guard<std::mutex> lk(joint_state_mutex_); js = last_joint_state_; }
    for (size_t i = 0; i < js.name.size() && i < js.position.size(); ++i) {
      if (js.name[i] == "gripper_close_open_joint") { return std::fabs(js.position[i]) < 0.1; }
    }
    return true;
  }

  // Picks start from the reset target ("home"); a background check from
  // anywhere else verifies nothing and its "none feasible" verdict made the
  // pipeline recompute good objects (2026-09-11 17:15, checks run while the
  // arm stood at the drop pose). Joint distance to the named target,
  // 0.05 rad on every joint.
  bool arm_at_reset_target(std::string * why = nullptr)
  {
    std::vector<double> now;
    if (!arm_joints_now(&now)) { if (why) { *why = "no fresh joint state"; } return false; }
    std::map<std::string, double> target;
    try { target = arm_group_.getNamedTargetValues(reset_target_name_); } catch (...) {}
    if (target.empty()) { if (why) { *why = "named target '" + reset_target_name_ + "' unknown"; } return false; }
    const auto names = arm_group_.getJointNames();
    double worst = 0.0;
    for (size_t i = 0; i < names.size() && i < now.size(); ++i) {
      auto it = target.find(names[i]);
      if (it == target.end()) { continue; }
      worst = std::max(worst, std::fabs(now[i] - it->second));
    }
    if (worst > 0.05) {
      if (why) {
        char b[96]; std::snprintf(b, sizeof(b), "arm is %.2f rad from '%s'", worst, reset_target_name_.c_str());
        *why = b;
      }
      return false;
    }
    return true;
  }

  static bool same_pose(
    const geometry_msgs::msg::Pose & a, const geometry_msgs::msg::Pose & b,
    double tol_pos, double tol_q)
  {
    const auto & pa = a.position; const auto & pb = b.position;
    const auto & qa = a.orientation; const auto & qb = b.orientation;
    return std::fabs(pa.x - pb.x) <= tol_pos && std::fabs(pa.y - pb.y) <= tol_pos &&
           std::fabs(pa.z - pb.z) <= tol_pos &&
           std::fabs(qa.x - qb.x) <= tol_q && std::fabs(qa.y - qb.y) <= tol_q &&
           std::fabs(qa.z - qb.z) <= tol_q && std::fabs(qa.w - qb.w) <= tol_q;
  }

  // Same candidate set: same count, every pose within 2 mm / 5e-3 in the
  // quaternion. The pipeline rebuilds the array from the stored grasps
  // with the live transform, so a still arm gives identical numbers.
  static bool same_set(
    const std::vector<geometry_msgs::msg::PoseStamped> & a,
    const std::vector<geometry_msgs::msg::PoseStamped> & b)
  {
    if (a.size() != b.size() || a.empty()) { return false; }
    for (size_t i = 0; i < a.size(); ++i) {
      if (!same_pose(a[i].pose, b[i].pose, 2e-3, 5e-3)) { return false; }
    }
    return true;
  }

  bool select_first_lin_feasible(
    const std::vector<geometry_msgs::msg::PoseStamped> & candidates,
    geometry_msgs::msg::PoseStamped * selected)
  {
    const bool ok = choose_first_lin_feasible(candidates, selected);
    // The winner just passed a FULL Pilz LIN plan from the current start
    // state — execute_grasp's pre-check of the same pose milliseconds
    // later is a duplicate ~7 s plan (measured 2026-08-26). Skip it once.
    skip_next_precheck_ = ok;
    return ok;
  }

  bool execute_grasp(const geometry_msgs::msg::PoseStamped & grasp_pose_input)
  {
    RCLCPP_INFO(node_->get_logger(),
      "Received grasp pose in frame '%s'", grasp_pose_input.header.frame_id.c_str());

    geometry_msgs::msg::PoseStamped grasp_pose_world;
    try {
      grasp_pose_world = tf_buffer_.transform(
        grasp_pose_input,
        target_frame_,
        tf2::durationFromSec(tf_timeout_));

      RCLCPP_INFO(node_->get_logger(),
        "Grasp pose in '%s': [%.3f, %.3f, %.3f]",
        target_frame_.c_str(),
        grasp_pose_world.pose.position.x,
        grasp_pose_world.pose.position.y,
        grasp_pose_world.pose.position.z);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_ERROR(node_->get_logger(),
        "TF transform failed: %s", ex.what());
      return false;
    }

    if (use_current_orientation_if_identity_) {
      const auto & q = grasp_pose_world.pose.orientation;
      const bool identity_like =
        std::abs(q.x) < 1e-3 && std::abs(q.y) < 1e-3 &&
        std::abs(q.z) < 1e-3 && std::abs(q.w - 1.0) < 1e-3;
      if (identity_like) {
        const auto current = arm_group_.getCurrentPose(arm_group_.getEndEffectorLink());
        grasp_pose_world.pose.orientation = current.pose.orientation;
        RCLCPP_INFO(
          node_->get_logger(),
          "Incoming grasp orientation is identity-like; using current EE orientation for planning.");
      }
    }

    // Clear stray perception collision objects before approaching. The
    // pipeline no longer publishes the target's own box, so this has never
    // fired since; it is a blocking service call, hence gated (see ctor).
    if (clear_stray_scene_objects_) {
      moveit::planning_interface::PlanningSceneInterface psi;
      auto obj_ids = psi.getKnownObjectNames();
      // NEVER wipe the static environment (table/wall/mount/...): the
      // planner needs it for every transport move — wiping it sent the
      // arm THROUGH the table on the way to drop right (2026-08-26).
      // Only stale perception boxes are fair game here.
      // "scene_obj_*" are the per-object neighbour boxes the pipeline
      // publishes for the OTHER objects on the table (2026-08-27) — the
      // planner must keep avoiding them, so preserve them as well.
      obj_ids.erase(
        std::remove_if(obj_ids.begin(), obj_ids.end(),
          [](const std::string & oid) {
            return oid == "table" || oid == "wall" ||
                   oid == "mount_surface" || oid == "flower" ||
                   oid == "static_table" ||
                   oid.rfind("scene_obj_", 0) == 0;
          }),
        obj_ids.end());
      if (!obj_ids.empty()) {
        RCLCPP_INFO(node_->get_logger(),
          "Clearing %zu collision objects before grasp approach.", obj_ids.size());
        psi.removeCollisionObjects(obj_ids);
        rclcpp::sleep_for(std::chrono::milliseconds(300));
      }
    }

    // Pre-check LIN feasibility before opening the gripper.
    // This avoids unnecessary gripper motion when arm motion is impossible.
    // Skipped when the pose just won the candidate probe (identical check,
    // identical start state, ~7 s of duplicate Pilz LIN planning).
    if (skip_next_precheck_) {
      skip_next_precheck_ = false;
      RCLCPP_INFO(node_->get_logger(),
        "Pre-check skipped — pose already LIN-verified by the candidate probe.");
    } else {
      std::string reason;
      if (!can_plan_lin_pose(grasp_pose_world, &reason)) {
        RCLCPP_ERROR(
          node_->get_logger(),
          "Pre-check failed: grasp LIN is not feasible from current state (%s). "
          "Skipping gripper open.",
          reason.c_str());
        return false;
      }
      RCLCPP_INFO(node_->get_logger(), "Pre-check: grasp LIN is feasible.");
    }

    // Refresh runtime-tunable params at the START of every pick cycle.
    // intent_router calls SetParameters BEFORE dispatching, but the C++
    // class members are populated only once at construction. Without
    // this re-read, the executor ignores per-pick overrides and the
    // vitamin-tube mode / wait_for_destination / drop_target_name stay
    // at whatever they were when the node started.
    refresh_runtime_params();
    last_failure_reason_.clear();
    cycle_done_note_.clear();
    open_pending_ = false; open_done_ = false; early_close_fired_.store(false);

    // Operator-stop check used between the irreversible-ish steps. Sets
    // the narration reason FIRST so recover_to_reset_target keeps it.
    auto aborted_at = [&](const char * where) {
      if (!abort_requested_.load()) {
        return false;
      }
      last_failure_reason_ = "stopped by your command";
      // INVALIDATION: the operator froze the arm mid-cycle.
      invalidate_cached_lin_plan("operator STOP during the pick");
      // FREEZE in place — no recovery move. "Stop means stop instantly in
      // its position, whatever it is doing" (Majd, 2026-08-12). The
      // operator resumes with go_home / drop commands when ready.
      RCLCPP_WARN(node_->get_logger(),
        "Operator STOP honoured at %s — freezing in place (no recovery move).",
        where);
      slip_monitor_armed_.store(false);
      return true;
    };

    // THINK FIRST, THEN OPEN, THEN MOVE (Majd 2026-08-26: "it should
    // open the gripper and then directly move" — the old order opened
    // the fingers and THEN sat planning). The gripper is IO-driven and
    // not in the arm planning group, so a plan computed now stays valid
    // after the fingers open — and if planning fails, the fingers never
    // open at all.
    moveit::planning_interface::MoveGroupInterface::Plan approach_plan;
    bool approach_planned = false;
    if (!use_pregrasp_staging_) {
      // FIRST: can we simply execute the plan the candidate probe already
      // paid 6-12 s for? (2026-08-28 — this duplicate was ~9.8 s of the
      // wall clock of every pick.) try_reuse_cached_lin_plan refuses
      // unless the pose, the LIVE start state and the planning scene all
      // still agree with what was planned; every refusal names itself.
      std::string reuse_reject;
      if (try_reuse_cached_lin_plan(grasp_pose_world, approach_plan, &reuse_reject)) {
        approach_planned = true;
      } else {
        RCLCPP_WARN(node_->get_logger(),
          "Step 2 - not reusing the probe's plan: %s. Planning fresh.",
          reuse_reject.c_str());
        // INVALIDATION: we rejected it, so nothing downstream may pick it
        // up later on a laxer path.
        invalidate_cached_lin_plan("Step 2 fell back to a fresh plan");
        for (int s2 = 0; s2 < 3 && !approach_planned; ++s2) {
          if (s2 > 0) {
            RCLCPP_WARN(node_->get_logger(),
              "Step 2 pre-plan failed — retry %d/2.", s2);
            rclcpp::sleep_for(std::chrono::milliseconds(50));
          }
          approach_planned = plan_arm_lin_pose(
            grasp_pose_world, "Step 2 - Grasp LIN (planned before open)",
            approach_plan);
        }
      }
      if (!approach_planned) {
        RCLCPP_ERROR(node_->get_logger(),
          "Step 2 planning failed — gripper stays closed.");
        recover_to_reset_target("Step 2 (LIN approach) failed");
        return false;
      }
    }

    // 1) The open goes out with the first arm motion, never before it
    // (send_open_with_motion in every branch below). Whether the fingers
    // really parted is checked by the finger sensor before the close.
    RCLCPP_INFO(node_->get_logger(), "Step 1 - Open gripper: goes out with the approach, not before it.");

    // 2) LIN-only approach to grasp (no PTP retries in this stage).
    if (aborted_at("the approach")) {
      return false;
    }
    bool staged_lin_done = false;
    bool staged_lin_ok = false;
    if (use_pregrasp_staging_ && (staging_mode_ == "blend" || staging_mode_ == "cartesian") &&
        cached_lin_plan_valid_ && !cached_final_plan_valid_) {
      // One blended trajectory (home -> pre-grasp corner -> grasp), from the
      // probe / background check, or planned now when nothing matches.
      moveit::planning_interface::MoveGroupInterface::Plan blended;
      std::string reuse_reject;
      if (!try_reuse_cached_lin_plan(grasp_pose_world, blended, &reuse_reject)) {
        RCLCPP_INFO(node_->get_logger(), "Step 2 - blended two-line plan not reusable (%s); planning now.", reuse_reject.c_str());
        std::string why;
        bool planned = (staging_mode_ == "cartesian") && plan_cartesian_two_lines(grasp_pose_world, &why);
        if (!planned) { planned = plan_blended_two_lines(grasp_pose_world, &why); }
        if (!planned) {
          RCLCPP_ERROR(node_->get_logger(), "Step 2 - one-motion planning failed: %s", why.c_str());
          recover_to_reset_target("Step 2 (blended two lines) planning failed");
          return false;
        }
        blended = cached_lin_plan_;
      }
      staged_lin_done = true;
      if (!execute_) {
        staged_lin_ok = true;
      } else {
        {
          stretch_plan_to_min_duration(blended, approach_floor_s(), "Step 2 - one motion");
          const auto & pts = blended.trajectory_.joint_trajectory.points;
          const double dur = pts.empty() ? 0.0 : rclcpp::Duration(pts.back().time_from_start).seconds();
          send_open_with_motion("Step 2 - one motion");
          // Close on the way in: the pulse goes out close_lead_s before the
          // end — but never before the open is done (gripper_open_time_s +
          // margin after the send) and never while the finger sensor still
          // reads "touching". Fingers not apart by the arrival: the close is
          // left to Step 3, after the usual sensor check and second pulse.
          double pause_s = 0.0;
          try { pause_s = node_->get_parameter("pause_before_close_s").as_double(); } catch (...) {}
          if (close_lead_s_ > 0.0 && pause_s <= 0.0 && !vitamin_mode_active_ && dur > close_lead_s_ + 0.2) {
            if (early_close_thread_.joinable()) { early_close_thread_.join(); }
            using ns = std::chrono::nanoseconds;
            const auto t_start = std::chrono::steady_clock::now();
            const auto open_ready = open_sent_t_ + std::chrono::duration_cast<ns>(
              std::chrono::duration<double>(gripper_open_time_s_ + open_before_arrival_margin_s_));
            const auto lead_at = t_start + std::chrono::duration_cast<ns>(std::chrono::duration<double>(dur - close_lead_s_));
            const auto arrival = t_start + std::chrono::duration_cast<ns>(std::chrono::duration<double>(dur));
            early_close_thread_ = std::thread([this, open_ready, lead_at, arrival]() {
              const auto earliest = std::max(open_ready, lead_at);
              while (std::chrono::steady_clock::now() < earliest && !abort_requested_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
              }
              while (!abort_requested_.load() && std::chrono::steady_clock::now() < arrival) {
                if (!check_grasp_sensor_ || !grasp_detected_.load()) {
                  early_close_fired_.store(true);
                  begin_gripper_close_async();
                  return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
              }
              if (!abort_requested_.load()) {
                RCLCPP_WARN(node_->get_logger(),
                  "Close on the way in skipped: the finger sensor did not read 'apart' before the arrival — closing after the sensor check.");
              }
            });
          }
        }
        const auto t2 = std::chrono::steady_clock::now();
        auto r = execute_with_deadline(blended, __func__);
        if (early_close_thread_.joinable()) { early_close_thread_.join(); }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
        if (r != moveit::core::MoveItErrorCode::SUCCESS) {
          if (close_pending_) { finish_gripper_close("Step 3 - Close gripper (after a failed approach)"); }
          RCLCPP_ERROR(node_->get_logger(), "Step 2 - blended two lines: execution failed: %s (%d)",
            moveit::core::error_code_to_string(r).c_str(), r.val);
        } else {
          RCLCPP_INFO(node_->get_logger(), "Step 2 - blended two lines: execution OK (%.2f s, no stop at the corner).", dt);
          log_arrival_error(grasp_pose_world, "Step 2 - blended two lines");
          staged_lin_ok = true;
        }
      }
    } else if (use_pregrasp_staging_ && (staging_mode_ == "lin" || staging_mode_ == "blend" || staging_mode_ == "cartesian")) {
      // Two straight lines: LIN to the pre-grasp (fingers apart on arrival),
      // then LIN the last stretch along the gripper's z. Both plans come from
      // the probe / background check when they match; otherwise planned now.
      const auto pre_pose = compute_pregrasp_pose(grasp_pose_world, pregrasp_approach_distance_);
      moveit::planning_interface::MoveGroupInterface::Plan pre_plan, final_plan;
      std::string reuse_reject;
      bool have_plans = try_reuse_cached_lin_plan(grasp_pose_world, pre_plan, &reuse_reject) && cached_final_plan_valid_;
      if (have_plans) {
        final_plan = cached_final_plan_;
        RCLCPP_INFO(node_->get_logger(), "Step 2a/2b - both lines reused from the probe.");
      } else {
        RCLCPP_INFO(node_->get_logger(), "Step 2a/2b - planning both lines now (%s).",
          reuse_reject.empty() ? "no final plan held" : reuse_reject.c_str());
        std::string why;
        if (can_reach_via_pregrasp(grasp_pose_world, &why) && !cached_final_plan_valid_) {
          why = "two-line planner kept no plans (staging_mode=" + staging_mode_ + ")";
        }
        if (!cached_lin_plan_valid_ || !cached_final_plan_valid_) {
          RCLCPP_ERROR(node_->get_logger(), "Step 2a/2b planning failed: %s", why.c_str());
          recover_to_reset_target("Step 2a/2b (two lines) planning failed");
          return false;
        }
        pre_plan = cached_lin_plan_;
        final_plan = cached_final_plan_;
      }
      staged_lin_done = true;
      if (!execute_) {
        RCLCPP_INFO(node_->get_logger(), "Step 2a/2b: execute:=false, skipping execution.");
        staged_lin_ok = true;
      } else {
        stretch_plan_to_min_duration(pre_plan, approach_floor_s(), "Step 2a - LIN to pre-grasp");
        RCLCPP_INFO(node_->get_logger(), "Step 2a - LIN to the pre-grasp %.0f cm behind the grasp.",
          pregrasp_approach_distance_ * 100.0);
        send_open_with_motion("Step 2a - LIN to the pre-grasp");
        auto r1 = execute_with_deadline(pre_plan, __func__);
        if (r1 != moveit::core::MoveItErrorCode::SUCCESS) {
          RCLCPP_ERROR(node_->get_logger(), "Step 2a - LIN to pre-grasp: execution failed: %s (%d)",
            moveit::core::error_code_to_string(r1).c_str(), r1.val);
        } else {
          log_arrival_error(pre_pose, "Step 2a - LIN to pre-grasp");
          if (aborted_at("the pre-grasp")) { return false; }
          const auto t2 = std::chrono::steady_clock::now();
          auto r2 = execute_with_deadline(final_plan, __func__);
          const double dt2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
          if (r2 != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "Step 2b - LIN to the grasp: execution failed: %s (%d)",
              moveit::core::error_code_to_string(r2).c_str(), r2.val);
          } else {
            RCLCPP_INFO(node_->get_logger(), "Step 2b - LIN to the grasp: execution OK (%.2f s for the last %.0f cm).",
              dt2, pregrasp_approach_distance_ * 100.0);
            log_arrival_error(grasp_pose_world, "Step 2b - LIN to the grasp");
            staged_lin_ok = true;
          }
        }
      }
    } else if (use_pregrasp_staging_) {
      const auto pre_pose =
        compute_pregrasp_pose(grasp_pose_world, pregrasp_approach_distance_);
      RCLCPP_INFO(node_->get_logger(),
        "Step 2a - PTP to stand-off %.0f cm behind the grasp.",
        pregrasp_approach_distance_ * 100.0);
      send_open_with_motion("Step 2a - PTP to stand-off");
      if (!execute_arm_ptp_pose(pre_pose, "Step 2a - PTP to stand-off")) {
        if (aborted_at("the approach")) {
          return false;
        }
        RCLCPP_ERROR(node_->get_logger(), "Step 2a failed (stand-off PTP).");
        recover_to_reset_target("Step 2a (stand-off PTP) failed");
        return false;
      }
      if (aborted_at("the approach")) {
        return false;
      }
    }
    bool step2_ok = staged_lin_ok;
    if (staged_lin_done) {
      // both lines already ran above
    } else if (approach_planned) {
      // Execute the PRE-COMPUTED plan immediately — no thinking pause.
      if (!execute_) {
        RCLCPP_INFO(node_->get_logger(),
          "Step 2 - Grasp LIN: execute:=false, skipping execution.");
        step2_ok = true;
      } else {
        stretch_plan_to_min_duration(approach_plan, approach_floor_s(), "Step 2 - Grasp LIN");
        send_open_with_motion("Step 2 - Grasp LIN");
        auto exec_result = execute_with_deadline(approach_plan, __func__);
        step2_ok = (exec_result == moveit::core::MoveItErrorCode::SUCCESS);
        if (step2_ok) {
          RCLCPP_INFO(node_->get_logger(), "Step 2 - Grasp LIN: execution OK.");
          log_arrival_error(grasp_pose_world, "Step 2 - Grasp LIN");
        } else {
          RCLCPP_ERROR(node_->get_logger(),
            "Step 2 - Grasp LIN: execution failed: %s (%d)",
            moveit::core::error_code_to_string(exec_result).c_str(),
            exec_result.val);
        }
      }
    } else {
      // Pregrasp-staging path: the LIN start pose only exists after the
      // stand-off PTP, so plan+execute here (3 attempts, flake guard).
      min_exec_duration_s_ = approach_floor_s();
      send_open_with_motion("Step 2 - Grasp LIN (fresh plan)");
      for (int s2 = 0; s2 < 3 && !step2_ok; ++s2) {
        if (s2 > 0) {
          RCLCPP_WARN(node_->get_logger(),
            "Step 2 plan failed — retry %d/2.", s2);
          rclcpp::sleep_for(std::chrono::milliseconds(50));
        }
        step2_ok = execute_arm_lin_pose(grasp_pose_world, "Step 2 - Grasp LIN");
      }
      min_exec_duration_s_ = 0.0;
    }
    if (step2_ok && !approach_planned && !staged_lin_done) {
      log_arrival_error(grasp_pose_world, "Step 2 - Grasp LIN");
    }
    if (!step2_ok) {
      if (aborted_at("the approach")) {
        return false;
      }
      RCLCPP_ERROR(node_->get_logger(), "Step 2 failed (LIN-only mode).");
      recover_to_reset_target("Step 2 (LIN approach) failed");
      return false;
    }

    // 3) Close gripper on object.
    // Vitamin-tube mode uses a shorter first-close (partial grip);
    // the harder re-close happens mid-trajectory during the move to
    // drop_target. See vitamin block further down.
    if (aborted_at("before closing the gripper")) {
      return false;
    }
    if (!finish_gripper_open("Step 1 - Open gripper")) {
      recover_to_reset_target("Step 1 (open gripper) failed");
      return false;
    }
    if (!early_close_fired_.load()) {   // otherwise the sensor was read 'apart' before the close went out
      // The fingers get their full opening time whatever the approach did
      // (a stretched plan, a fallback plan, a cached one): wait out the rest.
      {
        const auto ready_at = open_sent_t_ + std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(gripper_open_time_s_ + open_before_arrival_margin_s_));
        while (rclcpp::ok() && std::chrono::steady_clock::now() < ready_at && !abort_requested_.load()) {
          rclcpp::spin_some(node_);
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
      }
      if (!ensure_fingers_open("Step 1b - Fingers apart?")) {
        recover_to_reset_target("Step 1b: the fingers did not open");
        return false;
      }
    }
    double saved_close_pulse_s = -1.0;
    if (vitamin_mode_active_) {
      RCLCPP_INFO(node_->get_logger(),
        "Vitamin mode: temporarily shortening close pulse to %.2fs for "
        "Step 3 (partial grip).", vitamin_first_close_s_);
      saved_close_pulse_s = set_gripper_adapter_close_pulse(vitamin_first_close_s_);
    }
    // Calibration pause (2026-09-11): with pause_before_close_s > 0 the arm
    // stands at the grasp pose with the gripper open and logs its tool pose,
    // so the fingers can be measured against the object (GraspGen frame vs
    // the printed gripper). Off by default; set live:
    //   ros2 param set /grasp_executor_lin_llm pause_before_close_s 5.0
    try {
      const double pause_s = node_->get_parameter("pause_before_close_s").as_double();
      if (pause_s > 0.0) {
        moveit::core::RobotState st(arm_group_.getRobotModel());
        if (robot_state_from_ear(st)) {
          const Eigen::Isometry3d T = st.getGlobalLinkTransform(arm_group_.getEndEffectorLink());
          RCLCPP_WARN(node_->get_logger(),
            "Calibration pause: %.1f s at the grasp pose, gripper open, TCP [%.3f, %.3f, %.3f] in the "
            "planning frame — measure the fingers against the object now.",
            pause_s, T.translation().x(), T.translation().y(), T.translation().z());
        } else {
          RCLCPP_WARN(node_->get_logger(), "Calibration pause: %.1f s at the grasp pose, gripper open.", pause_s);
        }
        const auto t_end = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(pause_s));
        while (rclcpp::ok() && std::chrono::steady_clock::now() < t_end && !abort_requested_.load()) {
          rclcpp::spin_some(node_);
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (aborted_at("the calibration pause")) { return false; }
      }
    } catch (...) {}
    if (!finish_gripper_close("Step 3 - Close gripper")) {
      // Restore before bailing.
      if (vitamin_mode_active_ && saved_close_pulse_s > 0.0) {
        set_gripper_adapter_close_pulse(saved_close_pulse_s);
      }
      recover_to_reset_target("Step 3 (close gripper) failed");
      return false;
    }
    // Restore the normal close pulse on the adapter; the mid-trajectory
    // re-tighten in vitamin mode applies its own temporary override.
    if (vitamin_mode_active_ && saved_close_pulse_s > 0.0) {
      set_gripper_adapter_close_pulse(saved_close_pulse_s);
    }

    // ── No-grasp safety check ────────────────────────────────────────
    // Sample the digital input shortly after the close pulse finishes
    // so we read the steady-state. If the sensor says nothing is in
    // the gripper, broadcast the lifecycle event (intent_router speaks
    // a warning), then run the safety retreats (3a-up + 3a-back) and
    // go home — same path the operator can trust to clear the table.
    if (check_no_grasp_after_close_ && !vitamin_mode_active_) {
      // Let the DI line settle for ~150 ms after the pulse.
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      // Pump the IO subscriber so the latest pin state is reflected.
      rclcpp::spin_some(node_);
      // Time from the close command to the pin's rise. The pin only says
      // "fingers stalled": on an object they stall EARLY, on each other
      // (empty gripper) LATE, at the end of the stroke (Majd 2026-09-11:
      // "the fingers are touching each other, that is very common").
      // grasp_stall_max_s > 0 turns that into the verdict; 0 keeps the
      // pin alone. Calibrate from the "stall after" values of a few empty
      // and a few real closes in this log.
      double stall_s = -1.0;
      if (grasp_detected_.load() && grasp_rise_t_ >= close_cmd_t_) {
        stall_s = std::chrono::duration<double>(grasp_rise_t_ - close_cmd_t_).count();
      }
      double stall_max = 0.0;
      try { stall_max = node_->get_parameter("grasp_stall_max_s").as_double(); } catch (...) {}
      bool holding = grasp_detected_.load();
      if (holding && stall_max > 0.0 && stall_s > stall_max) {
        holding = false;   // stalled late = fingers met each other, nothing between them
      }
      if (!check_grasp_sensor_) {
        RCLCPP_INFO(node_->get_logger(), "Grasp sensor check is OFF — the object is assumed to be held.");
        holding = true;
      }
      RCLCPP_INFO(node_->get_logger(),
        "Post-close DI check: grasp_pin(TI0)=%d=%s (stall after %.2f s%s), slip_pin(TI1)=%d=%s "
        "(TI1 is informational — gripper hardware self-compensates).",
        grasp_pin_, holding ? "HIGH(holding)" : (grasp_detected_.load() ? "HIGH but LATE(empty)" : "low(empty)"),
        stall_s, stall_max > 0.0 ? (std::string(", limit ") + std::to_string(stall_max).substr(0, 4) + " s").c_str() : "",
        slip_pin_, slip_active_.load() ? "HIGH(slip→hw retighten)" : "low");
      held_object_.store(holding);
      grasp_verdict_holding_ = holding;
      if (!holding) {
        RCLCPP_WARN(node_->get_logger(),
          "No-grasp detected after close (grasp pin %d=LOW). Lifting "
          "clear, then asking before going home (empty gripper).",
          grasp_pin_);
        try {
          std_msgs::msg::String s;
          s.data = "no_grasp";
          interactive_status_pub_->publish(s);
        } catch (...) {}
        // Open the gripper so it's in a known state (we didn't grab
        // anything anyway).
        execute_gripper_named_target(gripper_open_name_,
          "No-grasp recovery - Open gripper");
        rclcpp::sleep_for(std::chrono::milliseconds(150));
        // TRUE-UP clearance jog (Majd 2026-08-26: "let it go up before
        // it goes back home"). Replaces the old base+z 'vertical'
        // retreat + toward-base retreat, which both move HORIZONTALLY
        // on this rig (base +z points at the table).
        if (!execute_fast_joint_lift(lift_height_m_)) {
          RCLCPP_WARN(node_->get_logger(),
            "No-grasp recovery: clearance lift failed — continuing.");
        }
        // Ask before going home in the interactive voice flow (Majd
        // 2026-08-26: "it shouldn't execute that alone — ask 'should I
        // go back home?'"). Reuses the release-stage home-confirm
        // protocol verbatim: router narrates the question; yes/home →
        // 'home', stop/stay → 'stay' on the resume topic. One-shot
        // compound picks (wait_for_destination=false) keep auto-home.
        bool nog_go_home = true;
        if (wait_for_destination_) {
          {
            std::lock_guard<std::mutex> lk(resume_mutex_);
            resume_received_ = false;
            resume_destination_.clear();
          }
          try {
            std_msgs::msg::String s;
            s.data = "awaiting_home_confirm";
            interactive_status_pub_->publish(s);
          } catch (...) {}
          const auto ng_deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::duration<double>(resume_timeout_s_));
          bool ng_got = false;
          while (rclcpp::ok() && std::chrono::steady_clock::now() < ng_deadline) {
            rclcpp::spin_some(node_);
            {
              std::lock_guard<std::mutex> lk(resume_mutex_);
              if (resume_received_) { ng_got = true; break; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
          }
          std::string ng;
          {
            std::lock_guard<std::mutex> lk(resume_mutex_);
            ng = resume_destination_;
            resume_received_ = false;
            resume_destination_.clear();
          }
          for (auto & c : ng) c = std::tolower(c);
          if (ng_got && (ng == "stay" || ng == "stop" || ng == "wait" ||
                         ng == "cancel" || ng == "hold")) {
            RCLCPP_INFO(node_->get_logger(),
              "No-grasp recovery: '%s' → staying put above the object.",
              ng.c_str());
            nog_go_home = false;
          } else if (!ng_got) {
            RCLCPP_WARN(node_->get_logger(),
              "No-grasp recovery: no answer within %.0fs — going home.",
              resume_timeout_s_);
          }
        }
        bool nog_home_ok = true;
        if (nog_go_home) {
          // MUST check the result (Majd 2026-08-28: "when he say 'Going
          // home' he just close the gripper but he doesnt back home").
          // The collision pre-check legitimately REFUSES this move when
          // the arm is still parked at the object it failed to grasp —
          // measured: "waypoint 5/10 would collide (gripper_finger_left
          // <-> scene_obj_small_bottle_0)". The return value used to be
          // DISCARDED, so the gripper then closed while the arm was
          // still down at the table and the operator was told nothing.
          nog_home_ok = execute_arm_named_target(reset_target_name_,
            "No-grasp recovery - PTP home");
          if (!nog_home_ok) {
            // Majd 2026-08-31: "it should back home whatever is
            // happening when it fail — ignore objects collision". The
            // gripper is EMPTY and OPEN and we already lifted clear, so
            // the neighbour boxes (approximations that block their own
            // escape path) may be sacrificed to get home. Delete ONLY
            // scene_obj_* — the TABLE stays in the scene and stays
            // checked. The pipeline republishes fresh boxes on the next
            // selection cycle anyway.
            RCLCPP_WARN(node_->get_logger(),
              "No-grasp recovery: home path blocked by object boxes — "
              "removing scene_obj_* and retrying (empty gripper, "
              "operator-mandated).");
            {
              moveit::planning_interface::PlanningSceneInterface psi;
              auto all_ids = psi.getKnownObjectNames();
              std::vector<std::string> obj_ids;
              for (const auto & oid : all_ids) {
                if (oid.rfind("scene_obj_", 0) == 0) obj_ids.push_back(oid);
              }
              if (!obj_ids.empty()) {
                psi.removeCollisionObjects(obj_ids);
                rclcpp::sleep_for(std::chrono::milliseconds(300));
              }
            }
            nog_home_ok = execute_arm_named_target(reset_target_name_,
              "No-grasp recovery - PTP home (boxes cleared)");
          }
          if (nog_home_ok) {
            // Re-close the gripper at home so it returns to the "ready
            // for next pick" posture instead of being left open.
            execute_gripper_named_target(gripper_close_name_,
              "No-grasp recovery - Close gripper at home");
          } else {
            // Do NOT close: an open gripper parked over the scene is
            // the safe posture, and closing here would pinch whatever
            // the arm is still hovering on.
            RCLCPP_ERROR(node_->get_logger(),
              "No-grasp recovery: the move to '%s' was REFUSED — the arm "
              "is still at the object, gripper left OPEN. Not closing it "
              "here: closing away from home would grip the scene.",
              reset_target_name_.c_str());
          }
        }
        RCLCPP_ERROR(node_->get_logger(),
          "Pick aborted: nothing in the gripper.");
        last_failure_reason_ = nog_home_ok
          ? "the gripper closed but nothing was caught"
          : "the gripper closed but nothing was caught, and I could not "
            "get back home - the path was blocked by an object, so I am "
            "still out here with the gripper open";
        return false;
      }
    }

    // From this point on, the gripper is closed on the object.
    // If any step fails, we attempt a fallback recovery to initgrasp.
    slip_monitor_armed_.store(check_slip_);

    if (stop_after_grasp_) {
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      if (execute_fast_joint_lift(lift_height_m_)) {
        RCLCPP_INFO(node_->get_logger(),
          "stop_after_grasp: fast lift OK.");
      }
      RCLCPP_INFO(node_->get_logger(),
        "stop_after_grasp: object grasped and HELD (lifted) — "
        "manual mode, no transport/drop. Cycle ends here.");
      try {
        std_msgs::msg::String s;
        s.data = "grasped_holding";
        interactive_status_pub_->publish(s);
      } catch (...) {}
      return true;
    }
    bool sequence_ok = true;

    // Clear collision objects — the object is now in the gripper, so
    // perception obstacles are stale and would block the lift.
    bool lifted_after_grasp = false;
    if (lift_height_m_ > 0.0) {
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      lifted_after_grasp = lift_object(grasp_pose_world, "Post-grasp lift");
      if (lifted_after_grasp) {
        RCLCPP_INFO(node_->get_logger(),
          "Post-grasp lift OK — holding clear of the table.");
      } else {
        RCLCPP_WARN(node_->get_logger(),
          "Post-grasp lift failed (jog and LIN); Step 3a will retry after the drop pause.");
      }
    }

    // ─── Interactive drop pause ────────────────────────────────────────
    // After the grasp closes, optionally pause here at the grasp pose
    // and wait for the LLM to deliver a destination via resume_topic_.
    // Voice flow: user says "pick the cup" → executor runs Steps 1-3
    // and pauses. Morty asks "where should I drop it?" The user replies
    // "drop right" → intent_router publishes "drop_right" on
    // resume_topic_. We wake up and continue with lift + drop.
    // For one-shot "pick X and place on Y" intents, wait_for_destination
    // is set to false at goal dispatch and we skip the pause entirely.
    // Compound-plan flags — set while parsing the destination answer,
    // consumed at the release/home stages below (function scope!).
    // Scene objects are cleared AFTER the planner-free lift (2026-09-16,
    // Majd: "a lot of time until it lifts"): the two planning-scene calls
    // and their 300 ms settle used to sit between the close and the lift.
    {
      moveit::planning_interface::PlanningSceneInterface psi;
      auto obj_ids = psi.getKnownObjectNames();
      // NEVER wipe the static environment (table/wall/mount/...): the
      // planner needs it for every transport move — wiping it sent the
      // arm THROUGH the table on the way to drop right (2026-08-26).
      // Only stale perception boxes are fair game here.
      // "scene_obj_*" are the per-object neighbour boxes the pipeline
      // publishes for the OTHER objects on the table (2026-08-27) — the
      // planner must keep avoiding them, so preserve them as well.
      obj_ids.erase(
        std::remove_if(obj_ids.begin(), obj_ids.end(),
          [](const std::string & oid) {
            return oid == "table" || oid == "wall" ||
                   oid == "mount_surface" || oid == "flower" ||
                   oid == "static_table" ||
                   oid.rfind("scene_obj_", 0) == 0;
          }),
        obj_ids.end());
      if (!obj_ids.empty()) {
        RCLCPP_INFO(node_->get_logger(),
          "Clearing %zu collision objects after the lift.", obj_ids.size());
        psi.removeCollisionObjects(obj_ids);
        rclcpp::sleep_for(std::chrono::milliseconds(300));
      }
    }

    // ─── Post-grasp lift (immediate) ───────────────────────────────────
    // The lift belongs to the GRASP, not to the transport (Majd,
    // 2026-08-26: "when I press execute it should go to the object,
    // grasp it and then lift it up"). It used to live after the
    // destination pause, so a manual EXECUTE — or any voice pick before
    // the operator named a destination — sat at table level holding the
    // object. Step 3a below is skipped when this succeeds.
    bool auto_release = false;
    bool auto_home = false;
    if (wait_for_destination_) {
      RCLCPP_INFO(node_->get_logger(),
        "Interactive drop: object grasped, pausing at grasp pose. "
        "Waiting for destination on '%s' (timeout %.1fs).",
        resume_topic_.c_str(), resume_timeout_s_);
      // Reset the wait state BEFORE publishing 'paused' (and before
      // anyone else can publish a resume) so we don't miss a fast
      // reply that lands between publish and wait setup.
      {
        std::lock_guard<std::mutex> lk(resume_mutex_);
        resume_received_ = false;
        resume_destination_.clear();
      }
      // Broadcast the lifecycle event so intent_router (and through it
      // chat_node) can speak the "Where should I drop it?" prompt at
      // exactly the moment the pause begins.
      try {
        std_msgs::msg::String s;
        s.data = "paused";
        interactive_status_pub_->publish(s);
      } catch (...) {}

      // We CANNOT use resume_cv_.wait_until() here: this whole sequence
      // runs on the main thread, the resume subscription is on the same
      // node, and main() spins the node via rclcpp::spin_some() in its
      // outer loop — which is BLOCKED right now because we're inside
      // execute_full_pick_place_sequence. Sleeping on the CV therefore
      // sleeps the spinner too, callbacks never fire, and the resume
      // message sits in the DDS queue forever. Poll-spin instead so
      // the resume subscriber's callback can run while we wait.
      const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(resume_timeout_s_));
      bool got = false;
      while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
        rclcpp::spin_some(node_);
        {
          std::lock_guard<std::mutex> lk(resume_mutex_);
          if (resume_received_) {
            got = true;
            break;
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      std::string dest;
      {
        std::lock_guard<std::mutex> lk(resume_mutex_);
        dest = resume_destination_;
      }

      if (!got) {
        RCLCPP_WARN(node_->get_logger(),
          "Interactive drop: timed out after %.1fs waiting for "
          "destination. Recovering to '%s' and reporting failure.",
          resume_timeout_s_, reset_target_name_.c_str());
        recover_to_reset_target("Interactive drop: destination timeout");
        return false;
      }

      // Normalise destination string.
      std::string dest_lc;
      dest_lc.reserve(dest.size());
      for (char c : dest) dest_lc.push_back(std::tolower(c));
      // Compound plan suffixes (2026-08-25, Majd: "drop right, open the
      // gripper, then go home" in ONE sentence must not pause again):
      // "<dest>+release" = auto-open on arrival, then ask about home;
      // "<dest>+release_home" = auto-open and go straight home.
      auto _strip = [&](const std::string & suf) {
        if (dest_lc.size() > suf.size() &&
            dest_lc.compare(dest_lc.size() - suf.size(), suf.size(), suf) == 0) {
          dest_lc.erase(dest_lc.size() - suf.size());
          return true;
        }
        return false;
      };
      if (_strip("+release_home")) { auto_release = true; auto_home = true; }
      else if (_strip("+release")) { auto_release = true; }

      if (abort_requested_.load() || dest_lc == "abort") {
        RCLCPP_WARN(node_->get_logger(),
          "Interactive drop: operator STOP — freezing at the grasp pose, "
          "gripper as-is, NO move home.");
        last_failure_reason_ = "stopped by your command";
        slip_monitor_armed_.store(false);
        return false;
      }
      // The GUI's Reset pick while the robot holds and waits (Majd
      // 2026-09-15: "reset pick doesn't work" — it froze in the air):
      // reset means "go home with the object, be ready again", not
      // "freeze". Distinct from stop/abort so the voice rule stays.
      if (dest_lc == "reset_home" || dest_lc == "reset") {
        RCLCPP_WARN(node_->get_logger(),
          "Interactive drop: operator RESET — returning home with the "
          "object, gripper as-is; the cycle ends here.");
        last_failure_reason_ = "reset by the operator";
        slip_monitor_armed_.store(false);
        try {
          std_msgs::msg::String s_evt;
          s_evt.data = "reset_home";
          interactive_status_pub_->publish(s_evt);
        } catch (...) {}
        recover_to_reset_target("Operator reset while holding");
        return false;
      }
      if (dest_lc == "stop" || dest_lc == "cancel") {
        RCLCPP_INFO(node_->get_logger(),
          "Interactive drop: keep-holding stop received. Leaving gripper "
          "as-is and moving to '%s'.", reset_target_name_.c_str());
        // Gripper stays closed/whatever — DO NOT open it on stop.
        execute_arm_named_target(reset_target_name_,
          "Interactive drop - stop: PTP to reset");
        return false;
      }

      // Map dest to a named pose. Empty → fall back to the
      // configured drop_target_name_ so old behaviour still works.
      if (dest_lc == "drop_right" || dest_lc == "right" ||
          dest_lc == "drop right" || dest_lc == "drop_right_side") {
        drop_target_name_ = "drop_right_side";
      } else if (dest_lc == "drop_left" || dest_lc == "left" ||
                 dest_lc == "drop left" || dest_lc == "drop_left_side") {
        drop_target_name_ = "drop_left_side";
      } else if (dest_lc == "home") {
        drop_target_name_ = "home";
      } else if (!dest_lc.empty()) {
        RCLCPP_WARN(node_->get_logger(),
          "Interactive drop: unknown destination '%s'; using configured "
          "default '%s'.", dest.c_str(), drop_target_name_.c_str());
      }
      RCLCPP_INFO(node_->get_logger(),
        "Interactive drop: destination set to '%s'. Continuing with "
        "lift + drop sequence.", drop_target_name_.c_str());
      try {
        std_msgs::msg::String s;
        s.data = "resumed:" + drop_target_name_;
        interactive_status_pub_->publish(s);
      } catch (...) {}
    }

    // 3a) Lift object along table +Y using LIN motion.
    // If lift_table_frame is provided, use that frame's +Y axis transformed
    // into target_frame_. Otherwise use yaw-based fallback (lift_table_yaw_deg).
    // Skipped when the object was already lifted right after the grasp —
    // a second 5 cm here would be pure altitude for nothing.
    if (lifted_after_grasp) {
      RCLCPP_INFO(node_->get_logger(),
        "Step 3a - already lifted right after the grasp; skipping.");
    } else if (lift_height_m_ > 0.0) {
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      if (!lift_object(grasp_pose_world, "Step 3a - Lift object")) {
        RCLCPP_WARN(node_->get_logger(),
          "All lift variants failed — CONTINUING WITHOUT LIFT (object "
          "held; transport move will provide the clearance).");
      }
    }

    // Both 3a-up and 3a-back are SAFETY-ONLY retreats. They make the
    // subsequent PTP to home safer, but if LIN can't draw the line
    // (singularity, IK gap, workspace edge) we log and continue. Failing
    // the whole pick over the bonus retreat would be a regression — the
    // grasp already succeeded, the object is in the gripper, and Step 3b
    // (PTP to home) used to be the next move anyway.
    //
    // Give the joint-state monitor a moment to catch up after the lift.
    // Without this small wait, getCurrentPose() returns a stale snapshot
    // and Pilz's LIN planner rejects the start state.
    if (sequence_ok && !lifted_after_grasp &&
        (post_lift_up_m_ > 0.0 || post_lift_back_m_ > 0.0)) {
      rclcpp::sleep_for(std::chrono::milliseconds(150));
    }

    // 3a-up) Extra straight-up retreat in base_link so the next moves
    // start from above the table.
    if (sequence_ok && post_lift_up_m_ > 0.0 && !lifted_after_grasp) {
      // (skipped when the post-grasp TRUE-UP lift succeeded — this
      // base+z retreat is horizontal on this rig and always NO_IK'd,
      // costing ~2.5 s per cycle for nothing.)
      geometry_msgs::msg::PoseStamped current = arm_group_.getCurrentPose(arm_group_.getEndEffectorLink());
      arm_group_.setMaxVelocityScalingFactor(std::min(velocity_scaling_, 0.1));
      arm_group_.setMaxAccelerationScalingFactor(std::min(acceleration_scaling_, 0.1));
      bool up_ok = execute_pose_base_offset_lin(
        current, 0.0, 0.0, post_lift_up_m_, "Step 3a-up - Vertical retreat");
      arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
      arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
      if (!up_ok) {
        // Soft failure: original logic went straight to 3b without this
        // step, so falling back to that is safe.
        RCLCPP_WARN(node_->get_logger(),
          "Step 3a-up failed (likely near wrist singularity or workspace "
          "edge); continuing without vertical retreat.");
      }
    }

    // 3a-back) Horizontal retreat toward the robot base in base_link XY.
    // Direction = unit vector from current TCP XY toward (0,0). For
    // typical mountings, home sits roughly above the base, so moving
    // toward base XY is moving "in the direction of home" without
    // committing to home's full joint config yet.
    // Skipped after a successful true-up lift: post-lift it only burned a
    // 1 s stale-state timeout and then silently did nothing (x=y=0 from
    // the failed pose fetch) — measured 2026-08-26.
    if (sequence_ok && post_lift_back_m_ > 0.0 && !lifted_after_grasp) {
      geometry_msgs::msg::PoseStamped current = arm_group_.getCurrentPose(arm_group_.getEndEffectorLink());
      const double x = current.pose.position.x;
      const double y = current.pose.position.y;
      const double xy_norm = std::sqrt(x * x + y * y);
      if (xy_norm > 1e-6) {
        const double dx = -x / xy_norm * post_lift_back_m_;
        const double dy = -y / xy_norm * post_lift_back_m_;
        RCLCPP_INFO(node_->get_logger(),
          "Step 3a-back - Retreat toward base by %.3f m: d=[%.3f, %.3f, 0.000]",
          post_lift_back_m_, dx, dy);
        arm_group_.setMaxVelocityScalingFactor(std::min(velocity_scaling_, 0.15));
        arm_group_.setMaxAccelerationScalingFactor(std::min(acceleration_scaling_, 0.15));
        bool back_ok = execute_pose_base_offset_lin(
          current, dx, dy, 0.0, "Step 3a-back - Retreat toward base");
        arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
        arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
        if (!back_ok) {
          RCLCPP_WARN(node_->get_logger(),
            "Step 3a-back failed; continuing without horizontal retreat.");
        }
      }
    }

    // 3b+4 BLENDED) One continuous lift -> home -> drop.
    // The canonical route is unchanged (still THROUGH home, never
    // direct-to-drop) — what changes is that it is ONE action goal, so
    // the ~2 s of fixed per-goal overhead is paid once and the arm keeps
    // moving through home instead of decelerating to a full stop there.
    // Any failure falls straight through to the proven two-step sequence
    // below, which starts by going to home from wherever the arm is.
    bool blended_ok = false;
    if (sequence_ok && blend_via_home_ && !skip_reset_before_drop_ &&
        !vitamin_mode_active_ && use_fast_replay_for_named_targets_ &&
        use_named_pose_action_ &&
        // "take it home" makes the drop target BE home; blending home
        // through home is a zero-length second segment that would just
        // stretch a one-leg move over two tier durations.
        drop_target_name_ != reset_target_name_)
    {
      blended_ok = execute_arm_named_target_blended(
        drop_target_name_, {reset_target_name_},
        "Step 3b+4 - Blended transport", "lift");
      if (!blended_ok) {
        RCLCPP_WARN(node_->get_logger(),
          "[blend] blended transport unavailable or refused — falling back "
          "to the two-step '%s' then '%s' sequence.",
          reset_target_name_.c_str(), drop_target_name_.c_str());
      }
    }

    // Wall clock around the two-step path so the blended and unblended
    // totals can be compared from the log alone.
    const bool two_step_ran = sequence_ok && !blended_ok;
    const auto two_step_t0 = std::chrono::steady_clock::now();

    // 3b) Move directly to reset target (initgrasp) before drop sequence.
    // Skipped when skip_reset_before_drop_ is true — the 3a retreats
    // already moved the arm to a safe pose, so going to home first is
    // wasted motion and may even put the arm into a worse position to
    // plan a PTP to drop_right_side from.
    if (sequence_ok && !blended_ok && !skip_reset_before_drop_) {
      if (!execute_arm_named_target(reset_target_name_, "Step 3b - Move to reset target")) {
        sequence_ok = false;
      }
    } else if (sequence_ok && !blended_ok) {
      RCLCPP_INFO(node_->get_logger(),
        "Step 3b skipped (skip_reset_before_drop=true). Going straight "
        "from retreat waypoint to '%s'.", drop_target_name_.c_str());
    }

    // 4) Move to drop_right_side and hold.
    // Vitamin-tube mode: use the special "slow + retighten at 20%"
    // variant. Other objects use the standard execute path.
    if (sequence_ok && !blended_ok) {
      bool step4_ok;
      if (vitamin_mode_active_) {
        step4_ok = execute_arm_to_drop_with_vitamin_retighten(drop_target_name_);
      } else {
        // FAST DIRECT DROP (Majd 2026-08-26: "avoid the table AND very
        // fast"): the collision-checked named-pose action moves from ANY
        // current pose — the same ~2 s channel that takes the arm home
        // in Step 6. The MoveIt PTP round trip (observed 6-12 s
        // client-side wall time for a 3 ms plan) stays as fallback when
        // the server refuses the direct joint path.
        step4_ok = false;
        if (use_fast_replay_for_named_targets_) {
          step4_ok = execute_arm_named_target_fast(drop_target_name_,
            "Step 4 - Move to drop target (fast)");
          if (!step4_ok) {
            RCLCPP_WARN(node_->get_logger(),
              "Step 4 fast path refused/failed — falling back to MoveIt PTP.");
          }
        }
        if (!step4_ok && use_fast_replay_for_named_targets_) {
          // Majd's remembered flow as automatic fallback (2026-08-26):
          // when the direct joint path would clip the table (e.g. the
          // lift was refused and we're still at table level), go UP AND
          // OVER via home — both legs on the fast collision-checked
          // channel. Slower than direct (~4 s) but geometrically always
          // clear, exactly like the old grasp→home→drop behaviour.
          RCLCPP_INFO(node_->get_logger(),
            "Step 4 - direct path refused; trying via home (up and over).");
          if (execute_arm_named_target_fast(reset_target_name_,
                "Step 4a - Via home") &&
              execute_arm_named_target_fast(drop_target_name_,
                "Step 4b - Home to drop")) {
            step4_ok = true;
          }
        }
        if (!step4_ok) {
          step4_ok = execute_arm_named_target(drop_target_name_,
            "Step 4 - Move to drop target", &last_drop_plan_trajectory_);
        }
      }
      if (!step4_ok) {
        sequence_ok = false;
      }
    }

    // Same measurement line for the unblended route, so the researcher can
    // diff the two totals straight out of the log (grep '\[blend\]').
    if (two_step_ran) {
      const double two_step_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - two_step_t0).count();
      RCLCPP_INFO(node_->get_logger(),
        "[blend] two-step lift->%s->%s %s in %.2f s (2 goals, "
        "1 waypoint each).",
        reset_target_name_.c_str(), drop_target_name_.c_str(),
        sequence_ok ? "completed" : "aborted", two_step_s);
    }

    // ─── Interactive release pause ─────────────────────────────────────
    // After arrival at the drop pose, optionally pause again BEFORE
    // opening the gripper. User can say "release" / "open" / "give it
    // to me" to drop, or "stop" / "cancel" to keep holding and go home.
    bool skip_release = false;
    // 2026-08-25 (Majd): after an explicit release, ASK before the
    // surprise move home ("other people don't know that he will move
    // directly back and crash with them"). A compound release_home
    // resume keeps the old immediate behaviour.
    bool ask_home_before_reset = false;
    bool stay_in_place = false;
    // ALREADY HOME ⇒ never ask "Should I go back home?" (Majd 2026-08-28:
    // "why does it say should I go back home when he is already at
    // home?"). When the operator chose HOME as the drop destination the
    // arm released right there, so both the question and the Step 6 move
    // are meaningless. Compared by name — drop_target_name_ is whatever
    // the destination answer resolved to.
    bool already_at_reset_pose = (drop_target_name_ == reset_target_name_);
    if (already_at_reset_pose) {
      RCLCPP_INFO(node_->get_logger(),
        "Drop destination IS the reset pose ('%s') — skipping the "
        "ask-before-home pause and the return move; the arm is already "
        "there.", reset_target_name_.c_str());
    }
    if (sequence_ok && auto_release && execute_) {
      RCLCPP_INFO(node_->get_logger(),
        "Interactive release: PRE-AUTHORIZED by the compound command — "
        "opening on arrival%s.", auto_home ? ", then straight home" : "");
      ask_home_before_reset = !auto_home;
      // v2 (2026-08-31): "released" was published here BEFORE the open
      // was even commanded, so TTS claimed "The gripper is open." while
      // it was still closed (and even when the open later FAILED).
      // "releasing" narrates the intent; "released" now comes from
      // Step 5 after the hardware reports success.
      try {
        std_msgs::msg::String s_evt;
        s_evt.data = "releasing";
        interactive_status_pub_->publish(s_evt);
      } catch (...) {}
    } else if (sequence_ok && wait_for_release_ && execute_) {
      RCLCPP_INFO(node_->get_logger(),
        "Interactive release: arrived at '%s' holding the object. "
        "Waiting for release confirmation on '%s' (timeout %.1fs).",
        drop_target_name_.c_str(), resume_topic_.c_str(),
        resume_timeout_s_);
      // Reset before broadcasting so a fast user reply isn't lost.
      {
        std::lock_guard<std::mutex> lk(resume_mutex_);
        resume_received_ = false;
        resume_destination_.clear();
      }
      try {
        std_msgs::msg::String s;
        s.data = "awaiting_release";
        interactive_status_pub_->publish(s);
      } catch (...) {}

      auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(resume_timeout_s_));
      bool got = false;
      std::string dest;
      while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
        rclcpp::spin_some(node_);
        std::string incoming;
        {
          std::lock_guard<std::mutex> lk(resume_mutex_);
          if (resume_received_) {
            incoming = resume_destination_;
            resume_received_ = false;
            resume_destination_.clear();
          }
        }
        if (!incoming.empty()) {
          std::string inc_lc;
          for (char c : incoming) inc_lc.push_back(std::tolower(c));
          // ── RELEASE-PAUSE v2 (2026-08-31, Majd: "it should be more
          // natural") ──────────────────────────────────────────────
          // A *different* destination now REDIRECTS the held object
          // there (via home — the canonical transport route) and the
          // open/keep question is re-asked on arrival. The destination
          // we are already at is the old stale-echo case (2026-08-26)
          // and is still ignored. Bare "drop"/"drop it" carry no side,
          // so they fall through to the release-synonym branch below
          // (the old prefix filter swallowed them as dead code).
          // F7 (review): compound FAST tokens "drop_left+release[_home]" /
          // "drop_right+release[_home]" used to lose their release half
          // to the redirect scanner (or vanish as a stale echo when
          // already at that pose). Normalize them into the atomic carry
          // form, which also opens IN PLACE when already there.
          if (inc_lc.rfind("drop", 0) == 0 &&
              inc_lc.find("+release") != std::string::npos) {
            if (inc_lc.find("left") != std::string::npos) {
              inc_lc = "carry_release:drop_left_side";
            } else if (inc_lc.find("right") != std::string::npos) {
              inc_lc = "carry_release:drop_right_side";
            }
          }
          // ATOMIC carry+release (v2): "carry_release:<named_target>"
          // moves the held object there (via home) and opens on arrival
          // with NO further question — the operator already said "and
          // give it to me". One token, immune to the single-slot
          // mailbox race that two separate tokens would hit.
          if (inc_lc.rfind("carry_release:", 0) == 0) {
            const std::string tgt =
              inc_lc.substr(std::string("carry_release:").size());
            const bool known = (tgt == reset_target_name_ ||
                                tgt == "drop_left_side" ||
                                tgt == "drop_right_side");
            if (!known) {
              RCLCPP_WARN(node_->get_logger(),
                "carry_release: unknown target '%s' — ignoring.",
                tgt.c_str());
              continue;
            }
            if (tgt == drop_target_name_) {
              // Already there: just open here, no motion needed.
              dest = "release";
              got = true;
              break;
            }
            RCLCPP_INFO(node_->get_logger(),
              "Interactive release: CARRY+RELEASE — taking the object "
              "from '%s' to '%s' (via '%s'), opening on arrival.",
              drop_target_name_.c_str(), tgt.c_str(),
              reset_target_name_.c_str());
            try {
              std_msgs::msg::String s_r;
              s_r.data = "redirect:" + tgt;
              interactive_status_pub_->publish(s_r);
            } catch (...) {}
            bool carry_ok = true;
            bool leg1_done = false;
            if (drop_target_name_ != reset_target_name_) {
              carry_ok = execute_arm_named_target(
                reset_target_name_, "Carry+release - via home (holding)");
              leg1_done = carry_ok;
            }
            if (carry_ok && abort_requested_.load()) {
              // F8: a spoken stop landed between the legs — freeze HERE.
              if (leg1_done) drop_target_name_ = reset_target_name_;
              RCLCPP_WARN(node_->get_logger(),
                "Carry+release: STOP between legs — freezing at '%s', "
                "gripper closed.", drop_target_name_.c_str());
              last_failure_reason_ = "stopped by your command";
              slip_monitor_armed_.store(false);
              try {
                std_msgs::msg::String s_p;
                s_p.data = "resumed:" + drop_target_name_;
                interactive_status_pub_->publish(s_p);
              } catch (...) {}
              try {
                std_msgs::msg::String s_evt;
                s_evt.data = "stopped_holding";
                interactive_status_pub_->publish(s_evt);
              } catch (...) {}
              return false;
            }
            if (carry_ok && tgt != reset_target_name_) {
              carry_ok = execute_arm_named_target(
                tgt, "Carry+release - to target (holding)");
            }
            if (!carry_ok && leg1_done) {
              // F6: leg 1 reached home, leg 2 failed — the ARM IS AT
              // HOME; the books must say so or a later "release" opens
              // "at drop_right" that is really home (or vice versa).
              drop_target_name_ = reset_target_name_;
            }
            if (!carry_ok) {
              RCLCPP_ERROR(node_->get_logger(),
                "Carry+release to '%s' failed/refused — the object "
                "stays held at '%s'; asking again.",
                tgt.c_str(), drop_target_name_.c_str());
              try {
                std_msgs::msg::String s_p;
                s_p.data = "resumed:" + drop_target_name_;
                interactive_status_pub_->publish(s_p);
              } catch (...) {}
              try {
                std_msgs::msg::String s_f;
                s_f.data = "redirect_failed";
                interactive_status_pub_->publish(s_f);
              } catch (...) {}
              try {
                std_msgs::msg::String s_a;
                s_a.data = "awaiting_release";
                interactive_status_pub_->publish(s_a);
              } catch (...) {}
              deadline = std::chrono::steady_clock::now() +
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::duration<double>(resume_timeout_s_));
              continue;
            }
            drop_target_name_ = tgt;
            dest = "release";
            got = true;
            break;
          }
          std::string redirect_target;
          {
            const std::string & t = inc_lc;
            const bool droppy = (t.rfind("drop", 0) == 0);
            if (t == "home" || t == "go_home" || t == "reset") {
              redirect_target = reset_target_name_;
            } else if (t == "left" ||
                       (droppy && t.find("left") != std::string::npos)) {
              redirect_target = "drop_left_side";
            } else if (t == "right" ||
                       (droppy && t.find("right") != std::string::npos)) {
              redirect_target = "drop_right_side";
            }
          }
          if (!redirect_target.empty()) {
            if (redirect_target == drop_target_name_) {
              RCLCPP_INFO(node_->get_logger(),
                "Interactive release: ignoring stale destination echo "
                "'%s' — already at '%s'.",
                incoming.c_str(), drop_target_name_.c_str());
              continue;
            }
            RCLCPP_INFO(node_->get_logger(),
              "Interactive release: REDIRECT while holding — carrying "
              "the object from '%s' to '%s' (via '%s').",
              drop_target_name_.c_str(), redirect_target.c_str(),
              reset_target_name_.c_str());
            try {
              std_msgs::msg::String s_r;
              s_r.data = "redirect:" + redirect_target;
              interactive_status_pub_->publish(s_r);
            } catch (...) {}
            bool move_ok = true;
            bool r_leg1_done = false;
            if (drop_target_name_ != reset_target_name_) {
              move_ok = execute_arm_named_target(
                reset_target_name_, "Redirect - via home (holding)");
              r_leg1_done = move_ok;
            }
            if (move_ok && abort_requested_.load()) {
              // F8: spoken stop between the legs — freeze at home.
              if (r_leg1_done) drop_target_name_ = reset_target_name_;
              RCLCPP_WARN(node_->get_logger(),
                "Redirect: STOP between legs — freezing at '%s', "
                "gripper closed.", drop_target_name_.c_str());
              last_failure_reason_ = "stopped by your command";
              slip_monitor_armed_.store(false);
              try {
                std_msgs::msg::String s_p;
                s_p.data = "resumed:" + drop_target_name_;
                interactive_status_pub_->publish(s_p);
              } catch (...) {}
              try {
                std_msgs::msg::String s_evt;
                s_evt.data = "stopped_holding";
                interactive_status_pub_->publish(s_evt);
              } catch (...) {}
              return false;
            }
            if (move_ok && redirect_target != reset_target_name_) {
              move_ok = execute_arm_named_target(
                redirect_target, "Redirect - to new target (holding)");
            }
            if (move_ok) {
              drop_target_name_ = redirect_target;
            } else {
              if (r_leg1_done) {
                // F6: reached home, second leg failed — books = home.
                drop_target_name_ = reset_target_name_;
              }
              RCLCPP_ERROR(node_->get_logger(),
                "Redirect to '%s' failed/refused — the object stays "
                "held at '%s'; the question stays open.",
                redirect_target.c_str(), drop_target_name_.c_str());
              try {
                std_msgs::msg::String s_f;
                s_f.data = "redirect_failed";
                interactive_status_pub_->publish(s_f);
              } catch (...) {}
            }
            // Authoritative pose AFTER the motion settles (F6): the
            // router books _executor_drop_dest from this, so the
            // re-asked question names where the arm REALLY is.
            try {
              std_msgs::msg::String s_p;
              s_p.data = "resumed:" + drop_target_name_;
              interactive_status_pub_->publish(s_p);
            } catch (...) {}
            // Majd 2026-09-02: "No, go home. I'll open the gripper there."
            // reaches us as TWO tokens (home, then release). If the
            // release answer is ALREADY in the mailbox when the redirect
            // settles, the operator has answered — do not ask again,
            // open. Only exact release synonyms qualify; anything else
            // (another destination, a stray word) keeps the normal
            // re-ask so a stale echo can never open the gripper here.
            if (move_ok) {
              std::string pending;
              {
                std::lock_guard<std::mutex> lk(resume_mutex_);
                if (resume_received_) pending = resume_destination_;
              }
              std::string plc;
              for (char c : pending) plc.push_back(std::tolower(c));
              const bool pending_release =
                (plc == "release" || plc == "open" || plc == "open_gripper" ||
                 plc == "yes" || plc == "give" || plc == "give it to me" ||
                 plc == "release_home" || plc == "release and home");
              if (pending_release) {
                {
                  std::lock_guard<std::mutex> lk(resume_mutex_);
                  resume_received_ = false;
                  resume_destination_.clear();
                }
                RCLCPP_INFO(node_->get_logger(),
                  "Redirect to '%s' settled with release answer '%s' "
                  "already waiting — opening without re-asking.",
                  drop_target_name_.c_str(), pending.c_str());
                dest = pending;
                got = true;
                break;
              }
            }
            try {
              std_msgs::msg::String s_a;
              s_a.data = "awaiting_release";
              interactive_status_pub_->publish(s_a);
            } catch (...) {}
            deadline = std::chrono::steady_clock::now() +
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(resume_timeout_s_));
            continue;
          }
          if (inc_lc == "stop" || inc_lc == "cancel" || inc_lc == "hold") {
            // v2: keep-holding STAYS HERE with the question open. It
            // used to silently carry the object home, which read as
            // disobedience ("Okay, I'll keep holding it" ... then the
            // arm drives off). Timeout still recovers to home, and a
            // spoken stop/abort still freezes via the abort topic.
            RCLCPP_INFO(node_->get_logger(),
              "Interactive release: keep-holding — STAYING at '%s', "
              "pause stays open.", drop_target_name_.c_str());
            try {
              std_msgs::msg::String s_k;
              s_k.data = "keeping";
              interactive_status_pub_->publish(s_k);
            } catch (...) {}
            deadline = std::chrono::steady_clock::now() +
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(resume_timeout_s_));
            continue;
          }
          dest = incoming;
          got = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      if (!got) {
        RCLCPP_WARN(node_->get_logger(),
          "Interactive release: timed out after %.1fs. Recovering to "
          "'%s' WITHOUT releasing the object.",
          resume_timeout_s_, reset_target_name_.c_str());
        skip_release = true;
      } else {
        std::string dest_lc;
        dest_lc.reserve(dest.size());
        for (char c : dest) dest_lc.push_back(std::tolower(c));
        if (abort_requested_.load() || dest_lc == "abort") {
          RCLCPP_WARN(node_->get_logger(),
            "Interactive release: operator STOP — freezing at the drop "
            "pose, gripper stays closed, NO move home.");
          last_failure_reason_ = "stopped by your command";
          slip_monitor_armed_.store(false);
          try {
            std_msgs::msg::String s_evt;
            // v2: distinct from "release_skipped" — the arm FROZE at
            // the drop pose still holding; the router must not book the
            // pose as "home" (it used to, making "where are you?" lie).
            s_evt.data = "stopped_holding";
            interactive_status_pub_->publish(s_evt);
          } catch (...) {}
          return false;
        }
        if (dest_lc == "release_home" ||
                   dest_lc == "release and home") {
          RCLCPP_INFO(node_->get_logger(),
            "Interactive release: compound '%s' → opening gripper, then "
            "straight home (no extra question).", dest.c_str());
        } else if (dest_lc == "release" || dest_lc == "open" ||
                   dest_lc == "open_gripper" || dest_lc == "yes" ||
                   dest_lc == "drop" || dest_lc == "drop it" ||
                   dest_lc == "give" || dest_lc == "give it to me") {
          ask_home_before_reset = true;
          // The ONLY words that open the gripper. Thesis §4.9.5 /
          // abstract: "Explicit user confirmation is required before
          // ... releasing an object."
          RCLCPP_INFO(node_->get_logger(),
            "Interactive release: explicit '%s' received → opening gripper.",
            dest.c_str());
        } else {
          // 2026-08-25 (Majd: "my command was clear 'go home' — and it
          // opened the gripper alone!"): the old default opened on ANY
          // unrecognised word, so "home" dropped the object. Safe
          // default is the opposite: KEEP HOLDING and move to reset.
          RCLCPP_WARN(node_->get_logger(),
            "Interactive release: '%s' is not an explicit release "
            "command — keeping the object HELD and moving to '%s'. "
            "Say 'open'/'release' to drop.",
            dest.c_str(), reset_target_name_.c_str());
          skip_release = true;
        }
      }
      try {
        std_msgs::msg::String s;
        // "releasing" = about to command the open (router narrates
        // "Opening the gripper."). "released" moved to AFTER Step 5
        // reports success — the old pre-action publish made TTS claim
        // an open that had not happened (2026-08-27 review, fixed v2).
        s.data = skip_release ? "release_skipped" : "releasing";
        interactive_status_pub_->publish(s);
      } catch (...) {}
    } else if (sequence_ok && execute_) {
      // Backwards-compat path when wait_for_release is disabled: hold
      // briefly to let inertia settle, then release.
      RCLCPP_INFO(node_->get_logger(), "Holding at '%s' for %.2f seconds.", drop_target_name_.c_str(), drop_hold_s_);
      rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(drop_hold_s_)));
    }

    // A redirect during the release pause may have moved the arm, so the
    // pre-pause "am I at the reset pose" snapshot can be stale (v2).
    already_at_reset_pose = (drop_target_name_ == reset_target_name_);
    // 5) Open gripper at drop pose, then return to initgrasp and close gripper.
    if (sequence_ok && !skip_release) {
      if (execute_gripper_named_target(gripper_open_name_, "Step 5 - Open gripper at drop") &&
          ensure_fingers_open("Step 5b - Object released?")) {
        // Only now is "The gripper is open." true.
        try {
          std_msgs::msg::String s_ok;
          s_ok.data = "released";
          interactive_status_pub_->publish(s_ok);
        } catch (...) {}
        if (execute_ && post_release_wait_s_ > 0.0 && !ask_home_before_reset) {
          RCLCPP_INFO(node_->get_logger(),
            "Step 5c - Released; staying %.1f s at the drop pose before going home.", post_release_wait_s_);
          const auto t_end = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(post_release_wait_s_));
          while (rclcpp::ok() && std::chrono::steady_clock::now() < t_end && !abort_requested_.load()) {
            rclcpp::spin_some(node_);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
          }
        }
      } else {
        try {
          std_msgs::msg::String s_bad;
          s_bad.data = "release_failed";
          interactive_status_pub_->publish(s_bad);
        } catch (...) {}
        sequence_ok = false;
      }
    }
    // ── Ask-before-home pause (2026-08-25) ─────────────────────────
    if (sequence_ok && !skip_release && ask_home_before_reset &&
        !already_at_reset_pose && execute_) {
      rclcpp::sleep_for(std::chrono::seconds(3));
      // 2026-08-25 (Majd's WTF transcript): if the operator ALREADY
      // answered while we were opening ("open the gripper, then go
      // home" split into two intents), the answer sits in the mailbox
      // — consume it instead of wiping it and asking again.
      {
        std::lock_guard<std::mutex> lk(resume_mutex_);
        if (resume_received_) {
          std::string early = resume_destination_;
          for (auto & c : early) c = std::tolower(c);
          resume_received_ = false;
          resume_destination_.clear();
          if (early == "home" || early == "go_home" || early == "yes" ||
              early == "release_home") {
            RCLCPP_INFO(node_->get_logger(),
              "Home-confirm: early '%s' already received — going home "
              "without asking.", early.c_str());
            ask_home_before_reset = false;
          }
        }
      }
    }
    if (sequence_ok && !skip_release && ask_home_before_reset &&
        !already_at_reset_pose && execute_) {
      {
        std::lock_guard<std::mutex> lk(resume_mutex_);
        resume_received_ = false;
        resume_destination_.clear();
      }
      try {
        std_msgs::msg::String s;
        s.data = "awaiting_home_confirm";
        interactive_status_pub_->publish(s);
      } catch (...) {}
      const auto hc_deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(resume_timeout_s_));
      bool hc_got = false;
      while (rclcpp::ok() && std::chrono::steady_clock::now() < hc_deadline) {
        rclcpp::spin_some(node_);
        {
          std::lock_guard<std::mutex> lk(resume_mutex_);
          if (resume_received_) { hc_got = true; break; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      std::string hc;
      {
        std::lock_guard<std::mutex> lk(resume_mutex_);
        hc = resume_destination_;
      }
      for (auto & c : hc) c = std::tolower(c);
      if (hc_got && (hc == "stay" || hc == "stop" || hc == "wait" ||
                     hc == "cancel" || hc == "hold")) {
        RCLCPP_INFO(node_->get_logger(),
          "Home-confirm: '%s' → STAYING at '%s' (no reset move).",
          hc.c_str(), drop_target_name_.c_str());
        stay_in_place = true;
        cycle_done_note_ = std::string("stayed:") + drop_target_name_;
        try {
          std_msgs::msg::String s;
          s.data = std::string("stayed:") + drop_target_name_;
          interactive_status_pub_->publish(s);
        } catch (...) {}
      } else if (!hc_got) {
        RCLCPP_WARN(node_->get_logger(),
          "Home-confirm: no answer within %.0fs — going home.",
          resume_timeout_s_);
      }
    }
    if (sequence_ok && !stay_in_place && !already_at_reset_pose &&
        !execute_arm_named_target(reset_target_name_, "Step 6 - Move to reset target")) {
      sequence_ok = false;
    }
    // Step 7 closes the gripper at home as a "ready for next pick"
    // posture. Skip if we kept the object held — gripper is already
    // closed.
    if (sequence_ok && !skip_release && !stay_in_place && !execute_gripper_named_target(gripper_close_name_, "Step 7 - Close gripper at reset")) {
      sequence_ok = false;
    }

    // Cycle over — the executor no longer drives the arm, so slip must
    // not slow down unrelated later motions (covers the skip_release
    // "go home still holding" path where no open ever fires).
    slip_monitor_armed_.store(false);

    if (!sequence_ok) {
      recover_to_reset_target("Pick-place sequence step failed");
      RCLCPP_ERROR(node_->get_logger(), "Grasp execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Pick-place sequence complete.");
    return true;
  }

  // Latest MoveIt diagnosis (collision pair / joint-limit text),
  // captured from /rosout by the dedicated listener thread and attached
  // to candidate rejections so the operator sees the EXACT cause.
  void note_moveit_hint(const std::string & hint)
  {
    std::lock_guard<std::mutex> lk(hint_mutex_);
    last_hint_ = hint;
    last_hint_t_ = std::chrono::steady_clock::now();
    RCLCPP_INFO(node_->get_logger(), "[hint] captured: %s", hint.c_str());
  }

  std::string fresh_moveit_hint()
  {
    std::lock_guard<std::mutex> lk(hint_mutex_);
    if (last_hint_.empty()) { return ""; }
    if (std::chrono::steady_clock::now() - last_hint_t_ >
        std::chrono::seconds(3)) { return ""; }
    return last_hint_;
  }

  void clear_moveit_hint()
  {
    std::lock_guard<std::mutex> lk(hint_mutex_);
    last_hint_.clear();
    last_hint_t_ = std::chrono::steady_clock::time_point{};
  }

  void request_abort()
  {
    RCLCPP_WARN(node_->get_logger(),
      "ABORT requested by operator — stopping the pick cycle.");
    abort_requested_.store(true);
    // Wake a resume wait so a paused cycle reacts instantly.
    {
      std::lock_guard<std::mutex> lk(resume_mutex_);
      resume_destination_ = "abort";  // freeze-in-place, NOT legacy stop
      resume_received_ = true;
    }
    resume_cv_.notify_all();
    // Halt whatever trajectory MoveIt is executing right now.
    try {
      arm_group_.stop();
    } catch (...) {}
    // INVALIDATION: a stop leaves the arm mid-trajectory, i.e. nowhere
    // near any cached start point, and the operator may drive it by hand
    // before the next command.
    invalidate_cached_lin_plan("operator abort");
  }

  const std::string & last_failure_reason() const { return last_failure_reason_; }
  // v2 (review finding 3): success-side note attached to "done|id|<note>"
  // so the router never has to race two topics to learn the arm STAYED
  // at the drop pose instead of returning home.
  const std::string & cycle_done_note() const { return cycle_done_note_; }
  void set_failure_reason(const std::string & reason) { last_failure_reason_ = reason; }
  // A stale abort must not kill the NEXT cycle — main clears it when a
  // fresh candidate set starts processing.
  void clear_abort() { abort_requested_.store(false); }
  // 2026-09-08: last_failure_reason_ was only cleared inside execute_grasp(),
  // so a cycle that died in the candidate search re-reported the PREVIOUS
  // cycle's reason — "stopped by your command" was spoken twice for
  // "nothing reachable" failures (Majd: "why does it keep saying that").
  void clear_failure_reason() { last_failure_reason_.clear(); }
  bool abort_pending() const { return abort_requested_.load(); }

  // ── Probe-plan cache: the invalidation entry point ────────────────
  // PUBLIC on purpose: main()'s cycle_id watchdog must be able to drop
  // the cache the instant a new selection cycle begins, before any of
  // this class's own code runs. Cheap and idempotent — call it from
  // anywhere the cached start state might have stopped being true. It
  // only logs when something was actually dropped.
  void invalidate_cached_lin_plan(const char * why)
  {
    if (!cached_lin_plan_valid_) {
      return;
    }
    cached_lin_plan_valid_ = false;
    cached_lin_plan_.trajectory_ = moveit_msgs::msg::RobotTrajectory();
    cached_lin_plan_wall_s_ = 0.0;
    cached_final_plan_valid_ = false;
    RCLCPP_INFO(node_->get_logger(), "Probe-plan cache dropped: %s.", why);
  }
  bool has_cached_lin_plan() const { return cached_lin_plan_valid_; }

private:
  void recover_to_reset_target(const std::string & reason)
  {
    // INVALIDATION: recovery moves the arm (or the cycle just died).
    invalidate_cached_lin_plan("recovery to the reset target");
    // Operator STOP outranks recovery: when the abort flag is up, the
    // step "failure" IS the halt we caused — moving home afterwards is
    // exactly what "stop" forbids (Majd, 2026-08-12: "it should stop
    // only"). Freeze; the operator commands the next move.
    if (abort_requested_.load()) {
      if (last_failure_reason_.empty()) {
        last_failure_reason_ = "stopped by your command";
      }
      RCLCPP_WARN(node_->get_logger(),
        "%s — operator STOP active: freezing in place, skipping the "
        "recovery move.", reason.c_str());
      return;
    }
    // Remember WHY the pick died so publish_completion can attach it and
    // Morty can narrate it (Majd: "when it fails it didn't say the reason").
    if (last_failure_reason_.empty()) {
      last_failure_reason_ = reason;
    }
    RCLCPP_WARN(node_->get_logger(),
      "%s. Attempting recovery to '%s'.", reason.c_str(), reset_target_name_.c_str());

    // Clear collision objects that might block the recovery move.
    {
      moveit::planning_interface::PlanningSceneInterface psi;
      auto obj_ids = psi.getKnownObjectNames();
      // NEVER wipe the static environment (table/wall/mount/...): the
      // planner needs it for every transport move — wiping it sent the
      // arm THROUGH the table on the way to drop right (2026-08-26).
      // Only stale perception boxes are fair game here.
      // "scene_obj_*" are the per-object neighbour boxes the pipeline
      // publishes for the OTHER objects on the table (2026-08-27) — the
      // planner must keep avoiding them, so preserve them as well.
      obj_ids.erase(
        std::remove_if(obj_ids.begin(), obj_ids.end(),
          [](const std::string & oid) {
            return oid == "table" || oid == "wall" ||
                   oid == "mount_surface" || oid == "flower" ||
                   oid == "static_table" ||
                   oid.rfind("scene_obj_", 0) == 0;
          }),
        obj_ids.end());
      if (!obj_ids.empty()) {
        RCLCPP_WARN(node_->get_logger(),
          "Recovery: clearing %zu collision objects.", obj_ids.size());
        psi.removeCollisionObjects(obj_ids);
        rclcpp::sleep_for(std::chrono::milliseconds(500));
      }
    }

    // Try PTP first
    if (!execute_arm_named_target(reset_target_name_, "Recovery - PTP to reset target")) {
      RCLCPP_WARN(node_->get_logger(), "PTP recovery failed. Trying OMPL planner.");
      arm_group_.clearPoseTargets();
      arm_group_.setStartStateToCurrentState();
      arm_group_.setPlanningPipelineId("ompl");
      arm_group_.setPlannerId("RRTConnect");
      if (arm_group_.setNamedTarget(reset_target_name_)) {
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = timed_plan(plan, "Recovery - OMPL to reset target");
        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
          RCLCPP_INFO(node_->get_logger(), "Recovery - OMPL plan OK, executing.");
          if (execute_) {
            execute_with_deadline(plan, __func__);
          }
        } else {
          RCLCPP_ERROR(node_->get_logger(),
            "Recovery - OMPL also failed. Robot may need manual recovery.");
        }
      }
    }
  }

  void apply_gripper_adapter_overrides_if_enabled()
  {
    if (!override_gripper_pulse_from_executor_) {
      return;
    }

    std::vector<rclcpp::Parameter> params_to_set;
    params_to_set.emplace_back("backend", "io");
    if (executor_pulse_hold_s_open_ > 0.0) {
      params_to_set.emplace_back("pulse_hold_s_open", executor_pulse_hold_s_open_);
    }
    if (executor_pulse_hold_s_close_ > 0.0) {
      params_to_set.emplace_back("pulse_hold_s_close", executor_pulse_hold_s_close_);
    }
    if (params_to_set.empty()) {
      RCLCPP_WARN(
        node_->get_logger(),
        "override_gripper_pulse_from_executor=true but no valid override values provided "
        "(executor_pulse_hold_s_open/close must be > 0).");
      return;
    }

    auto params_client =
      std::make_shared<rclcpp::AsyncParametersClient>(node_, gripper_adapter_node_name_);
    const auto wait_timeout = std::chrono::duration<double>(gripper_action_timeout_s_);
    if (!params_client->wait_for_service(wait_timeout)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Could not reach gripper adapter param service at '%s'; keeping current pulse timings.",
        gripper_adapter_node_name_.c_str());
      return;
    }

    auto future = params_client->set_parameters(params_to_set);
    if (rclcpp::spin_until_future_complete(node_, future, wait_timeout) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_WARN(node_->get_logger(), "Timed out applying gripper pulse overrides.");
      return;
    }

    const auto results = future.get();
    bool any_failed = false;
    for (size_t i = 0; i < results.size(); ++i) {
      if (!results[i].successful) {
        any_failed = true;
        RCLCPP_WARN(
          node_->get_logger(),
          "Failed to set gripper param '%s': %s",
          params_to_set[i].get_name().c_str(),
          results[i].reason.c_str());
      }
    }
    if (!any_failed) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Applied gripper overrides on '%s' (backend=io, open=%.3fs, close=%.3fs).",
        gripper_adapter_node_name_.c_str(),
        (executor_pulse_hold_s_open_ > 0.0 ? executor_pulse_hold_s_open_ : -1.0),
        (executor_pulse_hold_s_close_ > 0.0 ? executor_pulse_hold_s_close_ : -1.0));

      // Readback verification: confirm the adapter actually accepted the values.
      std::vector<std::string> names_to_read;
      if (executor_pulse_hold_s_open_ > 0.0) names_to_read.push_back("pulse_hold_s_open");
      if (executor_pulse_hold_s_close_ > 0.0) names_to_read.push_back("pulse_hold_s_close");
      if (!names_to_read.empty()) {
        auto get_future = params_client->get_parameters(names_to_read);
        if (rclcpp::spin_until_future_complete(node_, get_future, wait_timeout) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
          const auto readback = get_future.get();
          for (size_t i = 0; i < readback.size(); ++i) {
            double expected = (names_to_read[i] == "pulse_hold_s_open")
              ? executor_pulse_hold_s_open_ : executor_pulse_hold_s_close_;
            double actual = readback[i].as_double();
            if (std::abs(actual - expected) > 1e-6) {
              RCLCPP_WARN(
                node_->get_logger(),
                "Readback mismatch for '%s': expected=%.3f, actual=%.3f. "
                "Retrying set_parameters once.",
                names_to_read[i].c_str(), expected, actual);
              // Retry once.
              auto retry_future = params_client->set_parameters(params_to_set);
              rclcpp::spin_until_future_complete(node_, retry_future, wait_timeout);
              break;
            } else {
              RCLCPP_INFO(
                node_->get_logger(),
                "Readback OK for '%s': %.3fs.", names_to_read[i].c_str(), actual);
            }
          }
        } else {
          RCLCPP_WARN(node_->get_logger(), "Timed out reading back gripper pulse overrides.");
        }
      }
    }
  }

  // Plan-only variant of execute_arm_lin_pose — used to THINK BEFORE the
  // gripper opens so open->move is one continuous motion (2026-08-26).
  bool plan_arm_lin_pose(
    const geometry_msgs::msg::PoseStamped & target_pose,
    const std::string & step_name,
    moveit::planning_interface::MoveGroupInterface::Plan & plan_out)
  {
    // INVALIDATION: a fresh plan supersedes the cached one; keeping both
    // would leave two candidate trajectories for one motion.
    invalidate_cached_lin_plan("a fresh LIN plan superseded it");
    arm_group_.clearPoseTargets();
    set_start_state_from_ear(step_name.c_str());
    arm_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    arm_group_.setPlannerId("LIN");
    if (!arm_group_.setPoseTarget(target_pose, arm_group_.getEndEffectorLink())) {
      RCLCPP_ERROR(node_->get_logger(),
        "%s: setPoseTarget() failed (IK/joint limits).", step_name.c_str());
      return false;
    }
    auto plan_result = timed_plan(plan_out, step_name);
    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: planning failed: %s (%d)",
        step_name.c_str(),
        moveit::core::error_code_to_string(plan_result).c_str(),
        plan_result.val);
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "%s: plan OK.", step_name.c_str());
    return true;
  }

  // START STATE FROM OUR OWN EAR (2026-09-02). setStartStateToCurrentState()
  // asks MoveGroupInterface's CurrentStateMonitor for a state NEWER than
  // now(); nothing spins that monitor while this thread blocks, so every
  // plan paid "Didn't receive robot state ... within 1.000000 seconds" +
  // a 0.1 s retry = 1.1 s dead per plan (2 per cycle in today's logs),
  // then used a 20 ms-old state anyway. We already hold the latest
  // /joint_states in last_joint_state_; ingest it (spin_some) and hand
  // MoveIt an explicit start state instead. Falls back to the monitor
  // if the ear is empty or stale (> 0.5 s), so behaviour can only
  // improve, never regress.
  // RobotState from the executor's own /joint_states subscription. The
  // MoveGroupInterface state monitor is never spun in this single-threaded
  // program (every getCurrentState/getCurrentPose burned its full timeout:
  // "latest received state has time 0.000000" — 1.0 s + 0.1 s per batch in
  // the 2026-09-08 logs). Returns false if the ear is empty/stale.
  bool robot_state_from_ear(moveit::core::RobotState & st, double max_age_s = 0.5)
  {
    rclcpp::spin_some(node_);
    sensor_msgs::msg::JointState js;
    {
      std::lock_guard<std::mutex> lk(joint_state_mutex_);
      js = last_joint_state_;
    }
    if (js.name.empty() || js.position.size() < js.name.size()) { return false; }
    const double age = (node_->now() - rclcpp::Time(js.header.stamp)).seconds();
    if (js.header.stamp.sec != 0 && (age < 0.0 || age > max_age_s)) { return false; }
    const auto model = arm_group_.getRobotModel();
    st.setToDefaultValues();
    size_t applied = 0;
    for (size_t i = 0; i < js.name.size(); ++i) {
      if (model->hasJointModel(js.name[i])) {
        st.setVariablePosition(js.name[i], js.position[i]);
        ++applied;
      }
    }
    if (applied < 6) { return false; }
    st.update();
    return true;
  }

  bool set_start_state_from_ear(const char * who)
  {
    rclcpp::spin_some(node_);
    sensor_msgs::msg::JointState js;
    {
      std::lock_guard<std::mutex> lk(joint_state_mutex_);
      js = last_joint_state_;
    }
    if (js.name.empty() || js.position.size() < js.name.size()) {
      arm_group_.setStartStateToCurrentState();
      return false;
    }
    const double age = (node_->now() - rclcpp::Time(js.header.stamp)).seconds();
    if (js.header.stamp.sec != 0 && (age < 0.0 || age > 0.5)) {
      RCLCPP_WARN(node_->get_logger(),
        "%s: joint ear is %.2f s stale — falling back to the state monitor.",
        who, age);
      arm_group_.setStartStateToCurrentState();
      return false;
    }
    const auto model = arm_group_.getRobotModel();
    moveit::core::RobotState st(model);
    st.setToDefaultValues();
    size_t applied = 0;
    for (size_t i = 0; i < js.name.size(); ++i) {
      if (model->hasJointModel(js.name[i])) {
        st.setVariablePosition(js.name[i], js.position[i]);
        ++applied;
      }
    }
    if (applied < 6) {
      arm_group_.setStartStateToCurrentState();
      return false;
    }
    st.update();
    arm_group_.setStartState(st);
    return true;
  }

  bool execute_arm_lin_pose(const geometry_msgs::msg::PoseStamped & target_pose, const std::string & step_name)
  {
    // INVALIDATION: this call plans and moves on its own.
    invalidate_cached_lin_plan("the arm is running another LIN move");
    arm_group_.clearPoseTargets();
    set_start_state_from_ear(step_name.c_str());
    arm_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    arm_group_.setPlannerId("LIN");
    if (!arm_group_.setPoseTarget(target_pose, arm_group_.getEndEffectorLink())) {
      RCLCPP_ERROR(node_->get_logger(), "%s: setPoseTarget() failed (IK/joint limits).", step_name.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto plan_result = timed_plan(plan, step_name);
    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: planning failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(plan_result).c_str(), plan_result.val);
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "%s: plan OK.", step_name.c_str());
    if (min_exec_duration_s_ > 0.0) {
      stretch_plan_to_min_duration(plan, min_exec_duration_s_, step_name);
    }
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(), "%s: execute:=false, skipping execution.", step_name.c_str());
      return true;
    }

    auto exec_result = execute_with_deadline(plan, __func__);
    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: execution failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(exec_result).c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "%s: execution OK.", step_name.c_str());
    return true;
  }

  // Fast kinematic feasibility check — just runs IK against the current
  // robot state. Returns false if no IK solution exists; this lets us
  // drop unreachable candidates in O(ms) without invoking the planner.
  //
  // We cache the RobotState across one batch evaluation so we don't pay
  // the joint-state-monitor wait cost on every candidate. A short
  // timeout on the first lookup (and a "give up if it never arrives"
  // path) prevents the executor from blocking 25+ seconds on cold start
  // when the state monitor hasn't seen a /joint_states message yet.
  bool has_ik_solution(const geometry_msgs::msg::PoseStamped & target_pose)
  {
    // If we already tried and failed once this batch, don't retry per
    // candidate — that produces ~50 "Failed to fetch current robot state"
    // log lines and ~5s of wasted time when joint_states isn't ready.
    if (ik_seed_unavailable_) {
      return true;
    }
    if (!ik_seed_state_) {
      // Seed from our own ear first (instant). The state monitor never
      // delivered here, so this pre-filter silently passed every candidate
      // ("27/27 passed IK") for weeks (2026-09-08).
      auto ear = std::make_shared<moveit::core::RobotState>(arm_group_.getRobotModel());
      if (robot_state_from_ear(*ear)) {
        ik_seed_state_ = ear;
      } else {
        auto state = arm_group_.getCurrentState(0.1);
        if (!state) {
          ik_seed_unavailable_ = true;  // skip IK probe for the rest of this batch
          return true;
        }
        ik_seed_state_ = state;
      }
    }
    const auto * jmg = ik_seed_state_->getJointModelGroup(arm_group_.getName());
    if (!jmg) {
      return true;
    }
    geometry_msgs::msg::Pose pose = target_pose.pose;
    moveit::core::RobotState probe(*ik_seed_state_);
    return probe.setFromIK(jmg, pose, arm_group_.getEndEffectorLink(), prefilter_ik_timeout_s_);
  }

  void reset_ik_seed_state()
  {
    ik_seed_state_.reset();
    ik_seed_unavailable_ = false;
  }

  // Two-stage reachability: free-form PTP to a stand-off pose behind the
  // grasp, then a SHORT straight-line approach from there. Mirrors what
  // execute_grasp() actually drives, so a candidate that passes here is
  // genuinely executable (2026-08-21).
  // Plan the LIN already set up on arm_group_ (target + start state), and
  // if the planner refuses for the joint acceleration/velocity limits — a
  // SHORT line at full scaling wants the elbow at 10 rad/s^2 (2026-09-16:
  // every 5 cm final stretch failed in 0 ms) — plan again at 1/2, then 1/4
  // of the scaling. Returns the factor that worked (1.0 = full).
  double plan_lin_with_slowdown(
    moveit::planning_interface::MoveGroupInterface::Plan & plan, const std::string & what,
    double * wall_s, moveit::core::MoveItErrorCode * code_out)
  {
    const double factors[3] = {1.0, 0.5, 0.25};
    moveit::core::MoveItErrorCode code = moveit::core::MoveItErrorCode::FAILURE;
    double wall_total = 0.0;
    for (double f : factors) {
      arm_group_.setMaxVelocityScalingFactor(velocity_scaling_ * f);
      arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_ * f);
      double w = 0.0;
      code = timed_plan(plan, f == 1.0 ? what : what + " (scaling x" + std::to_string(f).substr(0, 4) + ")", &w);
      wall_total += w;
      if (code == moveit::core::MoveItErrorCode::SUCCESS) {
        arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
        arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
        if (wall_s) { *wall_s = wall_total; }
        if (code_out) { *code_out = code; }
        return f;
      }
      if (code != moveit::core::MoveItErrorCode::PLANNING_FAILED) { break; }
    }
    arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
    arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
    if (wall_s) { *wall_s = wall_total; }
    if (code_out) { *code_out = code; }
    return 0.0;
  }

  // ONE Cartesian path home -> pre-grasp -> grasp, one time profile over it
  // (TOTG within the joint limits x the scaling). Cached under the grasp
  // pose like a plain line. Retries at 1/2 and 1/4 scaling like the others.
  bool plan_cartesian_two_lines(
    const geometry_msgs::msg::PoseStamped & target_pose, std::string * reason)
  {
    moveit::core::RobotState start_state(arm_group_.getRobotModel());
    if (!robot_state_from_ear(start_state)) { start_state.setToDefaultValues(); }
    arm_group_.setStartState(start_state);
    arm_group_.setPoseReferenceFrame(target_pose.header.frame_id);
    // Shape (2026-09-17 evening, from the recorder: the quadratic arc
    // dipped to 0.08 m/s where it turned into the straight run):
    //   bend  = cubic Bezier from the current TCP to the pre-grasp point
    //           (pregrasp_approach_distance_ behind the grasp), starting
    //           along the chord and ending along the fingers' axis, with a
    //           tangent length of cartesian_curve_lead_m_ so the turn is
    //           spread over that length instead of concentrated;
    //   slide = a genuinely straight last stretch into the grasp.
    // The two join with a continuous tangent, so there is no corner. The
    // wrist finishes its turn by half of the bend, so the joint that hits
    // its limit at the turn has nothing left to do there.
    const std::string ee = arm_group_.getEndEffectorLink();
    Eigen::Isometry3d T0 = start_state.getGlobalLinkTransform(ee);
    if (target_pose.header.frame_id != arm_group_.getPlanningFrame()) {
      try {
        const auto tf = tf_buffer_.lookupTransform(target_pose.header.frame_id, arm_group_.getPlanningFrame(), tf2::TimePointZero);
        T0 = tf2::transformToEigen(tf.transform) * T0;
      } catch (const std::exception & e) {
        if (reason) { *reason = std::string("no TF for the curve: ") + e.what(); }
        return false;
      }
    }
    const auto pre_pose = compute_pregrasp_pose(target_pose, pregrasp_approach_distance_);
    const Eigen::Vector3d P0 = T0.translation();
    const Eigen::Vector3d Ppre(pre_pose.pose.position.x, pre_pose.pose.position.y, pre_pose.pose.position.z);
    const Eigen::Vector3d P2(target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z);
    const Eigen::Vector3d axis = (P2 - Ppre).norm() > 1e-6 ? (P2 - Ppre).normalized() : Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d chord = Ppre - P0;
    const double chord_len = chord.norm();
    if (chord_len < 1e-3) { if (reason) { *reason = "already at the pre-grasp point"; } return false; }
    // Tangent length: the requested lead, but never more than half the chord.
    const double tl = std::min(cartesian_curve_lead_m_, 0.5 * chord_len);
    const Eigen::Vector3d C1 = P0 + chord.normalized() * tl;   // leave along the chord
    const Eigen::Vector3d C2 = Ppre - axis * tl;               // arrive along the fingers' axis
    tf2::Quaternion q0, q2;
    {
      const Eigen::Quaterniond eq0(T0.rotation());
      q0 = tf2::Quaternion(eq0.x(), eq0.y(), eq0.z(), eq0.w());
      const auto & o = target_pose.pose.orientation;
      q2 = tf2::Quaternion(o.x, o.y, o.z, o.w);
    }
    std::vector<geometry_msgs::msg::Pose> waypoints;
    auto push = [&](const Eigen::Vector3d & P, const tf2::Quaternion & q) {
      geometry_msgs::msg::Pose wp;
      wp.position.x = P.x(); wp.position.y = P.y(); wp.position.z = P.z();
      wp.orientation.x = q.x(); wp.orientation.y = q.y(); wp.orientation.z = q.z(); wp.orientation.w = q.w();
      waypoints.push_back(wp);
    };
    // Samples: cartesian_curve_samples_ over the bend (its length is about
    // the chord), the slide at the same spacing.
    const int nb = std::max(8, cartesian_curve_samples_);
    for (int i = 1; i <= nb; ++i) {
      const double u = static_cast<double>(i) / nb, v = 1.0 - u;
      const Eigen::Vector3d P = v * v * v * P0 + 3 * v * v * u * C1 + 3 * v * u * u * C2 + u * u * u * Ppre;
      const double w = std::min(1.0, u / 0.5);   // wrist turn done by half of the bend
      push(P, q0.slerp(q2, w).normalized());
    }
    const int ns = std::max(2, static_cast<int>(std::ceil((P2 - Ppre).norm() / (chord_len / nb))));
    for (int i = 1; i <= ns; ++i) {
      const double u = static_cast<double>(i) / ns;
      push(Ppre + (P2 - Ppre) * u, q2);
    }
    moveit_msgs::msg::RobotTrajectory path_msg;
    const auto t0 = std::chrono::steady_clock::now();
    // Jump test: MoveIt's own is RELATIVE (a step larger than 2x the mean
    // step of the path), and at the bend the joints legitimately move
    // faster than on the straight parts — the post-mortem showed refused
    // points that were reachable and collision-free. Disabled here (0.0)
    // and replaced by an absolute check below: a real configuration flip
    // moves a joint by far more than cartesian_max_joint_step_rad in one
    // 15 mm step.
    const double fraction = arm_group_.computeCartesianPath(
      waypoints, cartesian_eef_step_m_, 0.0, path_msg, true);
    arm_group_.setStartStateToCurrentState();
    if (fraction < 0.999) {
      if (reason) {
        *reason = "cartesian path covers only " + std::to_string(static_cast<int>(fraction * 100.0)) + " % — " +
          curve_autopsy(start_state, path_msg, waypoints, fraction);
      }
      return false;
    }
    {
      // Two tests (2026-09-18: a flip spread over several 15 mm steps, each
      // under the per-step limit, slipped through and the arm went upside
      // down into the bottle — 8 s of motion, upper_arm <-> camera_mount):
      //  (1) per step: one joint moving more than cartesian_max_joint_step_rad;
      //  (2) whole path: one joint travelling more than
      //      cartesian_max_joint_travel_rad in total. An approach from home
      //      never needs a joint to turn more than ~150 deg; a flipped
      //      configuration needs 300 deg or more.
      const auto & jp = path_msg.joint_trajectory.points;
      std::vector<double> travel(jp.empty() ? 0 : jp.front().positions.size(), 0.0);
      for (size_t i = 1; i < jp.size(); ++i) {
        double dmax = 0.0;
        for (size_t k = 0; k < jp[i].positions.size() && k < jp[i - 1].positions.size(); ++k) {
          const double d = std::fabs(jp[i].positions[k] - jp[i - 1].positions[k]);
          dmax = std::max(dmax, d);
          if (k < travel.size()) { travel[k] += d; }
        }
        if (dmax > cartesian_max_joint_step_rad_) {
          if (reason) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "joint flip of %.2f rad in one step at %d %% of the curve", dmax,
                          static_cast<int>(100.0 * i / std::max<size_t>(1, jp.size() - 1)));
            *reason = buf;
          }
          return false;
        }
      }
      for (size_t k = 0; k < travel.size(); ++k) {
        if (travel[k] > cartesian_max_joint_travel_rad_) {
          if (reason) {
            char buf[200];
            std::snprintf(buf, sizeof(buf), "configuration flip along the curve: %s would turn %.0f deg in total (limit %.0f)",
                          k < path_msg.joint_trajectory.joint_names.size() ? path_msg.joint_trajectory.joint_names[k].c_str() : "a joint",
                          travel[k] * 180.0 / M_PI, cartesian_max_joint_travel_rad_ * 180.0 / M_PI);
            *reason = buf;
          }
          return false;
        }
      }
    }
    robot_trajectory::RobotTrajectory rt(arm_group_.getRobotModel(), arm_group_.getName());
    rt.setRobotTrajectoryMsg(start_state, path_msg);
    trajectory_processing::TimeOptimalTrajectoryGeneration totg(cartesian_path_tolerance_rad_, 0.01);
    const double factors[3] = {1.0, 0.5, 0.25};
    for (double f : factors) {
      if (!totg.computeTimeStamps(rt, velocity_scaling_ * f, acceleration_scaling_ * f)) { continue; }
      if (f >= 1.0 && rt.getDuration() > cartesian_max_motion_s_) {
        if (reason) {
          char buf[160];
          std::snprintf(buf, sizeof(buf), "the curve would take %.1f s at full speed (limit %.1f s): a wrong arm configuration, not an approach",
                        rt.getDuration(), cartesian_max_motion_s_);
          *reason = buf;
        }
        return false;
      }
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      rt.getRobotTrajectoryMsg(plan.trajectory_);
      moveit::core::robotStateToRobotStateMsg(start_state, plan.start_state_);
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      plan.planning_time_ = wall;
      cache_lin_plan(target_pose, plan, wall);
      cached_final_plan_valid_ = false;
      const auto & pts = plan.trajectory_.joint_trajectory.points;
      const double dur = pts.empty() ? 0.0 : rclcpp::Duration(pts.back().time_from_start).seconds();
      RCLCPP_INFO(node_->get_logger(),
        "one-motion approach OK: %zu points, %.2f s, bend into a straight %.0f cm slide along the fingers' axis, scaling x%.2f, planned in %.2f s.",
        pts.size(), dur, pregrasp_approach_distance_ * 100.0, f, wall);
      return true;
    }
    if (reason) { *reason = "time parameterisation failed"; }
    return false;
  }

  // ONE blended trajectory for home -> pre-grasp -> grasp (Pilz sequence,
  // LIN + LIN, blend radius at the corner). Retries at 1/2 and 1/4 scaling
  // like the separate lines. On success the trajectory is cached under the
  // GRASP pose (plan reuse and the background record work unchanged).
  // Two straight lines blended at the corner (mode "blend", and the
  // curve's fallback). A "swing" variant (PTP into the slide) was tried
  // and deleted on 2026-09-17: from this home pose the joint path to any
  // pre-grasp sweeps the upper arm through the wall object (0 of 28
  // planned, 1.7 s per candidate).
  bool plan_blended_two_lines(
    const geometry_msgs::msg::PoseStamped & target_pose, std::string * reason)
  {
    if (!sequence_client_ || !sequence_client_->wait_for_service(std::chrono::milliseconds(300))) {
      if (reason) { *reason = "sequence planning service /plan_sequence_path not available"; }
      return false;
    }
    const auto pre_pose = compute_pregrasp_pose(target_pose, pregrasp_approach_distance_);
    const std::string ee = arm_group_.getEndEffectorLink();
    moveit::core::RobotState start_state(arm_group_.getRobotModel());
    if (!robot_state_from_ear(start_state)) { start_state.setToDefaultValues(); }
    moveit_msgs::msg::RobotState start_msg;
    moveit::core::robotStateToRobotStateMsg(start_state, start_msg);

    // Per-line factors (2026-09-16, Majd: "the first LIN is slow"): only the
    // short last stretch needs the slow-down for the joint acceleration
    // limits; the long line keeps its speed as long as it can.
    // {vel line, acc line, vel stretch, acc stretch}: the limit the short
    // stretch violates is the joint ACCELERATION, so its velocity is kept and
    // only its acceleration comes down first — it brakes into the object
    // instead of crawling (Majd: "it stops then moves again").
    // Both lines share one acceleration setting first (Majd 2026-09-16:
    // "get them both synced"): the same ramps on both sides of the corner
    // read as one motion. Mixed settings only when the shared ones fail.
    const double pairs[7][4] = {{1.0, 1.0, 1.0, 1.0}, {1.0, 0.5, 1.0, 0.5}, {1.0, 0.25, 1.0, 0.25},
                                {1.0, 0.25, 0.5, 0.25}, {0.5, 0.25, 0.5, 0.25}, {0.5, 0.5, 0.25, 0.25},
                                {0.25, 0.25, 0.25, 0.25}};
    const auto t0 = std::chrono::steady_clock::now();
    std::string last_err;
    for (const auto & pr : pairs) {
      const double f = pr[2];
      auto req = std::make_shared<moveit_msgs::srv::GetMotionSequence::Request>();
      for (int k = 0; k < 2; ++k) {
        moveit_msgs::msg::MotionSequenceItem item;
        item.req.group_name = arm_group_.getName();
        item.req.pipeline_id = "pilz_industrial_motion_planner";
        item.req.planner_id = "LIN";
        item.req.max_velocity_scaling_factor = velocity_scaling_ * pr[k == 0 ? 0 : 2];
        item.req.max_acceleration_scaling_factor = acceleration_scaling_ * pr[k == 0 ? 1 : 3];
        item.req.allowed_planning_time = std::max(2.0, lin_feasibility_planning_time_);
        if (k == 0) { item.req.start_state = start_msg; }
        item.req.goal_constraints.push_back(
          kinematic_constraints::constructGoalConstraints(ee, k == 0 ? pre_pose : target_pose));
        item.blend_radius = (k == 0) ? blend_radius_m_ : 0.0;
        req->request.items.push_back(item);
      }
      auto fut = sequence_client_->async_send_request(req);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline &&
             fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
      {
        rclcpp::spin_some(node_);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      if (fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        last_err = "sequence planning timed out";
        break;
      }
      const auto resp = fut.get();
      const int code = resp->response.error_code.val;
      if (code == moveit_msgs::msg::MoveItErrorCodes::SUCCESS && !resp->response.planned_trajectories.empty()) {
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        plan.trajectory_ = resp->response.planned_trajectories.front();
        plan.start_state_ = start_msg;
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        plan.planning_time_ = wall;
        cache_lin_plan(target_pose, plan, wall);
        cached_final_plan_valid_ = false;
        const auto & pts = plan.trajectory_.joint_trajectory.points;
        const double dur = pts.empty() ? 0.0 : rclcpp::Duration(pts.back().time_from_start).seconds();
        RCLCPP_INFO(node_->get_logger(),
          "blended two-line plan OK: %zu points, %.2f s of motion, blend %.0f mm, "
          "line vel x%.2f acc x%.2f / last stretch vel x%.2f acc x%.2f, planned in %.2f s.",
          pts.size(), dur, blend_radius_m_ * 1000.0, pr[0], pr[1], pr[2], pr[3], wall);
        return true;
      }
      last_err = "sequence planning failed: " + moveit::core::error_code_to_string(resp->response.error_code) +
                 " (" + std::to_string(code) + ")";
      if (code != moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED) { break; }
    }
    if (reason) { *reason = last_err; }
    return false;
  }

  bool can_reach_via_pregrasp(
    const geometry_msgs::msg::PoseStamped & target_pose,
    std::string * reason = nullptr)
  {
    const auto pre_pose =
      compute_pregrasp_pose(target_pose, pregrasp_approach_distance_);
    const std::string ee = arm_group_.getEndEffectorLink();

    // Stage A — PTP to the stand-off pose (joints free to curve).
    arm_group_.clearPoseTargets();
    arm_group_.setStartStateToCurrentState();
    arm_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    // Stage A is a straight line in every staged mode. Majd 2026-09-17:
    // "we don't need that PTP fallback when the two lines don't work" — a
    // candidate the curve and the two lines cannot reach is rejected and
    // the next one is tried; the arm only ever approaches on a curve or
    // on straight lines. Only the plain "ptp" mode keeps the PTP stand-off.
    // BOTH plans are cached below whatever stage A is: a fallback that
    // cached nothing made a verified candidate get planned again at
    // execution and fail there (2026-09-16 "blend", 2026-09-17 "cartesian").
    const bool lin_staging = (staging_mode_ != "ptp");
    arm_group_.setPlannerId(lin_staging ? "LIN" : "PTP");
    if (!arm_group_.setPoseTarget(pre_pose, ee)) {
      if (reason) { *reason = "LIN planning failed: NO_IK_SOLUTION (-31)"; }
      return false;
    }
    moveit::planning_interface::MoveGroupInterface::Plan pre_plan;
    double wall_a = 0.0;
    moveit::core::MoveItErrorCode pre_result = moveit::core::MoveItErrorCode::FAILURE;
    if (lin_staging) {
      const double f = plan_lin_with_slowdown(pre_plan, "candidate probe stage-A LIN to pre-grasp", &wall_a, &pre_result);
      if (f > 0.0 && f < 1.0) {
        RCLCPP_INFO(node_->get_logger(), "stage-A line planned at x%.2f of the scaling (limits at full speed).", f);
      }
    } else {
      pre_result = timed_plan(pre_plan, "candidate probe stage-A PTP", &wall_a);
    }
    if (pre_result != moveit::core::MoveItErrorCode::SUCCESS) {
      if (reason) {
        *reason = "stand-off unreachable: " +
          moveit::core::error_code_to_string(pre_result) +
          " (" + std::to_string(pre_result.val) + ")";
      }
      return false;
    }

    // Stage B — short LIN from the stand-off state to the grasp.
    const auto & pts = pre_plan.trajectory_.joint_trajectory.points;
    if (!pts.empty()) {
      // Stage B starts where stage A ends. getCurrentState() can never
      // work in this single-threaded program (its monitor needs someone
      // to spin while it waits) — it returned null and the executor
      // crashed on the first staged probe (2026-09-11, exit -11). Seed
      // from our own /joint_states ear, then overwrite the arm joints
      // with the PTP's last point.
      moveit::core::RobotState start_state(arm_group_.getRobotModel());
      if (!robot_state_from_ear(start_state)) { start_state.setToDefaultValues(); }
      start_state.setJointGroupPositions(arm_group_.getName(), pts.back().positions);
      start_state.update();
      arm_group_.setStartState(start_state);
    }
    arm_group_.clearPoseTargets();
    arm_group_.setPlannerId("LIN");
    bool ok = false;
    if (!arm_group_.setPoseTarget(target_pose, ee)) {
      if (reason) { *reason = "LIN planning failed: NO_IK_SOLUTION (-31)"; }
    } else {
      moveit::planning_interface::MoveGroupInterface::Plan lin_plan;
      double wall_b = 0.0;
      moveit::core::MoveItErrorCode lin_result = moveit::core::MoveItErrorCode::FAILURE;
      const double fb = plan_lin_with_slowdown(lin_plan, "candidate probe stage-B LIN", &wall_b, &lin_result);
      if (fb > 0.0 && fb < 1.0) {
        RCLCPP_INFO(node_->get_logger(), "stage-B line (last stretch) planned at x%.2f of the scaling.", fb);
      }
      if (lin_result == moveit::core::MoveItErrorCode::SUCCESS) {
        ok = true;
        // Keyed by the GRASP pose so plan reuse and the background record
        // match as before; the plan itself ends at the pre-grasp point.
        // Cached in every staged mode (PTP stand-off + LIN last stretch in
        // "cartesian"), never left for execution to plan again.
        cache_lin_plan(target_pose, pre_plan, wall_a + wall_b);
        cached_final_plan_ = lin_plan;
        cached_final_plan_valid_ = true;
      } else if (reason) {
        *reason = "LIN planning failed: " +
          moveit::core::error_code_to_string(lin_result) +
          " (" + std::to_string(lin_result.val) + ")";
      }
    }
    arm_group_.setStartStateToCurrentState();
    return ok;
  }

  // Lift the held object clear of the table: the planner-free joint jog
  // first, the LIN fallback the moment it fails. One owner for both call
  // sites (right after the grasp, and the retry after the drop pause).
  // Before 2026-09-17 a refused jog ("IK jumped 0.41 rad") left the object
  // standing on the table until the operator pressed Drop, and only then
  // did the LIN fallback run — "the lift sometimes takes too long".
  bool lift_object(const geometry_msgs::msg::PoseStamped & grasp_pose_world, const char * tag)
  {
    const auto dir = compute_lift_direction_in_target();
    const double dx = dir[0] * lift_height_m_;
    const double dy = dir[1] * lift_height_m_;
    const double dz = dir[2] * lift_height_m_;
    geometry_msgs::msg::PoseStamped lift_pose = grasp_pose_world;
    lift_pose.pose.position.x += dx;
    lift_pose.pose.position.y += dy;
    lift_pose.pose.position.z += dz;
    RCLCPP_INFO(node_->get_logger(),
      "%s - lifting %.3f m along dir=[%.3f, %.3f, %.3f] in %s: "
      "d=[%.3f, %.3f, %.3f] -> [%.3f, %.3f, %.3f]",
      tag, lift_height_m_, dir[0], dir[1], dir[2], target_frame_.c_str(),
      dx, dy, dz,
      lift_pose.pose.position.x, lift_pose.pose.position.y, lift_pose.pose.position.z);
    arm_group_.setMaxVelocityScalingFactor(std::min(velocity_scaling_, 0.1));
    arm_group_.setMaxAccelerationScalingFactor(std::min(acceleration_scaling_, 0.1));
    bool lift_ok = execute_fast_joint_lift(lift_height_m_);
    if (lift_ok) {
      RCLCPP_INFO(node_->get_logger(), "%s - fast joint lift OK (planner-free).", tag);
    } else {
      rclcpp::sleep_for(std::chrono::milliseconds(200));
      const std::string label = std::string(tag) + " (LIN fallback)";
      lift_ok = execute_arm_lin_pose(lift_pose, label);
    }
    arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
    arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
    return lift_ok;
  }

  // Why did the curve stop at `fraction`? computeCartesianPath logs nothing
  // (2026-09-17 night: 73 of 88 curves refused, every one "collision or
  // IK"). From the last joint state it did reach, solve the next waypoint
  // with that state as the seed: no solution = reach or a wrist flip;
  // a solution that /check_state_validity rejects = a named contact; a
  // clean solution = the step itself was refused (joint jump). The
  // validity call costs a service round trip, so it is made at the click
  // only, not inside the background check.
  std::string curve_autopsy(
    const moveit::core::RobotState & start_state,
    const moveit_msgs::msg::RobotTrajectory & partial,
    const std::vector<geometry_msgs::msg::Pose> & waypoints,
    double fraction)
  {
    if (waypoints.empty()) { return "no waypoints"; }
    moveit::core::RobotState st = start_state;
    const auto * jmg = st.getJointModelGroup(arm_group_.getName());
    if (!jmg) { return "no joint model group"; }
    const auto & pts = partial.joint_trajectory.points;
    if (!pts.empty()) {
      for (size_t i = 0; i < partial.joint_trajectory.joint_names.size() && i < pts.back().positions.size(); ++i) {
        st.setVariablePosition(partial.joint_trajectory.joint_names[i], pts.back().positions[i]);
      }
      st.update();
    }
    const size_t k = std::min(waypoints.size() - 1, static_cast<size_t>(fraction * waypoints.size()));
    const int pct = static_cast<int>(fraction * 100.0);
    char buf[256];
    if (!st.setFromIK(jmg, waypoints[k], arm_group_.getEndEffectorLink(), 0.05)) {
      std::snprintf(buf, sizeof(buf), "no joint solution for the point at %d %% (reach edge or a wrist flip)", pct);
      return buf;
    }
    st.update();
    if (precheck_mode_) {
      std::snprintf(buf, sizeof(buf), "a joint solution exists at %d %% (collision or joint jump; detail at the click)", pct);
      return buf;
    }
    std::string why;
    const int v = check_state_valid(st, &why);
    if (v == 0) { std::snprintf(buf, sizeof(buf), "at %d %% the arm %s", pct, why.c_str()); return buf; }
    if (v == 1) { std::snprintf(buf, sizeof(buf), "the point at %d %% is reachable and collision-free — the step to it was refused (joint jump)", pct); return buf; }
    std::snprintf(buf, sizeof(buf), "a joint solution exists at %d %% (%s)", pct, why.c_str());
    return buf;
  }

  // ── Instrumented plan() ───────────────────────────────────────────
  // EVERY plan this executor issues goes through here, so the client-side
  // cost is always printed next to what the server claims it spent. The
  // 2026-08-28 forensic pass found the two differ by ~1900x on a
  // successful Pilz LIN (11.933 s of wall clock against a reported
  // ~0.006 s of "planning time"): the money is in the round trip, not in
  // the planner's own accounting, and until now no log line said so.
  moveit::core::MoveItErrorCode timed_plan(
    moveit::planning_interface::MoveGroupInterface::Plan & plan,
    const std::string & what,
    double * wall_s_out = nullptr)
  {
    const auto t0 = std::chrono::steady_clock::now();
    const auto rc = arm_group_.plan(plan);
    const double wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (wall_s_out) {
      *wall_s_out = wall_s;
    }
    RCLCPP_INFO(
      node_->get_logger(),
      "%s: plan() round trip: %.2f s (server reported %.3f s)%s",
      what.c_str(), wall_s, plan.planning_time_,
      (rc == moveit::core::MoveItErrorCode::SUCCESS) ? "" : " [plan FAILED]");
    return rc;
  }

  // ══ Probe-plan reuse cache ════════════════════════════════════════
  // WHY IT EXISTS: see the ctor comment on reuse_probe_plan_ — the
  // winning candidate's LIN plan was being computed twice, ~9.8 s of
  // waste per pick.
  //
  // WHY IT IS DANGEROUS: a stored plan is a JOINT-SPACE command. If its
  // first point is not where the arm actually stands, the controller
  // closes that gap at trajectory speed — an uncommanded lurch through
  // whatever is in between. Every guard below exists for that single
  // failure mode, and the default answer is always "plan again": a
  // fresh plan costs 11 s, a wrong plan costs a robot.
  void cache_lin_plan(
    const geometry_msgs::msg::PoseStamped & pose,
    const moveit::planning_interface::MoveGroupInterface::Plan & plan,
    double wall_s)
  {
    if (!reuse_probe_plan_) {
      return;
    }
    const auto & jt = plan.trajectory_.joint_trajectory;
    // Without a named, well-formed first point there is nothing we could
    // later prove about the start state, so there is nothing worth
    // keeping either.
    if (jt.points.empty() || jt.joint_names.empty() ||
        jt.points.front().positions.size() != jt.joint_names.size())
    {
      cached_lin_plan_valid_ = false;
      return;
    }
    cached_lin_plan_ = plan;
    cached_lin_plan_pose_ = pose;
    cached_lin_plan_wall_s_ = wall_s;
    cached_lin_plan_t_ = std::chrono::steady_clock::now();
    cached_lin_plan_valid_ = true;
    RCLCPP_INFO(
      node_->get_logger(),
      "Kept the probe's verified LIN plan (%zu points, cost %.2f s) so Step 2 "
      "does not have to recompute it.",
      jt.points.size(), wall_s);
  }

  // Fresh joints from OUR OWN /joint_states ear. Same idiom as the
  // lambda inside execute_fast_joint_lift — getCurrentState() can NEVER
  // work in this single-threaded program (its monitor needs somebody to
  // spin the node while it waits, and nobody does), so WE spin until a
  // NEW message naming every arm joint arrives. Freshness by
  // construction, not by timestamp arithmetic. Deliberately a separate
  // function rather than a refactor of the lift: the lift is a proven
  // safety path and this change does not touch it.
  bool wait_fresh_arm_joint_state(
    double timeout_s, sensor_msgs::msg::JointState & out)
  {
    const std::vector<std::string> arm_joint_names = arm_group_.getJointNames();
    uint64_t last_seen = joint_state_seq_.load();
    const auto t_end = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timeout_s));
    while (rclcpp::ok() && std::chrono::steady_clock::now() < t_end) {
      rclcpp::spin_some(node_);
      const uint64_t s = joint_state_seq_.load();
      if (s != last_seen) {
        last_seen = s;
        sensor_msgs::msg::JointState cand;
        {
          std::lock_guard<std::mutex> lk(joint_state_mutex_);
          cand = last_joint_state_;
        }
        bool has_all = (cand.position.size() >= cand.name.size());
        for (const auto & jn : arm_joint_names) {
          if (std::find(cand.name.begin(), cand.name.end(), jn) == cand.name.end()) {
            has_all = false;
            break;
          }
        }
        if (has_all) {
          out = cand;
          return true;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
  }

  // The RobotState the robot would occupy at trajectory point `idx`.
  // Seeded from the LIVE joint state first so everything outside the arm
  // group (the gripper fingers above all) sits where it really is, then
  // the arm joints are overwritten from the trajectory. Matching is BY
  // NAME throughout — /joint_states ordering is not the trajectory's.
  bool robot_state_at_point(
    const trajectory_msgs::msg::JointTrajectory & jt, size_t idx,
    const sensor_msgs::msg::JointState & live_js,
    moveit::core::RobotState & state_out) const
  {
    if (idx >= jt.points.size()) {
      return false;
    }
    const auto & pt = jt.points[idx];
    if (pt.positions.size() != jt.joint_names.size()) {
      return false;
    }
    state_out.setToDefaultValues();
    for (size_t i = 0; i < live_js.name.size() && i < live_js.position.size(); ++i) {
      if (state_out.getRobotModel()->hasJointModel(live_js.name[i])) {
        state_out.setVariablePosition(live_js.name[i], live_js.position[i]);
      }
    }
    for (size_t i = 0; i < jt.joint_names.size(); ++i) {
      if (!state_out.getRobotModel()->hasJointModel(jt.joint_names[i])) {
        return false;
      }
      state_out.setVariablePosition(jt.joint_names[i], pt.positions[i]);
    }
    state_out.update();
    return true;
  }

  // Ask move_group whether one state is collision-free, reusing the
  // client added for the rejection autopsy (state_validity_client_).
  // Returns 1 = valid, 0 = INVALID (named in *why), -1 = COULD NOT TELL.
  //
  // Note the asymmetry with collision_detail(): there, "could not tell"
  // means stay quiet and carry on, because it is only a diagnostic.
  // Here it is a SAFETY check, so the caller treats -1 exactly like 0
  // and replans. Unverifiable is rejected.
  // Execute a plan with a DEADLINE (2026-09-08). MoveGroupInterface::execute()
  // blocks with no timeout: when the trajectory controller is inactive or the
  // robot is paused (speed scaling 0), move_group ACCEPTS the goal and it never
  // finishes — the main thread wedged for 211 s, ignored every later cycle,
  // did not react to SIGTERM and had to be SIGKILLed ("it doesn't execute at
  // all"). This sends the same ExecuteTrajectory goal move_group uses, spins
  // while waiting, honours the operator abort, and gives up after
  // 2 × planned duration + 5 s (at least 10 s) with a reason that names the
  // likely cause.
  // Where did the arm really stop, against what was commanded? (Majd
  // 2026-09-16: "I think it is mostly the LIN when it arrives".) The TCP
  // from the joint-state ear versus the target, in millimetres, so the
  // arrival error of the straight line is a number in the log, not a guess.
  void log_arrival_error(const geometry_msgs::msg::PoseStamped & target, const std::string & step_name)
  {
    try {
      moveit::core::RobotState st(arm_group_.getRobotModel());
      if (!robot_state_from_ear(st)) { return; }
      const Eigen::Isometry3d T = st.getGlobalLinkTransform(arm_group_.getEndEffectorLink());
      const auto & p = target.pose.position;
      const double dx = (T.translation().x() - p.x) * 1000.0;
      const double dy = (T.translation().y() - p.y) * 1000.0;
      const double dz = (T.translation().z() - p.z) * 1000.0;
      RCLCPP_INFO(node_->get_logger(),
        "[arrival] %s: commanded TCP [%.3f, %.3f, %.3f], reached [%.3f, %.3f, %.3f] (%s) — "
        "error %.1f mm (dx %+.1f, dy %+.1f, dz %+.1f)",
        step_name.c_str(), p.x, p.y, p.z,
        T.translation().x(), T.translation().y(), T.translation().z(), target.header.frame_id.c_str(),
        std::sqrt(dx * dx + dy * dy + dz * dz), dx, dy, dz);
    } catch (...) {}
  }

  // Slow a planned trajectory down to at least min_s: times scale by k,
  // velocities by 1/k, accelerations by 1/k^2, the path itself is untouched.
  void stretch_plan_to_min_duration(
    moveit::planning_interface::MoveGroupInterface::Plan & plan, double min_s, const std::string & step_name)
  {
    auto & pts = plan.trajectory_.joint_trajectory.points;
    if (pts.empty() || min_s <= 0.0) { return; }
    const double dur = rclcpp::Duration(pts.back().time_from_start).seconds();
    if (dur <= 0.0 || dur >= min_s) { return; }
    const double k = min_s / dur;
    for (auto & pt : pts) {
      pt.time_from_start = rclcpp::Duration::from_seconds(rclcpp::Duration(pt.time_from_start).seconds() * k);
      for (auto & v : pt.velocities) { v /= k; }
      for (auto & a : pt.accelerations) { a /= (k * k); }
    }
    RCLCPP_INFO(node_->get_logger(),
      "%s: planned %.2f s, stretched to %.2f s (x%.2f slower): approach duration %.2f s, fingers %.2f + %.2f s.",
      step_name.c_str(), dur, min_s, k, approach_duration_s_, gripper_open_time_s_, open_before_arrival_margin_s_);
  }

  moveit::core::MoveItErrorCode execute_with_deadline(
    const moveit::planning_interface::MoveGroupInterface::Plan & plan, const char * what)
  {
    return execute_with_deadline(plan.trajectory_, what);
  }

  moveit::core::MoveItErrorCode execute_with_deadline(
    const moveit_msgs::msg::RobotTrajectory & trajectory, const char * what)
  {
    using ET = moveit_msgs::action::ExecuteTrajectory;
    if (!exec_traj_client_) {
      exec_traj_client_ = rclcpp_action::create_client<ET>(node_, "/execute_trajectory");
    }
    if (!exec_traj_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(node_->get_logger(), "%s: /execute_trajectory action server not available.", what);
      last_failure_reason_ = "move_group's execute_trajectory action is not available";
      return moveit::core::MoveItErrorCode(moveit_msgs::msg::MoveItErrorCodes::CONTROL_FAILED);
    }
    const auto & pts = trajectory.joint_trajectory.points;
    const double planned_s = pts.empty() ? 0.0 : rclcpp::Duration(pts.back().time_from_start).seconds();
    const double budget_s = std::max(10.0, 2.0 * planned_s + 5.0);
    ET::Goal goal;
    goal.trajectory = trajectory;
    auto gh_fut = exec_traj_client_->async_send_goal(goal);
    auto ready = [](auto & f) { return f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready; };
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    while (rclcpp::ok() && !ready(gh_fut) && elapsed() < 3.0) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!ready(gh_fut) || !gh_fut.get()) {
      RCLCPP_ERROR(node_->get_logger(),
        "%s: execution goal REJECTED or unanswered (%.1f s) — trajectory controller inactive? "
        "(press Play / re-arm; the GUI re-activates it while the program runs)", what, elapsed());
      last_failure_reason_ = "execution goal rejected — trajectory controller inactive";
      return moveit::core::MoveItErrorCode(moveit_msgs::msg::MoveItErrorCodes::CONTROL_FAILED);
    }
    auto gh = gh_fut.get();
    auto res_fut = exec_traj_client_->async_get_result(gh);
    while (rclcpp::ok() && !ready(res_fut)) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (abort_requested_.load()) {
        exec_traj_client_->async_cancel_goal(gh);
        arm_group_.stop();
        RCLCPP_WARN(node_->get_logger(), "%s: execution cancelled by operator abort.", what);
        return moveit::core::MoveItErrorCode(moveit_msgs::msg::MoveItErrorCodes::PREEMPTED);
      }
      if (elapsed() > budget_s) {
        exec_traj_client_->async_cancel_goal(gh);
        arm_group_.stop();
        char buf[200];
        std::snprintf(buf, sizeof(buf),
          "execution stalled: no completion after %.0f s for a %.1f s trajectory — "
          "trajectory controller inactive or robot paused (speed scaling 0)?", elapsed(), planned_s);
        RCLCPP_ERROR(node_->get_logger(), "%s: %s", what, buf);
        last_failure_reason_ = buf;
        return moveit::core::MoveItErrorCode(moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT);
      }
    }
    auto wrapped = res_fut.get();
    if (!wrapped.result) {
      return moveit::core::MoveItErrorCode(moveit_msgs::msg::MoveItErrorCodes::FAILURE);
    }
    return moveit::core::MoveItErrorCode(wrapped.result->error_code);
  }

  // Parallel variant of check_state_valid(): every request goes out
  // before any answer is awaited, so N states cost one round trip.
  // verdicts: 1 valid, 0 in collision, -1 not answered.
  void check_states_valid_parallel(
    std::vector<moveit::core::RobotState> & states,
    std::vector<int> & verdicts, std::vector<std::string> & whys)
  {
    verdicts.assign(states.size(), -1);
    whys.assign(states.size(), "/check_state_validity is not available");
    if (!state_validity_client_ || !state_validity_client_->service_is_ready()) { return; }
    using Fut = rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedFuture;
    std::vector<Fut> futs;
    for (auto & st : states) {
      auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
      moveit::core::robotStateToRobotStateMsg(st, req->robot_state);
      req->group_name = arm_group_.getName();
      futs.push_back(state_validity_client_->async_send_request(req).future.share());
    }
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(cached_plan_validate_timeout_s_));
    auto ready = [&](const Fut & f) { return f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready; };
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline &&
           !std::all_of(futs.begin(), futs.end(), ready)) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    for (size_t k = 0; k < futs.size(); ++k) {
      if (!ready(futs[k])) { whys[k] = "/check_state_validity did not answer in time"; continue; }
      auto resp = futs[k].get();
      if (!resp) { whys[k] = "/check_state_validity returned nothing"; continue; }
      if (resp->valid) { verdicts[k] = 1; whys[k].clear(); continue; }
      verdicts[k] = 0;
      const std::string pairs = format_contacts(*resp, nullptr);
      whys[k] = pairs.empty() ? "state is in collision" : "collides: " + pairs;
    }
  }

  int check_state_valid(moveit::core::RobotState & state, std::string * why)
  {
    if (!state_validity_client_ || !state_validity_client_->service_is_ready()) {
      if (why) { *why = "/check_state_validity is not available"; }
      return -1;
    }
    auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
    moveit::core::robotStateToRobotStateMsg(state, req->robot_state);
    req->group_name = arm_group_.getName();
    auto future = state_validity_client_->async_send_request(req);
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(cached_plan_validate_timeout_s_));
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline &&
           future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
    {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
      // Drop the pending call so a late answer can never be mistaken for
      // the next waypoint's.
      state_validity_client_->remove_pending_request(future);
      if (why) { *why = "/check_state_validity did not answer in time"; }
      return -1;
    }
    auto resp = future.get();
    if (!resp) {
      if (why) { *why = "/check_state_validity returned nothing"; }
      return -1;
    }
    if (resp->valid) {
      return 1;
    }
    if (why) {
      *why = "state is in collision";
      for (const auto & c : resp->contacts) {
        *why = "collides: " + c.contact_body_1 + " <-> " + c.contact_body_2;
        if (c.depth > 0.0) {
          char buf[32];
          std::snprintf(buf, sizeof(buf), " (%.0f mm deep)", c.depth * 1000.0);
          *why += buf;
        }
        break;
      }
    }
    return 0;
  }

  // ── The gate ──────────────────────────────────────────────────────
  // Hand back the cached probe plan ONLY if every one of these holds:
  //   (0) it is not stale,
  //   (a) it was planned for THIS pose, in THIS frame,
  //   (b) the arm is STILL standing on the trajectory's first point,
  //       measured against live /joint_states,
  //   (c) the sampled waypoints (endpoint always included) are still
  //       collision-free in the CURRENT planning scene.
  // Anything else — including "could not check" — returns false with a
  // human-readable reason and the caller plans fresh.
  bool try_reuse_cached_lin_plan(
    const geometry_msgs::msg::PoseStamped & pose,
    moveit::planning_interface::MoveGroupInterface::Plan & plan_out,
    std::string * reject_reason)
  {
    char buf[320];
    auto reject = [&](const std::string & r) {
      if (reject_reason) { *reject_reason = r; }
      return false;
    };
    if (!reuse_probe_plan_) {
      return reject("reuse_probe_plan is disabled");
    }
    if (!cached_lin_plan_valid_) {
      return reject("no cached probe plan is held");
    }

    // WORK ON A PRIVATE COPY. Checks (b) and (c) below spin the node
    // (rclcpp::spin_some) to get fresh joints and to talk to
    // /check_state_validity, and this program is single-threaded: a
    // cycle_id callback can therefore run INSIDE those waits and call
    // invalidate_cached_lin_plan(), which reassigns cached_lin_plan_'s
    // trajectory. A reference into the member would dangle right there.
    // So snapshot everything first, and re-confirm the cache is still
    // valid before handing the plan over.
    const moveit::planning_interface::MoveGroupInterface::Plan candidate_plan =
      cached_lin_plan_;
    const geometry_msgs::msg::PoseStamped cached_pose = cached_lin_plan_pose_;
    const double cached_wall_s = cached_lin_plan_wall_s_;
    const auto cached_t = cached_lin_plan_t_;

    // (0) Staleness.
    const double age_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - cached_t).count();
    if (age_s > cached_plan_max_age_s_) {
      std::snprintf(buf, sizeof(buf),
        "the cached plan is %.1f s old (limit %.1f s)", age_s, cached_plan_max_age_s_);
      return reject(buf);
    }

    // (a) Same pose. The probe sees the candidate in the frame the
    //     pipeline published it in (base_link, live), while Step 2 has
    //     already transformed it into target_frame_. Those are the same
    //     frame in the current cell, but they need not be: reconcile
    //     through the SAME tf_buffer_ execute_grasp used, and refuse if
    //     TF cannot do it. A pose we cannot place is a pose we cannot
    //     claim is the one we planned for.
    geometry_msgs::msg::PoseStamped cached_pose_here = cached_pose;
    if (cached_pose_here.header.frame_id != pose.header.frame_id) {
      try {
        cached_pose_here = tf_buffer_.transform(
          cached_pose, pose.header.frame_id,
          tf2::durationFromSec(tf_timeout_));
      } catch (const tf2::TransformException & ex) {
        return reject(
          std::string("the cached plan's pose is in frame '") +
          cached_pose.header.frame_id + "' and could not be "
          "transformed into '" + pose.header.frame_id + "': " + ex.what());
      }
    }
    double pos_err = 0.0, ori_err = 0.0;
    {
      const auto & a = cached_pose_here.pose;
      const auto & b = pose.pose;
      if (!plan_reuse::pose_matches(
          a.position.x, a.position.y, a.position.z,
          a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w,
          b.position.x, b.position.y, b.position.z,
          b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w,
          cached_plan_pos_tol_m_, cached_plan_ori_tol_rad_, &pos_err, &ori_err))
      {
        std::snprintf(buf, sizeof(buf),
          "it was planned for a different pose (%.2f mm / %.3f deg away; "
          "tol %.2f mm / %.3f deg)",
          pos_err * 1000.0, ori_err * 180.0 / M_PI,
          cached_plan_pos_tol_m_ * 1000.0,
          cached_plan_ori_tol_rad_ * 180.0 / M_PI);
        return reject(buf);
      }
    }

    // (b) Same start state. THIS is the guard that matters: everything
    //     else only decides whether the motion is sensible, this one
    //     decides whether the first thing the controller does is a jump.
    const auto & jt = candidate_plan.trajectory_.joint_trajectory;
    if (jt.points.empty() || jt.joint_names.empty() ||
        jt.points.front().positions.size() != jt.joint_names.size())
    {
      return reject("the cached trajectory has no usable first point");
    }
    sensor_msgs::msg::JointState js;
    if (!wait_fresh_arm_joint_state(1.0, js)) {
      return reject(
        "no fresh /joint_states arrived within 1.0 s, so the start state "
        "cannot be confirmed");
    }
    std::vector<double> live_pos, planned_pos;
    live_pos.reserve(jt.joint_names.size());
    planned_pos.reserve(jt.joint_names.size());
    for (size_t i = 0; i < jt.joint_names.size(); ++i) {
      const auto it = std::find(js.name.begin(), js.name.end(), jt.joint_names[i]);
      if (it == js.name.end()) {
        return reject(
          "joint '" + jt.joint_names[i] + "' is absent from /joint_states");
      }
      const size_t k = static_cast<size_t>(std::distance(js.name.begin(), it));
      if (k >= js.position.size()) {
        return reject("/joint_states is truncated");
      }
      live_pos.push_back(js.position[k]);
      planned_pos.push_back(jt.points.front().positions[i]);
    }
    double joint_err = 0.0;
    size_t worst_i = 0;
    if (!plan_reuse::joints_match(
        live_pos, planned_pos, cached_plan_start_joint_tol_rad_,
        &joint_err, &worst_i))
    {
      std::snprintf(buf, sizeof(buf),
        "the arm has moved since the probe: joint '%s' is %.4f rad off the "
        "cached start point (tol %.3f rad)",
        jt.joint_names[worst_i].c_str(), joint_err,
        cached_plan_start_joint_tol_rad_);
      return reject(buf);
    }

    // (c) Scene still permits the path. Re-check evenly spaced waypoints
    //     with /check_state_validity; the ENDPOINT is always included
    //     because that is where an object that moved onto the target
    //     would show up. This cannot see a swept-volume collision
    //     between samples — but neither could the original plan, whose
    //     scene we are re-testing, and the whole path is at most a few
    //     centimetres of straight-line approach.
    const size_t n = jt.points.size();
    const size_t want = std::min(
      n, static_cast<size_t>(std::max(1, cached_plan_validate_samples_)));
    std::vector<size_t> samples;
    if (want <= 1) {
      samples.push_back(n - 1);
    } else {
      for (size_t s = 0; s < want; ++s) {
        samples.push_back((n - 1) * s / (want - 1));
      }
    }
    samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
    {
      // All sampled waypoints are checked in ONE round trip (2026-09-08:
      // five serial /check_state_validity calls cost ~1.8 s under load —
      // the whole gap between "Pre-check skipped" and Step 2).
      std::vector<moveit::core::RobotState> probes;
      probes.reserve(samples.size());
      for (const size_t idx : samples) {
        moveit::core::RobotState probe_state(arm_group_.getRobotModel());
        if (!robot_state_at_point(jt, idx, js, probe_state)) {
          std::snprintf(buf, sizeof(buf),
            "the robot state at waypoint %zu/%zu could not be rebuilt",
            idx, n - 1);
          return reject(buf);
        }
        probes.push_back(probe_state);
      }
      std::vector<int> verdicts;
      std::vector<std::string> whys;
      check_states_valid_parallel(probes, verdicts, whys);
      for (size_t k = 0; k < samples.size(); ++k) {
        if (verdicts[k] != 1) {
          std::snprintf(buf, sizeof(buf),
            "%s at waypoint %zu/%zu (%s)",
            (verdicts[k] == 0) ? "the planning scene changed"
                               : "the planning scene could not be re-checked",
            samples[k], n - 1, whys[k].c_str());
          return reject(buf);
        }
      }
    }

    // Did anything invalidate the cache WHILE we were checking it? The
    // spins above can run a cycle_id callback or any other invalidation
    // site, and "this plan is stale" must outrank "this plan passed".
    if (!cached_lin_plan_valid_) {
      return reject(
        "the cache was invalidated while it was being checked (a new cycle "
        "or another motion began mid-validation)");
    }

    // Every guard passed. Hand it over and DROP the cache in the same
    // breath: one plan, one execution, never twice.
    // INVALIDATION: consumed. (Cleared inline rather than through
    // invalidate_cached_lin_plan() only so it does not print a "dropped"
    // line immediately under the reuse line that explains it.)
    plan_out = candidate_plan;
    const double saved_s = cached_wall_s;
    const size_t checked = samples.size();
    cached_lin_plan_valid_ = false;
    cached_lin_plan_.trajectory_ = moveit_msgs::msg::RobotTrajectory();
    cached_lin_plan_wall_s_ = 0.0;
    RCLCPP_INFO(
      node_->get_logger(),
      "Step 2 - reusing the probe's verified plan (saved ~%.1f s) "
      "[pose match %.3f mm / %.4f deg; start state within %.4f rad of the "
      "cached first point (tol %.3f); %zu of %zu waypoints re-validated; "
      "plan age %.1f s].",
      saved_s, pos_err * 1000.0, ori_err * 180.0 / M_PI, joint_err,
      cached_plan_start_joint_tol_rad_, checked, n, age_s);
    return true;
  }

  // The background check yields to a click between candidates AND between
  // the stages of one candidate's chain (curve -> blend -> two lines): a
  // chain can cost 2 s on a hard pose, and the click's candidates were
  // waiting behind it (2026-09-17).
  bool precheck_stop_requested()
  {
    if (!precheck_mode_) { return false; }
    rclcpp::spin_some(node_);
    if (precheck_interrupt_ && precheck_interrupt_()) {
      precheck_interrupted_ = true;
      return true;
    }
    return false;
  }

  bool can_plan_lin_pose(const geometry_msgs::msg::PoseStamped & target_pose, std::string * reason = nullptr)
  {
    // INVALIDATION: a new probe supersedes any earlier one. Candidate
    // N-1's trajectory must never survive into candidate N — the poses
    // differ and the guards would then be comparing against the wrong
    // plan. Dropping it here also covers the staging branch below, which
    // plans from the STAND-OFF state and so is never cacheable for
    // Step 2 (Step 2 executes from wherever the arm is now).
    invalidate_cached_lin_plan("a new LIN probe started");
    last_planner_.clear();
    // Apply a tight planning budget for the per-candidate feasibility
    // probe so a single hard pose can't burn 20-30s. Restore the normal
    // budget afterwards so the actual execution plan still has full time.
    arm_group_.setPlanningTime(lin_feasibility_planning_time_);
    arm_group_.setNumPlanningAttempts(lin_feasibility_attempts_);
    if (use_pregrasp_staging_) {
      bool staged_ok = false;
      if (staging_mode_ == "cartesian" || staging_mode_ == "blend") {
        std::string why;
        staged_ok = false;
        if (staging_mode_ == "cartesian") {
          staged_ok = plan_cartesian_two_lines(target_pose, &why);
          if (!staged_ok) {
            RCLCPP_INFO(node_->get_logger(), "one-motion approach not planned (%s) — trying the blend.", why.c_str());
          } else {
            last_planner_ = "curve";
          }
        }
        // A click outranks the rest of this chain (background check only).
        auto yield_to_selection = [&]() {
          if (!precheck_stop_requested()) { return false; }
          if (reason) { *reason = "background check interrupted by a selection"; }
          return true;
        };
        // The chain (Majd 2026-09-17): curve, blended lines, reject. The two
        // separate lines are out (they need the same straight line the blend
        // just failed to draw: 1 rescue in 85, at a crawl); the swing was
        // deleted (0 of 28, 1.7 s per candidate); no PTP anywhere.
        if (!staged_ok && !yield_to_selection()) {
          staged_ok = plan_blended_two_lines(target_pose, &why);
          if (!staged_ok) { RCLCPP_INFO(node_->get_logger(), "blend not planned (%s) — candidate rejected.", why.c_str()); }
          else { last_planner_ = "blended lines"; }
        }
        if (!staged_ok && reason && reason->empty()) { *reason = why; }
      } else {
        staged_ok = can_reach_via_pregrasp(target_pose, reason);
        if (staged_ok) { last_planner_ = "two lines"; }
      }
      arm_group_.setPlanningTime(planning_time_);
      arm_group_.setNumPlanningAttempts(num_attempts_);
      arm_group_.setStartStateToCurrentState();
      return staged_ok;
    }
    arm_group_.clearPoseTargets();
    arm_group_.setStartStateToCurrentState();
    arm_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    arm_group_.setPlannerId("LIN");
    bool ok = false;
    if (!arm_group_.setPoseTarget(target_pose, arm_group_.getEndEffectorLink())) {
      if (reason) {
        *reason = "setPoseTarget failed (IK/joint limits)";
      }
    } else {
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      double wall_s = 0.0;
      auto plan_result = timed_plan(plan, "candidate LIN probe", &wall_s);
      if (plan_result != moveit::core::MoveItErrorCode::SUCCESS &&
          plan_result == moveit::core::MoveItErrorCode::PLANNING_FAILED &&
          lin_slow_retry_factor_ > 0.0 && lin_slow_retry_factor_ < 1.0)
      {
        // Pilz refuses a line whose joint acceleration or speed would
        // exceed a limit at the requested scaling — at 50 % that was the
        // elbow at the start of the line for 2 of 9 picks on 2026-09-11,
        // the same object at the same place passing on the next try only
        // because GraspGen had sampled a slightly different grasp. Its own
        // hint is "set the scaling factor lower", so do that once for THIS
        // candidate: the kept plan then runs this one pick slower instead
        // of refusing it.
        arm_group_.setMaxVelocityScalingFactor(velocity_scaling_ * lin_slow_retry_factor_);
        arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_ * lin_slow_retry_factor_);
        double wall2 = 0.0;
        const auto slow = timed_plan(plan, "candidate LIN probe (slower retry)", &wall2);
        arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
        arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
        if (slow == moveit::core::MoveItErrorCode::SUCCESS) {
          RCLCPP_INFO(node_->get_logger(),
            "LIN refused at %.0f %% but planned at %.0f %% — this pick runs slower.",
            velocity_scaling_ * 100.0, velocity_scaling_ * lin_slow_retry_factor_ * 100.0);
          plan_result = slow;
          wall_s += wall2;
        }
      }
      if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
        ok = true;
        // KEEP IT (2026-08-28). This trajectory is planned from the
        // CURRENT state to THIS pose with the Pilz LIN planner — bit for
        // bit the artefact Step 2 used to spend another 6-12 s asking
        // move_group to recompute. try_reuse_cached_lin_plan() decides
        // later whether it is still safe to execute.
        cache_lin_plan(target_pose, plan, wall_s);
      } else if (reason) {
        *reason = "LIN planning failed: " + moveit::core::error_code_to_string(plan_result) +
                  " (" + std::to_string(plan_result.val) + ")";
      }
    }
    arm_group_.setPlanningTime(planning_time_);
    arm_group_.setNumPlanningAttempts(num_attempts_);
    return ok;
  }

  // ── Collision autopsy ─────────────────────────────────────────────
  // Majd, 2026-08-28: "if there is a collision then I need to know the
  // exact collision and with what". Pilz LIN only ever hands back
  // PLANNING_FAILED(-1) / INVALID_MOTION_PLAN(-2) — it never names the
  // obstacle. So ask MoveIt outright instead of scraping /rosout: solve
  // IK for the goal pose (seeded from the LIVE joints, exactly the way
  // execute_fast_joint_lift does it — getCurrentState() can never work
  // in this single-threaded program, see there), then hand that
  // RobotState to /check_state_validity and read the contact pairs back
  // BY NAME ("gripper_finger_left <-> scene_obj_vitamin_c_1").
  //
  // Returns "" whenever there is nothing trustworthy to say:
  //   * IK failed            — the caller's own NO_IK message covers it
  //   * state reported valid — the goal itself is clear, so something
  //                            MID-PATH blocks it (*goal_state_clear)
  //   * service missing/slow — never block a pick for a diagnostic
  // *blocker_out receives the environment side of the first contact
  // (the scene object), for the spoken "kept hitting X" summary.
  std::string collision_detail(
    const geometry_msgs::msg::PoseStamped & pose,
    std::string * blocker_out = nullptr,
    bool * goal_state_clear = nullptr)
  {
    if (blocker_out) { blocker_out->clear(); }
    if (goal_state_clear) { *goal_state_clear = false; }
    if (!state_validity_client_) { return ""; }

    // (a) Live joints from OUR OWN /joint_states ear (joint_states_sub_).
    const std::vector<std::string> arm_joint_names = arm_group_.getJointNames();
    auto has_all_arm_joints = [&](const sensor_msgs::msg::JointState & s) {
      if (s.name.empty() || s.position.size() < s.name.size()) { return false; }
      for (const auto & jn : arm_joint_names) {
        if (std::find(s.name.begin(), s.name.end(), jn) == s.name.end()) {
          return false;
        }
      }
      return true;
    };
    sensor_msgs::msg::JointState js;
    {
      std::lock_guard<std::mutex> lk(joint_state_mutex_);
      js = last_joint_state_;
    }
    if (!has_all_arm_joints(js)) {
      // Cold cache only: pump our own callbacks briefly, then give up.
      const auto js_deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
      while (rclcpp::ok() && std::chrono::steady_clock::now() < js_deadline) {
        rclcpp::spin_some(node_);
        {
          std::lock_guard<std::mutex> lk(joint_state_mutex_);
          js = last_joint_state_;
        }
        if (has_all_arm_joints(js)) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      if (!has_all_arm_joints(js)) { return ""; }
    }

    moveit::core::RobotState state(arm_group_.getRobotModel());
    state.setToDefaultValues();
    for (size_t i = 0; i < js.name.size() && i < js.position.size(); ++i) {
      if (state.getRobotModel()->hasJointModel(js.name[i])) {
        state.setVariablePosition(js.name[i], js.position[i]);
      }
    }
    state.update();
    const auto * jmg = state.getJointModelGroup(arm_group_.getName());
    if (!jmg) { return ""; }

    // (b) No IK ⇒ say nothing; the NO_IK rejection message owns that case.
    if (!state.setFromIK(jmg, pose.pose, arm_group_.getEndEffectorLink(), 0.05)) {
      return "";
    }
    state.update();

    // (c) Ask move_group what that state touches. service_is_ready() is a
    //     non-blocking graph check — no move_group, no diagnostic, no delay.
    if (!state_validity_client_->service_is_ready()) { return ""; }
    auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
    moveit::core::robotStateToRobotStateMsg(state, req->robot_state);
    req->group_name = arm_group_.getName();
    auto future = state_validity_client_->async_send_request(req);
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(collision_detail_timeout_s_));
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline &&
           future.wait_for(std::chrono::milliseconds(0)) !=
             std::future_status::ready) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (future.wait_for(std::chrono::milliseconds(0)) !=
          std::future_status::ready) {
      // Drop the pending call so a late answer can never be mistaken for
      // the next candidate's, and so the client stops tracking it.
      state_validity_client_->remove_pending_request(future);
      return "";
    }
    auto resp = future.get();
    if (!resp) { return ""; }
    if (resp->valid) {
      if (goal_state_clear) { *goal_state_clear = true; }
      return "";
    }

    // (d) Name the first few DISTINCT pairs — robot link first, object
    //     second, which is how the operator thinks about it.
    const std::string pairs = format_contacts(*resp, blocker_out);
    return pairs.empty()
      ? "goal state is invalid but MoveIt named no contact pair"
      : "collides: " + pairs;
  }

  // "A <-> B; C <-> D" (robot link first, environment object second, at
  // most three distinct pairs); blocker_out gets the first environment
  // object. Empty when the response names no contact.
  static std::string format_contacts(
    const moveit_msgs::srv::GetStateValidity::Response & resp,
    std::string * blocker_out)
  {
    std::vector<std::string> pairs;
    std::vector<std::string> keys;
    for (const auto & c : resp.contacts) {
      const std::string key = (c.contact_body_1 < c.contact_body_2)
        ? c.contact_body_1 + "|" + c.contact_body_2
        : c.contact_body_2 + "|" + c.contact_body_1;
      if (std::find(keys.begin(), keys.end(), key) != keys.end()) { continue; }
      keys.push_back(key);
      const bool a_is_link =
        (c.body_type_1 == moveit_msgs::msg::ContactInformation::ROBOT_LINK);
      const bool b_is_link =
        (c.body_type_2 == moveit_msgs::msg::ContactInformation::ROBOT_LINK);
      std::string a = c.contact_body_1, b = c.contact_body_2;
      std::string env;
      if (a_is_link && !b_is_link) {
        env = c.contact_body_2;
      } else if (!a_is_link && b_is_link) {
        env = c.contact_body_1;
        std::swap(a, b);  // robot link first
      }
      if (blocker_out && blocker_out->empty() && !env.empty()) {
        *blocker_out = env;
      }
      // How deep the overlap is (2026-09-08, Majd: "how near was the
      // collision?"): FCL's penetration depth, in mm. A few mm means the
      // 5 mm neighbour padding is what stopped it; centimetres mean the
      // gripper really sweeps through the object.
      std::string depth;
      if (c.depth > 0.0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " (%.0f mm deep)", c.depth * 1000.0);
        depth = buf;
      }
      pairs.push_back(a + " <-> " + b + depth);
      if (pairs.size() >= 3) { break; }
    }
    std::string out;
    for (size_t i = 0; i < pairs.size(); ++i) {
      if (i) { out += "; "; }
      out += pairs[i];
    }
    return out;
  }

  // Straight-approach autopsy (2026-09-08, Majd: "it should say what the
  // object is"). MoveIt DOES name the blocker of an invalid LIN path
  // ("Found a contact between 'forearm_link' and 'camera_mount'") — but
  // from free-standing loggers, which Humble never forwards to /rosout,
  // so the listener in main() cannot see them; only the console log does.
  // So ask /check_state_validity ourselves at a few points of the
  // straight line between the stand-off pose and the grasp (the goal
  // itself was already checked). All requests go out together and are
  // awaited once, so this costs one service round trip per candidate.
  // Returns e.g. "2 cm before the grasp: gripper_finger_left <-> scene_obj_x"
  // or "" when every sample is clear / the service is unavailable.
  std::string approach_collision_detail(
    const geometry_msgs::msg::PoseStamped & target,
    std::string * blocker_out = nullptr)
  {
    if (blocker_out) { blocker_out->clear(); }
    if (!state_validity_client_ || !state_validity_client_->service_is_ready()) { return ""; }
    sensor_msgs::msg::JointState js;
    {
      std::lock_guard<std::mutex> lk(joint_state_mutex_);
      js = last_joint_state_;
    }
    if (js.name.empty()) { return ""; }
    moveit::core::RobotState state(arm_group_.getRobotModel());
    state.setToDefaultValues();
    for (size_t i = 0; i < js.name.size() && i < js.position.size(); ++i) {
      if (state.getRobotModel()->hasJointModel(js.name[i])) {
        state.setVariablePosition(js.name[i], js.position[i]);
      }
    }
    state.update();
    const auto * jmg = state.getJointModelGroup(arm_group_.getName());
    if (!jmg) { return ""; }

    const auto pre = compute_pregrasp_pose(target, pregrasp_approach_distance_);
    const int n = std::max(1, approach_autopsy_samples_);
    struct Sample { double t; std::shared_future<std::shared_ptr<moveit_msgs::srv::GetStateValidity::Response>> fut; };
    std::vector<Sample> samples;
    for (int k = 1; k <= n; ++k) {
      const double t = static_cast<double>(k) / (n + 1);   // strictly between stand-off and grasp
      geometry_msgs::msg::Pose pose = target.pose;          // LIN keeps the orientation
      pose.position.x = pre.pose.position.x + t * (target.pose.position.x - pre.pose.position.x);
      pose.position.y = pre.pose.position.y + t * (target.pose.position.y - pre.pose.position.y);
      pose.position.z = pre.pose.position.z + t * (target.pose.position.z - pre.pose.position.z);
      // seeded by the previous sample: stays in the branch the LIN would use
      if (!state.setFromIK(jmg, pose, arm_group_.getEndEffectorLink(), 0.05)) { continue; }
      state.update();
      auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
      moveit::core::robotStateToRobotStateMsg(state, req->robot_state);
      req->group_name = arm_group_.getName();
      samples.push_back({t, state_validity_client_->async_send_request(req).share()});
    }
    if (samples.empty()) { return ""; }
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(collision_detail_timeout_s_));
    auto all_ready = [&]() {
      for (const auto & smp : samples) {
        if (smp.fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) { return false; }
      }
      return true;
    };
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline && !all_ready()) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    for (const auto & smp : samples) {           // nearest to the stand-off first
      if (smp.fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) { continue; }
      auto resp = smp.fut.get();
      if (!resp || resp->valid) { continue; }
      const std::string pairs = format_contacts(*resp, blocker_out);
      char where[48];
      std::snprintf(where, sizeof(where), "%.0f cm before the grasp",
        (1.0 - smp.t) * pregrasp_approach_distance_ * 100.0);
      return std::string(where) + ": " +
        (pairs.empty() ? "the arm is in collision but MoveIt named no pair" : pairs);
    }
    return "";
  }

  // Straight-line KINEMATICS along the whole approach (2026-09-11). Pilz
  // LIN refuses a line whose joint-space path exceeds a joint velocity or
  // acceleration limit — in practice the elbow sweeping past a singularity
  // ("Joint velocity limit of elbow_joint violated ... 12.6 rad/s", move_group
  // log, cycle 18 of that day; the limit is about 3.1). That message comes
  // from a free logger and never reaches /rosout, so the executor used to
  // call it "something along the straight line blocks it", which sent Majd
  // looking for an obstacle that does not exist. Measure it instead: IK at
  // one point per 2 cm from the current TCP to the grasp (position and
  // orientation interpolated like LIN does, each solve seeded by the
  // previous one), and the sharpest joint turn per centimetre on the way.
  std::string approach_kinematic_detail(const geometry_msgs::msg::PoseStamped & target)
  {
    moveit::core::RobotState state(arm_group_.getRobotModel());
    if (!robot_state_from_ear(state)) { return ""; }
    const auto * jmg = state.getJointModelGroup(arm_group_.getName());
    if (!jmg) { return ""; }
    const std::string & eef = arm_group_.getEndEffectorLink();
    const Eigen::Isometry3d T0 = state.getGlobalLinkTransform(eef);
    const Eigen::Vector3d p0 = T0.translation();
    const Eigen::Vector3d p1(target.pose.position.x, target.pose.position.y, target.pose.position.z);
    const double len = (p1 - p0).norm();
    if (len < 0.02) { return ""; }
    const Eigen::Quaterniond q0(T0.rotation());
    const Eigen::Quaterniond q1(target.pose.orientation.w, target.pose.orientation.x,
                                target.pose.orientation.y, target.pose.orientation.z);
    const int n = std::max(8, static_cast<int>(len / 0.02));
    const double step_cm = len / n * 100.0;
    std::vector<double> prev;
    state.copyJointGroupPositions(jmg, prev);
    const auto & names = jmg->getVariableNames();
    double worst = 0.0; size_t worst_j = 0; double worst_t = 0.0;
    for (int k = 1; k <= n; ++k) {
      const double t = static_cast<double>(k) / n;
      geometry_msgs::msg::Pose pose;
      const Eigen::Vector3d p = p0 + t * (p1 - p0);
      const Eigen::Quaterniond q = q0.slerp(t, q1);
      pose.position.x = p.x(); pose.position.y = p.y(); pose.position.z = p.z();
      pose.orientation.x = q.x(); pose.orientation.y = q.y();
      pose.orientation.z = q.z(); pose.orientation.w = q.w();
      if (!state.setFromIK(jmg, pose, eef, 0.05)) {
        char b[160];
        std::snprintf(b, sizeof(b),
          "no joint solution %.0f cm before the grasp — the straight line leaves the arm's workspace",
          (1.0 - t) * len * 100.0);
        return b;
      }
      std::vector<double> cur;
      state.copyJointGroupPositions(jmg, cur);
      for (size_t j = 0; j < cur.size() && j < prev.size(); ++j) {
        const double d = std::fabs(cur[j] - prev[j]) / step_cm;   // rad per cm of line
        if (d > worst) { worst = d; worst_j = j; worst_t = t; }
      }
      prev = cur;
    }
    char b[240];
    const char * jname = worst_j < names.size() ? names[worst_j].c_str() : "a joint";
    if (worst > 0.15) {
      std::snprintf(b, sizeof(b),
        "%s would turn %.2f rad per cm of line, %.0f cm before the grasp — the straight line "
        "passes a singularity, so the joint speed limit is exceeded (a pre-grasp approach or a "
        "lower velocity scaling avoids this)",
        jname, worst, (1.0 - worst_t) * len * 100.0);
    } else {
      std::snprintf(b, sizeof(b),
        "no collision and no sharp joint turn found on the line (worst %s %.2f rad/cm); "
        "MoveIt gave no reason", jname, worst);
    }
    return b;
  }

  bool choose_first_lin_feasible(
    const std::vector<geometry_msgs::msg::PoseStamped> & candidates_in,
    geometry_msgs::msg::PoseStamped * selected)
  {
    // Mutable copy: the wrist-twin pass below may replace orientations.
    std::vector<geometry_msgs::msg::PoseStamped> candidates = candidates_in;
    if (!selected) {
      return false;
    }
    if (candidates.empty()) {
      return false;
    }

    // INVALIDATION: a new candidate set means a new pick; nothing from
    // the previous evaluation may survive into it.
    invalidate_cached_lin_plan("a new candidate set arrived");
    // Refresh the IK probe state at the start of each batch.
    reset_ik_seed_state();
    // ...and the rejection-autopsy budget, so every cycle gets its
    // own first-N detailed reports. Re-read live so the limit can be
    // raised without restarting the node.
    collision_detail_reports_ = 0;
    try {
      collision_detail_max_reports_ = std::max(0, static_cast<int>(
        node_->get_parameter("collision_detail_max_reports").as_int()));
    } catch (...) {}

    // ── Fix 2 + Fix 3 — pre-filter unreachable poses and sort by ─────
    // proximity to the current TCP. Cheaper to fail fast on IK than to
    // run the LIN planner; cheaper to try the closest grasp first than
    // to walk the GraspGen-score order.
    const size_t total_in = candidates.size();
    std::vector<std::pair<double, size_t>> ranked;  // (cost, original_index)
    ranked.reserve(total_in);

    geometry_msgs::msg::Pose tcp_pose;
    bool have_tcp = false;
    {
      // From our own ear — getCurrentPose() waited its full 1.0 s here on
      // every batch and then failed anyway (2026-09-08).
      moveit::core::RobotState ear(arm_group_.getRobotModel());
      if (robot_state_from_ear(ear)) {
        const Eigen::Isometry3d T = ear.getGlobalLinkTransform(arm_group_.getEndEffectorLink());
        const Eigen::Quaterniond q(T.rotation());
        tcp_pose.position.x = T.translation().x();
        tcp_pose.position.y = T.translation().y();
        tcp_pose.position.z = T.translation().z();
        tcp_pose.orientation.x = q.x(); tcp_pose.orientation.y = q.y();
        tcp_pose.orientation.z = q.z(); tcp_pose.orientation.w = q.w();
        have_tcp = true;
      }
    }

    // Re-read live so Majd can flip probe ordering without a restart:
    // true  = probe nearest-to-gripper first (old behaviour),
    // false = probe in the pipeline's score order (centroid-weighted
    //         best first — "centered grasps first", 2026-08-20).
    bool sort_by_reach = sort_candidates_by_reachability_;
    try {
      sort_by_reach =
        node_->get_parameter("sort_candidates_by_reachability").as_bool();
    } catch (...) {}
    // Wrist-spin fix (Majd, 2026-08-20: "I don't want this to happen
    // again"): a parallel gripper grips IDENTICALLY when rotated 180
    // degrees about its approach axis, but GraspGen emits an arbitrary
    // one of the two twins — sometimes forcing a half-turn wrist
    // rotation. Swap each candidate to whichever twin is rotationally
    // closer to the CURRENT tcp orientation. Grip unchanged, motion
    // shorter.
    if (enable_wrist_twin_rewrite_ && have_tcp) {
      const auto & cq = tcp_pose.orientation;
      size_t flipped_upright = 0, flipped_closer = 0;
      for (auto & c : candidates) {
        auto & q = c.pose.orientation;
        // twin = q * Rz(pi): (x,y,z,w) -> (qy, -qx, qw, -qz)
        const double tx = q.y, ty = -q.x, tz = q.w, tw = -q.z;
        // z-component of the gripper's local +Y axis in base. The twin
        // negates it, so exactly one twin hangs "upside down".
        const double y_up = 2.0 * (q.y * q.z + q.x * q.w);
        // z of the finger-opening axis (local +X); also negated by the
        // twin. Decides which finger is on top for FLAT grasps.
        const double x_up = 2.0 * (q.x * q.z - q.y * q.w);
        bool take_twin;
        if (prefer_upright_twin_ && std::fabs(y_up) > upright_twin_dead_zone_) {
          take_twin = upright_twin_y_up_positive_ ? (y_up < 0.0)
                                                  : (y_up > 0.0);
          if (take_twin) { ++flipped_upright; }
        } else if (prefer_upright_twin_ &&
                   std::fabs(x_up) > upright_twin_dead_zone_) {
          // FLAT grasp: same finger on top, every time.
          take_twin = flat_twin_x_up_positive_ ? (x_up < 0.0) : (x_up > 0.0);
          if (take_twin) { ++flipped_upright; }
        } else {
          const double dot_orig = std::fabs(
            q.x * cq.x + q.y * cq.y + q.z * cq.z + q.w * cq.w);
          const double dot_twin = std::fabs(
            tx * cq.x + ty * cq.y + tz * cq.z + tw * cq.w);
          take_twin = (dot_twin > dot_orig);
          if (take_twin) { ++flipped_closer; }
        }
        if (take_twin) {
          q.x = tx; q.y = ty; q.z = tz; q.w = tw;
        }
      }
      RCLCPP_INFO(node_->get_logger(),
        "Wrist-twin pass (%s): %zu flipped upright, %zu flipped for "
        "shortest rotation, of %zu candidates.",
        prefer_upright_twin_ ? "upright-first" : "closest-only",
        flipped_upright, flipped_closer, candidates.size());
    } else if (!enable_wrist_twin_rewrite_) {
      RCLCPP_INFO(node_->get_logger(),
        "Wrist-twin rewrite disabled; preserving all pipeline TCP orientations.");
    }
    // 2026-09-08: the IK pre-filter ORDERS, it no longer EXCLUDES. A 20 ms
    // KDL attempt from wherever the arm happens to be has many false
    // negatives (today: 0/15 then 15/22 on the same object), and every
    // false negative used to cost a grasp. Candidates that fail the quick
    // probe go to the END of the probe order; the LIN probe stays the judge.
    if (!precheck_mode_) {
      // (2026-09-16) the open is no longer sent here: when no line was
      // drivable the gripper had opened for nothing. It goes with the approach.
    }
    size_t ik_demoted = 0;
    std::vector<std::pair<double, size_t>> demoted, untested;
    // Only the first batch (max_candidate_tries_) is ever probed, so the
    // 20 ms quick test stops once that many have passed (Majd 2026-09-17:
    // "sort the first 10 because we only execute them"). 48 tests were
    // 0.5 s of every click; 10-12 are 0.1 s. The untested rest keeps
    // GraspGen's order behind the passers, ahead of the demoted ones.
    const size_t want = static_cast<size_t>(std::max(1, max_candidate_tries_));
    for (size_t i = 0; i < total_in; ++i) {
      double cost = static_cast<double>(i);  // fall back to GraspGen order
      if (sort_by_reach && have_tcp) {
        const auto & p = candidates[i].pose.position;
        const double dx = p.x - tcp_pose.position.x;
        const double dy = p.y - tcp_pose.position.y;
        const double dz = p.z - tcp_pose.position.z;
        cost = std::sqrt(dx * dx + dy * dy + dz * dz);
      }
      if (!sort_by_reach && ranked.size() >= want) {
        untested.emplace_back(cost, i);
        continue;
      }
      if (has_ik_solution(candidates[i])) {
        ranked.emplace_back(cost, i);
      } else {
        ++ik_demoted;
        demoted.emplace_back(cost, i);
      }
    }

    if (sort_by_reach) {
      std::sort(ranked.begin(), ranked.end());
      std::sort(demoted.begin(), demoted.end());
    }
    const size_t ik_passed = ranked.size();
    ranked.insert(ranked.end(), untested.begin(), untested.end());
    ranked.insert(ranked.end(), demoted.begin(), demoted.end());

    RCLCPP_INFO(
      node_->get_logger(),
      "Pre-filter: %zu/%zu candidates have a quick IK solution; %zu demoted to "
      "the end of the probe order (none dropped; %zu untested: enough passed for the first batch). Sort by reachability: %s.",
      ik_passed, ik_passed + ik_demoted, ik_demoted, untested.size(),
      sort_by_reach ? "enabled" : "disabled");

    if (ranked.empty()) {
      return false;
    }

    const size_t total = ranked.size();
    // Give-up deadline for the whole feasibility search (Majd, 2026-08-12:
    // "if it doesn't execute after N seconds it must cancel and say why").
    double search_timeout_s = 60.0;  // Majd: 15 s was too tight; 1 min
    try {
      search_timeout_s =
        node_->get_parameter("candidate_search_timeout_s").as_double();
    } catch (...) {}
    const auto search_deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(search_timeout_s));
    // Rejection autopsy (Majd, 2026-08-12: "I need to know exactly WHY it
    // didn't succeed through this 1 min") — aggregated into the spoken
    // failure reason instead of being lost in per-candidate log lines.
    size_t n_tried = 0, n_no_ik = 0, n_invalid = 0, n_planfail = 0, n_other = 0;
    // What the autopsy actually found, tallied per obstacle so the
    // spoken summary can name the thing the gripper kept hitting.
    std::map<std::string, size_t> blocker_hits;
    auto reason_breakdown = [&]() {
      std::string out = "tried " + std::to_string(n_tried) + " grasps: ";
      bool first = true;
      auto add = [&](size_t n, const char * what) {
        if (!n) { return; }
        if (!first) { out += ", "; }
        out += std::to_string(n) + " " + what;
        first = false;
      };
      add(n_no_ik, "had no IK somewhere along the straight approach");
      add(n_invalid, "had an invalid LIN trajectory");
      add(n_planfail, "failed LIN planning");
      add(n_other, "failed for other reasons");
      if (n_no_ik > n_invalid + n_planfail + n_other) {
        out += " — mostly out of reach, try sending me home or moving the object closer";
      }
      // Spoken aloud — keep it to the single most common obstacle.
      if (!blocker_hits.empty()) {
        const std::string * top_name = nullptr;
        size_t top_n = 0;
        for (const auto & kv : blocker_hits) {
          if (kv.second > top_n) { top_n = kv.second; top_name = &kv.first; }
        }
        if (top_name) { out += " — the gripper kept hitting " + *top_name; }
      }
      return out;
    };
    // Compact per-candidate descriptor: height, distance from base
    // (candidates arrive in base_link), approach direction.
    auto describe_candidate = [](const geometry_msgs::msg::PoseStamped & ps) {
      const auto & p = ps.pose.position;
      const auto & q = ps.pose.orientation;
      const double az = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
      // z of the gripper local +Y (palm up/down; fingers open along
      // local X — measured from TF 2026-08-21). Negative = hanging.
      const double y_up = 2.0 * (q.y * q.z + q.x * q.w);
      // Labels follow the CALIBRATED natural side (#186: this gripper's
      // correct orientation is y_up<0) — "upright" = the side Majd
      // accepts, "FLIPPED" = the side he kept rejecting.
      const char * roll = (y_up < -0.15) ? "upright"
                        : (y_up > 0.15) ? "FLIPPED" : "flat";
      const char * dir = (az < -0.7) ? "top-down"
                       : (std::abs(az) < 0.4) ? "side" : "angled";
      const double tilt_deg = std::asin(
        std::clamp(std::abs(az), 0.0, 1.0)) * 180.0 / M_PI;
      const char * tilt_sign = (az < -0.02) ? "down"
                              : (az > 0.02) ? "up" : "level";
      const double roll_deg = std::acos(
        std::clamp(-y_up, -1.0, 1.0)) * 180.0 / M_PI;
      const double dist = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      char buf[160];
      std::snprintf(
        buf, sizeof(buf),
        "z=%.2fm, %.2fm from base, %s, tilt=%.0fdeg %s, roll=%.0fdeg, %s",
        p.z, dist, dir, tilt_deg, tilt_sign, roll_deg, roll);
      return std::string(buf);
    };
    const size_t batch_size = static_cast<size_t>(std::max(1, max_candidate_tries_));
    for (size_t batch_start = 0; batch_start < total; batch_start += batch_size) {
      const size_t batch_end = std::min(batch_start + batch_size, total);
      const size_t batch_idx = (batch_start / batch_size) + 1;
      const size_t batch_count = (total + batch_size - 1) / batch_size;
      // INVALIDATION: a new candidate batch begins.
      invalidate_cached_lin_plan("a new candidate batch started");
      RCLCPP_INFO(
        node_->get_logger(),
        "Evaluating candidate batch %zu/%zu (indices %zu..%zu).",
        batch_idx, batch_count, batch_start + 1, batch_end);
      for (size_t i = batch_start; i < batch_end; ++i) {
        // Let a real selection in, and stop for it between candidates.
        if (precheck_stop_requested()) { return false; }
        if (abort_requested_.load()) {
          last_failure_reason_ = "stopped by your command";
          // INVALIDATION: abort mid-evaluation.
          invalidate_cached_lin_plan("operator abort during candidate evaluation");
          RCLCPP_WARN(node_->get_logger(),
            "Operator abort during candidate evaluation — giving up.");
          return false;
        }
        if (std::chrono::steady_clock::now() > search_deadline) {
          last_failure_reason_ =
            "gave up after " +
            std::to_string(static_cast<int>(search_timeout_s)) +
            " seconds; " + reason_breakdown();
          RCLCPP_WARN(node_->get_logger(),
            "Candidate search timeout (%.0fs) after %zu/%zu candidates.",
            search_timeout_s, i, total);
          return false;
        }
        const size_t orig_idx = ranked[i].second;
        // A hint belongs to exactly one planning request. Clearing it here
        // prevents a velocity/collision diagnosis from candidate N being
        // accidentally printed for candidate N+1.
        clear_moveit_hint();
        std::string reason;
        if (can_plan_lin_pose(candidates[orig_idx], &reason)) {
          *selected = candidates[orig_idx];
          RCLCPP_INFO(
            node_->get_logger(),
            "Candidate %zu/%zu (orig=%zu, cost=%.3f) is LIN-feasible; selecting it.",
            i + 1, total, orig_idx + 1, ranked[i].first);
          try {
            std_msgs::msg::String cs;
            cs.data = "grasp " + std::to_string(i + 1) + "/" +
              std::to_string(total) + ": FEASIBLE — executing this one via the " +
              (last_planner_.empty() ? std::string("line") : last_planner_) + " (" +
              describe_candidate(candidates[orig_idx]) + ")";
            if (!precheck_mode_) { candidate_status_pub_->publish(cs); }
          } catch (...) {}
          precheck_tried_ = n_tried + 1;
          return true;
        }
        if (precheck_interrupted_) { return false; }
        ++n_tried;
        precheck_tried_ = n_tried;
        // MoveGroup returns only a broad error code. Its useful detail (for
        // example the exact joint and measured limit, or a collision pair)
        // is emitted separately on /rosout, often a few ms later.
        std::string hint = fresh_moveit_hint();
        for (int w = 0; hint.empty() && w < 20; ++w) {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          hint = fresh_moveit_hint();
        }

        std::string short_reason;
        if (reason.find("NO_IK") != std::string::npos) {
          ++n_no_ik;
          short_reason = hint.empty()
            ? "no IK along straight approach: TCP pose/orientation has no joint solution"
            : hint;
        } else if (reason.find("INVALID_MOTION") != std::string::npos ||
                   reason.find("PLANNING_FAILED") != std::string::npos) {
          // Planner failure. The -1/-2 code says nothing about WHAT
          // blocked the path, so run the collision autopsy (budgeted by
          // collision_detail_max_reports) and name the contact pair.
          const bool invalid_traj =
            reason.find("INVALID_MOTION") != std::string::npos;
          if (invalid_traj) { ++n_invalid; } else { ++n_planfail; }
          const char * bare_code_msg = invalid_traj
            ? "MoveIt rejected LIN trajectory (-2); no collision/limit detail was emitted"
            : "LIN planner found no complete straight path (-1); no detailed cause was emitted";
          std::string detail, blocker;
          bool goal_clear = false;
          if (!precheck_mode_ && collision_detail_reports_ < collision_detail_max_reports_) {
            ++collision_detail_reports_;
            detail = collision_detail(candidates[orig_idx], &blocker, &goal_clear);
            if (!blocker.empty()) { ++blocker_hits[blocker]; }
          }
          if (detail.empty() && goal_clear && approach_autopsy_samples_ > 0) {
            // Endpoint is clean, so the obstruction sits somewhere along
            // the straight line: walk it (one parallel service round trip).
            detail = approach_collision_detail(candidates[orig_idx], &blocker);
            if (!blocker.empty()) { ++blocker_hits[blocker]; }
            if (!detail.empty()) { detail = "mid-approach, " + detail; }
          }
          // The /rosout hint (joint-limit lines) may have landed during the
          // autopsy round trips; read it again before composing.
          if (hint.empty()) { hint = fresh_moveit_hint(); }
          std::string kin;
          if (!precheck_mode_ && detail.empty() && goal_clear) { kin = approach_kinematic_detail(candidates[orig_idx]); }
          if (!detail.empty()) {
            short_reason = "LIN path blocked " + detail;
          } else if (goal_clear) {
            short_reason = "straight line not drivable: " +
              (kin.empty() ? std::string("goal pose is collision-free and nothing was found on the line") : kin);
            if (!hint.empty()) { short_reason += " — MoveIt: " + hint; }
          } else {
            // Budget spent, IK failed, or move_group's service is down:
            // exactly the pre-autopsy wording.
            short_reason = hint.empty() ? bare_code_msg : hint;
          }
        } else {
          ++n_other;
          short_reason = reason.empty() ? "unknown planning failure" : reason;
          if (!hint.empty()) { short_reason += ": " + hint; }
        }
        try {
          std_msgs::msg::String cs;
          cs.data = "grasp " + std::to_string(i + 1) + "/" +
            std::to_string(total) + ": rejected — " + short_reason +
            " (" + describe_candidate(candidates[orig_idx]) + ")";
          if (!precheck_mode_) { candidate_status_pub_->publish(cs); }
        } catch (...) {}
        RCLCPP_WARN(
          node_->get_logger(),
          "Candidate %zu/%zu (orig=%zu) rejected for LIN: %s",
          i + 1, total, orig_idx + 1, reason.c_str());
      }
      if (!candidate_try_all_batches_) {
        break;
      }
    }
    if (last_failure_reason_.empty() && n_tried > 0) {
      last_failure_reason_ = reason_breakdown();
    }
    // The search's own verdict is the reported reason (unless something more specific was set).
    if (last_failure_reason_.empty()) { last_failure_reason_ = reason_breakdown(); }
    RCLCPP_WARN(node_->get_logger(), "Feasibility search failed: %s",
      reason_breakdown().c_str());
    return false;
  }

  bool execute_fast_joint_lift(double dz)
  {
    // INVALIDATION: the planner-free lift streams joints straight to the
    // controller, so nothing else would notice the arm had moved.
    invalidate_cached_lin_plan("the arm is running the fast joint lift");
    // Planner-free lift (Majd 2026-08-26: "fast simple, like the fast
    // observation — just go up 5 cm"): one IK solve seeded at the
    // current joints, then a 2-point joint trajectory streamed straight
    // to the controller. No MoveIt planning pipeline involved at all.
    // Fresh joints from OUR OWN /joint_states ear (joint_states_sub_).
    // getCurrentState() can NEVER work in this single-threaded program:
    // its monitor needs someone to spin the node WHILE it waits, and
    // nobody does — so it starved forever right after the gripper action
    // and every lift bailed before computing anything (2026-08-26; the
    // stream itself was healthy: ~433 Hz, ~1 ms stale, measured live).
    // Freshness by construction: WE spin until a NEW message that names
    // every arm joint arrives.
    const std::vector<std::string> arm_joint_names = arm_group_.getJointNames();
    auto wait_fresh_arm_js =
      [&](double timeout_s, sensor_msgs::msg::JointState & out) -> bool {
        uint64_t last_seen = joint_state_seq_.load();
        const auto t_end_js = std::chrono::steady_clock::now() +
          std::chrono::duration<double>(timeout_s);
        while (rclcpp::ok() && std::chrono::steady_clock::now() < t_end_js) {
          rclcpp::spin_some(node_);
          const uint64_t s = joint_state_seq_.load();
          if (s != last_seen) {
            last_seen = s;
            sensor_msgs::msg::JointState cand;
            {
              std::lock_guard<std::mutex> lk(joint_state_mutex_);
              cand = last_joint_state_;
            }
            bool has_all = true;
            for (const auto & jn : arm_joint_names) {
              if (std::find(cand.name.begin(), cand.name.end(), jn) ==
                    cand.name.end()) { has_all = false; break; }
            }
            if (has_all) { out = cand; return true; }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
      };
    sensor_msgs::msg::JointState js;
    if (!wait_fresh_arm_js(2.0, js)) {
      RCLCPP_WARN(node_->get_logger(),
        "fast-lift: no fresh /joint_states with the arm joints within 2 s "
        "— skipping the lift.");
      return false;
    }
    // Seed state + current TCP by FORWARD KINEMATICS — pure math on the
    // joints we just read; no monitors, no TF, no waiting on anyone.
    auto state = std::make_shared<moveit::core::RobotState>(
      arm_group_.getRobotModel());
    state->setToDefaultValues();
    for (size_t i = 0; i < js.name.size() && i < js.position.size(); ++i) {
      if (state->getRobotModel()->hasJointModel(js.name[i])) {
        state->setVariablePosition(js.name[i], js.position[i]);
      }
    }
    state->update();
    const auto * jmg = state->getJointModelGroup(arm_group_.getName());
    if (!jmg) { return false; }
    std::vector<double> seed;
    state->copyJointGroupPositions(jmg, seed);
    const Eigen::Isometry3d & tcp_now =
      state->getGlobalLinkTransform(arm_group_.getEndEffectorLink());
    geometry_msgs::msg::PoseStamped cur;
    cur.pose.position.x = tcp_now.translation().x();
    cur.pose.position.y = tcp_now.translation().y();
    cur.pose.position.z = tcp_now.translation().z();
    {
      const Eigen::Quaterniond q_now(tcp_now.rotation());
      cur.pose.orientation.x = q_now.x();
      cur.pose.orientation.y = q_now.y();
      cur.pose.orientation.z = q_now.z();
      cur.pose.orientation.w = q_now.w();
    }
    geometry_msgs::msg::Pose target = cur.pose;
    // TRUE UP, not base +z (Majd 2026-08-26: "the lift goes FORWARD in
    // z" — this cell's robot base is mounted with its z axis pointing
    // HORIZONTALLY, so base +z drives toward the table. The table
    // frame's up (same helper the LIN lift always used) is the real
    // vertical.)
    const auto up_dir = compute_lift_direction_in_target();
    target.position.x += up_dir[0] * dz;
    target.position.y += up_dir[1] * dz;
    target.position.z += up_dir[2] * dz;
    RCLCPP_INFO(node_->get_logger(),
      "fast-lift: %.3f m along up=[%.3f, %.3f, %.3f]: TCP [%.3f, %.3f, %.3f] "
      "-> [%.3f, %.3f, %.3f]",
      dz, up_dir[0], up_dir[1], up_dir[2],
      cur.pose.position.x, cur.pose.position.y, cur.pose.position.z,
      target.position.x, target.position.y, target.position.z);
    // STEPPED CLIMB (2026-08-26): one 10 cm IK often lands on a FLIPPED
    // arm configuration (observed: 0.97 rad jump, refused). Climb in
    // ~2.5 cm sub-steps instead — each step's IK is seeded by the
    // previous step, so the solution stays in the same configuration and
    // each joint move must be tiny or we refuse.
    const int n_steps = std::max(1, static_cast<int>(std::ceil(dz / 0.025)));
    std::vector<std::vector<double>> climb;
    for (int i = 1; i <= n_steps; ++i) {
      geometry_msgs::msg::Pose step_target = cur.pose;
      step_target.position.x += up_dir[0] * dz * i / n_steps;
      step_target.position.y += up_dir[1] * dz * i / n_steps;
      step_target.position.z += up_dir[2] * dz * i / n_steps;
      if (!state->setFromIK(jmg, step_target,
                            arm_group_.getEndEffectorLink(), 0.05)) {
        RCLCPP_WARN(node_->get_logger(),
          "fast-lift: IK found no solution at step %d/%d.", i, n_steps);
        return false;
      }
      std::vector<double> j;
      state->copyJointGroupPositions(jmg, j);
      const std::vector<double> & prev = climb.empty() ? seed : climb.back();
      double step_d = 0.0;
      for (size_t k = 0; k < j.size() && k < prev.size(); ++k) {
        step_d = std::max(step_d, std::fabs(j[k] - prev[k]));
      }
      if (step_d > 0.35) {
        // A 2.5 cm rise must be a tiny jog — a jump means IK flipped
        // the configuration. Refuse rather than swing wildly.
        RCLCPP_WARN(node_->get_logger(),
          "fast-lift: IK jumped %.2f rad at step %d/%d — refusing jog.",
          step_d, i, n_steps);
        return false;
      }
      climb.push_back(std::move(j));
    }
    using FJT = control_msgs::action::FollowJointTrajectory;
    auto client = rclcpp_action::create_client<FJT>(
      node_, "/scaled_joint_trajectory_controller/follow_joint_trajectory");
    if (!client->wait_for_action_server(std::chrono::seconds(1))) {
      return false;
    }
    FJT::Goal g;
    g.trajectory.joint_names = jmg->getVariableNames();
    const double total_t = std::max(0.8, 10.0 * dz);  // >= 0.8 s, ~1.0 s for 10 cm
    trajectory_msgs::msg::JointTrajectoryPoint p0;
    p0.positions = seed;
    p0.time_from_start = rclcpp::Duration::from_seconds(0.0);
    g.trajectory.points.push_back(p0);
    for (int i = 0; i < static_cast<int>(climb.size()); ++i) {
      trajectory_msgs::msg::JointTrajectoryPoint pi;
      pi.positions = climb[static_cast<size_t>(i)];
      pi.time_from_start = rclcpp::Duration::from_seconds(
        total_t * (i + 1) / static_cast<double>(climb.size()));
      g.trajectory.points.push_back(pi);
    }
    auto gh_f = client->async_send_goal(g);
    auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (rclcpp::ok() && std::chrono::steady_clock::now() < t_end &&
           gh_f.wait_for(std::chrono::milliseconds(0)) !=
             std::future_status::ready) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (gh_f.wait_for(std::chrono::milliseconds(0)) !=
          std::future_status::ready) { return false; }
    auto gh = gh_f.get();
    if (!gh) { return false; }
    auto res_f = client->async_get_result(gh);
    t_end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (rclcpp::ok() && std::chrono::steady_clock::now() < t_end &&
           res_f.wait_for(std::chrono::milliseconds(0)) !=
             std::future_status::ready) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (res_f.wait_for(std::chrono::milliseconds(0)) !=
          std::future_status::ready) { return false; }
    auto res = res_f.get();
    const bool lift_ok = res.code == rclcpp_action::ResultCode::SUCCEEDED &&
                         res.result->error_code == FJT::Result::SUCCESSFUL;
    if (lift_ok) {
      // Ground truth for the direction argument: what the TCP ACTUALLY
      // did — measured the same sensor-direct FK way.
      sensor_msgs::msg::JointState js2;
      if (wait_fresh_arm_js(0.5, js2)) {
        for (size_t i = 0; i < js2.name.size() && i < js2.position.size(); ++i) {
          if (state->getRobotModel()->hasJointModel(js2.name[i])) {
            state->setVariablePosition(js2.name[i], js2.position[i]);
          }
        }
        state->update();
        const Eigen::Isometry3d & tcp_after =
          state->getGlobalLinkTransform(arm_group_.getEndEffectorLink());
        RCLCPP_INFO(node_->get_logger(),
          "fast-lift: DONE — TCP now [%.3f, %.3f, %.3f], moved d=[%+.3f, %+.3f, %+.3f].",
          tcp_after.translation().x(), tcp_after.translation().y(),
          tcp_after.translation().z(),
          tcp_after.translation().x() - cur.pose.position.x,
          tcp_after.translation().y() - cur.pose.position.y,
          tcp_after.translation().z() - cur.pose.position.z);
      }
    }
    return lift_ok;
  }

  bool execute_arm_ptp_pose(const geometry_msgs::msg::PoseStamped & target_pose, const std::string & step_name)
  {
    // INVALIDATION: any other arm motion moves the arm off the cached
    // trajectory's first point.
    invalidate_cached_lin_plan("the arm is running a PTP pose move");
    arm_group_.clearPoseTargets();
    arm_group_.setStartStateToCurrentState();
    arm_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    arm_group_.setPlannerId("PTP");
    if (!arm_group_.setPoseTarget(target_pose, arm_group_.getEndEffectorLink())) {
      RCLCPP_ERROR(node_->get_logger(), "%s: setPoseTarget() failed (IK/joint limits).", step_name.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto plan_result = timed_plan(plan, step_name);
    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: planning failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(plan_result).c_str(), plan_result.val);
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "%s: plan OK.", step_name.c_str());
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(), "%s: execute:=false, skipping execution.", step_name.c_str());
      return true;
    }

    auto exec_result = execute_with_deadline(plan, __func__);
    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: execution failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(exec_result).c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "%s: execution OK.", step_name.c_str());
    return true;
  }

  bool execute_arm_tool_offset_lin(double dx, double dy, double dz, const std::string & step_name)
  {
    const auto current = arm_group_.getCurrentPose(arm_group_.getEndEffectorLink());

    tf2::Transform current_tf;
    tf2::fromMsg(current.pose, current_tf);

    tf2::Transform offset_tf;
    offset_tf.setIdentity();
    offset_tf.setOrigin(tf2::Vector3(dx, dy, dz));

    tf2::Transform target_tf = current_tf * offset_tf;  // local tool-frame offset

    geometry_msgs::msg::PoseStamped target = current;
    target.pose.position.x = target_tf.getOrigin().x();
    target.pose.position.y = target_tf.getOrigin().y();
    target.pose.position.z = target_tf.getOrigin().z();
    const tf2::Quaternion q = target_tf.getRotation();
    target.pose.orientation.x = q.x();
    target.pose.orientation.y = q.y();
    target.pose.orientation.z = q.z();
    target.pose.orientation.w = q.w();

    return execute_arm_lin_pose(target, step_name);
  }

  bool execute_pose_tool_offset_lin(
    const geometry_msgs::msg::PoseStamped & base_pose,
    double dx, double dy, double dz,
    const std::string & step_name)
  {
    tf2::Transform base_tf;
    tf2::fromMsg(base_pose.pose, base_tf);

    tf2::Transform offset_tf;
    offset_tf.setIdentity();
    offset_tf.setOrigin(tf2::Vector3(dx, dy, dz));

    tf2::Transform target_tf = base_tf * offset_tf;  // local tool-frame offset

    geometry_msgs::msg::PoseStamped target = base_pose;
    target.pose.position.x = target_tf.getOrigin().x();
    target.pose.position.y = target_tf.getOrigin().y();
    target.pose.position.z = target_tf.getOrigin().z();
    const tf2::Quaternion q = target_tf.getRotation();
    target.pose.orientation.x = q.x();
    target.pose.orientation.y = q.y();
    target.pose.orientation.z = q.z();
    target.pose.orientation.w = q.w();

    return execute_arm_lin_pose(target, step_name);
  }

  bool execute_pose_base_offset_lin(
    const geometry_msgs::msg::PoseStamped & base_pose,
    double dx, double dy, double dz,
    const std::string & step_name)
  {
    geometry_msgs::msg::PoseStamped target = base_pose;
    target.pose.position.x += dx;
    target.pose.position.y += dy;
    target.pose.position.z += dz;
    return execute_arm_lin_pose(target, step_name);
  }

  bool execute_pose_tool_offset_ptp(
    const geometry_msgs::msg::PoseStamped & base_pose,
    double dx, double dy, double dz,
    const std::string & step_name)
  {
    tf2::Transform base_tf;
    tf2::fromMsg(base_pose.pose, base_tf);

    tf2::Transform offset_tf;
    offset_tf.setIdentity();
    offset_tf.setOrigin(tf2::Vector3(dx, dy, dz));

    tf2::Transform target_tf = base_tf * offset_tf;  // local tool-frame offset

    geometry_msgs::msg::PoseStamped target = base_pose;
    target.pose.position.x = target_tf.getOrigin().x();
    target.pose.position.y = target_tf.getOrigin().y();
    target.pose.position.z = target_tf.getOrigin().z();
    const tf2::Quaternion q = target_tf.getRotation();
    target.pose.orientation.x = q.x();
    target.pose.orientation.y = q.y();
    target.pose.orientation.z = q.z();
    target.pose.orientation.w = q.w();

    return execute_arm_ptp_pose(target, step_name);
  }

  // Topic-based fast-replay path (no collision check). Kept for fallback
  // and for benchmarking against the action-based path.
  bool execute_arm_named_target_fast_topic(
    const std::string & target_name, const std::string & step_name)
  {
    // INVALIDATION: fast replay drives the arm outside MoveIt entirely.
    invalidate_cached_lin_plan("the arm is running a fast-replay named move");
    {
      std::lock_guard<std::mutex> lk(fast_replay_mutex_);
      fast_replay_last_status_.clear();
    }

    std::ostringstream oss;
    oss << "{\"label\":\"" << target_name << "\",\"duration_s\":"
        << fast_replay_duration_s_ << "}";
    std_msgs::msg::String msg;
    msg.data = oss.str();
    fast_replay_pub_->publish(msg);

    RCLCPP_INFO(node_->get_logger(),
      "%s: fast-replay (topic) -> '%s' (duration %.2fs).",
      step_name.c_str(), target_name.c_str(), fast_replay_duration_s_);

    const std::string done_prefix = "done:" + target_name;
    const std::string fail_prefix = "failed:" + target_name;
    const std::string cancelled_prefix = "cancelled:" + target_name;
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(fast_replay_timeout_s_);
    // F9 fix (2026-08-31, review finding confirmed LIVE): this wait
    // used to block on the condition variable with NOBODY spinning the
    // node — the status subscription callback that signals it could
    // never run, so every topic-fallback move "timed out" while the
    // replay node was actually DRIVING THE ARM. The operator then heard
    // "I could not get back home" while standing at home. Spin the node
    // between checks, exactly like the release-pause wait loop does.
    while (std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lk(fast_replay_mutex_);
        if (!fast_replay_last_status_.empty()) {
          if (fast_replay_last_status_.rfind(done_prefix, 0) == 0) {
            RCLCPP_INFO(node_->get_logger(),
              "%s: fast-replay completed: %s",
              step_name.c_str(), fast_replay_last_status_.c_str());
            return true;
          }
          if (fast_replay_last_status_.rfind(fail_prefix, 0) == 0 ||
              fast_replay_last_status_.rfind(cancelled_prefix, 0) == 0) {
            RCLCPP_ERROR(node_->get_logger(),
              "%s: fast-replay reported failure: %s",
              step_name.c_str(), fast_replay_last_status_.c_str());
            return false;
          }
        }
      }
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    RCLCPP_ERROR(node_->get_logger(),
      "%s: fast-replay timed out after %.1fs awaiting status for '%s'.",
      step_name.c_str(), fast_replay_timeout_s_, target_name.c_str());
    return false;
  }

  // Action-based fast path: routes through /move_to_named_pose, which
  // pre-checks the trajectory against the planning scene (table, wall,
  // mount, ...) before forwarding to the replay node. Same execution
  // speed; just one extra ~10-50ms collision check up front.
  bool execute_arm_named_target_fast_action(
    const std::string & target_name, const std::string & step_name)
  {
    // INVALIDATION: the named-pose action moves the arm.
    invalidate_cached_lin_plan("the arm is running a named-pose action move");
    using MoveToNamedPose = morty_interfaces::action::MoveToNamedPose;

    if (!named_pose_action_client_->wait_for_action_server(
        std::chrono::duration<double>(2.0)))
    {
      RCLCPP_WARN(node_->get_logger(),
        "%s: /move_to_named_pose action not available; "
        "falling back to topic publish (no collision check).",
        step_name.c_str());
      return execute_arm_named_target_fast_topic(target_name, step_name);
    }

    MoveToNamedPose::Goal goal;
    goal.pose_name = target_name;
    // THE missing line (2026-08-26): without velocity_scaling the server
    // falls back to its SLOW tier (4.0 s, ~20%) — every transport leg ran
    // slow no matter what fast_replay_duration_s said (that param only
    // feeds the topic fallback). 0.5 = the server's very_fast tier
    // (1.6 s, ~50%) — the documented UR3e safety ceiling; higher trips
    // protective stops.
    // Per-leg tiers (Majd 2026-08-26): to-HOME at the 1.6 s tier feels
    // right, but home→drop at the same tier swings the long arc too
    // violently ("toooo fast, it makes noise") — drop legs take 2.0 s.
    // MEDIUM EVERYWHERE (Majd 2026-08-28, second pass: "still kind too
    // fast... even when it goes from drop to home back it is fast
    // movement and a lot of sounds — make it medium as before and keep
    // it smooth"). BOTH legs now take the server's NORMAL tier
    // (0.3 -> 2.7 s); to-HOME no longer gets the 0.5/very_fast tier that
    // made the return leg the loudest move in the cycle. Live-settable:
    // the server maps this straight onto its five duration buckets.
    goal.velocity_scaling =
      (target_name == reset_target_name_) ? 0.3f : 0.3f;

    auto goal_future = named_pose_action_client_->async_send_goal(goal);
    if (rclcpp::spin_until_future_complete(
        node_, goal_future,
        std::chrono::duration<double>(fast_replay_timeout_s_))
        != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(),
        "%s: timeout waiting for /move_to_named_pose to accept goal.",
        step_name.c_str());
      return false;
    }
    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(node_->get_logger(),
        "%s: /move_to_named_pose rejected the goal.", step_name.c_str());
      return false;
    }

    auto result_future = named_pose_action_client_->async_get_result(goal_handle);
    if (rclcpp::spin_until_future_complete(
        node_, result_future,
        std::chrono::duration<double>(fast_replay_timeout_s_ + 30.0))
        != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(),
        "%s: timeout waiting for /move_to_named_pose result.",
        step_name.c_str());
      return false;
    }
    auto result = result_future.get();
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success) {
      RCLCPP_INFO(node_->get_logger(),
        "%s: action-fast-replay completed (collision-checked) for '%s'.",
        step_name.c_str(), target_name.c_str());
      return true;
    }
    RCLCPP_ERROR(node_->get_logger(),
      "%s: action-fast-replay failed: %s",
      step_name.c_str(),
      result.result ? result.result->error.c_str() : "(no result)");
    return false;
  }

  // BLENDED multi-pose move (2026-08-28). ONE MoveToNamedPose goal whose
  // via_poses carry the intermediate stops, so the server builds a single
  // dense, S-curve-timed trajectory current -> vias... -> target. The arm
  // flows THROUGH the vias at speed instead of decelerating to zero at
  // each one, and the ~2 s of per-goal fixed overhead (action round trip
  // + collision pre-check + controller goal acceptance) is paid once.
  //
  // Returns false on ANY problem. Every caller must be ready to fall back
  // to the one-pose-at-a-time sequence — a false here can mean "nothing
  // moved" (goal refused up front) OR "moved partway" (controller aborted
  // mid-path), and the fallback's first leg is a move to home from
  // wherever the arm actually is, which is safe either way.
  bool execute_arm_named_target_blended(
    const std::string & target_name,
    const std::vector<std::string> & via_names,
    const std::string & step_name,
    const std::string & origin_label = "current")
  {
    // INVALIDATION: the blended named-pose action moves the arm.
    invalidate_cached_lin_plan("the arm is running a blended named-pose move");
    using MoveToNamedPose = morty_interfaces::action::MoveToNamedPose;

    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(),
        "%s: execute:=false, skipping blended move to '%s'.",
        step_name.c_str(), target_name.c_str());
      return true;
    }
    if (via_names.empty()) {
      RCLCPP_WARN(node_->get_logger(),
        "%s: blended move asked for with no via poses; nothing to blend.",
        step_name.c_str());
      return false;
    }
    if (!named_pose_action_client_->wait_for_action_server(
        std::chrono::duration<double>(2.0)))
    {
      RCLCPP_WARN(node_->get_logger(),
        "%s: /move_to_named_pose action not available; cannot blend.",
        step_name.c_str());
      return false;
    }

    std::ostringstream path;
    path << origin_label;
    for (const auto & v : via_names) {
      path << "->" << v;
    }
    path << "->" << target_name;
    const std::string path_label = path.str();
    const size_t n_segments = via_names.size() + 1;

    MoveToNamedPose::Goal goal;
    goal.pose_name = target_name;
    goal.via_poses = via_names;
    // One tier for the whole path; the server charges it per segment.
    goal.velocity_scaling = static_cast<float>(blend_velocity_scaling_);

    RCLCPP_INFO(node_->get_logger(),
      "[blend] %s: ONE goal for %s (velocity_scaling=%.2f, %zu segments).",
      step_name.c_str(), path_label.c_str(), blend_velocity_scaling_,
      n_segments);

    const auto t_start = std::chrono::steady_clock::now();
    auto goal_future = named_pose_action_client_->async_send_goal(goal);
    if (rclcpp::spin_until_future_complete(
        node_, goal_future,
        std::chrono::duration<double>(fast_replay_timeout_s_))
        != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(),
        "[blend] %s: timeout waiting for /move_to_named_pose to accept the "
        "blended goal.", step_name.c_str());
      return false;
    }
    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(node_->get_logger(),
        "[blend] %s: /move_to_named_pose rejected the blended goal.",
        step_name.c_str());
      return false;
    }

    auto result_future =
      named_pose_action_client_->async_get_result(goal_handle);
    if (rclcpp::spin_until_future_complete(
        node_, result_future,
        std::chrono::duration<double>(fast_replay_timeout_s_ + 30.0))
        != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(),
        "[blend] %s: timeout waiting for the blended result.",
        step_name.c_str());
      return false;
    }
    auto result = result_future.get();
    const double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_start).count();

    if (result.code == rclcpp_action::ResultCode::SUCCEEDED &&
        result.result && result.result->success)
    {
      RCLCPP_INFO(node_->get_logger(),
        "[blend] %s completed in %.2f s (%zu segments, %d waypoints).",
        path_label.c_str(), elapsed_s, n_segments,
        result.result->waypoints);
      return true;
    }
    RCLCPP_ERROR(node_->get_logger(),
      "[blend] %s FAILED after %.2f s: %s",
      path_label.c_str(), elapsed_s,
      result.result ? result.result->error.c_str() : "(no result)");
    return false;
  }

  // Fast path: skip MoveIt's full PTP planner. Two implementations:
  //   1. action-based (default): /move_to_named_pose action, which does
  //      a collision pre-check against the planning scene before
  //      forwarding to the replay node. Same speed, with safety.
  //   2. topic-based (legacy): publish JSON directly to the replay topic.
  //      No collision check. Toggle via use_named_pose_action_for_fast_replay.
  bool execute_arm_named_target_fast(
    const std::string & target_name, const std::string & step_name)
  {
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(),
        "%s: execute:=false, skipping fast-replay for '%s'.",
        step_name.c_str(), target_name.c_str());
      return true;
    }
    if (use_named_pose_action_) {
      return execute_arm_named_target_fast_action(target_name, step_name);
    }
    return execute_arm_named_target_fast_topic(target_name, step_name);
  }

  bool execute_arm_named_target(
    const std::string & target_name,
    const std::string & step_name,
    moveit_msgs::msg::RobotTrajectory * planned_trajectory_out = nullptr)
  {
    // Fast path for trusted named poses — skip MoveIt entirely.
    // Only when (a) the param is on, (b) the caller doesn't need the
    // planned trajectory back (only Step 4 needs it for reverse-replay).
    if (use_fast_replay_for_named_targets_ && planned_trajectory_out == nullptr) {
      return execute_arm_named_target_fast(target_name, step_name);
    }

    arm_group_.clearPoseTargets();
    arm_group_.setStartStateToCurrentState();
    arm_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    arm_group_.setPlannerId("PTP");
    if (!arm_group_.setNamedTarget(target_name)) {
      RCLCPP_ERROR(node_->get_logger(), "%s: unknown arm named target '%s'.", step_name.c_str(), target_name.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto plan_result = timed_plan(plan, step_name);
    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: planning failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(plan_result).c_str(), plan_result.val);
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "%s: plan OK.", step_name.c_str());
    if (planned_trajectory_out != nullptr) {
      *planned_trajectory_out = plan.trajectory_;
    }
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(), "%s: execute:=false, skipping execution.", step_name.c_str());
      return true;
    }

    auto exec_result = execute_with_deadline(plan, __func__);
    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: execution failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(exec_result).c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "%s: execution OK.", step_name.c_str());
    return true;
  }

  bool execute_arm_reverse_last_drop(const std::string & step_name)
  {
    // INVALIDATION: replays the reversed Step 4 trajectory.
    invalidate_cached_lin_plan("the arm is replaying the reversed drop");
    const auto & orig = last_drop_plan_trajectory_.joint_trajectory;
    const auto n = orig.points.size();
    if (n < 2) {
      RCLCPP_ERROR(node_->get_logger(), "%s: no valid Step 4 trajectory to reverse.", step_name.c_str());
      return false;
    }

    moveit_msgs::msg::RobotTrajectory reversed;
    reversed.joint_trajectory.header = orig.header;
    reversed.joint_trajectory.joint_names = orig.joint_names;
    reversed.joint_trajectory.points.resize(n);

    const int64_t total_ns =
      static_cast<int64_t>(orig.points.back().time_from_start.sec) * 1000000000LL +
      static_cast<int64_t>(orig.points.back().time_from_start.nanosec);

    for (size_t i = 0; i < n; ++i) {
      const auto & src = orig.points[n - 1 - i];
      auto & dst = reversed.joint_trajectory.points[i];
      dst.positions = src.positions;
      dst.velocities = src.velocities;
      dst.accelerations = src.accelerations;
      dst.effort = src.effort;

      for (auto & v : dst.velocities) {
        v = -v;
      }
      for (auto & a : dst.accelerations) {
        a = -a;
      }
      const int64_t src_ns =
        static_cast<int64_t>(src.time_from_start.sec) * 1000000000LL +
        static_cast<int64_t>(src.time_from_start.nanosec);
      const int64_t dst_ns = total_ns - src_ns;
      dst.time_from_start.sec = static_cast<int32_t>(dst_ns / 1000000000LL);
      dst.time_from_start.nanosec = static_cast<uint32_t>(dst_ns % 1000000000LL);
    }

    RCLCPP_INFO(node_->get_logger(), "%s: reversed trajectory generated (%zu points).", step_name.c_str(), n);
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(), "%s: execute:=false, skipping execution.", step_name.c_str());
      return true;
    }

    auto exec_result = execute_with_deadline(reversed, __func__);
    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: execution failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(exec_result).c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "%s: execution OK.", step_name.c_str());
    return true;
  }

  void begin_gripper_open_async()
  {
    open_pending_ = false;
    open_done_ = false;
    if (!execute_ || !gripper_action_client_) { return; }
    if (!gripper_action_client_->action_server_is_ready()) { return; }   // finish() falls back
    slip_monitor_armed_.store(false);
    held_object_.store(false);
    control_msgs::action::GripperCommand::Goal goal;
    goal.command.position = gripper_open_position_;
    goal.command.max_effort = 0.0;
    open_goal_future_ = gripper_action_client_->async_send_goal(goal);
    open_pending_ = true;
    open_sent_t_ = std::chrono::steady_clock::now();
    RCLCPP_INFO(node_->get_logger(),
      "Gripper open sent (target=%.3f) — it opens while the arm travels.", gripper_open_position_);
  }

  void begin_gripper_close_async()
  {
    close_pending_ = false;
    if (!execute_ || !gripper_action_client_) { return; }
    if (!gripper_action_client_->action_server_is_ready()) { return; }
    control_msgs::action::GripperCommand::Goal goal;
    goal.command.position = gripper_close_position_;
    goal.command.max_effort = 0.0;
    close_cmd_t_ = std::chrono::steady_clock::now();
    close_goal_future_ = gripper_action_client_->async_send_goal(goal);
    close_pending_ = true;
    RCLCPP_INFO(node_->get_logger(), "Gripper close sent on the way in (%.2f s before the arm stops).", close_lead_s_);
  }

  bool finish_gripper_close(const std::string & step_name)
  {
    if (!close_pending_) {
      return execute_gripper_named_target(gripper_close_name_, step_name);
    }
    close_pending_ = false;
    const auto wait_timeout = std::chrono::duration<double>(gripper_action_timeout_s_);
    if (rclcpp::spin_until_future_complete(node_, close_goal_future_, wait_timeout) !=
      rclcpp::FutureReturnCode::SUCCESS || !close_goal_future_.get())
    {
      RCLCPP_WARN(node_->get_logger(), "%s: early close goal not accepted — sending it again.", step_name.c_str());
      return execute_gripper_named_target(gripper_close_name_, step_name);
    }
    auto result_future = gripper_action_client_->async_get_result(close_goal_future_.get());
    if (rclcpp::spin_until_future_complete(node_, result_future, wait_timeout) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "%s: timed out waiting for the early close result.", step_name.c_str());
      return false;
    }
    const auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_WARN(node_->get_logger(), "%s: early close failed (code %d) — sending it again.", step_name.c_str(), static_cast<int>(wrapped.code));
      return execute_gripper_named_target(gripper_close_name_, step_name);
    }
    RCLCPP_INFO(node_->get_logger(), "%s: closed on the way in (reached=%.3f).", step_name.c_str(), wrapped.result->position);
    return true;
  }

  bool finish_gripper_open(const std::string & step_name)
  {
    // Once the open of this pick has completed, a second call is a no-op
    // (2026-09-17: the pre-close check re-sent an OPEN after the close had
    // been sent on the way in — it released the object just grasped).
    if (open_done_) { return true; }
    if (!open_pending_) {
      const bool ok = execute_gripper_named_target(gripper_open_name_, step_name);
      open_done_ = ok;
      return ok;
    }
    open_pending_ = false;
    const auto wait_timeout = std::chrono::duration<double>(gripper_action_timeout_s_);
    if (rclcpp::spin_until_future_complete(node_, open_goal_future_, wait_timeout) !=
      rclcpp::FutureReturnCode::SUCCESS || !open_goal_future_.get())
    {
      RCLCPP_WARN(node_->get_logger(), "%s: early open goal not accepted — sending it again.", step_name.c_str());
      return execute_gripper_named_target(gripper_open_name_, step_name);
    }
    auto result_future = gripper_action_client_->async_get_result(open_goal_future_.get());
    if (rclcpp::spin_until_future_complete(node_, result_future, wait_timeout) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "%s: timed out waiting for the early open result.", step_name.c_str());
      return false;
    }
    const auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_WARN(node_->get_logger(), "%s: early open failed (code %d) — sending it again.", step_name.c_str(), static_cast<int>(wrapped.code));
      return execute_gripper_named_target(gripper_open_name_, step_name);
    }
    open_done_ = true;
    RCLCPP_INFO(node_->get_logger(), "%s: already open (reached=%.3f, opened during the probing).", step_name.c_str(), wrapped.result->position);
    return true;
  }

  // The finger sensor (pin 16) is HIGH whenever the fingers are stalled —
  // on an object or on each other. Before a close nothing is held, so HIGH
  // means the open pulse did not part them (2026-09-16: the arm went,
  // closed on air, "empty"). One more pulse, then give up rather than
  // grasp nothing.
  bool ensure_fingers_open(const std::string & step_name)
  {
    if (!execute_ || !check_grasp_sensor_) { return true; }
    auto settled_low = [this](double window_s) {
      const auto t_end = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(window_s));
      while (rclcpp::ok() && std::chrono::steady_clock::now() < t_end) {
        rclcpp::spin_some(node_);
        if (!grasp_detected_.load()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }
      return !grasp_detected_.load();
    };
    if (settled_low(0.3)) { return true; }
    RCLCPP_WARN(node_->get_logger(),
      "%s: pin %d still HIGH — the fingers are still touching; opening again.", step_name.c_str(), grasp_pin_);
    if (!execute_gripper_named_target(gripper_open_name_, step_name + " (second pulse)")) { return false; }
    if (settled_low(0.8)) { return true; }
    RCLCPP_ERROR(node_->get_logger(),
      "%s: the fingers did not open after two pulses (pin %d HIGH) — not closing on air.", step_name.c_str(), grasp_pin_);
    last_failure_reason_ = "the gripper did not open";
    return false;
  }

  bool execute_gripper_named_target(const std::string & target_name, const std::string & step_name)
  {
    double target_pos = 0.0;
    if (target_name == gripper_open_name_) {
      // Opening releases whatever we held — slip monitoring stands down
      // (covers Step 1, Step 5 at drop, and every recovery open).
      slip_monitor_armed_.store(false);
      held_object_.store(false);
      target_pos = gripper_open_position_;
    } else if (target_name == gripper_close_name_) {
      target_pos = gripper_close_position_;
      close_cmd_t_ = std::chrono::steady_clock::now();
    } else {
      RCLCPP_ERROR(
        node_->get_logger(), "%s: unknown gripper named target '%s'.", step_name.c_str(),
        target_name.c_str());
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "%s: target=%.3f via action '%s'.", step_name.c_str(), target_pos, gripper_action_name_.c_str());
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(), "%s: execute:=false, skipping execution.", step_name.c_str());
      return true;
    }

    const auto wait_timeout = std::chrono::duration<double>(gripper_action_timeout_s_);
    if (!gripper_action_client_->wait_for_action_server(wait_timeout)) {
      RCLCPP_ERROR(node_->get_logger(), "%s: gripper action server unavailable: %s", step_name.c_str(), gripper_action_name_.c_str());
      return false;
    }

    control_msgs::action::GripperCommand::Goal goal;
    goal.command.position = target_pos;
    goal.command.max_effort = 0.0;

    auto goal_future = gripper_action_client_->async_send_goal(goal);
    if (rclcpp::spin_until_future_complete(node_, goal_future, wait_timeout) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "%s: timed out sending gripper goal.", step_name.c_str());
      return false;
    }

    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(node_->get_logger(), "%s: gripper goal rejected.", step_name.c_str());
      return false;
    }

    auto result_future = gripper_action_client_->async_get_result(goal_handle);
    if (rclcpp::spin_until_future_complete(node_, result_future, wait_timeout) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "%s: timed out waiting gripper result.", step_name.c_str());
      return false;
    }

    const auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_ERROR(node_->get_logger(), "%s: gripper action failed with code %d.", step_name.c_str(), static_cast<int>(wrapped.code));
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "%s: execution OK (reached=%.3f).", step_name.c_str(), wrapped.result->position);
    return true;
  }

  void set_speed_slider(double fraction)
  {
    if (!speed_slider_client_) return;
    if (!speed_slider_client_->wait_for_service(std::chrono::milliseconds(50))) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
        "speed_slider service '%s' unavailable; cannot pause/resume.",
        speed_slider_srv_.c_str());
      return;
    }
    auto req = std::make_shared<
      ur_msgs::srv::SetSpeedSliderFraction::Request>();
    req->speed_slider_fraction =
      std::max(0.0, std::min(1.0, fraction));
    // Fire-and-forget — we don't need to block here; the IO callback
    // already drives state transitions.
    speed_slider_client_->async_send_request(req);
  }

  // Pull the latest values for parameters that intent_router updates
  // per-pick via SetParameters. We can't rely on the constructor-time
  // values because they're stale after the first param override. Called
  // once at the start of every pick cycle.
  void refresh_runtime_params()
  {
    try {
      vitamin_mode_active_ =
        node_->get_parameter("vitamin_mode_active").as_bool();
    } catch (...) {}
    try {
      wait_for_destination_ =
        node_->get_parameter("wait_for_destination").as_bool();
    } catch (...) {}
    try {
      wait_for_release_ =
        node_->get_parameter("wait_for_release").as_bool();
    } catch (...) {}
    try {
      drop_target_name_ =
        node_->get_parameter("drop_target_name").as_string();
    } catch (...) {}
    // Blend knobs are re-read per pick so the A/B can be flipped live
    // with `ros2 param set` between cycles, without a relaunch.
    try {
      blend_via_home_ =
        node_->get_parameter("blend_via_home").as_bool();
    } catch (...) {}
    try {
      blend_velocity_scaling_ =
        node_->get_parameter("blend_velocity_scaling").as_double();
    if (node_->has_parameter("drop_hold_s")) {
      drop_hold_s_ = node_->get_parameter("drop_hold_s").as_double();
    }
    if (node_->has_parameter("check_grasp_sensor")) {
      check_grasp_sensor_ = node_->get_parameter("check_grasp_sensor").as_bool();
    }
    if (node_->has_parameter("check_slip")) {
      check_slip_ = node_->get_parameter("check_slip").as_bool();
    }
    if (node_->has_parameter("post_release_wait_s")) {
      post_release_wait_s_ = node_->get_parameter("post_release_wait_s").as_double();
    }
    if (node_->has_parameter("fast_replay_duration_s")) {
      fast_replay_duration_s_ = node_->get_parameter("fast_replay_duration_s").as_double();
    }
    // Approach speed / accel from the GUI's Pick-speeds Apply (2026-09-17:
    // "the execution is way too fast" — Apply changed the node parameter,
    // the GUI read it back, but the curve was still planned with the launch
    // values). A cached plan or background verdict was timed at the old
    // speed, so both are dropped when the value changes.
    {
      double v = velocity_scaling_, a = acceleration_scaling_;
      if (node_->has_parameter("velocity_scaling")) { v = node_->get_parameter("velocity_scaling").as_double(); }
      if (node_->has_parameter("acceleration_scaling")) { a = node_->get_parameter("acceleration_scaling").as_double(); }
      if (v > 0.0 && a > 0.0 && (v != velocity_scaling_ || a != acceleration_scaling_)) {
        RCLCPP_INFO(node_->get_logger(),
          "Approach speed changed in the GUI: velocity %.2f -> %.2f, acceleration %.2f -> %.2f (cached plans dropped).",
          velocity_scaling_, v, acceleration_scaling_, a);
        velocity_scaling_ = v; acceleration_scaling_ = a;
        arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
        arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
        invalidate_cached_lin_plan("approach speed changed in the GUI");
        prechecks_.clear();
      }
    }
    if (node_->has_parameter("staging_mode")) {
      const std::string m = node_->get_parameter("staging_mode").as_string();
      if (!m.empty() && m != staging_mode_) {
        RCLCPP_INFO(node_->get_logger(), "staging_mode changed: %s -> %s (cached plans dropped).", staging_mode_.c_str(), m.c_str());
        staging_mode_ = m;
        invalidate_cached_lin_plan("staging_mode changed");
        prechecks_.clear();
      }
    }
    if (node_->has_parameter("approach_duration_s")) {
      approach_duration_s_ = node_->get_parameter("approach_duration_s").as_double();
    }
    if (node_->has_parameter("gripper_open_time_s")) {
      gripper_open_time_s_ = node_->get_parameter("gripper_open_time_s").as_double();
    }
    if (node_->has_parameter("open_before_arrival_margin_s")) {
      open_before_arrival_margin_s_ = node_->get_parameter("open_before_arrival_margin_s").as_double();
    }
      if (blend_velocity_scaling_ > 0.5) {
        RCLCPP_WARN(node_->get_logger(),
          "blend_velocity_scaling=%.2f exceeds the 0.50 UR3e ceiling; "
          "clamping.", blend_velocity_scaling_);
        blend_velocity_scaling_ = 0.5;
      }
    } catch (...) {}
    // Approach speed, live (2026-09-11, Majd: "let us try 50 %"): the
    // scaling factors were read once at construction, so a parameter set
    // from the GUI or the shell changed nothing until a restart.
    try {
      const double v = node_->get_parameter("velocity_scaling").as_double();
      const double a = node_->get_parameter("acceleration_scaling").as_double();
      if (v > 0.0 && v <= 1.0 && std::fabs(v - velocity_scaling_) > 1e-6) {
        RCLCPP_INFO(node_->get_logger(), "velocity_scaling %.2f -> %.2f (live)", velocity_scaling_, v);
        velocity_scaling_ = v;
        arm_group_.setMaxVelocityScalingFactor(v);
      }
      if (a > 0.0 && a <= 1.0 && std::fabs(a - acceleration_scaling_) > 1e-6) {
        RCLCPP_INFO(node_->get_logger(), "acceleration_scaling %.2f -> %.2f (live)", acceleration_scaling_, a);
        acceleration_scaling_ = a;
        arm_group_.setMaxAccelerationScalingFactor(a);
      }
    } catch (...) {}
    RCLCPP_INFO(node_->get_logger(),
      "refresh_runtime_params: vitamin_mode_active=%s, "
      "wait_for_destination=%s, wait_for_release=%s, drop_target_name='%s', "
      "blend_via_home=%s (velocity_scaling=%.2f).",
      vitamin_mode_active_ ? "true" : "false",
      wait_for_destination_ ? "true" : "false",
      wait_for_release_ ? "true" : "false",
      drop_target_name_.c_str(),
      blend_via_home_ ? "true" : "false",
      blend_velocity_scaling_);
  }

  // ─── Vitamin-tube helpers ──────────────────────────────────────────
  // Set the gripper adapter's close-pulse duration and return the
  // PREVIOUS value (so the caller can restore). Returns -1 on failure
  // — the caller should treat that as "didn't change anything".
  double set_gripper_adapter_close_pulse(double new_seconds)
  {
    auto params_client = std::make_shared<rclcpp::AsyncParametersClient>(
      node_, gripper_adapter_node_name_);
    if (!params_client->wait_for_service(std::chrono::milliseconds(300))) {
      RCLCPP_WARN(node_->get_logger(),
        "set_gripper_adapter_close_pulse: param service not ready.");
      return -1.0;
    }
    // Read current value first so we can return it for restoration.
    double prev = -1.0;
    auto get_fut = params_client->get_parameters({"pulse_hold_s_close"});
    if (rclcpp::spin_until_future_complete(node_, get_fut,
        std::chrono::milliseconds(300)) ==
        rclcpp::FutureReturnCode::SUCCESS) {
      const auto vals = get_fut.get();
      if (!vals.empty()) prev = vals[0].as_double();
    }
    // Apply the new value.
    auto set_fut = params_client->set_parameters({
      rclcpp::Parameter("pulse_hold_s_close", new_seconds)});
    if (rclcpp::spin_until_future_complete(node_, set_fut,
        std::chrono::milliseconds(300)) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_WARN(node_->get_logger(),
        "set_gripper_adapter_close_pulse: timed out applying %.3fs.",
        new_seconds);
      return prev;
    }
    return prev;
  }

  // Fire an asynchronous close pulse so we don't block the trajectory
  // monitor thread waiting for the action result. The result is
  // observed indirectly via the IO state.
  void fire_close_async()
  {
    if (!gripper_action_client_ ||
        !gripper_action_client_->wait_for_action_server(
          std::chrono::milliseconds(200))) {
      RCLCPP_WARN(node_->get_logger(),
        "fire_close_async: gripper action server not ready.");
      return;
    }
    control_msgs::action::GripperCommand::Goal goal;
    goal.command.position = gripper_close_position_;
    goal.command.max_effort = 0.0;
    gripper_action_client_->async_send_goal(goal);
  }

  // Vitamin-tube special Step 4: plan and execute a PTP move to
  // drop_target_name, then while the trajectory is executing watch the
  // controller state; when ~20% of the planned duration has elapsed,
  // slam the speed slider to vitamin_retighten_speed_scaling_ and fire
  // a second (harder) close pulse. After the close pulse completes,
  // restore the speed slider. Re-enables the TI0 no-grasp check at the
  // end so a downstream stage can catch the object slipping out.
  bool execute_arm_to_drop_with_vitamin_retighten(const std::string & target)
  {
    // INVALIDATION: drives the arm to the drop target.
    invalidate_cached_lin_plan("the arm is moving to the drop target");
    // Plan the PTP first so we know the trajectory duration. Use the
    // same path as execute_arm_named_target's PTP branch.
    arm_group_.clearPoseTargets();
    arm_group_.setStartStateToCurrentState();
    arm_group_.setPlanningPipelineId("ompl");
    arm_group_.setPlannerId("RRTConnect");
    if (!arm_group_.setNamedTarget(target)) {
      RCLCPP_ERROR(node_->get_logger(),
        "Vitamin Step 4: setNamedTarget('%s') failed.", target.c_str());
      return false;
    }
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (timed_plan(plan, "Vitamin Step 4 - OMPL to drop") !=
        moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(),
        "Vitamin Step 4: PTP planning to '%s' failed.", target.c_str());
      return false;
    }
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(),
        "Vitamin Step 4: execute:=false; not actually moving.");
      return true;
    }

    double total_dur_s = 0.0;
    if (!plan.trajectory_.joint_trajectory.points.empty()) {
      const auto & last_pt =
        plan.trajectory_.joint_trajectory.points.back();
      total_dur_s = static_cast<double>(last_pt.time_from_start.sec) +
        static_cast<double>(last_pt.time_from_start.nanosec) * 1e-9;
    }
    const double retighten_t_s =
      std::max(0.1, total_dur_s * vitamin_retighten_at_fraction_);
    RCLCPP_INFO(node_->get_logger(),
      "Vitamin Step 4: total_dur=%.2fs, retighten at %.2fs (%.0f%%), "
      "speed_scaling during retighten=%.2f.",
      total_dur_s, retighten_t_s,
      vitamin_retighten_at_fraction_ * 100.0,
      vitamin_retighten_speed_scaling_);

    // SAFER design: spawn a one-shot helper thread that sleeps the
    // retighten delay then performs the slow-and-retighten side effects
    // (speed-slider + extra close pulse). The MAIN thread does the
    // standard synchronous execute(plan). No asyncExecute, no goal-
    // handle ownership issues, no MoveIt re-entrancy.
    std::atomic<bool> trajectory_done{false};
    std::thread retighten_worker([this, retighten_t_s, &trajectory_done]() {
      const auto t0 = std::chrono::steady_clock::now();
      while (!trajectory_done.load() && rclcpp::ok()) {
        auto elapsed_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
        if (elapsed_s >= retighten_t_s) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      if (trajectory_done.load() || !rclcpp::ok()) return;
      RCLCPP_INFO(node_->get_logger(),
        "Vitamin Step 4: hit %.0f%% mark — slowing arm and re-closing "
        "the gripper.", vitamin_retighten_at_fraction_ * 100.0);
      try {
        std_msgs::msg::String s;
        s.data = "vitamin_retighten";
        interactive_status_pub_->publish(s);
      } catch (...) {}
      set_speed_slider(vitamin_retighten_speed_scaling_);
      double saved = set_gripper_adapter_close_pulse(vitamin_retighten_close_s_);
      fire_close_async();
      std::this_thread::sleep_for(std::chrono::duration_cast<
        std::chrono::milliseconds>(std::chrono::duration<double>(
          vitamin_retighten_close_s_ + 0.1)));
      set_speed_slider(nominal_speed_slider_);
      if (saved > 0.0) {
        set_gripper_adapter_close_pulse(saved);
      }
      RCLCPP_INFO(node_->get_logger(),
        "Vitamin Step 4: retighten complete, normal speed restored.");
    });

    // Synchronous execute on the main thread — this is the same call
    // the rest of the executor uses, no special handling needed.
    auto exec_result = execute_with_deadline(plan, __func__);
    trajectory_done.store(true);
    if (retighten_worker.joinable()) {
      retighten_worker.join();
    }
    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(),
        "Vitamin Step 4: execute failed (%d).", exec_result.val);
      return false;
    }
    return true;
  }

  std::array<double, 3> compute_lift_direction_in_target() const
  {
    // Preferred mode: table frame +Y transformed into target frame.
    if (!lift_table_frame_.empty()) {
      try {
        const auto tf = tf_buffer_.lookupTransform(
          target_frame_, lift_table_frame_, tf2::TimePointZero,
          tf2::durationFromSec(tf_timeout_));
        tf2::Quaternion q;
        tf2::fromMsg(tf.transform.rotation, q);
        tf2::Matrix3x3 R(q);
        const tf2::Vector3 y_axis = R * tf2::Vector3(0.0, 1.0, 0.0);
        const double n = y_axis.length();
        if (n > 1e-9) {
          return {y_axis.x() / n, y_axis.y() / n, y_axis.z() / n};
        }
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Lift direction TF lookup failed for lift_table_frame='%s' -> '%s': %s. "
          "Falling back to lift_table_yaw_deg.",
          lift_table_frame_.c_str(), target_frame_.c_str(), ex.what());
      }
    }

    // Fallback: yaw-only in XY plane (table +Y after yaw about +Z).
    const double yaw_rad = lift_table_yaw_deg_ * M_PI / 180.0;
    return {-std::sin(yaw_rad), std::cos(yaw_rad), 0.0};
  }

  rclcpp::Node::SharedPtr node_;
  moveit::planning_interface::MoveGroupInterface & arm_group_;
  tf2_ros::Buffer & tf_buffer_;
  rclcpp_action::Client<control_msgs::action::GripperCommand>::SharedPtr gripper_action_client_;
  moveit_msgs::msg::RobotTrajectory last_drop_plan_trajectory_;

  bool execute_;
  double pregrasp_distance_;
  bool use_pregrasp_staging_{true};
  std::string staging_mode_{"ptp"};
  double blend_radius_m_{0.02};
  double cartesian_path_tolerance_rad_{0.05};
  double cartesian_curve_lead_m_{0.12};
  double close_lead_s_{0.4};
  double approach_duration_s_{2.5};
  std::string last_planner_;   // which approach the last probe planned ("curve", "blended lines", "two lines")
  double cartesian_max_joint_step_rad_{0.8};
  double cartesian_max_joint_travel_rad_{4.36};
  double cartesian_max_motion_s_{4.0};
  double approach_floor_s() const { return std::max(approach_duration_s_, gripper_open_time_s_ + open_before_arrival_margin_s_); }
  bool check_grasp_sensor_{true};
  bool check_slip_{true};
  std::atomic<bool> early_close_fired_{false};   // the close went out on the way in (sensor said apart)
  bool close_pending_{false};
  bool open_done_{false};      // the open of this pick has completed
  std::thread early_close_thread_;
  int cartesian_curve_samples_{24};
  double cartesian_eef_step_m_{0.005};
  rclcpp::Client<moveit_msgs::srv::GetMotionSequence>::SharedPtr sequence_client_;
  moveit::planning_interface::MoveGroupInterface::Plan cached_final_plan_;   // pre-grasp -> grasp (staging_mode lin)
  bool cached_final_plan_valid_{false};
  double pregrasp_approach_distance_{0.10};
  std::string target_frame_;
  double tf_timeout_;
  double planning_time_;
  double lin_feasibility_planning_time_;
  int lin_feasibility_attempts_;
  bool sort_candidates_by_reachability_;
  // Cached for the duration of one candidate-batch evaluation so we
  // don't pay the joint-state-monitor wait on every IK probe.
  mutable moveit::core::RobotStatePtr ik_seed_state_;
  mutable bool ik_seed_unavailable_ = false;
  int num_attempts_;
  bool one_shot_;
  double velocity_scaling_;
  double acceleration_scaling_;
  double drop_hold_s_;
  std::string drop_target_name_;
  std::string reset_target_name_;
  std::string gripper_open_name_;
  std::string gripper_close_name_;
  double gripper_open_position_;
  double gripper_close_position_;
  std::string gripper_action_name_;
  double gripper_action_timeout_s_;
  double gripper_open_time_s_{1.2};
  double post_release_wait_s_{2.0};
  double open_before_arrival_margin_s_{0.5};
  double min_exec_duration_s_{0.0};   // consumed by execute_arm_lin_pose for the approach
  std::chrono::steady_clock::time_point open_sent_t_{};
  std::string gripper_adapter_node_name_;
  bool override_gripper_pulse_from_executor_;
  double executor_pulse_hold_s_open_;
  double executor_pulse_hold_s_close_;
  double lift_height_m_;
  double lift_table_yaw_deg_;
  std::string lift_table_frame_;
  double post_lift_up_m_;
  double post_lift_back_m_;
  bool skip_reset_before_drop_;
  bool blend_via_home_;
  double blend_velocity_scaling_;

  // Interactive drop mode: pause after Step 3 (close gripper) at the
  // grasp pose until a destination arrives on resume_topic_.
  bool wait_for_destination_;
  bool wait_for_release_;
  std::string resume_topic_;
  double resume_timeout_s_;

  // Slip slow-down watchdog
  bool slip_watchdog_enabled_;
  double slip_slow_slider_;            // 0..1 — slider value during slip
  std::string slip_event_topic_;       // topic where slip_start/clear is published
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr slip_event_pub_;
  int slip_pin_;
  // Pin that goes HIGH when the gripper is holding an object. The
  // "no-grasp" condition is logical NOT of this pin.
  int grasp_pin_;
  std::string io_states_topic_;
  std::string speed_slider_srv_;
  double nominal_speed_slider_;
  bool check_no_grasp_after_close_;
  // Vitamin-tube special mode knobs.
  bool vitamin_mode_active_;
  double vitamin_first_close_s_;
  double vitamin_retighten_close_s_;
  double vitamin_retighten_at_fraction_;
  double vitamin_retighten_speed_scaling_;
  rclcpp::Subscription<ur_msgs::msg::IOStates>::SharedPtr io_states_sub_;
  rclcpp::Client<ur_msgs::srv::SetSpeedSliderFraction>::SharedPtr
    speed_slider_client_;
  std::atomic<bool> slip_active_{false};
  // Slip watchdog is only ARMED while the gripper is closed on an object
  // mid-pick. Unarmed, TI1 chatter from manual open/close tests (closing
  // on empty air reads as "slip") must not slow the arm or trigger the
  // "Slip detected" narration (2026-08-11).
  std::atomic<bool> slip_monitor_armed_{false};
  // Human-readable cause of the last failed pick, attached to the
  // "failed|<cycle_id>|<reason>" completion payload for narration.
  std::string last_failure_reason_;
  std::string cycle_done_note_;
  std::mutex hint_mutex_;
  std::string last_hint_;
  std::chrono::steady_clock::time_point last_hint_t_{};
  // Operator abort ("stop" / "cancel" by voice or GUI). Checked between
  // candidates and between sequence steps; also wakes the resume waits.
  // Before this existed, "stop" only cancelled router-owned goals — the
  // executor kept planning/moving while Morty said "stopping" (2026-08-11).
  std::atomic<bool> abort_requested_{false};
  bool stop_after_grasp_ = false;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr abort_sub_;
  bool enable_wrist_twin_rewrite_{false};
  bool prefer_upright_twin_{true};
  double upright_twin_dead_zone_{0.15};
  bool flat_twin_x_up_positive_{true};
  bool upright_twin_y_up_positive_{false};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr candidate_status_pub_;
  // True when the grasp sensor reports holding an object (TI0=HIGH).
  // no-grasp condition = !grasp_detected_.
  std::atomic<bool> grasp_detected_{false};
  std::atomic<bool> slip_paused_{false};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr resume_sub_;
  std::mutex resume_mutex_;
  std::condition_variable resume_cv_;
  std::string resume_destination_;
  bool resume_received_ = false;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr interactive_status_pub_;

  bool use_current_orientation_if_identity_;
  bool use_candidate_pose_array_;
  int max_candidate_tries_;
  bool candidate_try_all_batches_;

  // Fast-replay path (skip MoveIt for trusted named poses).
  bool use_fast_replay_for_named_targets_;
  std::string fast_replay_topic_;
  std::string fast_replay_status_topic_;
  double fast_replay_timeout_s_;
  double fast_replay_duration_s_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fast_replay_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fast_replay_status_sub_;
  // One-shot: the pose about to be executed already passed the candidate
  // LIN probe — skip the duplicate pre-check plan (2026-08-26).
  bool skip_next_precheck_ = false;
  // Rejection autopsy (see collision_detail): ONE service client for
  // the whole node, plus the per-cycle report budget.
  rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedPtr
    state_validity_client_;
  int approach_autopsy_samples_ = 4;
  double prefilter_ik_timeout_s_ = 0.02;
  double lin_slow_retry_factor_ = 0.5;   // second LIN attempt at this fraction of the scaling; 0 = off
  bool clear_stray_scene_objects_ = false;
  // Gripper open started while the candidates are still being probed
  // (2026-09-09): the 0.56 s open used to sit on the critical path after
  // the feasible grasp was found. begin_gripper_open_async() at the start
  // of the search, finish_gripper_open() where Step 1 used to block.
  using GripperGoalHandle = rclcpp_action::ClientGoalHandle<control_msgs::action::GripperCommand>;
  bool open_pending_ = false;
  std::shared_future<GripperGoalHandle::SharedPtr> open_goal_future_;
  std::shared_future<GripperGoalHandle::SharedPtr> close_goal_future_;   // close sent on the way in
  rclcpp_action::Client<moveit_msgs::action::ExecuteTrajectory>::SharedPtr exec_traj_client_;
  int collision_detail_max_reports_{5};
  int collision_detail_reports_ = 0;
  // Hard ceiling on the /check_state_validity wait. A diagnostic may
  // never stall the pick.
  double collision_detail_timeout_s_{0.25};

  // ── Probe-plan reuse cache (2026-08-28) ───────────────────────────
  // The winning candidate's LIN trajectory, kept between the probe and
  // Step 2 so the ~11 s plan is paid for once instead of twice. Guarded
  // by try_reuse_cached_lin_plan(); dropped by
  // invalidate_cached_lin_plan() from every site that could move the arm
  // or change the scene underneath it.
  bool reuse_probe_plan_{true};
  bool cached_lin_plan_valid_ = false;
  geometry_msgs::msg::PoseStamped cached_lin_plan_pose_;
  moveit::planning_interface::MoveGroupInterface::Plan cached_lin_plan_;
  double cached_lin_plan_wall_s_ = 0.0;
  std::chrono::steady_clock::time_point cached_lin_plan_t_{};
  double cached_plan_pos_tol_m_{1e-4};
  double cached_plan_ori_tol_rad_{1e-3};
  // MoveIt's own trajectory_execution/allowed_start_tolerance default.
  double cached_plan_start_joint_tol_rad_{0.01};
  double cached_plan_max_age_s_{30.0};
  int cached_plan_validate_samples_{5};
  double cached_plan_validate_timeout_s_{0.5};

  // background LIN check state (see PrecheckRecord)
  bool precheck_mode_ = false;
  bool precheck_interrupted_ = false;
  std::function<bool()> precheck_interrupt_;
  size_t precheck_tried_ = 0;
  std::deque<PrecheckRecord> prechecks_;
  size_t precheck_max_sets_ = 8;

  bool arm_joints_now(std::vector<double> * out)
  {
    sensor_msgs::msg::JointState js;
    if (!wait_fresh_arm_joint_state(0.5, js)) { return false; }
    out->clear();
    for (const auto & name : arm_group_.getJointNames()) {
      bool found = false;
      for (size_t i = 0; i < js.name.size() && i < js.position.size(); ++i) {
        if (js.name[i] == name) { out->push_back(js.position[i]); found = true; break; }
      }
      if (!found) { return false; }
    }
    return true;
  }

  // Sensor-direct joint feed for execute_fast_joint_lift (see there).
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  std::mutex joint_state_mutex_;
  sensor_msgs::msg::JointState last_joint_state_;
  std::atomic<uint64_t> joint_state_seq_{0};
  std::mutex fast_replay_mutex_;
  std::condition_variable fast_replay_status_cv_;
  std::string fast_replay_last_status_;

  // Collision-checked fast-replay path via /move_to_named_pose action.
  bool use_named_pose_action_;
  std::string named_pose_action_name_;
  rclcpp_action::Client<morty_interfaces::action::MoveToNamedPose>::SharedPtr
    named_pose_action_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("grasp_executor_lin_llm", options);

  const auto planning_group = get_or_declare_param<std::string>(node, "planning_group", "ur3e_manipulator");
  const auto end_effector_link = get_or_declare_param<std::string>(node, "end_effector_link", "gripper_tcp");
  const auto move_group_node = get_or_declare_param<std::string>(node, "move_group_node", "move_group");
  const auto grasp_pose_topic = get_or_declare_param<std::string>(node, "grasp_pose_topic", "/grasp_pose");
  const auto use_candidate_pose_array = get_or_declare_param<bool>(node, "use_candidate_pose_array", true);
  const auto candidate_wait_timeout_s =
    std::max(0.0, get_or_declare_param<double>(node, "candidate_wait_timeout_s", 0.35));
  const auto candidate_pose_topic =
    get_or_declare_param<std::string>(node, "candidate_pose_topic", "/graspgen/candidate_poses_base_link");
  const auto completion_topic =
    get_or_declare_param<std::string>(node, "completion_topic", "/grasp_executor_lin_llm/cycle_done");
  const auto selection_cycle_id_topic =
    get_or_declare_param<std::string>(node, "selection_cycle_id_topic", "/graspgen/selection_cycle_id");
  const auto execution_gate_topic =
    get_or_declare_param<std::string>(
      node, "execution_gate_topic", "/grasp_executor_lin_llm/execution_enabled");

  copy_moveit_params_from_move_group(node, move_group_node, planning_group);

  tf2_ros::Buffer tf_buffer(node->get_clock());
  tf2_ros::TransformListener tf_listener(tf_buffer);

  moveit::planning_interface::MoveGroupInterface arm_group(node, planning_group);
  arm_group.setEndEffectorLink(end_effector_link);

  RCLCPP_INFO(node->get_logger(),
    "Grasp Executor (LIN-only) ready. pose='%s', candidates='%s' (enabled=%s), planning_group=%s, ee=%s",
    grasp_pose_topic.c_str(), candidate_pose_topic.c_str(), (use_candidate_pose_array ? "true" : "false"),
    planning_group.c_str(), end_effector_link.c_str());
  auto pub_completion = node->create_publisher<std_msgs::msg::String>(completion_topic, 10);
  auto pub_selected_pose = node->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/grasp_executor_lin_llm/selected_pose", rclcpp::QoS(1).reliable().transient_local());
  std::atomic<int64_t> latest_cycle_id{0};
  std::atomic<int64_t> last_processed_cycle_id{-1};
  std::atomic_bool execution_enabled{true};

  GraspExecutorLin grasp_executor(node, arm_group, tf_buffer);

  // Dedicated abort listener on its OWN node + spinner thread. The main
  // node's callbacks only run at explicit spin points — during candidate
  // probing the main thread is wall-to-wall plan() calls, so a voice
  // "cancel" sat unread while candidates 42..46 kept churning (Majd,
  // 2026-08-12). This thread hears the abort within milliseconds, always.
  auto abort_listener = std::make_shared<rclcpp::Node>(
    "grasp_executor_abort_listener");
  auto abort_sub_dedicated =
    abort_listener->create_subscription<std_msgs::msg::String>(
      "/grasp_executor_lin_llm/abort", 10,
      [&grasp_executor](const std_msgs::msg::String::SharedPtr) {
        grasp_executor.request_abort();
      });
  // Same node also watches /rosout for MoveIt's own diagnosis lines
  // (exact collision pair, violated joint limit) so rejections can say
  // WHAT collided instead of just "path collides".
  auto rosout_sub = abort_listener->create_subscription<
    rcl_interfaces::msg::Log>(
    // Best-effort subscriber is compatible with both best-effort and
    // reliable rosout publishers. A reliable-only subscription silently
    // missed diagnostics from some MoveIt launch configurations.
    "/rosout", rclcpp::QoS(200).best_effort().durability_volatile(),
    [&grasp_executor](const rcl_interfaces::msg::Log::SharedPtr log) {
      const std::string & m = log->msg;
      // "Found a contact between 'A' (type ...) and 'B' (type ...)"
      auto a0 = m.find("Found a contact between '");
      if (a0 != std::string::npos) {
        a0 += std::strlen("Found a contact between '");
        const auto a1 = m.find('\'', a0);
        auto b0 = m.find('\'', m.find(" and '", a1));
        if (a1 != std::string::npos && b0 != std::string::npos) {
          ++b0;
          const auto b1 = m.find('\'', b0);
          if (b1 != std::string::npos) {
            grasp_executor.note_moveit_hint(
              "contact " + m.substr(a0, a1 - a0) + " <-> " +
              m.substr(b0, b1 - b0));
            return;
          }
        }
      }
      // Pilz format (real example):
      // "Joint acceleration limit of elbow_joint violated. ... Actual joint
      // acceleration is -6.01448, while the limit is 5."
      const char * limit_kind =
        (m.find("Joint acceleration limit") != std::string::npos) ? "acceleration" :
        (m.find("Joint velocity limit") != std::string::npos) ? "velocity" :
        (m.find("Joint deceleration limit") != std::string::npos) ? "deceleration" : nullptr;
      if (limit_kind && m.find("violated") != std::string::npos) {
        std::string joint = "unknown joint";
        const auto j0 = m.find("limit of ");
        if (j0 != std::string::npos) {
          const auto begin = j0 + std::strlen("limit of ");
          const auto end = m.find(" violated", begin);
          if (end != std::string::npos) { joint = m.substr(begin, end - begin); }
        }
        std::string detail = joint + " " + limit_kind + " limit exceeded";
        try {
          const std::string actual_key = std::string("Actual joint ") + limit_kind + " is ";
          const auto a0 = m.find(actual_key);
          const auto l0 = m.find("while the limit is ");
          if (a0 != std::string::npos && l0 != std::string::npos) {
            const double actual = std::abs(std::stod(m.substr(a0 + actual_key.size())));
            const double limit = std::abs(std::stod(m.substr(l0 + std::strlen("while the limit is "))));
            char values[96];
            std::snprintf(values, sizeof(values), " (actual %.2f > limit %.2f)", actual, limit);
            detail += values;
          }
        } catch (...) {}
        grasp_executor.note_moveit_hint(detail);
        return;
      }
      if (m.find("Unable to find IK solution") != std::string::npos) {
        grasp_executor.note_moveit_hint(
          "no IK at an intermediate point of the straight approach");
        return;
      }
      const auto ik_link = m.find("Failed to compute inverse kinematics for link: ");
      if (ik_link != std::string::npos) {
        const auto begin = ik_link + std::strlen("Failed to compute inverse kinematics for link: ");
        const auto end = m.find_first_of(" .", begin);
        grasp_executor.note_moveit_hint(
          "no IK for " + m.substr(begin, end - begin) + " along the straight approach");
        return;
      }
      if (m.find("trajectory is in collision") != std::string::npos ||
          m.find("Trajectory is in collision") != std::string::npos) {
        grasp_executor.note_moveit_hint(
          "LIN trajectory is in collision (MoveIt did not name the contact pair)");
      }
    });
  std::thread([abort_listener]() {
    rclcpp::spin(abort_listener);
  }).detach();

  std::mutex mutex;
  std::optional<geometry_msgs::msg::PoseStamped> pending_grasp;
  std::optional<std::chrono::steady_clock::time_point> pending_grasp_received_at;
  std::optional<std::vector<geometry_msgs::msg::PoseStamped>> pending_candidates;
  // Cycle ID observed at the moment candidates/grasp were buffered. The
  // pipeline announces a new cycle first and publishes its final candidates
  // last, so an arrival tagged with an unprocessed cycle is ready to evaluate.
  int64_t pending_candidates_cycle_at_arrival = -1;
  int64_t pending_grasp_cycle_at_arrival = -1;
  bool processing = false;

  auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  auto sub_execution_gate = node->create_subscription<std_msgs::msg::Bool>(
    execution_gate_topic,
    latched_qos,
    [&](const std_msgs::msg::Bool::SharedPtr msg) {
      const bool previous = execution_enabled.exchange(
        static_cast<bool>(msg->data), std::memory_order_relaxed);
      if (previous != static_cast<bool>(msg->data)) {
        RCLCPP_INFO(
          node->get_logger(),
          "Execution gate %s — %s.",
          msg->data ? "OPEN" : "CLOSED",
          msg->data ? "buffered manual grasp may execute" :
          "manual grasp will wait for the GUI Execute button");
      }
    });
  // Everything on the candidate topics is TRANSIENT_LOCAL, so a (re)started
  // executor gets the PREVIOUS cycle's candidates replayed as if fresh —
  // it then drives to where an object USED to be and poisons the current
  // cycle_id as "processed" (vitamin-tube ghost pick, 2026-08-12). Reject
  // any candidate message stamped before this executor came up.
  const rclcpp::Time executor_started_at = node->now();
  // Receipt time of the active cycle-id announcement. Candidate topics are
  // transient-local and may deliver the previous cycle's final sample just
  // after a new cycle begins. A candidate generated for this cycle must have
  // a header stamp newer than this boundary.
  rclcpp::Time active_cycle_started_at = executor_started_at;
  auto is_stale_replay = [executor_started_at, node](
      const std_msgs::msg::Header & h, const char * what) {
    const rclcpp::Time stamp(h.stamp);
    if (stamp.nanoseconds() > 0 && stamp < executor_started_at) {
      RCLCPP_WARN_ONCE(node->get_logger(),
        "Ignoring latched %s stamped before executor start "
        "(stale replay from a previous cycle).", what);
      return true;
    }
    return false;
  };
  auto sub_cycle_id = node->create_subscription<std_msgs::msg::Int64>(
    selection_cycle_id_topic,
    latched_qos,
    [&](const std_msgs::msg::Int64::SharedPtr msg) {
      const int64_t new_id = static_cast<int64_t>(msg->data);
      const int64_t old_id =
        latest_cycle_id.exchange(new_id, std::memory_order_relaxed);
      if (new_id != old_id) {
        // A NEW cycle begins: whatever sits in the buffers belongs to the
        // previous object — the keep-alive republish restamps old
        // candidates as fresh, so timestamps can't be trusted across the
        // bump (ghost pick #2, 2026-08-12). Drop them; the pipeline will
        // publish this cycle's candidates when generation finishes.
        std::lock_guard<std::mutex> lock(mutex);
        active_cycle_started_at = node->now();
        // INVALIDATION: a new selection cycle targets a different object.
        // Any trajectory kept from the previous cycle's probe is now for
        // the wrong grasp and must never be reachable.
        grasp_executor.invalidate_cached_lin_plan("a new selection cycle_id arrived");
        // A NEW pick supersedes any stale stop (2026-08-31, live repro:
        // "Stop, reset." produced MULTIPLE abort publishes — stop intent
        // + reset intent — and the stragglers arriving after clear_abort
        // latched the flag, so the operator's next TWO picks died
        // instantly with "stopped by your command" while the arm never
        // moved. A fresh selection is an explicit operator command; a
        // stop belongs to the cycle it was said in.)
        grasp_executor.clear_abort();
        grasp_executor.clear_failure_reason();
        if (pending_candidates.has_value() || pending_grasp.has_value()) {
          RCLCPP_INFO(node->get_logger(),
            "New selection cycle_id=%ld — discarding buffered "
            "candidates/grasp from the previous cycle.",
            static_cast<long>(new_id));
        }
        pending_candidates.reset();
        pending_grasp.reset();
        pending_candidates_cycle_at_arrival = -1;
        pending_grasp_cycle_at_arrival = -1;
      }
    });

  auto publish_completion = [&](bool ok, int64_t cycle_id) {
    std_msgs::msg::String msg;
    msg.data = std::string(ok ? "done" : "failed") + "|" + std::to_string(cycle_id);
    if (!ok && !grasp_executor.last_failure_reason().empty()) {
      msg.data += "|" + grasp_executor.last_failure_reason();
    }
    if (ok && !grasp_executor.cycle_done_note().empty()) {
      // e.g. "done|7|stayed:drop_left_side" — payload beats topic races.
      msg.data += "|" + grasp_executor.cycle_done_note();
    }
    pub_completion->publish(msg);
  };

  // Background LIN check (2026-09-11): candidate sets from the pipeline's
  // precompute cache, checked while the arm is idle; verdicts back to the
  // pipeline so a set with no feasible grasp is recomputed before anyone
  // selects it. Off with lin_precheck_enabled:=false.
  const bool precheck_enabled =
    get_or_declare_param<bool>(node, "lin_precheck_enabled", true);
  std::deque<std::pair<int64_t, std::vector<geometry_msgs::msg::PoseStamped>>> pending_prechecks;
  bool precheck_paused_logged = false;
  auto pub_precheck_result = node->create_publisher<std_msgs::msg::String>(
    "/grasp_executor_lin_llm/precheck_result", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
  auto sub_precheck = node->create_subscription<geometry_msgs::msg::PoseArray>(
    "/graspgen/precheck_candidates", rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
    [&](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
      if (!precheck_enabled || msg->poses.empty()) { return; }
      const int64_t key = static_cast<int64_t>(msg->header.stamp.sec);   // the pipeline's entry id
      std::vector<geometry_msgs::msg::PoseStamped> cands;
      cands.reserve(msg->poses.size());
      for (const auto & p : msg->poses) {
        geometry_msgs::msg::PoseStamped ps; ps.header = msg->header; ps.pose = p;
        cands.push_back(ps);
      }
      std::lock_guard<std::mutex> lock(mutex);
      pending_prechecks.erase(
        std::remove_if(pending_prechecks.begin(), pending_prechecks.end(),
          [&](const auto & j) { return j.first == key; }),
        pending_prechecks.end());
      pending_prechecks.emplace_back(key, std::move(cands));
      while (pending_prechecks.size() > 16) { pending_prechecks.pop_front(); }
    });

  auto sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    grasp_pose_topic,
    latched_qos,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex);
      if (processing) {
        return;
      }
      if (is_stale_replay(msg->header, "grasp pose")) {
        return;
      }
      const rclcpp::Time pose_stamp(msg->header.stamp);
      if (pose_stamp.nanoseconds() > 0 && pose_stamp <= active_cycle_started_at) {
        RCLCPP_WARN(node->get_logger(),
          "Ignoring grasp pose stamped before cycle_id=%ld began.",
          static_cast<long>(latest_cycle_id.load(std::memory_order_relaxed)));
        return;
      }
      // Buffer the grasp for the currently announced, unprocessed cycle.
      // The dispatcher gives the candidate array a short chance to arrive
      // before falling back to this single pose.
      pending_grasp = *msg;
      pending_grasp_received_at = std::chrono::steady_clock::now();
      pending_grasp_cycle_at_arrival =
        latest_cycle_id.load(std::memory_order_relaxed);
    });

  std::shared_ptr<rclcpp::Subscription<geometry_msgs::msg::PoseArray>> sub_candidates;
  if (use_candidate_pose_array) {
    sub_candidates = node->create_subscription<geometry_msgs::msg::PoseArray>(
      candidate_pose_topic,
      latched_qos,
      [&](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex);
        if (processing) {
          return;
        }
        // Ignore empty PoseArrays — these are pipeline status frames,
        // not real grasp candidates. The pipeline publishes a real
        // candidate set right after the final filter runs.
        if (msg->poses.empty()) {
          return;
        }
        if (is_stale_replay(msg->header, "candidate PoseArray")) {
          return;
        }
        const rclcpp::Time candidate_stamp(msg->header.stamp);
        if (candidate_stamp.nanoseconds() > 0 &&
            candidate_stamp <= active_cycle_started_at)
        {
          RCLCPP_WARN(node->get_logger(),
            "Ignoring cached candidate PoseArray stamped before cycle_id=%ld began.",
            static_cast<long>(latest_cycle_id.load(std::memory_order_relaxed)));
          return;
        }
        const int64_t cid_now =
          latest_cycle_id.load(std::memory_order_relaxed);
        // Don't re-buffer candidates from a cycle we've already
        // processed — avoids log spam from latched republishes.
        if (cid_now == last_processed_cycle_id) {
          return;
        }
        std::vector<geometry_msgs::msg::PoseStamped> cands;
        cands.reserve(msg->poses.size());
        for (const auto & p : msg->poses) {
          geometry_msgs::msg::PoseStamped ps;
          ps.header = msg->header;
          ps.pose = p;
          cands.push_back(ps);
        }
        // Only log when the buffer state actually changes — avoids
        // hundreds of identical "Buffered N candidates" lines.
        const bool first_buffer = !pending_candidates.has_value();
        const bool size_changed = pending_candidates.has_value() &&
          pending_candidates->size() != cands.size();
        pending_candidates = std::move(cands);
        pending_candidates_cycle_at_arrival = cid_now;
        if (first_buffer || size_changed) {
          RCLCPP_INFO(
            node->get_logger(),
            "Buffered %zu candidates for active cycle_id=%ld; %s.",
            pending_candidates->size(),
            static_cast<long>(pending_candidates_cycle_at_arrival),
            execution_enabled.load(std::memory_order_relaxed) ?
            "ready for evaluation" : "waiting for manual approval");
        }
      });
  }

  rclcpp::Rate rate(50);
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    // RESET-WHILE-GATED (2026-08-27): pressing Reset during "waiting for
    // manual approval" set abort_requested_, but this loop only watched
    // the execution gate — the buffered cycle survived every press (Majd
    // clicked 4x, nothing happened) and the STALE flag would then have
    // killed the NEXT legitimate pick at its first checkpoint. Consume
    // the abort here: discard the gated cycle and publish its cycle_done
    // so the pipeline unblocks; with nothing buffered, just clear the
    // stray flag.
    if (!processing && grasp_executor.abort_pending()) {
      std::lock_guard<std::mutex> lock(mutex);
      if (pending_candidates.has_value()) {
        const int64_t cid = pending_candidates_cycle_at_arrival;
        RCLCPP_WARN(node->get_logger(),
          "Reset while awaiting approval — discarding %zu buffered "
          "candidates for cycle_id=%ld.",
          pending_candidates->size(), static_cast<long>(cid));
        pending_candidates.reset();
        pending_candidates_cycle_at_arrival = -1;
        pending_grasp.reset();
        last_processed_cycle_id = cid;
        grasp_executor.set_failure_reason("stopped by your command");
        publish_completion(false, cid);
      }
      grasp_executor.clear_abort();
    }

    std::optional<geometry_msgs::msg::PoseStamped> grasp;
    std::optional<std::vector<geometry_msgs::msg::PoseStamped>> grasp_candidates;
    {
      std::lock_guard<std::mutex> lock(mutex);

      // Pipeline contract: cycle_id is published FIRST, candidates LAST.
      // The arrival of the PoseArray itself is the "ready" signal. The
      // cycle_id at arrival must just be a fresh cycle (i.e. different
      // from the last one we processed) — we don't need a strictly-newer
      // bump after, because the cycle_id is already final by the time
      // the pipeline reaches the publish step.
      const bool candidates_ready =
        pending_candidates.has_value() &&
        pending_candidates_cycle_at_arrival != last_processed_cycle_id &&
        execution_enabled.load(std::memory_order_relaxed);

      if (candidates_ready && !processing) {
        processing = true;
        grasp_candidates = pending_candidates;
        pending_candidates.reset();
        pending_candidates_cycle_at_arrival = -1;
        pending_grasp.reset();
        pending_grasp_received_at.reset();
        pending_grasp_cycle_at_arrival = -1;
      } else if (pending_grasp.has_value() && !processing) {
        const bool grasp_ready =
          pending_grasp_cycle_at_arrival != last_processed_cycle_id &&
          execution_enabled.load(std::memory_order_relaxed);
        if (use_candidate_pose_array) {
          if (!pending_grasp_received_at.has_value()) {
            pending_grasp_received_at = std::chrono::steady_clock::now();
          }
          const double waited_s =
            std::chrono::duration<double>(
            std::chrono::steady_clock::now() - *pending_grasp_received_at).count();
          if (grasp_ready && waited_s >= candidate_wait_timeout_s) {
            RCLCPP_WARN(
              node->get_logger(),
              "No candidate PoseArray received within %.2fs; falling back to single grasp pose topic.",
              candidate_wait_timeout_s);
            processing = true;
            grasp = pending_grasp;
            pending_grasp.reset();
            pending_grasp_received_at.reset();
            pending_grasp_cycle_at_arrival = -1;
          }
        } else if (grasp_ready) {
          processing = true;
          grasp = pending_grasp;
          pending_grasp.reset();
          pending_grasp_received_at.reset();
          pending_grasp_cycle_at_arrival = -1;
        }
      }
    }

    if (grasp_candidates.has_value()) {
      const int64_t cycle_id = latest_cycle_id.load(std::memory_order_relaxed);
      if (cycle_id == last_processed_cycle_id) {
        RCLCPP_WARN(node->get_logger(),
          "Skipping already-processed cycle_id=%ld (waiting for new selection).",
          static_cast<long>(cycle_id));
        std::lock_guard<std::mutex> lock(mutex);
        processing = false;
        rate.sleep();
        continue;
      }
      geometry_msgs::msg::PoseStamped selected;
      const auto & cands = *grasp_candidates;
      RCLCPP_INFO(
        node->get_logger(),
        "Received %zu grasp candidates; evaluating LIN feasibility (cycle_id=%ld).",
        cands.size(), static_cast<long>(cycle_id));
      // Open the fingers NOW, when the candidates arrive — not at the click
      // (Majd 2026-09-17: "we open the gripper before we execute, that is
      // wrong": the click ran 3-5 s ahead of the move, so the fingers stood
      // open in front of a still arm). Candidates mean a pick is really
      // coming; the probe (~1.3 s) and the approach (~1 s) cover the
      // 1.2 s + 0.5 s the fingers need. Never while an object is held.
      grasp_executor.clear_abort();
      if (grasp_executor.holding_object()) {
        // The previous pick was stopped while holding (Reset during the
        // "where do I drop it?" wait). Probing would open the gripper early
        // and drop the object wherever the arm is (2026-09-11: a bottle
        // released mid-air), and every plan from that pose fails anyway.
        RCLCPP_ERROR(node->get_logger(),
          "Still holding an object (grasp pin HIGH) — not starting a new pick. "
          "Choose where to drop it (Drop left / Drop right / Release in the GUI, "
          "or say it), or open the gripper yourself.");
        grasp_executor.set_failure_reason(
          "still holding the previous object — tell me where to drop it first");
        last_processed_cycle_id = cycle_id;
        publish_completion(false, cycle_id);
        { std::lock_guard<std::mutex> lock(mutex); processing = false; }
        rate.sleep();
        continue;
      }
      bool feasible = false;
      bool used_precheck = false;
      if (precheck_enabled) {
        const auto * rec = grasp_executor.match_precheck(cands);
        std::string why;
        if (rec == nullptr) {
          RCLCPP_INFO(node->get_logger(),
            "[precheck] MISS: no background check for this candidate set — probing now.");
        } else if (!grasp_executor.precheck_usable(*rec, &why)) {
          RCLCPP_INFO(node->get_logger(), "[precheck] MISS: %s — probing now.", why.c_str());
        } else if (rec->feasible_idx < 0) {
          RCLCPP_INFO(node->get_logger(),
            "[precheck] HIT but none of the first %zu candidates was feasible %.0f s ago — probing again.",
            rec->n_tried, grasp_executor.precheck_age_s(*rec));
        } else {
          selected = cands[static_cast<size_t>(rec->feasible_idx)];
          grasp_executor.apply_precheck(*rec);
          grasp_executor.announce_precheck_hit(rec->feasible_idx, cands.size(), selected, rec->planner);
          feasible = true;
          used_precheck = true;
          RCLCPP_INFO(node->get_logger(),
            "[precheck] HIT: candidate %d of %zu was LIN-verified %.0f s ago — executing without probing (%s).",
            rec->feasible_idx + 1, cands.size(), grasp_executor.precheck_age_s(*rec),
            rec->has_plan ? "plan reused" : "plan recomputed");
        }
      }
      if (!used_precheck) {
        feasible = grasp_executor.select_first_lin_feasible(cands, &selected);
      }
      bool ok = false;
      if (!feasible) {
        RCLCPP_ERROR(node->get_logger(), "No LIN-feasible grasp candidate found.");
        if (grasp_executor.last_failure_reason().empty()) {
          grasp_executor.set_failure_reason(
            "none of the grasp candidates was reachable in a straight line "
            "from where my arm is now — try sending me home first");
        }
      } else {
        selected.header.stamp = node->now();
        pub_selected_pose->publish(selected);
        RCLCPP_INFO(node->get_logger(),
          "Published selected pose on '/grasp_executor_lin_llm/selected_pose': [%.3f, %.3f, %.3f]",
          selected.pose.position.x, selected.pose.position.y, selected.pose.position.z);
        ok = grasp_executor.execute_grasp(selected);
      }
      last_processed_cycle_id = cycle_id;
      publish_completion(ok, cycle_id);
      if (!ok) {
        RCLCPP_ERROR(node->get_logger(), "Grasp execution failed.");
      } else {
        RCLCPP_INFO(node->get_logger(), "Grasp sequence finished.");
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        processing = false;
      }
      if (grasp_executor.is_one_shot()) {
        RCLCPP_INFO(node->get_logger(), "One-shot mode: exiting after first execution.");
        break;
      }
    } else if (grasp.has_value()) {
      const int64_t cycle_id = latest_cycle_id.load(std::memory_order_relaxed);
      if (cycle_id == last_processed_cycle_id) {
        RCLCPP_WARN(node->get_logger(),
          "Skipping already-processed cycle_id=%ld (waiting for new selection).",
          static_cast<long>(cycle_id));
        std::lock_guard<std::mutex> lock(mutex);
        processing = false;
        rate.sleep();
        continue;
      }
      RCLCPP_INFO(node->get_logger(), "Grasp pose received, will process.");
      bool ok = grasp_executor.execute_grasp(*grasp);
      last_processed_cycle_id = cycle_id;
      publish_completion(ok, cycle_id);
      if (!ok) {
        RCLCPP_ERROR(node->get_logger(), "Grasp execution failed.");
      } else {
        RCLCPP_INFO(node->get_logger(), "Grasp sequence finished.");
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        processing = false;
      }
      if (grasp_executor.is_one_shot()) {
        RCLCPP_INFO(node->get_logger(), "One-shot mode: exiting after first execution.");
        break;
      }
    }

    if (precheck_enabled && !grasp_candidates.has_value() && !grasp.has_value()) {
      std::optional<std::pair<int64_t, std::vector<geometry_msgs::msg::PoseStamped>>> job;
      {
        std::lock_guard<std::mutex> lock(mutex);
        // Only a real CANDIDATE set stops a background check. pending_grasp
        // is the single-pose fallback topic, which the pipeline republishes
        // continuously (measured 2026-09-11: every check "stopped after 0
        // candidates" because of it); it is consumed only after the
        // candidate-wait timeout and never while candidate sets flow.
        if (!processing && !pending_candidates.has_value() &&
            !pending_prechecks.empty() && !grasp_executor.abort_pending())
        {
          job = std::move(pending_prechecks.front());
          pending_prechecks.pop_front();
        }
      }
      if (job.has_value()) {
        std::string not_home;
        if (!grasp_executor.arm_at_reset_target(&not_home)) {
          // Put it back for later, keep the queue order, and say it once.
          std::lock_guard<std::mutex> lock(mutex);
          pending_prechecks.push_front(std::move(*job));
          if (!precheck_paused_logged) {
            RCLCPP_INFO(node->get_logger(),
              "[precheck] paused: %s — background checks resume at home.", not_home.c_str());
            precheck_paused_logged = true;
          }
          rate.sleep();
          continue;
        }
        precheck_paused_logged = false;
        auto interrupt = [&]() {
          std::lock_guard<std::mutex> lk(mutex);
          return pending_candidates.has_value();
        };
        GraspExecutorLin::PrecheckRecord rec;
        const auto t0 = std::chrono::steady_clock::now();
        const bool done = grasp_executor.run_precheck(job->second, job->first, interrupt, &rec);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!done) {
          RCLCPP_INFO(node->get_logger(),
            "[precheck] set %ld stopped after %zu candidates (%.2f s): a selection or a stop came first.",
            static_cast<long>(job->first), rec.n_tried, dt);
        } else {
          std_msgs::msg::String r;
          r.data = "key=" + std::to_string(job->first) + " feasible=" + std::to_string(rec.feasible_idx) +
            " tried=" + std::to_string(rec.n_tried);
          if (rec.feasible_idx >= 0 && !rec.planner.empty()) {
            std::string tok = rec.planner;
            std::replace(tok.begin(), tok.end(), ' ', '_');
            r.data += " planner=" + tok;     // parsed by the pipeline's precompute (no spaces)
          }
          if (rec.feasible_idx < 0 && !rec.reason.empty()) { r.data += " reason=" + rec.reason; }
          pub_precheck_result->publish(r);
          if (rec.feasible_idx >= 0) {
            RCLCPP_INFO(node->get_logger(),
              "[precheck] set %ld: candidate %d of %zu is LIN-feasible (%zu probed, %.2f s) — verdict kept.",
              static_cast<long>(job->first), rec.feasible_idx + 1, job->second.size(), rec.n_tried, dt);
          } else {
            RCLCPP_WARN(node->get_logger(),
              "[precheck] set %ld: none of the first %zu candidates is LIN-feasible (%.2f s) — pipeline told to recompute.",
              static_cast<long>(job->first), rec.n_tried, dt);
          }
          grasp_executor.remember_precheck(std::move(rec));
        }
      }
    }

    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
