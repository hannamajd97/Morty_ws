#include <algorithm>
#include <chrono>
#include <cmath>
#include <atomic>
#include <array>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/string.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/parameter_client.hpp>
#include <control_msgs/control_msgs/action/gripper_command.hpp>
#include <control_msgs/control_msgs/action/follow_joint_trajectory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <morty_interfaces/action/move_to_named_pose.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/utils/moveit_error_code.h>
#include <moveit_msgs/msg/contact_information.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
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
    sort_candidates_by_reachability_ = get_or_declare_param<bool>(
      node_, "sort_candidates_by_reachability", true);
    num_attempts_ = get_or_declare_param<int>(node_, "num_attempts", 10);
    // Manual-button mode (Majd 2026-08-26): grasp and HOLD at the
    // grasp pose — no lift/transport/drop/reset; the operator drives
    // everything after with the GUI buttons.
    stop_after_grasp_ = get_or_declare_param<bool>(
      node_, "stop_after_grasp", false);
    one_shot_ = get_or_declare_param<bool>(node_, "one_shot", true);
    velocity_scaling_ = get_or_declare_param<double>(node_, "velocity_scaling", 0.5);
    acceleration_scaling_ = get_or_declare_param<double>(node_, "acceleration_scaling", 0.5);
    drop_hold_s_ = get_or_declare_param<double>(node_, "drop_hold_s", 5.0);
    drop_target_name_ = get_or_declare_param<std::string>(node_, "drop_target_name", "drop_right_side");
    reset_target_name_ = get_or_declare_param<std::string>(node_, "reset_target_name", "home");
    // BLENDED transport (2026-08-28), mirrored from grasp_executor_lin_llm.
    // Step 3b (to home) and Step 4 (to drop) become ONE MoveToNamedPose
    // goal with via_poses=[home]: the named-pose server turns it into a
    // single dense, S-curve timed trajectory, so the arm flows through
    // home instead of stopping there and the ~2 s of per-goal fixed
    // overhead is paid once instead of twice. Automatic fallback to the
    // two-step sequence on any failure.
    blend_via_home_ = get_or_declare_param<bool>(
      node_, "blend_via_home", true);
    // Speed tier charged once per blended segment. MEDIUM (Majd
    // 2026-08-28, second pass: "make it medium as before and keep it
    // smooth") -> 0.3 = the NORMAL tier (2.7 s) x 2 segments = 5.4 s of
    // motion under ONE overhead. Measured worst joint against its UR3e
    // limit on the real taught poses: 76% at 0.50, 61% at 0.40, 45% at
    // 0.30 — the default now sits at that 45%. Raise to 0.4 only if the
    // researcher asks for more pace. Never above 0.50.
    blend_velocity_scaling_ = get_or_declare_param<double>(
      node_, "blend_velocity_scaling", 0.3);
    if (blend_velocity_scaling_ > 0.5) {
      RCLCPP_WARN(node_->get_logger(),
        "blend_velocity_scaling=%.2f exceeds the 0.50 UR3e ceiling; "
        "clamping.", blend_velocity_scaling_);
      blend_velocity_scaling_ = 0.5;
    }
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
    use_current_orientation_if_identity_ =
      get_or_declare_param<bool>(node_, "use_current_orientation_if_identity", true);
    use_candidate_pose_array_ = get_or_declare_param<bool>(node_, "use_candidate_pose_array", true);
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
    // Created ONCE: a client per candidate would be slow and would
    // leak service endpoints for the life of the node.
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

    // Clear perception collision objects before approaching — the target object
    // itself is published as a collision obstacle and would block the grasp LIN.
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

    // THINK FIRST, THEN OPEN, THEN MOVE (Majd 2026-08-26) — see the
    // llm executor for the rationale. Plan the approach while the
    // fingers are still closed; open only when the path exists.
    moveit::planning_interface::MoveGroupInterface::Plan approach_plan;
    bool approach_planned = false;
    // FIRST: can we simply execute the plan the candidate probe already
    // paid 6-12 s for? (2026-08-28 — this duplicate was ~9.8 s of the
    // wall clock of every pick.) try_reuse_cached_lin_plan refuses unless
    // the pose, the LIVE start state and the planning scene all still
    // agree with what was planned; every refusal names itself.
    {
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
      }
    }
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
    if (!approach_planned) {
      RCLCPP_ERROR(node_->get_logger(),
        "Step 2 planning failed — gripper stays closed.");
      recover_to_reset_target("Step 2 (LIN approach) failed");
      return false;
    }

    // 1) Open gripper
    if (!execute_gripper_named_target(gripper_open_name_, "Step 1 - Open gripper")) {
      recover_to_reset_target("Step 1 (open gripper) failed");
      return false;
    }

    // 2) Execute the PRE-COMPUTED approach immediately — no pause.
    bool step2_ok = false;
    {
      if (!execute_) {
        RCLCPP_INFO(node_->get_logger(),
          "Step 2 - Grasp LIN: execute:=false, skipping execution.");
        step2_ok = true;
      } else {
        auto exec_result = arm_group_.execute(approach_plan);
        step2_ok = (exec_result == moveit::core::MoveItErrorCode::SUCCESS);
        if (step2_ok) {
          RCLCPP_INFO(node_->get_logger(), "Step 2 - Grasp LIN: execution OK.");
        } else {
          RCLCPP_ERROR(node_->get_logger(),
            "Step 2 - Grasp LIN: execution failed: %s (%d)",
            moveit::core::error_code_to_string(exec_result).c_str(),
            exec_result.val);
        }
      }
    }
    if (!step2_ok) {
      RCLCPP_ERROR(node_->get_logger(), "Step 2 failed (LIN-only mode).");
      recover_to_reset_target("Step 2 (LIN approach) failed");
      return false;
    }

    // 3) Close gripper on object
    if (!execute_gripper_named_target(gripper_close_name_, "Step 3 - Close gripper")) {
      recover_to_reset_target("Step 3 (close gripper) failed");
      return false;
    }

    // From this point on, the gripper is closed on the object.
    if (stop_after_grasp_) {
      // Majd 2026-08-26: manual mode = grasp -> LIFT -> hold. The fast
      // table-up jog clears the table; failure is non-fatal (still
      // holding safely at the grasp pose).
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      if (execute_fast_joint_lift(lift_height_m_)) {
        RCLCPP_INFO(node_->get_logger(),
          "stop_after_grasp: fast lift +5 cm OK — HOLDING, waiting for "
          "manual commands.");
      } else {
        RCLCPP_WARN(node_->get_logger(),
          "stop_after_grasp: fast lift unavailable — HOLDING at the "
          "grasp pose.");
      }
      return true;
    }
    // If any step fails, we attempt a fallback recovery to initgrasp.
    bool sequence_ok = true;

    // Clear collision objects — the object is now in the gripper, so
    // perception obstacles are stale and would block the lift.
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
          "Clearing %zu collision objects before lift.", obj_ids.size());
        psi.removeCollisionObjects(obj_ids);
        rclcpp::sleep_for(std::chrono::milliseconds(300));
      }
    }

    // 3a) Lift object along table +Y using LIN motion.
    // If lift_table_frame is provided, use that frame's +Y axis transformed
    // into target_frame_. Otherwise use yaw-based fallback (lift_table_yaw_deg).
    if (lift_height_m_ > 0.0) {
      const auto dir = compute_lift_direction_in_target();
      const double dx = dir[0] * lift_height_m_;
      const double dy = dir[1] * lift_height_m_;
      const double dz = dir[2] * lift_height_m_;
      geometry_msgs::msg::PoseStamped lift_pose = grasp_pose_world;
      lift_pose.pose.position.x += dx;
      lift_pose.pose.position.y += dy;
      lift_pose.pose.position.z += dz;
      RCLCPP_INFO(node_->get_logger(),
        "Step 3a - Lifting %.3f m along dir=[%.3f, %.3f, %.3f] in %s: "
        "d=[%.3f, %.3f, %.3f] -> [%.3f, %.3f, %.3f]",
        lift_height_m_, dir[0], dir[1], dir[2], target_frame_.c_str(),
        dx, dy, dz,
        lift_pose.pose.position.x, lift_pose.pose.position.y, lift_pose.pose.position.z);
      // Use lower velocity/acceleration for the short lift to avoid joint limits.
      arm_group_.setMaxVelocityScalingFactor(std::min(velocity_scaling_, 0.1));
      arm_group_.setMaxAccelerationScalingFactor(std::min(acceleration_scaling_, 0.1));
      // FAST LIFT FIRST, same as the LLM executor (2026-08-26): every
      // LIN lift variant here died PLANNING_FAILED within 200 ms even
      // though the motion is trivial in freedrive. LIN stays as fallback.
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      bool lift_ok = execute_fast_joint_lift(lift_height_m_);
      if (lift_ok) {
        RCLCPP_INFO(node_->get_logger(),
          "Step 3a - fast joint lift OK (planner-free).");
      } else {
        rclcpp::sleep_for(std::chrono::milliseconds(200));
        lift_ok = execute_arm_lin_pose(lift_pose, "Step 3a - Lift object (LIN fallback)");
      }
      arm_group_.setMaxVelocityScalingFactor(velocity_scaling_);
      arm_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);
      if (!lift_ok) {
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
    if (sequence_ok && (post_lift_up_m_ > 0.0 || post_lift_back_m_ > 0.0)) {
      rclcpp::sleep_for(std::chrono::milliseconds(150));
    }

    // 3a-up) Extra straight-up retreat in base_link so the next moves
    // start from above the table.
    if (sequence_ok && post_lift_up_m_ > 0.0) {
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
    if (sequence_ok && post_lift_back_m_ > 0.0) {
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

    // 3b+4 BLENDED) One continuous lift -> home -> drop. Same route as
    // the two steps below (still THROUGH home), but as ONE action goal so
    // the fixed per-goal overhead is paid once and the arm does not
    // decelerate to zero at home. Falls through to the two-step sequence
    // on any failure, which starts by going home from wherever the arm is.
    bool blended_ok = false;
    if (sequence_ok && blend_via_home_ &&
        use_fast_replay_for_named_targets_ && use_named_pose_action_ &&
        // Blending home through home would be a zero-length second
        // segment stretched over two tier durations.
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
    if (sequence_ok && !blended_ok &&
        !execute_arm_named_target(reset_target_name_, "Step 3b - Move to reset target")) {
      sequence_ok = false;
    }

    // 4) Move to drop_right_side and hold
    if (sequence_ok && !blended_ok &&
        !execute_arm_named_target(drop_target_name_, "Step 4 - Move to drop target", &last_drop_plan_trajectory_)) {
      sequence_ok = false;
    }

    // Same measurement line for the unblended route (grep '[blend]').
    if (two_step_ran) {
      const double two_step_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - two_step_t0).count();
      RCLCPP_INFO(node_->get_logger(),
        "[blend] two-step lift->%s->%s %s in %.2f s (2 goals, "
        "1 waypoint each).",
        reset_target_name_.c_str(), drop_target_name_.c_str(),
        sequence_ok ? "completed" : "aborted", two_step_s);
    }
    if (sequence_ok && execute_) {
      RCLCPP_INFO(node_->get_logger(), "Holding at '%s' for %.2f seconds.", drop_target_name_.c_str(), drop_hold_s_);
      rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(drop_hold_s_)));
    }

    // 5) Open gripper at drop pose, then return to initgrasp and close gripper.
    if (sequence_ok && !execute_gripper_named_target(gripper_open_name_, "Step 5 - Open gripper at drop")) {
      sequence_ok = false;
    }
    if (sequence_ok && !execute_arm_named_target(reset_target_name_, "Step 6 - Move to reset target")) {
      sequence_ok = false;
    }
    if (sequence_ok && !execute_gripper_named_target(gripper_close_name_, "Step 7 - Close gripper at reset")) {
      sequence_ok = false;
    }

    if (!sequence_ok) {
      recover_to_reset_target("Pick-place sequence step failed");
      RCLCPP_ERROR(node_->get_logger(), "Grasp execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Pick-place sequence complete.");
    return true;
  }

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
    RCLCPP_INFO(node_->get_logger(), "Probe-plan cache dropped: %s.", why);
  }
  bool has_cached_lin_plan() const { return cached_lin_plan_valid_; }

private:
  void recover_to_reset_target(const std::string & reason)
  {
    // INVALIDATION: recovery moves the arm (or the cycle just died).
    invalidate_cached_lin_plan("recovery to the reset target");
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
            arm_group_.execute(plan);
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
    arm_group_.setStartStateToCurrentState();
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

  bool execute_arm_lin_pose(const geometry_msgs::msg::PoseStamped & target_pose, const std::string & step_name)
  {
    // INVALIDATION: this call plans and moves on its own.
    invalidate_cached_lin_plan("the arm is running another LIN move");
    arm_group_.clearPoseTargets();
    arm_group_.setStartStateToCurrentState();
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
    if (!execute_) {
      RCLCPP_INFO(node_->get_logger(), "%s: execute:=false, skipping execution.", step_name.c_str());
      return true;
    }

    auto exec_result = arm_group_.execute(plan);
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
  bool has_ik_solution(const geometry_msgs::msg::PoseStamped & target_pose) const
  {
    // If we already tried and failed once this batch, don't retry per
    // candidate — that produces ~50 "Failed to fetch current robot state"
    // log lines and ~5s of wasted time when joint_states isn't ready.
    if (ik_seed_unavailable_) {
      return true;
    }
    if (!ik_seed_state_) {
      auto state = arm_group_.getCurrentState(0.1);
      if (!state) {
        ik_seed_unavailable_ = true;  // skip IK probe for the rest of this batch
        return true;
      }
      ik_seed_state_ = state;
    }
    const auto * jmg = ik_seed_state_->getJointModelGroup(arm_group_.getName());
    if (!jmg) {
      return true;
    }
    geometry_msgs::msg::Pose pose = target_pose.pose;
    moveit::core::RobotState probe(*ik_seed_state_);
    return probe.setFromIK(jmg, pose, arm_group_.getEndEffectorLink(), 0.02);
  }

  void reset_ik_seed_state()
  {
    ik_seed_state_.reset();
    ik_seed_unavailable_ = false;
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
      moveit::core::RobotState probe_state(arm_group_.getRobotModel());
      for (const size_t idx : samples) {
        if (!robot_state_at_point(jt, idx, js, probe_state)) {
          std::snprintf(buf, sizeof(buf),
            "the robot state at waypoint %zu/%zu could not be rebuilt",
            idx, n - 1);
          return reject(buf);
        }
        std::string why;
        const int v = check_state_valid(probe_state, &why);
        if (v != 1) {
          std::snprintf(buf, sizeof(buf),
            "%s at waypoint %zu/%zu (%s)",
            (v == 0) ? "the planning scene changed"
                     : "the planning scene could not be re-checked",
            idx, n - 1, why.c_str());
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

  bool can_plan_lin_pose(const geometry_msgs::msg::PoseStamped & target_pose, std::string * reason = nullptr)
  {
    // INVALIDATION: a new probe supersedes any earlier one. Candidate
    // N-1's trajectory must never survive into candidate N — the poses
    // differ and the guards would then be comparing against the wrong
    // plan.
    invalidate_cached_lin_plan("a new LIN probe started");
    // Apply a tight planning budget for the per-candidate feasibility
    // probe so a single hard pose can't burn 20-30s. Restore the normal
    // budget afterwards so the actual execution plan still has full time.
    arm_group_.setPlanningTime(lin_feasibility_planning_time_);
    arm_group_.setNumPlanningAttempts(lin_feasibility_attempts_);
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
    std::vector<std::string> pairs;
    std::vector<std::string> keys;
    for (const auto & c : resp->contacts) {
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
      pairs.push_back(a + " <-> " + b);
      if (pairs.size() >= 3) { break; }
    }
    if (pairs.empty()) {
      return "goal state is invalid but MoveIt named no contact pair";
    }
    std::string out = "collides: ";
    for (size_t i = 0; i < pairs.size(); ++i) {
      if (i) { out += "; "; }
      out += pairs[i];
    }
    return out;
  }

  bool choose_first_lin_feasible(
    const std::vector<geometry_msgs::msg::PoseStamped> & candidates,
    geometry_msgs::msg::PoseStamped * selected)
  {
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
    try {
      auto current = arm_group_.getCurrentPose(arm_group_.getEndEffectorLink());
      tcp_pose = current.pose;
      have_tcp = true;
    } catch (...) {
      have_tcp = false;
    }

    size_t ik_dropped = 0;
    for (size_t i = 0; i < total_in; ++i) {
      if (!has_ik_solution(candidates[i])) {
        ++ik_dropped;
        continue;
      }
      double cost = static_cast<double>(i);  // fall back to GraspGen order
      if (sort_candidates_by_reachability_ && have_tcp) {
        const auto & p = candidates[i].pose.position;
        const double dx = p.x - tcp_pose.position.x;
        const double dy = p.y - tcp_pose.position.y;
        const double dz = p.z - tcp_pose.position.z;
        cost = std::sqrt(dx * dx + dy * dy + dz * dz);
      }
      ranked.emplace_back(cost, i);
    }

    if (sort_candidates_by_reachability_) {
      std::sort(ranked.begin(), ranked.end());
    }

    RCLCPP_INFO(
      node_->get_logger(),
      "Pre-filter: %zu/%zu candidates passed IK (dropped %zu unreachable). "
      "Sort by reachability: %s.",
      ranked.size(), total_in, ik_dropped,
      sort_candidates_by_reachability_ ? "enabled" : "disabled");

    if (ranked.empty()) {
      return false;
    }

    const size_t total = ranked.size();
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
        const size_t orig_idx = ranked[i].second;
        std::string reason;
        if (can_plan_lin_pose(candidates[orig_idx], &reason)) {
          *selected = candidates[orig_idx];
          RCLCPP_INFO(
            node_->get_logger(),
            "Candidate %zu/%zu (orig=%zu, cost=%.3f) is LIN-feasible; selecting it.",
            i + 1, total, orig_idx + 1, ranked[i].first);
          return true;
        }
        // Same collision autopsy as grasp_executor_lin_llm. This executor
        // has no candidate_status publisher and no spoken failure
        // summary, so the named contact pair goes into the log line.
        std::string rejected_detail = reason;
        if (reason.find("NO_IK") == std::string::npos &&
            (reason.find("INVALID_MOTION") != std::string::npos ||
             reason.find("PLANNING_FAILED") != std::string::npos) &&
            collision_detail_reports_ < collision_detail_max_reports_) {
          ++collision_detail_reports_;
          bool goal_clear = false;
          const std::string detail =
            collision_detail(candidates[orig_idx], nullptr, &goal_clear);
          if (!detail.empty()) {
            rejected_detail += " — LIN path blocked: " + detail;
          } else if (goal_clear) {
            rejected_detail +=
              " — LIN path blocked mid-approach (goal pose itself is "
              "collision-free; something along the straight line blocks it)";
          }
        }
        RCLCPP_WARN(
          node_->get_logger(),
          "Candidate %zu/%zu (orig=%zu) rejected for LIN: %s",
          i + 1, total, orig_idx + 1, rejected_detail.c_str());
      }
      if (!candidate_try_all_batches_) {
        break;
      }
    }
    return false;
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

    auto exec_result = arm_group_.execute(plan);
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
    std::unique_lock<std::mutex> lk(fast_replay_mutex_);
    while (std::chrono::steady_clock::now() < deadline) {
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
      fast_replay_status_cv_.wait_for(lk, std::chrono::milliseconds(100));
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

    auto exec_result = arm_group_.execute(plan);
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

    auto exec_result = arm_group_.execute(reversed);
    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(), "%s: execution failed: %s (%d)", step_name.c_str(),
        moveit::core::error_code_to_string(exec_result).c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "%s: execution OK.", step_name.c_str());
    return true;
  }

  bool execute_gripper_named_target(const std::string & target_name, const std::string & step_name)
  {
    double target_pos = 0.0;
    if (target_name == gripper_open_name_) {
      target_pos = gripper_open_position_;
    } else if (target_name == gripper_close_name_) {
      target_pos = gripper_close_position_;
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

  bool execute_fast_joint_lift(double dz)
  {
    // INVALIDATION: the planner-free lift streams joints straight to the
    // controller, so nothing else would notice the arm had moved.
    invalidate_cached_lin_plan("the arm is running the fast joint lift");
    // Planner-free lift along TABLE-UP (ported from the LLM executor,
    // 2026-08-26): one seeded IK solve + a 2-point joint trajectory
    // streamed straight to the controller. Base +z is HORIZONTAL in
    // this cell — true vertical comes from the table frame.
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
  bool stop_after_grasp_ = false;
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
  std::string gripper_adapter_node_name_;
  bool override_gripper_pulse_from_executor_;
  double executor_pulse_hold_s_open_;
  double executor_pulse_hold_s_close_;
  double lift_height_m_;
  double lift_table_yaw_deg_;
  std::string lift_table_frame_;
  double post_lift_up_m_;
  double post_lift_back_m_;
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
  bool blend_via_home_;
  double blend_velocity_scaling_;
  std::string named_pose_action_name_;
  rclcpp_action::Client<morty_interfaces::action::MoveToNamedPose>::SharedPtr
    named_pose_action_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("grasp_executor_lin", options);

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
    get_or_declare_param<std::string>(node, "completion_topic", "/grasp_executor_lin/cycle_done");
  const auto selection_cycle_id_topic =
    get_or_declare_param<std::string>(node, "selection_cycle_id_topic", "/graspgen/selection_cycle_id");

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
    "/grasp_executor_lin/selected_pose", rclcpp::QoS(1).reliable().transient_local());
  std::atomic<int64_t> latest_cycle_id{0};
  std::atomic<int64_t> last_processed_cycle_id{-1};

  GraspExecutorLin grasp_executor(node, arm_group, tf_buffer);

  std::mutex mutex;
  std::optional<geometry_msgs::msg::PoseStamped> pending_grasp;
  std::optional<std::chrono::steady_clock::time_point> pending_grasp_received_at;
  std::optional<std::vector<geometry_msgs::msg::PoseStamped>> pending_candidates;
  // Cycle ID observed at the moment candidates/grasp were buffered.
  // We only release them for execution when a NEWER cycle_id arrives —
  // that's the pipeline's "ready, candidates are final" handshake.
  int64_t pending_candidates_cycle_at_arrival = -1;
  int64_t pending_grasp_cycle_at_arrival = -1;
  bool processing = false;

  auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  auto sub_cycle_id = node->create_subscription<std_msgs::msg::Int64>(
    selection_cycle_id_topic,
    latched_qos,
    [&](const std_msgs::msg::Int64::SharedPtr msg) {
      const int64_t new_id = static_cast<int64_t>(msg->data);
      const int64_t old_id =
        latest_cycle_id.exchange(new_id, std::memory_order_relaxed);
      if (new_id != old_id) {
        // INVALIDATION: a new selection cycle targets a different object.
        // Any trajectory kept from the previous cycle's probe is now for
        // the wrong grasp and must never be reachable.
        grasp_executor.invalidate_cached_lin_plan("a new selection cycle_id arrived");
      }
    });

  auto publish_completion = [&](bool ok, int64_t cycle_id) {
    std_msgs::msg::String msg;
    msg.data = std::string(ok ? "done" : "failed") + "|" + std::to_string(cycle_id);
    pub_completion->publish(msg);
  };

  auto sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    grasp_pose_topic,
    latched_qos,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex);
      if (processing) {
        return;
      }
      // Buffer the grasp; the dispatcher loop will only act on it once
      // the pipeline bumps the cycle_id past pending_grasp_cycle_at_arrival.
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
            "Buffered %zu candidates (cycle_id at arrival=%ld); waiting for newer cycle_id "
            "before evaluating.",
            pending_candidates->size(),
            static_cast<long>(pending_candidates_cycle_at_arrival));
        }
      });
  }

  rclcpp::Rate rate(50);
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

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
        pending_candidates_cycle_at_arrival != last_processed_cycle_id;

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
          pending_grasp_cycle_at_arrival != last_processed_cycle_id;
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
      bool feasible = grasp_executor.select_first_lin_feasible(cands, &selected);
      bool ok = false;
      if (!feasible) {
        RCLCPP_ERROR(node->get_logger(), "No LIN-feasible grasp candidate found.");
      } else {
        selected.header.stamp = node->now();
        pub_selected_pose->publish(selected);
        RCLCPP_INFO(node->get_logger(),
          "Published selected pose on '/grasp_executor_lin/selected_pose': [%.3f, %.3f, %.3f]",
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

    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
