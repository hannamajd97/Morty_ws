#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/utils/moveit_error_code.h>

namespace
{

struct AttemptResult
{
  moveit::core::MoveItErrorCode plan_result;
  moveit::core::MoveItErrorCode exec_result;
};

}  // namespace


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
      node->has_parameter(kinematics_solver_param))
  {
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

  try {
    auto future = params_client->get_parameters(names);
    while (rclcpp::spin_some(node), future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    for (const auto & p : future.get()) {
      if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
        continue;
      }
      if (!node->has_parameter(p.get_name())) {
        node->declare_parameter(p.get_name(), p.get_parameter_value());
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(node->get_logger(), "Failed to read params from '%s': %s", move_group_node.c_str(), e.what());
  }
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("observation_ptp_commander", options);

  const auto planning_group =
    get_or_declare_param<std::string>(node, "planning_group", "ur3e_manipulator");
  const auto end_effector_link =
    get_or_declare_param<std::string>(node, "end_effector_link", "gripper_tcp");
  const auto move_group_node =
    get_or_declare_param<std::string>(node, "move_group_node", "move_group");
  const auto target_name =
    get_or_declare_param<std::string>(node, "target_name", "initgrasp");
  const auto use_pose_target =
    get_or_declare_param<bool>(node, "use_pose_target", false);
  const auto use_joint_target =
    get_or_declare_param<bool>(node, "use_joint_target", false);
  const auto use_cartesian_path =
    get_or_declare_param<bool>(node, "use_cartesian_path", false);
  const auto cartesian_num_waypoints =
    get_or_declare_param<int>(node, "cartesian_num_waypoints", 0);
  const auto cartesian_eef_step =
    get_or_declare_param<double>(node, "cartesian_eef_step", 0.005);
  const auto cartesian_jump_threshold =
    get_or_declare_param<double>(node, "cartesian_jump_threshold", 0.0);
  const auto reference_frame =
    get_or_declare_param<std::string>(node, "reference_frame", "base_link");
  const auto pose_x =
    get_or_declare_param<double>(node, "pose_x", 0.0);
  const auto pose_y =
    get_or_declare_param<double>(node, "pose_y", 0.0);
  const auto pose_z =
    get_or_declare_param<double>(node, "pose_z", 0.0);
  const auto pose_qx =
    get_or_declare_param<double>(node, "pose_qx", 0.0);
  const auto pose_qy =
    get_or_declare_param<double>(node, "pose_qy", 0.0);
  const auto pose_qz =
    get_or_declare_param<double>(node, "pose_qz", 0.0);
  const auto pose_qw =
    get_or_declare_param<double>(node, "pose_qw", 1.0);
  const auto execute =
    get_or_declare_param<bool>(node, "execute", false);
  const auto one_shot =
    get_or_declare_param<bool>(node, "one_shot", true);
  const auto planning_time =
    get_or_declare_param<double>(node, "planning_time", 5.0);
  const auto num_attempts =
    get_or_declare_param<int>(node, "num_attempts", 10);
  const auto velocity_scaling =
    get_or_declare_param<double>(node, "velocity_scaling", 0.15);
  const auto acceleration_scaling =
    get_or_declare_param<double>(node, "acceleration_scaling", 0.15);
  const auto planning_pipeline =
    get_or_declare_param<std::string>(node, "planning_pipeline", "pilz_industrial_motion_planner");
  const auto planner_id =
    get_or_declare_param<std::string>(node, "planner_id", "PTP");
  const auto allow_lin_fallback =
    get_or_declare_param<bool>(node, "allow_lin_fallback", true);
  const auto fallback_planning_pipeline =
    get_or_declare_param<std::string>(
    node, "fallback_planning_pipeline", "pilz_industrial_motion_planner");
  const auto fallback_planner_id =
    get_or_declare_param<std::string>(node, "fallback_planner_id", "LIN");
  const auto allow_ompl_fallback =
    get_or_declare_param<bool>(node, "allow_ompl_fallback", true);
  const auto ompl_planning_pipeline =
    get_or_declare_param<std::string>(node, "ompl_planning_pipeline", "ompl");
  const auto ompl_planner_id =
    get_or_declare_param<std::string>(node, "ompl_planner_id", "RRTConnect");
  const auto ompl_planning_time =
    get_or_declare_param<double>(node, "ompl_planning_time", 5.0);

  copy_moveit_params_from_move_group(node, move_group_node, planning_group);

  moveit::planning_interface::MoveGroupInterface arm_group(node, planning_group);
  arm_group.setEndEffectorLink(end_effector_link);
  arm_group.setPlanningTime(planning_time);
  arm_group.setNumPlanningAttempts(num_attempts);
  arm_group.setMaxVelocityScalingFactor(velocity_scaling);
  arm_group.setMaxAccelerationScalingFactor(acceleration_scaling);
  arm_group.setPlanningPipelineId(planning_pipeline);
  arm_group.setPlannerId(planner_id);
  RCLCPP_INFO(
    node->get_logger(), "Planner: pipeline='%s', id='%s'",
    planning_pipeline.c_str(), planner_id.c_str());

  const auto named_targets = arm_group.getNamedTargets();
  std::string target_list;
  for (size_t i = 0; i < named_targets.size(); ++i) {
    target_list += named_targets[i];
    if (i + 1 < named_targets.size()) {
      target_list += ", ";
    }
  }

  geometry_msgs::msg::Pose pose;
  pose.position.x = pose_x;
  pose.position.y = pose_y;
  pose.position.z = pose_z;
  pose.orientation.x = pose_qx;
  pose.orientation.y = pose_qy;
  pose.orientation.z = pose_qz;
  pose.orientation.w = pose_qw;

  // Joint target support: read 6 joint values by index
  std::vector<double> joint_target_values;
  if (use_joint_target) {
    const auto joint_names = arm_group.getJointNames();
    joint_target_values.resize(joint_names.size(), 0.0);
    for (size_t i = 0; i < joint_names.size(); ++i) {
      const auto param_name = "joint_target_" + std::to_string(i);
      joint_target_values[i] = get_or_declare_param<double>(node, param_name, 0.0);
    }
    std::string jt_str;
    for (size_t i = 0; i < joint_names.size(); ++i) {
      jt_str += joint_names[i] + "=" + std::to_string(joint_target_values[i]);
      if (i + 1 < joint_names.size()) jt_str += ", ";
    }
    RCLCPP_INFO(
      node->get_logger(),
      "Observation PTP commander ready. joint target: [%s], execute=%s",
      jt_str.c_str(), execute ? "true" : "false");
  } else if (use_cartesian_path) {
    RCLCPP_INFO(
      node->get_logger(),
      "Observation PTP commander ready. cartesian path with %d waypoints, "
      "eef_step=%.4f, execute=%s",
      cartesian_num_waypoints, cartesian_eef_step, execute ? "true" : "false");
  } else if (use_pose_target) {
    RCLCPP_INFO(
      node->get_logger(),
      "Observation PTP commander ready. pose target in frame '%s', execute=%s",
      reference_frame.c_str(), execute ? "true" : "false");
  } else {
    RCLCPP_INFO(
      node->get_logger(),
      "Observation PTP commander ready. target='%s', execute=%s, available targets=[%s]",
      target_name.c_str(), execute ? "true" : "false", target_list.c_str());
    if (std::find(named_targets.begin(), named_targets.end(), target_name) == named_targets.end()) {
      RCLCPP_ERROR(
        node->get_logger(),
        "Unknown named target '%s'. Check the SRDF group_state names.",
        target_name.c_str());
      rclcpp::shutdown();
      return 1;
    }
  }

  const auto apply_target = [&]() -> bool {
    arm_group.clearPoseTargets();
    arm_group.setStartStateToCurrentState();
    if (use_joint_target) {
      return arm_group.setJointValueTarget(joint_target_values);
    }
    if (use_pose_target) {
      arm_group.setPoseReferenceFrame(reference_frame);
      arm_group.setPoseTarget(pose, end_effector_link);
      return true;
    }
    return arm_group.setNamedTarget(target_name);
  };

  const auto run_attempt =
    [&](const std::string & pipeline_id, const std::string & planner_name) -> AttemptResult {
      arm_group.setPlanningPipelineId(pipeline_id);
      arm_group.setPlannerId(planner_name);
      RCLCPP_INFO(
        node->get_logger(), "Attempting observation move with pipeline='%s', planner='%s'",
        pipeline_id.c_str(), planner_name.c_str());

      if (!apply_target()) {
        RCLCPP_ERROR(
          node->get_logger(),
          "Failed to apply %s target before planning.",
          use_joint_target ? "joint" : (use_pose_target ? "pose" : "named"));
        return {moveit::core::MoveItErrorCode::FAILURE, moveit::core::MoveItErrorCode::FAILURE};
      }

      moveit::planning_interface::MoveGroupInterface::Plan plan;
      const auto plan_result = arm_group.plan(plan);
      if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(
          node->get_logger(),
          "%s planning failed: %s (%d)",
          planner_name.c_str(),
          moveit::core::error_code_to_string(plan_result).c_str(),
          plan_result.val);
        return {plan_result, moveit::core::MoveItErrorCode::FAILURE};
      }

      if (use_joint_target) {
        RCLCPP_INFO(node->get_logger(), "%s planning to joint target succeeded.", planner_name.c_str());
      } else if (use_pose_target) {
        RCLCPP_INFO(node->get_logger(), "%s planning to pose target succeeded.", planner_name.c_str());
      } else {
        RCLCPP_INFO(node->get_logger(), "%s planning to '%s' succeeded.", planner_name.c_str(), target_name.c_str());
      }
      if (!execute) {
        return {plan_result, moveit::core::MoveItErrorCode::SUCCESS};
      }

      const auto exec_result = arm_group.execute(plan);
      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(
          node->get_logger(),
          "%s execution failed: %s (%d)",
          planner_name.c_str(),
          moveit::core::error_code_to_string(exec_result).c_str(),
          exec_result.val);
        return {plan_result, exec_result};
      }

      if (use_joint_target) {
        RCLCPP_INFO(node->get_logger(), "%s execution to joint target succeeded.", planner_name.c_str());
      } else if (use_pose_target) {
        RCLCPP_INFO(node->get_logger(), "%s execution to pose target succeeded.", planner_name.c_str());
      } else {
        RCLCPP_INFO(node->get_logger(), "%s execution to '%s' succeeded.", planner_name.c_str(), target_name.c_str());
      }
      return {plan_result, exec_result};
    };

  // ── Cartesian waypoint path mode ──────────────────────────────────
  if (use_cartesian_path) {
    arm_group.setPoseReferenceFrame(reference_frame);
    std::vector<geometry_msgs::msg::Pose> waypoints;
    for (int i = 0; i < cartesian_num_waypoints; ++i) {
      const auto prefix = "wp_" + std::to_string(i) + "_";
      geometry_msgs::msg::Pose wp;
      wp.position.x = get_or_declare_param<double>(node, prefix + "x", 0.0);
      wp.position.y = get_or_declare_param<double>(node, prefix + "y", 0.0);
      wp.position.z = get_or_declare_param<double>(node, prefix + "z", 0.0);
      wp.orientation.x = get_or_declare_param<double>(node, prefix + "qx", 0.0);
      wp.orientation.y = get_or_declare_param<double>(node, prefix + "qy", 0.0);
      wp.orientation.z = get_or_declare_param<double>(node, prefix + "qz", 0.0);
      wp.orientation.w = get_or_declare_param<double>(node, prefix + "qw", 1.0);
      waypoints.push_back(wp);
    }
    RCLCPP_INFO(
      node->get_logger(),
      "Computing Cartesian path through %zu waypoints (eef_step=%.4f, jump=%.2f).",
      waypoints.size(), cartesian_eef_step, cartesian_jump_threshold);

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = arm_group.computeCartesianPath(
      waypoints, cartesian_eef_step, cartesian_jump_threshold, trajectory);
    RCLCPP_INFO(
      node->get_logger(),
      "Cartesian path fraction achieved: %.3f (%zu points in trajectory).",
      fraction, trajectory.joint_trajectory.points.size());

    if (fraction < 0.90) {
      RCLCPP_ERROR(
        node->get_logger(),
        "Cartesian path only achieved %.1f%% — aborting (need >= 90%%).", fraction * 100.0);
      rclcpp::shutdown();
      return 2;
    }

    // MoveIt's computeCartesianPath already uses the velocity/acceleration
    // scaling factors set on the MoveGroupInterface, so no manual scaling needed.

    if (execute) {
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      plan.trajectory_ = trajectory;
      const auto exec_result = arm_group.execute(plan);
      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(
          node->get_logger(),
          "Cartesian path execution failed: %s (%d)",
          moveit::core::error_code_to_string(exec_result).c_str(), exec_result.val);
        rclcpp::shutdown();
        return 3;
      }
      RCLCPP_INFO(node->get_logger(), "Cartesian path execution succeeded.");
    } else {
      RCLCPP_INFO(node->get_logger(), "Cartesian path planned (execute:=false).");
    }

    if (one_shot) {
      rclcpp::shutdown();
      return 0;
    }
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
  }

  // ── Standard PTP / fallback chain ──────────────────────────────────
  const auto primary_result = run_attempt(planning_pipeline, planner_id);
  if (primary_result.plan_result == moveit::core::MoveItErrorCode::SUCCESS &&
      (!execute || primary_result.exec_result == moveit::core::MoveItErrorCode::SUCCESS))
  {
    if (!execute) {
      RCLCPP_INFO(node->get_logger(), "execute:=false, stopping after successful planning.");
    }
  } else {
    const bool should_try_lin =
      allow_lin_fallback &&
      !(fallback_planning_pipeline == planning_pipeline && fallback_planner_id == planner_id);
    if (!should_try_lin) {
      rclcpp::shutdown();
      return primary_result.plan_result == moveit::core::MoveItErrorCode::SUCCESS ? 3 : 2;
    }

    RCLCPP_WARN(
      node->get_logger(),
      "Primary observation move failed; retrying with fallback pipeline='%s', planner='%s'.",
      fallback_planning_pipeline.c_str(), fallback_planner_id.c_str());
    const auto fallback_result = run_attempt(fallback_planning_pipeline, fallback_planner_id);
    const bool lin_ok =
      fallback_result.plan_result == moveit::core::MoveItErrorCode::SUCCESS &&
      (!execute || fallback_result.exec_result == moveit::core::MoveItErrorCode::SUCCESS);

    if (!lin_ok) {
      if (fallback_result.plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(
          node->get_logger(),
          "Fallback %s planning failed: %s (%d)",
          fallback_planner_id.c_str(),
          moveit::core::error_code_to_string(fallback_result.plan_result).c_str(),
          fallback_result.plan_result.val);
      } else {
        RCLCPP_WARN(
          node->get_logger(),
          "Fallback %s execution failed: %s (%d)",
          fallback_planner_id.c_str(),
          moveit::core::error_code_to_string(fallback_result.exec_result).c_str(),
          fallback_result.exec_result.val);
      }

      if (!allow_ompl_fallback) {
        RCLCPP_ERROR(node->get_logger(), "OMPL fallback disabled. Giving up.");
        rclcpp::shutdown();
        return fallback_result.plan_result == moveit::core::MoveItErrorCode::SUCCESS ? 3 : 2;
      }

      RCLCPP_WARN(
        node->get_logger(),
        "LIN fallback failed; retrying with OMPL pipeline='%s', planner='%s'.",
        ompl_planning_pipeline.c_str(), ompl_planner_id.c_str());

      const auto saved_planning_time = arm_group.getPlanningTime();
      arm_group.setPlanningTime(ompl_planning_time);

      const auto ompl_result = run_attempt(ompl_planning_pipeline, ompl_planner_id);

      arm_group.setPlanningTime(saved_planning_time);

      if (ompl_result.plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(
          node->get_logger(),
          "OMPL %s planning failed: %s (%d)",
          ompl_planner_id.c_str(),
          moveit::core::error_code_to_string(ompl_result.plan_result).c_str(),
          ompl_result.plan_result.val);
        rclcpp::shutdown();
        return 2;
      }
      if (execute && ompl_result.exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(
          node->get_logger(),
          "OMPL %s execution failed: %s (%d)",
          ompl_planner_id.c_str(),
          moveit::core::error_code_to_string(ompl_result.exec_result).c_str(),
          ompl_result.exec_result.val);
        rclcpp::shutdown();
        return 3;
      }
      if (!execute) {
        RCLCPP_INFO(
          node->get_logger(),
          "OMPL %s planning succeeded with execute:=false.",
          ompl_planner_id.c_str());
      }
    } else {
      if (!execute) {
        RCLCPP_INFO(
          node->get_logger(),
          "Fallback %s planning succeeded with execute:=false.",
          fallback_planner_id.c_str());
      }
    }
  }

  if (one_shot) {
    rclcpp::shutdown();
    return 0;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
