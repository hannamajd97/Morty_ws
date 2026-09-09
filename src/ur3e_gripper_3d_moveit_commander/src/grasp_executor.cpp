#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <thread>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/utils/moveit_error_code.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

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
  // This means we back off from the grasp pose along the gripper's approach vector

  tf2::Transform grasp_tf;
  tf2::fromMsg(grasp_pose.pose, grasp_tf);

  // Create offset transform: move back along local Z axis
  tf2::Transform offset;
  offset.setIdentity();
  offset.setOrigin(tf2::Vector3(0.0, 0.0, -pregrasp_distance));

  // Apply offset in grasp frame: pregrasp = grasp * offset
  tf2::Transform pregrasp_tf = grasp_tf * offset;

  geometry_msgs::msg::PoseStamped pregrasp_pose;
  pregrasp_pose.header = grasp_pose.header;

  // Convert tf2::Transform to geometry_msgs::Pose
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

class GraspExecutor
{
public:
  GraspExecutor(
    const rclcpp::Node::SharedPtr & node,
    moveit::planning_interface::MoveGroupInterface & move_group,
    tf2_ros::Buffer & tf_buffer)
  : node_(node), move_group_(move_group), tf_buffer_(tf_buffer)
  {
    // Get parameters
    execute_ = get_or_declare_param<bool>(node_, "execute", false);
    pregrasp_distance_ = get_or_declare_param<double>(node_, "pregrasp_distance", 0.03);
    target_frame_ = get_or_declare_param<std::string>(node_, "target_frame", "world");
    cartesian_step_ = get_or_declare_param<double>(node_, "cartesian_step", 0.005);
    cartesian_jump_threshold_ = get_or_declare_param<double>(node_, "cartesian_jump_threshold", 0.0);
    tf_timeout_ = get_or_declare_param<double>(node_, "tf_timeout", 2.0);
    planning_time_ = get_or_declare_param<double>(node_, "planning_time", 10.0);
    num_attempts_ = get_or_declare_param<int>(node_, "num_attempts", 10);

    move_group_.setPlanningTime(planning_time_);
    move_group_.setNumPlanningAttempts(num_attempts_);
  }

  bool execute_grasp(const geometry_msgs::msg::PoseStamped & grasp_pose_input)
  {
    RCLCPP_INFO(node_->get_logger(),
      "Received grasp pose in frame '%s'", grasp_pose_input.header.frame_id.c_str());

    // Step 1: Transform grasp pose to target frame (world)
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

    // Step 2: Compute pre-grasp pose
    geometry_msgs::msg::PoseStamped pregrasp_pose = compute_pregrasp_pose(
      grasp_pose_world, pregrasp_distance_);

    RCLCPP_INFO(node_->get_logger(),
      "Pre-grasp pose (%.3fm back): [%.3f, %.3f, %.3f]",
      pregrasp_distance_,
      pregrasp_pose.pose.position.x,
      pregrasp_pose.pose.position.y,
      pregrasp_pose.pose.position.z);

    // Step 3: Plan to pre-grasp using OMPL
    move_group_.setStartStateToCurrentState();
    if (!move_group_.setPoseTarget(pregrasp_pose, move_group_.getEndEffectorLink())) {
      RCLCPP_ERROR(node_->get_logger(), "setPoseTarget() failed for pre-grasp (IK/joint limits).");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan pregrasp_plan;
    auto pregrasp_result = move_group_.plan(pregrasp_plan);
    if (pregrasp_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(),
        "Pre-grasp planning failed: %s (%d)",
        moveit::core::error_code_to_string(pregrasp_result).c_str(),
        pregrasp_result.val);
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Pre-grasp plan OK.");

    // Execute pre-grasp if enabled
    if (execute_) {
      auto exec_result = move_group_.execute(pregrasp_plan);
      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(),
          "Pre-grasp execution failed: %s (%d)",
          moveit::core::error_code_to_string(exec_result).c_str(),
          exec_result.val);
        return false;
      }
      RCLCPP_INFO(node_->get_logger(), "Pre-grasp execution OK.");
    } else {
      RCLCPP_INFO(node_->get_logger(), "execute:=false, skipping pre-grasp execution.");
    }

    // Step 4: Cartesian path from pre-grasp to grasp
    move_group_.setStartStateToCurrentState();

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(grasp_pose_world.pose);

    moveit_msgs::msg::RobotTrajectory cartesian_trajectory;
    double fraction = move_group_.computeCartesianPath(
      waypoints,
      cartesian_step_,
      cartesian_jump_threshold_,
      cartesian_trajectory,
      true);  // avoid_collisions

    if (fraction < 0.95) {
      RCLCPP_ERROR(node_->get_logger(),
        "Cartesian path to grasp failed (fraction=%.3f). "
        "Likely collision or IK limits.", fraction);
      return false;
    }

    RCLCPP_INFO(node_->get_logger(),
      "Cartesian path to grasp OK (fraction=%.3f).", fraction);

    // Execute Cartesian path if enabled
    if (execute_) {
      auto exec_result = move_group_.execute(cartesian_trajectory);
      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(),
          "Grasp Cartesian execution failed: %s (%d)",
          moveit::core::error_code_to_string(exec_result).c_str(),
          exec_result.val);
        return false;
      }
      RCLCPP_INFO(node_->get_logger(), "Grasp execution OK. Robot at grasp pose.");
    } else {
      RCLCPP_INFO(node_->get_logger(), "execute:=false, skipping grasp execution.");
    }

    return true;
  }

private:
  rclcpp::Node::SharedPtr node_;
  moveit::planning_interface::MoveGroupInterface & move_group_;
  tf2_ros::Buffer & tf_buffer_;

  bool execute_;
  double pregrasp_distance_;
  std::string target_frame_;
  double cartesian_step_;
  double cartesian_jump_threshold_;
  double tf_timeout_;
  double planning_time_;
  int num_attempts_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("grasp_executor", options);

  // Parameters
  const auto planning_group = get_or_declare_param<std::string>(node, "planning_group", "ur3e_manipulator");
  const auto end_effector_link = get_or_declare_param<std::string>(node, "end_effector_link", "gripper_tcp");
  const auto move_group_node = get_or_declare_param<std::string>(node, "move_group_node", "move_group");
  const auto grasp_pose_topic = get_or_declare_param<std::string>(node, "grasp_pose_topic", "/grasp_pose");

  // Copy MoveIt parameters from move_group node
  copy_moveit_params_from_move_group(node, move_group_node, planning_group);

  // TF2 setup
  tf2_ros::Buffer tf_buffer(node->get_clock());
  tf2_ros::TransformListener tf_listener(tf_buffer);

  // MoveIt setup
  moveit::planning_interface::MoveGroupInterface move_group(node, planning_group);
  move_group.setEndEffectorLink(end_effector_link);

  RCLCPP_INFO(node->get_logger(),
    "Grasp Executor ready. Listening on '%s' (planning_group=%s, ee=%s)",
    grasp_pose_topic.c_str(), planning_group.c_str(), end_effector_link.c_str());

  // Create executor
  GraspExecutor grasp_executor(node, move_group, tf_buffer);

  // Subscription state
  std::mutex mutex;
  std::optional<geometry_msgs::msg::PoseStamped> pending_grasp;
  bool processing = false;

  // Subscribe to grasp poses
  auto sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    grasp_pose_topic,
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex);
      if (processing) {
        RCLCPP_WARN(node->get_logger(), "Already processing a grasp, ignoring new pose.");
        return;
      }
      pending_grasp = *msg;
      RCLCPP_INFO(node->get_logger(), "Grasp pose received, will process.");
    });

  // Processing timer
  auto timer = node->create_wall_timer(
    std::chrono::milliseconds(100),
    [&]() {
      geometry_msgs::msg::PoseStamped grasp_to_process;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (!pending_grasp || processing) {
          return;
        }
        grasp_to_process = *pending_grasp;
        pending_grasp.reset();
        processing = true;
      }

      bool success = grasp_executor.execute_grasp(grasp_to_process);

      {
        std::lock_guard<std::mutex> lock(mutex);
        processing = false;
      }

      if (success) {
        RCLCPP_INFO(node->get_logger(), "Grasp execution completed successfully.");
      } else {
        RCLCPP_ERROR(node->get_logger(), "Grasp execution failed.");
      }
    });

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
