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
 
class GraspExecutorPilz
{
public:
  GraspExecutorPilz(
    const rclcpp::Node::SharedPtr & node,
    moveit::planning_interface::MoveGroupInterface & move_group,
    tf2_ros::Buffer & tf_buffer)
  : node_(node), move_group_(move_group), tf_buffer_(tf_buffer)
  {
    execute_ = get_or_declare_param<bool>(node_, "execute", false);
    pregrasp_distance_ = get_or_declare_param<double>(node_, "pregrasp_distance", 0.03);
    target_frame_ = get_or_declare_param<std::string>(node_, "target_frame", "world");
    tf_timeout_ = get_or_declare_param<double>(node_, "tf_timeout", 2.0);
    planning_time_ = get_or_declare_param<double>(node_, "planning_time", 10.0);
    num_attempts_ = get_or_declare_param<int>(node_, "num_attempts", 10);
    one_shot_ = get_or_declare_param<bool>(node_, "one_shot", true);

    move_group_.setPlanningTime(planning_time_);
    move_group_.setNumPlanningAttempts(num_attempts_);
  }

  bool is_one_shot() const { return one_shot_; }

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
 
    geometry_msgs::msg::PoseStamped pregrasp_pose = compute_pregrasp_pose(
      grasp_pose_world, pregrasp_distance_);
 
    RCLCPP_INFO(node_->get_logger(),
      "Pre-grasp pose (%.3fm back): [%.3f, %.3f, %.3f]",
      pregrasp_distance_,
      pregrasp_pose.pose.position.x,
      pregrasp_pose.pose.position.y,
      pregrasp_pose.pose.position.z);
 
    // Step 1: PTP to pre-grasp
    move_group_.clearPoseTargets();
    move_group_.setStartStateToCurrentState();
    move_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    move_group_.setPlannerId("PTP");
    if (!move_group_.setPoseTarget(pregrasp_pose, move_group_.getEndEffectorLink())) {
      RCLCPP_ERROR(node_->get_logger(), "setPoseTarget() failed for pre-grasp (IK/joint limits).");
      return false;
    }
 
    moveit::planning_interface::MoveGroupInterface::Plan pregrasp_plan;
    auto pregrasp_result = move_group_.plan(pregrasp_plan);
    if (pregrasp_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(),
        "Pre-grasp PTP planning failed: %s (%d)",
        moveit::core::error_code_to_string(pregrasp_result).c_str(),
        pregrasp_result.val);
      return false;
    }
 
    RCLCPP_INFO(node_->get_logger(), "Pre-grasp PTP plan OK.");
 
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
 
    // Step 2: LIN to grasp
    move_group_.clearPoseTargets();
    move_group_.setStartStateToCurrentState();
    move_group_.setPlanningPipelineId("pilz_industrial_motion_planner");
    move_group_.setPlannerId("LIN");
    if (!move_group_.setPoseTarget(grasp_pose_world, move_group_.getEndEffectorLink())) {
      RCLCPP_ERROR(node_->get_logger(), "setPoseTarget() failed for grasp (IK/joint limits).");
      return false;
    }
 
    moveit::planning_interface::MoveGroupInterface::Plan grasp_plan;
    auto grasp_result = move_group_.plan(grasp_plan);
    if (grasp_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(node_->get_logger(),
        "Grasp LIN planning failed: %s (%d)",
        moveit::core::error_code_to_string(grasp_result).c_str(),
        grasp_result.val);
      return false;
    }
 
    RCLCPP_INFO(node_->get_logger(), "Grasp LIN plan OK.");
 
    if (execute_) {
      auto exec_result = move_group_.execute(grasp_plan);
      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(),
          "Grasp LIN execution failed: %s (%d)",
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
  double tf_timeout_;
  double planning_time_;
  int num_attempts_;
  bool one_shot_;
};
 
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("grasp_executor_pilz", options);
 
  const auto planning_group = get_or_declare_param<std::string>(node, "planning_group", "ur3e_manipulator");
  const auto end_effector_link = get_or_declare_param<std::string>(node, "end_effector_link", "gripper_tcp");
  const auto move_group_node = get_or_declare_param<std::string>(node, "move_group_node", "move_group");
  const auto grasp_pose_topic = get_or_declare_param<std::string>(node, "grasp_pose_topic", "/grasp_pose");
 
  copy_moveit_params_from_move_group(node, move_group_node, planning_group);
 
  tf2_ros::Buffer tf_buffer(node->get_clock());
  tf2_ros::TransformListener tf_listener(tf_buffer);
 
  moveit::planning_interface::MoveGroupInterface move_group(node, planning_group);
  move_group.setEndEffectorLink(end_effector_link);
 
  RCLCPP_INFO(node->get_logger(),
    "Grasp Executor (Pilz) ready. Listening on '%s' (planning_group=%s, ee=%s, pregrasp=PTP, grasp=LIN)",
    grasp_pose_topic.c_str(), planning_group.c_str(), end_effector_link.c_str());
 
  GraspExecutorPilz grasp_executor(node, move_group, tf_buffer);
 
  std::mutex mutex;
  std::optional<geometry_msgs::msg::PoseStamped> pending_grasp;
  bool processing = false;
 
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
    });
 
  rclcpp::Rate rate(50);
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
 
    std::optional<geometry_msgs::msg::PoseStamped> grasp;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (pending_grasp.has_value() && !processing) {
        processing = true;
        grasp = pending_grasp;
        pending_grasp.reset();
      }
    }
 
    if (grasp.has_value()) {
      RCLCPP_INFO(node->get_logger(), "Grasp pose received, will process.");
      bool ok = grasp_executor.execute_grasp(*grasp);
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
