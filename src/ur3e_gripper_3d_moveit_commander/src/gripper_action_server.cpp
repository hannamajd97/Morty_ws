// gripper_action_server.cpp
//
// C++ port of the original Python gripper_action_server. Exposes a high-level
// OperateGripper action that translates voice / intent-router commands
// (OPEN / CLOSE / SET_WIDTH + force) into low-level control_msgs/GripperCommand
// goals sent to the ros2_control gripper controller. Behavior-identical to
// the original Python implementation; same node name, same action name,
// same parameter set.
//
// Parameters (with defaults):
//   downstream_action     : "/gripper_controller/gripper_cmd"
//   server_action_name    : "/gripper"
//   open_position         : 0.5
//   close_position        : 0.0
//   default_force_n       : 50.0
//   connect_timeout_s     : 5.0
//   goal_timeout_s        : 15.0  (currently informational; downstream handles its own timing)

#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <control_msgs/action/gripper_command.hpp>
#include <morty_interfaces/action/operate_gripper.hpp>


using OperateGripper = morty_interfaces::action::OperateGripper;
using GripperCommand = control_msgs::action::GripperCommand;
using GoalHandle = rclcpp_action::ServerGoalHandle<OperateGripper>;


class GripperActionServer : public rclcpp::Node
{
public:
  GripperActionServer()
  : rclcpp::Node("gripper_action_server")
  {
    downstream_name_ = this->declare_parameter<std::string>(
      "downstream_action", "/gripper_controller/gripper_cmd");
    server_action_name_ = this->declare_parameter<std::string>(
      "server_action_name", "/gripper");
    open_pos_ = this->declare_parameter<double>("open_position", 0.5);
    close_pos_ = this->declare_parameter<double>("close_position", 0.0);
    default_force_n_ = this->declare_parameter<double>("default_force_n", 50.0);
    connect_timeout_s_ = this->declare_parameter<double>("connect_timeout_s", 5.0);
    goal_timeout_s_ = this->declare_parameter<double>("goal_timeout_s", 15.0);

    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    client_ = rclcpp_action::create_client<GripperCommand>(
      this, downstream_name_, cb_group_);

    server_ = rclcpp_action::create_server<OperateGripper>(
      this,
      server_action_name_,
      std::bind(&GripperActionServer::handle_goal, this,
        std::placeholders::_1, std::placeholders::_2),
      std::bind(&GripperActionServer::handle_cancel, this,
        std::placeholders::_1),
      std::bind(&GripperActionServer::handle_accepted, this,
        std::placeholders::_1),
      rcl_action_server_get_default_options(),
      cb_group_);

    RCLCPP_INFO(
      this->get_logger(),
      "OperateGripper server ready; forwarding to %s",
      downstream_name_.c_str());
  }

private:
  // ── Goal lifecycle callbacks ────────────────────────────────────────────
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & /*uuid*/,
    std::shared_ptr<const OperateGripper::Goal> /*goal*/)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      RCLCPP_WARN(
        this->get_logger(),
        "Rejecting goal: another OperateGripper goal is in flight");
      return rclcpp_action::GoalResponse::REJECT;
    }
    active_ = true;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandle> /*goal_handle*/)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    // Run the execution body on a detached thread so the executor thread
    // returns immediately. Matches the Python version's async semantics
    // (executor stays free to dispatch downstream callbacks).
    std::thread{
      std::bind(&GripperActionServer::execute, this, std::placeholders::_1),
      goal_handle
    }.detach();
  }

  // ── Helpers ─────────────────────────────────────────────────────────────
  bool resolve_target(
    const OperateGripper::Goal & goal,
    double & out_target,
    std::string & out_label) const
  {
    switch (goal.command) {
      case OperateGripper::Goal::OPEN:
        out_target = open_pos_;
        out_label = "open";
        return true;
      case OperateGripper::Goal::CLOSE:
        out_target = close_pos_;
        out_label = "close";
        return true;
      case OperateGripper::Goal::SET_WIDTH:
        out_target = static_cast<double>(goal.width_m);
        out_label = "set_width(" + std::to_string(goal.width_m) + ")";
        return true;
      default:
        return false;
    }
  }

  void publish_feedback(
    const std::shared_ptr<GoalHandle> & goal_handle,
    const std::string & phase,
    double width) const
  {
    auto fb = std::make_shared<OperateGripper::Feedback>();
    fb->phase = phase;
    fb->current_width_m = static_cast<float>(width);
    goal_handle->publish_feedback(fb);
  }

  void fail(
    const std::shared_ptr<GoalHandle> & goal_handle,
    const std::string & error)
  {
    RCLCPP_ERROR(this->get_logger(), "%s", error.c_str());
    auto result = std::make_shared<OperateGripper::Result>();
    result->success = false;
    result->error = error;
    goal_handle->abort(result);
  }

  // ── Main execution ──────────────────────────────────────────────────────
  void execute(const std::shared_ptr<GoalHandle> goal_handle)
  {
    // Always release the active flag, even on early return / exception.
    struct ActiveGuard
    {
      GripperActionServer * self;
      ~ActiveGuard()
      {
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->active_ = false;
      }
    } guard{this};

    const auto goal = goal_handle->get_goal();

    double target = 0.0;
    std::string label;
    if (!resolve_target(*goal, target, label)) {
      fail(goal_handle,
        "unknown command: " + std::to_string(static_cast<int>(goal->command)));
      return;
    }

    const double force = (goal->force_n != 0.0f)
      ? static_cast<double>(goal->force_n)
      : default_force_n_;

    RCLCPP_INFO(
      this->get_logger(),
      "OperateGripper: %s, force=%.1fN",
      label.c_str(), force);

    if (!client_->wait_for_action_server(
        std::chrono::duration<double>(connect_timeout_s_)))
    {
      fail(goal_handle,
        "downstream action " + downstream_name_ + " not available");
      return;
    }

    publish_feedback(goal_handle, "moving", target);

    GripperCommand::Goal downstream_goal;
    downstream_goal.command.position = static_cast<float>(target);
    downstream_goal.command.max_effort = static_cast<float>(force);

    // send_goal_async returns a future<GoalHandle::SharedPtr>. Block on it
    // here; we're already in a detached thread, so this does not block the
    // executor.
    auto send_future = client_->async_send_goal(downstream_goal);
    if (send_future.wait_for(
        std::chrono::duration<double>(connect_timeout_s_))
      == std::future_status::timeout)
    {
      fail(goal_handle, "downstream did not accept goal in time");
      return;
    }
    auto downstream_handle = send_future.get();
    if (!downstream_handle) {
      fail(goal_handle, "downstream rejected the gripper goal");
      return;
    }

    auto result_future = client_->async_get_result(downstream_handle);
    if (result_future.wait_for(
        std::chrono::duration<double>(goal_timeout_s_))
      == std::future_status::timeout)
    {
      fail(goal_handle, "downstream timed out");
      return;
    }
    const auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED &&
        wrapped.code != rclcpp_action::ResultCode::ABORTED)
    {
      // Treat anything other than SUCCEEDED/ABORTED as a transport failure;
      // we still inspect the downstream result fields below for ABORTED to
      // mirror the Python version's lenient success-detection logic.
      fail(goal_handle, "downstream returned an unexpected result code");
      return;
    }

    const auto & downstream_result = wrapped.result;

    publish_feedback(goal_handle, "settled", downstream_result->position);

    auto result = std::make_shared<OperateGripper::Result>();
    result->success = downstream_result->reached_goal ||
      !downstream_result->stalled;
    result->final_width_m = static_cast<float>(downstream_result->position);
    result->object_detected = downstream_result->stalled;
    result->contact_force_n = static_cast<float>(downstream_result->effort);
    result->error = result->success
      ? std::string{}
      : std::string{"did not reach goal and not stalled"};

    goal_handle->succeed(result);
  }

  // ── State ───────────────────────────────────────────────────────────────
  std::string downstream_name_;
  std::string server_action_name_;
  double open_pos_;
  double close_pos_;
  double default_force_n_;
  double connect_timeout_s_;
  double goal_timeout_s_;

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp_action::Client<GripperCommand>::SharedPtr client_;
  rclcpp_action::Server<OperateGripper>::SharedPtr server_;

  std::mutex mutex_;
  bool active_{false};
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GripperActionServer>();
  // MultiThreadedExecutor so the reentrant callback group can dispatch
  // downstream client callbacks while the action server's execute()
  // thread is blocked on its futures.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
