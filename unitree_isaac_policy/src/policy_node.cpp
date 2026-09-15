// Observation building + policy inference, reusing unitree_rl_lab's deploy runtime verbatim.
//
// ManagerBasedRLEnv, ObservationManager, ActionManager, the mdp term registries and OrtRunner
// are the same headers the DDS hardware/unitree_mujoco path runs, so the observation vector
// and action processing are shared code rather than a re-implementation. Only two things are
// substituted: where robot state comes from (ROS topics instead of DDS LowState) and where the
// velocity command comes from (/cmd_vel instead of the handheld joystick).
//
// This node emits joint position targets. Converting them to torque is the actuator's job -
// motor firmware on hardware, MuJoCo's actuator in unitree_mujoco, pd_node here.
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace unitree_isaac_policy
{

// The ROS counterpart of unitree_rl_lab's BaseArticulation (deploy/include/
// unitree_articulation.h), which reads the same fields out of a DDS LowState.
class RosArticulation : public isaaclab::Articulation
{
public:
  void update() override
  {
    for (size_t i = 0; i < q_.size(); ++i) {
      data.joint_pos[i] = q_[i];
      data.joint_vel[i] = dq_[i];
    }
    data.root_ang_vel_b = gyro_;
    data.projected_gravity_b = quat_.conjugate() * data.GRAVITY_VEC_W;
  }

  std::vector<float> q_, dq_;
  Eigen::Vector3f gyro_{0.0f, 0.0f, 0.0f};
  Eigen::Quaternionf quat_{1.0f, 0.0f, 0.0f, 0.0f};
};

class PolicyNode : public rclcpp::Node
{
public:
  PolicyNode()
  : Node("policy_node")
  {
    const auto deploy_path = declare_parameter<std::string>("deploy_yaml_path");
    const auto onnx_path = declare_parameter<std::string>("policy_onnx_path");
    joint_names_ = declare_parameter<std::vector<std::string>>("joint_names");
    const auto target_topic = declare_parameter<std::string>("target_topic", "/joint_targets");
    // Simulation only: gz_reset_node teleports the robot to the trained start pose once the
    // stack is live, so the policy must not engage before that. On hardware reaching the
    // default pose is the controller FSM's job upstream of this node, so it stays false there.
    wait_for_reset_ = declare_parameter<bool>("wait_for_reset", false);

    n_ = joint_names_.size();
    for (size_t i = 0; i < n_; ++i) {
      index_by_name_[joint_names_[i]] = i;
    }

    art_ = std::make_shared<RosArticulation>();
    art_->q_.assign(n_, 0.0f);
    art_->dq_.assign(n_, 0.0f);

    // Replaces the stock term, which reads a Unitree handheld remote. Registered before the
    // env is built, since ObservationManager resolves every term name at construction. A
    // Twist already follows ROS convention, so unlike the joystick version there is no sign
    // flip - and going through UnitreeJoystick would also apply its axis smoothing/deadzone.
    isaaclab::observations_map()["velocity_commands"] =
      [this](isaaclab::ManagerBasedRLEnv * env, YAML::Node) {
        const auto r = env->cfg["commands"]["base_velocity"]["ranges"];
        return std::vector<float>{
          std::clamp(cmd_[0], r["lin_vel_x"][0].as<float>(), r["lin_vel_x"][1].as<float>()),
          std::clamp(cmd_[1], r["lin_vel_y"][0].as<float>(), r["lin_vel_y"][1].as<float>()),
          std::clamp(cmd_[2], r["ang_vel_z"][0].as<float>(), r["ang_vel_z"][1].as<float>())};
      };

    env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(YAML::LoadFile(deploy_path), art_);
    env_->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path);

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      declare_parameter<std::string>("joint_states_topic", "/joint_states"), 10,
      std::bind(&PolicyNode::on_joint_state, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      declare_parameter<std::string>("imu_topic", "/imu"), rclcpp::SensorDataQoS(),
      std::bind(&PolicyNode::on_imu, this, std::placeholders::_1));
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel"), 10,
      [this](geometry_msgs::msg::Twist::SharedPtr m) {
        cmd_ = {static_cast<float>(m->linear.x), static_cast<float>(m->linear.y),
          static_cast<float>(m->angular.z)};
      });
    // Latched by gz_reset_node, so a reset that fires before this node is up isn't missed.
    reset_sub_ = create_subscription<std_msgs::msg::Empty>(
      declare_parameter<std::string>("reset_topic", "/policy_reset"),
      rclcpp::QoS(1).transient_local(),
      std::bind(&PolicyNode::on_reset, this, std::placeholders::_1));
    target_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(target_topic, 10);

    // A ROS-time timer, not a wall timer: with use_sim_time the policy must step once per
    // step_dt of *simulation* time the way it did in training. A wall timer ignores
    // use_sim_time, so it fires repeatedly on unchanged observations whenever the sim runs
    // slower than real time.
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(env_->step_dt),
      std::bind(&PolicyNode::on_step, this));

    RCLCPP_INFO(
      get_logger(), "policy_node ready: %zu joints, step_dt=%.4fs, wait_for_reset=%s", n_,
      env_->step_dt, wait_for_reset_ ? "true" : "false");
  }

private:
  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    ++joint_msgs_;
    for (size_t k = 0; k < msg->name.size(); ++k) {
      const auto it = index_by_name_.find(msg->name[k]);
      if (it == index_by_name_.end()) continue;
      if (k < msg->position.size()) art_->q_[it->second] = msg->position[k];
      if (k < msg->velocity.size()) art_->dq_[it->second] = msg->velocity[k];
    }
  }

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    ++imu_msgs_;
    art_->gyro_ = {static_cast<float>(msg->angular_velocity.x),
      static_cast<float>(msg->angular_velocity.y), static_cast<float>(msg->angular_velocity.z)};
    art_->quat_ = Eigen::Quaternionf(
      msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
  }

  // The robot was just teleported, so everything remembered about its motion describes a
  // trajectory that no longer happened. The re-seed is deferred because at this instant the
  // cached state is still pre-reset; seeding on that would hand the policy exactly the
  // discontinuity this avoids.
  void on_reset(const std_msgs::msg::Empty::SharedPtr)
  {
    reset_received_ = reseed_pending_ = true;
    joint_msgs_at_reset_ = joint_msgs_;
    imu_msgs_at_reset_ = imu_msgs_;
    RCLCPP_INFO(get_logger(), "Reset received - re-seeding on post-reset data.");
  }

  void on_step()
  {
    ++steps_;
    if (reseed_pending_ && joint_msgs_ > joint_msgs_at_reset_ && imu_msgs_ > imu_msgs_at_reset_) {
      env_->reset();
      reseed_pending_ = false;
    }
    if (!ready()) return;

    env_->step();
    const auto action = env_->action_manager->processed_actions();

    std_msgs::msg::Float64MultiArray msg;
    msg.data.assign(action.begin(), action.end());
    target_pub_->publish(msg);

    if (steps_ % 50 == 1) {
      RCLCPP_INFO(
        get_logger(),
        "diag: joint_msgs=%zu imu_msgs=%zu | cmd=[%.2f,%.2f,%.2f] | proj_grav_b=[%.3f,%.3f,%.3f]"
        " | target[0]=%.3f",
        joint_msgs_, imu_msgs_, cmd_[0], cmd_[1], cmd_[2], art_->data.projected_gravity_b[0],
        art_->data.projected_gravity_b[1], art_->data.projected_gravity_b[2], action[0]);
    }
  }

  bool ready()
  {
    const char * blocked = nullptr;
    if (joint_msgs_ == 0 || imu_msgs_ == 0) {
      blocked = "waiting for /joint_states and /imu";
    } else if (wait_for_reset_ && !reset_received_) {
      blocked = "waiting for the reset notification before starting the policy";
    } else if (reseed_pending_) {
      blocked = "waiting for post-reset sensor data before re-seeding";
    }
    if (blocked) {
      if (steps_ % 50 == 1) RCLCPP_INFO(get_logger(), "%s", blocked);
      return false;
    }
    if (!started_) {
      started_ = true;
      RCLCPP_INFO(get_logger(), "Starting the trained policy from the reset default pose.");
    }
    return true;
  }

  std::vector<std::string> joint_names_;
  std::unordered_map<std::string, size_t> index_by_name_;
  size_t n_ = 0;

  std::shared_ptr<RosArticulation> art_;
  std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;
  std::vector<float> cmd_{0.0f, 0.0f, 0.0f};

  bool wait_for_reset_ = false, reset_received_ = false, reseed_pending_ = false,
    started_ = false;
  size_t joint_msgs_ = 0, imu_msgs_ = 0, steps_ = 0, joint_msgs_at_reset_ = 0,
    imu_msgs_at_reset_ = 0;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace unitree_isaac_policy

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<unitree_isaac_policy::PolicyNode>());
  rclcpp::shutdown();
  return 0;
}
