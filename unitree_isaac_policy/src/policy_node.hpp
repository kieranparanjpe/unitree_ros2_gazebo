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
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "isaaclab/envs/manager_based_rl_env.h"
// These two are included for their side effects, not for any symbol named below:
// REGISTER_ACTION / REGISTER_OBSERVATION in them are what populate actions_map() and
// observations_map() with joint_pos, base_ang_vel, projected_gravity, joint_pos_rel,
// joint_vel_rel and last_action. They look unused to a linter, and dropping either one still
// compiles - then throws "Observation term 'base_ang_vel' is not registered." at startup.
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "ros_articulation.hpp"

namespace unitree_isaac_policy
{

  class PolicyNode : public rclcpp::Node
  {
    public:
      PolicyNode(): Node("policy_node")
      {
        const auto deploy_path = declare_parameter<std::string>("deploy_yaml_path");
        const auto onnx_path = declare_parameter<std::string>("policy_onnx_path");
        joint_names_ = declare_parameter<std::vector<std::string>>("joint_names");
        const auto target_topic =
          declare_parameter<std::string>("target_topic", "/joint_targets");

        // Simulation only: gz_reset_node teleports the robot to the trained start pose once
        // the stack is live, so the policy must not engage before that. On hardware reaching
        // the default pose is the controller FSM's job upstream of this node, so it stays
        // false there and the policy seeds as soon as sensor data arrives.
        const auto wait_for_reset = declare_parameter<bool>("wait_for_reset", false);
        phase_ = wait_for_reset ? Phase::AwaitingReset : Phase::ResetTriggered;

        joint_count_ = joint_names_.size();
        for (size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
          joint_index_by_name_[joint_names_[joint_index]] = joint_index;
        }

        articulation_ = std::make_shared<RosArticulation>(joint_count_);

        // Replaces the stock term, which reads a Unitree handheld remote. Registered before
        // the env is built, since ObservationManager resolves every term name at
        // construction. A Twist already follows ROS convention, so unlike the joystick
        // version there is no sign flip - and going through UnitreeJoystick would also apply
        // its axis smoothing/deadzone.
        isaaclab::observations_map()["velocity_commands"] =
          [this](isaaclab::ManagerBasedRLEnv* env, YAML::Node) {
            const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
            std::vector<float> command{
              std::clamp(
                velocity_command_[0],
                ranges["lin_vel_x"][0].as<float>(),
                ranges["lin_vel_x"][1].as<float>()),
              std::clamp(
                velocity_command_[1],
                ranges["lin_vel_y"][0].as<float>(),
                ranges["lin_vel_y"][1].as<float>()),
              std::clamp(
                velocity_command_[2],
                ranges["ang_vel_z"][0].as<float>(),
                ranges["ang_vel_z"][1].as<float>())};
            // Every velocity_commands implementation publishes here; gait_phase reads it to
            // gate its clock the way training does.
            env->command = command;
            return command;
          };

        env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
          YAML::LoadFile(deploy_path), articulation_);
        env_->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path);

        joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
          declare_parameter<std::string>("joint_states_topic", "/joint_states"), 10,
          [this](const sensor_msgs::msg::JointState::ConstSharedPtr & msg) {
            on_joint_state(msg);
          });

        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
          declare_parameter<std::string>("imu_topic", "/imu"), rclcpp::SensorDataQoS(),
          [this](const sensor_msgs::msg::Imu::ConstSharedPtr & msg) {
            on_imu(msg);
          });

        cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
          declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel"), 10,
          [this](const geometry_msgs::msg::Twist::ConstSharedPtr & msg) {
            velocity_command_ = {
              static_cast<float>(msg->linear.x),
              static_cast<float>(msg->linear.y),
              static_cast<float>(msg->angular.z)};
          });

        // Latched by gz_reset_node, so a reset that fires before this node is up isn't
        // missed.
        reset_sub_ = create_subscription<std_msgs::msg::Empty>(
          declare_parameter<std::string>("reset_topic", "/policy_reset"),
          rclcpp::QoS(1).transient_local(),
          [this](const std_msgs::msg::Empty::ConstSharedPtr & msg) {
            on_reset(msg);
          });

        target_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(target_topic, 10);

        // A ROS-time timer, not a wall timer: with use_sim_time the policy must step once per
        // step_dt of *simulation* time the way it did in training. A wall timer ignores
        // use_sim_time, so it fires repeatedly on unchanged observations whenever the sim
        // runs slower than real time.
        timer_ = rclcpp::create_timer(
          this, get_clock(), rclcpp::Duration::from_seconds(env_->step_dt),
          [this]() {
            on_step();
          });

        RCLCPP_INFO(
          get_logger(), "policy_node ready: %zu joints, step_dt=%.4fs, wait_for_reset=%s",
          joint_count_, env_->step_dt, wait_for_reset ? "true" : "false");
      }

    private:

      enum class Phase
      {
        AwaitingReset,
        ResetTriggered,
        Running,
      };

      void on_joint_state(const sensor_msgs::msg::JointState::ConstSharedPtr & msg)
      {
        ++joint_state_msg_count_;

        for (size_t msg_index = 0; msg_index < msg->name.size(); ++msg_index) {
          const auto found = joint_index_by_name_.find(msg->name[msg_index]);

          if (found == joint_index_by_name_.end()) {
            continue;
          }

          const size_t joint_index = found->second;

          if (msg_index < msg->position.size()) {
            articulation_->set_joint_pos(joint_index, static_cast<float>(msg->position[msg_index]));
          }

          if (msg_index < msg->velocity.size()) {
            articulation_->set_joint_vel(joint_index, static_cast<float>(msg->velocity[msg_index]));
          }
        }
      }

      void on_imu(const sensor_msgs::msg::Imu::ConstSharedPtr & msg)
      {
        ++imu_msg_count_;

        articulation_->set_imu(
          Eigen::Quaternionf(
            static_cast<float>(msg->orientation.w),
            static_cast<float>(msg->orientation.x),
            static_cast<float>(msg->orientation.y),
            static_cast<float>(msg->orientation.z)
          ),
        Eigen::Vector3f(
            static_cast<float>(msg->angular_velocity.x),
            static_cast<float>(msg->angular_velocity.y),
            static_cast<float>(msg->angular_velocity.z)
          )
        );
      }

      // The robot was just teleported, so everything remembered about its motion describes a
      // trajectory that no longer happened. Zeroing the counters is what defers the re-seed:
      // at this instant the cached state is still pre-reset, and seeding on that would hand
      // the policy exactly the discontinuity this avoids.
      void on_reset(const std_msgs::msg::Empty::ConstSharedPtr &)
      {
        phase_ = Phase::ResetTriggered;
        joint_state_msg_count_ = 0;
        imu_msg_count_ = 0;
        RCLCPP_INFO(get_logger(), "Reset received - re-seeding on post-reset data.");
      }

      void on_step()
      {
        if (phase_ == Phase::ResetTriggered && have_sensor_data()) {
          // Fills every observation term's history history_length deep with the current
          // state, and zeroes last_action - so it has to run on post-reset data.
          env_->reset();
          phase_ = Phase::Running;
          RCLCPP_INFO_ONCE(
            get_logger(), "Starting the trained policy from the reset default pose.");
        }

        if (phase_ != Phase::Running) {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000, "%s",
            phase_ == Phase::AwaitingReset
              ? "waiting for the reset notification before starting the policy"
              : "waiting for /joint_states and /imu");
          return;
        }

        env_->step();
        const auto action = env_->action_manager->processed_actions();

        std_msgs::msg::Float64MultiArray msg;
        msg.data.assign(action.begin(), action.end());
        target_pub_->publish(msg);

        // Counts are per reset epoch, not lifetime - see on_reset().
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "diag: joint_msgs=%zu imu_msgs=%zu | cmd=[%.2f,%.2f,%.2f] | "
          "proj_grav_b=[%.3f,%.3f,%.3f] | target[0]=%.3f",
          joint_state_msg_count_, imu_msg_count_,
          velocity_command_[0], velocity_command_[1], velocity_command_[2],
          articulation_->data.projected_gravity_b[0],
          articulation_->data.projected_gravity_b[1],
          articulation_->data.projected_gravity_b[2],
          action[0]);
      }

      bool have_sensor_data() const
      {
        return joint_state_msg_count_ > 0 && imu_msg_count_ > 0;
      }

      // Robot config
      std::vector<std::string> joint_names_;
      std::unordered_map<std::string, size_t> joint_index_by_name_;
      size_t joint_count_ = 0;

      // Reused upstream runtime, plus the state it reads from
      std::shared_ptr<RosArticulation> articulation_;
      std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;
      std::vector<float> velocity_command_{0.0f, 0.0f, 0.0f};

      // Set from wait_for_reset in the constructor. Messages received since the current reset
      // epoch began, which is what defers the re-seed - see on_reset().
      Phase phase_ = Phase::ResetTriggered;
      size_t joint_state_msg_count_ = 0;
      size_t imu_msg_count_ = 0;

      // ROS interfaces
      rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
      rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
      rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
      rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
      rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_pub_;
      rclcpp::TimerBase::SharedPtr timer_;
  };

}  // namespace unitree_isaac_policy
