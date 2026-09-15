// Actuator model: turns the policy's joint position targets into joint torques.
//
// Stands in for what closes this loop on every other target - motor firmware on hardware,
// MuJoCo's actuator inside unitree_mujoco, Isaac's implicit-PD actuator in training. Gazebo's
// effort interface has nothing behind it, so this is the one piece of the stack with no
// upstream equivalent to reuse. It runs far faster than the policy: an explicit PD held across
// a whole 20ms control period is unstable for the lightest joints (H2's wrists pair kd=1.0
// with ~1e-3 kg*m^2 of inertia).
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "unitree_isaac_policy/pd.hpp"

namespace unitree_isaac_policy
{

  class PdNode : public rclcpp::Node
  {
    public:
      PdNode()
      : Node("pd_node")
      {
        const auto deploy_path = declare_parameter<std::string>("deploy_yaml_path");
        joint_names_ = declare_parameter<std::vector<std::string>>("joint_names");

        // Defaults to the rate real motor firmware closes this loop at, which is also
        // gz_ros2_control's controller_manager rate.
        const double rate_hz = declare_parameter<double>("pd_update_rate_hz", 1000.0);

        joint_count_ = joint_names_.size();
        for (size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
          joint_index_by_name_[joint_names_[joint_index]] = joint_index;
        }

        gains_ = load_gains(YAML::LoadFile(deploy_path), joint_count_);
        joint_pos_.assign(joint_count_, 0.0);
        joint_vel_.assign(joint_count_, 0.0);
        torque_.assign(joint_count_, 0.0);

        joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
          declare_parameter<std::string>("joint_states_topic", "/joint_states"), 10,
          [this](const sensor_msgs::msg::JointState::ConstSharedPtr & msg) {
            on_joint_state(msg);
          });

        target_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
          declare_parameter<std::string>("target_topic", "/joint_targets"), 10,
          [this](const std_msgs::msg::Float64MultiArray::ConstSharedPtr & msg) {
            if (msg->data.size() != joint_count_) {
              RCLCPP_ERROR_ONCE(
                get_logger(), "target has %zu entries, expected %zu - ignoring",
                msg->data.size(), joint_count_);
              return;
            }
            target_pos_ = msg->data;
          });

        effort_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
          declare_parameter<std::string>(
            "effort_command_topic", "/effort_controller/commands"), 10);

        timer_ = rclcpp::create_timer(
          this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_hz),
          [this]() {
            on_tick();
          });

        RCLCPP_INFO(
          get_logger(), "pd_node ready: %zu joints at %.0f Hz", joint_count_, rate_hz);
      }

    private:
      void on_joint_state(const sensor_msgs::msg::JointState::ConstSharedPtr & msg)
      {
        for (size_t msg_index = 0; msg_index < msg->name.size(); ++msg_index) {
          const auto found = joint_index_by_name_.find(msg->name[msg_index]);

          if (found == joint_index_by_name_.end()) {
            continue;
          }

          const size_t joint_index = found->second;

          if (msg_index < msg->position.size()) {
            joint_pos_[joint_index] = msg->position[msg_index];
          }

          if (msg_index < msg->velocity.size()) {
            joint_vel_[joint_index] = msg->velocity[msg_index];
          }
        }
      }

      void on_tick()
      {
        if (target_pos_.empty()) {
          return;
        }

        compute_effort(gains_, target_pos_, joint_pos_, joint_vel_, torque_);

        for (size_t joint_index = 0; joint_index < joint_count_; ++joint_index) {
          if (sane(torque_[joint_index])) {
            continue;
          }

          // Not a real control decision - a runaway feedback loop about to put a physically
          // nonsensical force into the simulator. Fail safe and stop loudly.
          std_msgs::msg::Float64MultiArray zero_effort;
          zero_effort.data.assign(joint_count_, 0.0);
          effort_pub_->publish(zero_effort);

          RCLCPP_FATAL(
            get_logger(),
            "Refusing to command joint '%s' with tau=%.6g N*m. Published zero effort and "
            "stopping. This is almost always bad/missing sensor input - check /joint_states "
            "and /imu.",
            joint_names_[joint_index].c_str(), torque_[joint_index]);

          throw std::runtime_error("pd_node: commanded effort out of sane bounds");
        }

        std_msgs::msg::Float64MultiArray msg;
        msg.data = torque_;
        effort_pub_->publish(msg);
      }

      // Robot config
      std::vector<std::string> joint_names_;
      std::unordered_map<std::string, size_t> joint_index_by_name_;
      size_t joint_count_ = 0;
      Gains gains_;

      // Live state, all indexed in deploy.yaml's environment joint order
      std::vector<double> joint_pos_;
      std::vector<double> joint_vel_;
      std::vector<double> torque_;
      std::vector<double> target_pos_;

      // ROS interfaces
      rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
      rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr target_sub_;
      rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_pub_;
      rclcpp::TimerBase::SharedPtr timer_;
  };

}  // namespace unitree_isaac_policy

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<unitree_isaac_policy::PdNode>());
  rclcpp::shutdown();
  return 0;
}
