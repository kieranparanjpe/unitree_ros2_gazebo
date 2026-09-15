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
      PdNode(): Node("pd_node")
      {
        const auto deploy_path = declare_parameter<std::string>("deploy_yaml_path");
        joint_names_ = declare_parameter<std::vector<std::string>>("joint_names");
        // Defaults to the rate real motor firmware closes this loop at, which is also
        // gz_ros2_control's controller_manager rate.
        const double rate_hz = declare_parameter<double>("pd_update_rate_hz", 1000.0);

        n_ = joint_names_.size();
        for (size_t i = 0; i < n_; ++i) {
          index_by_name_[joint_names_[i]] = i;
        }
        gains_ = load_gains(YAML::LoadFile(deploy_path), n_);
        q_.assign(n_, 0.0);
        dq_.assign(n_, 0.0);
        tau_.assign(n_, 0.0);

        joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
          declare_parameter<std::string>("joint_states_topic", "/joint_states"), 10,
          std::bind(&PdNode::on_joint_state, this, std::placeholders::_1));
        target_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
          declare_parameter<std::string>("target_topic", "/joint_targets"), 10,
          [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
            if (m->data.size() != n_) {
              RCLCPP_ERROR_ONCE(
                get_logger(), "target has %zu entries, expected %zu - ignoring", m->data.size(), n_);
              return;
            }
            target_ = m->data;
          });
        effort_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
          declare_parameter<std::string>("effort_command_topic", "/effort_controller/commands"), 10);

        timer_ = rclcpp::create_timer(
          this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_hz),
          std::bind(&PdNode::on_tick, this));

        RCLCPP_INFO(get_logger(), "pd_node ready: %zu joints at %.0f Hz", n_, rate_hz);
      }

    private:
      void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
      {
        for (size_t k = 0; k < msg->name.size(); ++k) {
          const auto it = index_by_name_.find(msg->name[k]);
          if (it == index_by_name_.end()) continue;
          if (k < msg->position.size()) q_[it->second] = msg->position[k];
          if (k < msg->velocity.size()) dq_[it->second] = msg->velocity[k];
        }
      }

      void on_tick()
      {
        if (target_.empty()) return;
        compute_effort(gains_, target_, q_, dq_, tau_);

        for (size_t i = 0; i < n_; ++i) {
          if (sane(tau_[i])) continue;
          // Not a real control decision - a runaway feedback loop about to put a physically
          // nonsensical force into the simulator. Fail safe and stop loudly.
          std_msgs::msg::Float64MultiArray zero;
          zero.data.assign(n_, 0.0);
          effort_pub_->publish(zero);
          RCLCPP_FATAL(
            get_logger(),
            "Refusing to command joint '%s' with tau=%.6g N*m. Published zero effort and stopping. "
            "This is almost always bad/missing sensor input - check /joint_states and /imu.",
            joint_names_[i].c_str(), tau_[i]);
          throw std::runtime_error("pd_node: commanded effort out of sane bounds");
        }

        std_msgs::msg::Float64MultiArray msg;
        msg.data = tau_;
        effort_pub_->publish(msg);
      }

      std::vector<std::string> joint_names_;
      std::unordered_map<std::string, size_t> index_by_name_;
      size_t n_ = 0;
      Gains gains_;
      std::vector<double> q_, dq_, tau_, target_;

      rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
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
