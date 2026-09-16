// Hold-to-drive keyboard teleop: WASD for the linear axes, left/right arrows for yaw, space to
// stop. Publishes geometry_msgs/Twist on /cmd_vel at a steady rate.
//
// Terminals deliver key *press* events but never key *release*, so "hold to drive" is
// reconstructed from the terminal's auto-repeat: every press refreshes a timestamp, and the
// command decays to zero once no key has arrived for key_timeout. That timeout has to clear
// X11's initial auto-repeat delay (~500ms by default), which is why releasing a key coasts for
// about half a second rather than stopping instantly. Space is the immediate stop.
//
// Each axis times out on its own. Sharing one timestamp across all three was tried first and
// is worse: any held key keeps every axis alive, so tapping D and then S leaves you driving
// sideways and forwards with no way to release just the sideways part.
//
// The cost is that terminals only auto-repeat the most recently pressed key, so a combined
// command (forward while turning) holds only for key_timeout after the last press of the
// axis you are not holding - to sustain an arc you tap W while holding the arrow. That is a
// terminal limitation, not a choice: nothing in a tty reports which keys are currently down.
#include <atomic>
#include <chrono>
#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "keyboard_handler/keyboard_handler.hpp"
#include "rclcpp/rclcpp.hpp"

namespace unitree_keyboard_teleop
{

  class KeyboardController : public rclcpp::Node
  {
    public:
      KeyboardController(): Node("keyboard_controller")
      {
        // The policy clamps whatever arrives to the ranges it was trained on - for the current
        // H2 checkpoint lin_x [-0.5, 1.0], lin_y [-0.3, 0.3], ang_z [-0.2, 0.2]. These defaults
        // sit inside that so what you command is what the policy acts on; raise linear_x
        // towards 1.0 if you want the trained maximum.
        linear_x_ = declare_parameter<double>("linear_x", 1);
        linear_y_ = declare_parameter<double>("linear_y", 0.5);
        angular_z_ = declare_parameter<double>("angular_z", 0.5);

        // Must exceed the terminal's initial auto-repeat delay or a held key stutters: one
        // press, a gap, then the repeat stream.
        key_timeout_ = declare_parameter<double>("key_timeout", 0.6);

        const auto publish_rate = declare_parameter<double>("publish_rate_hz", 20.0);
        const auto cmd_vel_topic = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");

        cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic, 10);

        bind_key(KeyboardHandler::KeyCode::W, &x_, linear_x_);
        bind_key(KeyboardHandler::KeyCode::S, &x_, -linear_x_);
        bind_key(KeyboardHandler::KeyCode::A, &y_, linear_y_);
        bind_key(KeyboardHandler::KeyCode::D, &y_, -linear_y_);
        bind_key(KeyboardHandler::KeyCode::CURSOR_LEFT, &yaw_, angular_z_);
        bind_key(KeyboardHandler::KeyCode::CURSOR_RIGHT, &yaw_, -angular_z_);

        keyboard_handler_.add_key_press_callback(
          [this](KeyboardHandler::KeyCode, KeyboardHandler::KeyModifiers) {
            stop();
          },
          KeyboardHandler::KeyCode::SPACE);

        // A wall timer, not a ROS-time timer: the operator's key presses happen in wall time
        // whatever the simulation is doing, and this node deliberately does not set
        // use_sim_time. A sim-time timer would stop publishing whenever the world is paused,
        // which is exactly when you might be trying to stop the robot.
        timer_ = create_wall_timer(
          std::chrono::duration<double>(1.0 / publish_rate),
          [this]() {
            publish();
          });

        RCLCPP_INFO(
          get_logger(),
          "keyboard_controller ready on %s\n"
          "  W/S  forward / back   (%.2f m/s)\n"
          "  A/D  left / right     (%.2f m/s)\n"
          "  <-/->  turn left / right  (%.2f rad/s)\n"
          "  space  stop\n"
          "Hold a key to keep moving; release and it stops after %.2fs.",
          cmd_vel_topic.c_str(), linear_x_, linear_y_, angular_z_, key_timeout_);
      }

    private:
      // KeyboardHandler runs its own reader thread and invokes callbacks from it, so both
      // fields are shared with the publish timer's thread and have to be atomic.
      struct Axis
      {
        std::atomic<double> value{0.0};
        std::atomic<int64_t> last_press_ns{0};
      };

      // Every binding is the same shape: set one axis to a fixed value and mark that axis
      // fresh. Written from the keyboard handler's own thread - see the Axis comment.
      void bind_key(KeyboardHandler::KeyCode key, Axis * axis, double value)
      {
        keyboard_handler_.add_key_press_callback(
          [axis, value](KeyboardHandler::KeyCode, KeyboardHandler::KeyModifiers) {
            axis->value.store(value);
            axis->last_press_ns.store(now_ns());
          },
          key);
      }

      void stop()
      {
        for (Axis * axis : {&x_, &y_, &yaw_}) {
          axis->value.store(0.0);
          // Deliberately stale, so held() reads false immediately rather than treating the
          // space bar as one more press keeping the last velocity alive.
          axis->last_press_ns.store(0);
        }
      }

      void publish()
      {
        geometry_msgs::msg::Twist msg;
        msg.linear.x = held(x_);
        msg.linear.y = held(y_);
        msg.angular.z = held(yaw_);
        cmd_vel_pub_->publish(msg);
      }

      double held(const Axis & axis) const
      {
        const auto last = axis.last_press_ns.load();
        if (last == 0) {
          return 0.0;
        }
        const auto age_seconds = static_cast<double>(now_ns() - last) * 1e-9;
        return age_seconds > key_timeout_ ? 0.0 : axis.value.load();
      }

      static int64_t now_ns()
      {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      }

      double linear_x_ = 0.0;
      double linear_y_ = 0.0;
      double angular_z_ = 0.0;
      double key_timeout_ = 0.0;

      Axis x_;
      Axis y_;
      Axis yaw_;

      // Declared after the state it writes to, so it is constructed last and destroyed first -
      // the reader thread is joined before the atomics it touches go away.
      KeyboardHandler keyboard_handler_;

      rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
      rclcpp::TimerBase::SharedPtr timer_;
  };

}  // namespace unitree_keyboard_teleop

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<unitree_keyboard_teleop::KeyboardController>());
  rclcpp::shutdown();
  return 0;
}
