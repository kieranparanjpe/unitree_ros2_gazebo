// Robot-agnostic manifest-driven policy bridge.
//
// Parses an Isaac Lab-exported deploy.yaml (the same manifest unitree_rl_lab's C++ hardware
// deploy path reads - see unitree_rl_lab/deploy/include/isaaclab/manager/*.h), runs the
// exported ONNX policy, and commands raw joint effort computed with the same PD gains as
// training (tau = kp*(q_des - q) - kd*dq). All per-robot data (joint order, gains, default
// pose, observation/action term spec) comes from deploy.yaml + the `joint_names` parameter -
// nothing here is robot-specific.
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "onnxruntime_cxx_api.h"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "unitree_policy_bridge/observation_terms.hpp"
#include "yaml-cpp/yaml.h"

namespace unitree_policy_bridge
{

namespace
{

std::vector<float> as_float_vec(const YAML::Node & node)
{
  if (!node || node.IsNull()) {
    return {};
  }
  return node.as<std::vector<float>>();
}

// Far beyond any real actuator's torque limit (H2's are all under 400 N*m per
// unitree_rl_lab's UNITREE_H2_CFG actuator configs) - this only ever trips on a genuine
// numerical blow-up (e.g. an unclipped observation/action feedback loop fed bad sensor
// input), never on a legitimate command.
constexpr float kMaxSaneEffortNm = 1.0e6f;

std::pair<float, float> as_range_or_unbounded(const YAML::Node & node)
{
  if (!node || node.IsNull()) {
    return {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
  }
  return {node[0].as<float>(), node[1].as<float>()};
}

// Active rotation of vector v by quaternion (w,x,y,z): v' = v + 2w(q x v) + 2(q x (q x v)).
// To project a world-frame vector into the body frame (what the IMU-based hardware deploy
// path does via `root_quat_w.conjugate() * v`, see unitree_articulation.h), call this with
// the conjugate (w,-x,-y,-z).
std::array<float, 3> rotate_vector(
  float w, float x, float y, float z, const std::array<float, 3> & v)
{
  const float cx = y * v[2] - z * v[1];
  const float cy = z * v[0] - x * v[2];
  const float cz = x * v[1] - y * v[0];
  const float c2x = y * cz - z * cy;
  const float c2y = z * cx - x * cz;
  const float c2z = x * cy - y * cx;
  return {
    v[0] + 2.0f * w * cx + 2.0f * c2x,
    v[1] + 2.0f * w * cy + 2.0f * c2y,
    v[2] + 2.0f * w * cz + 2.0f * c2z,
  };
}

}  // namespace

class PolicyBridgeNode : public rclcpp::Node
{
public:
  PolicyBridgeNode(): Node("policy_bridge_node")
  {
    const auto deploy_yaml_path = declare_parameter<std::string>("deploy_yaml_path");
    const auto policy_onnx_path = declare_parameter<std::string>("policy_onnx_path");
    joint_names_ = declare_parameter<std::vector<std::string>>("joint_names");
    const auto joint_states_topic = declare_parameter<std::string>("joint_states_topic", "/joint_states");
    const auto imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    const auto cmd_vel_topic = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    const auto effort_command_topic =
      declare_parameter<std::string>("effort_command_topic", "/effort_controller/commands");
    const auto reset_topic = declare_parameter<std::string>("reset_topic", "/policy_reset");
    // Simulation only: something outside this node (unitree_gz_bringup's gz_reset_node)
    // teleports the robot to default_joint_pos once the stack is live, so the trained policy
    // must not engage before that has happened. On real hardware nothing can teleport the
    // robot - getting it to the default pose is the controller FSM's stand-up state, upstream
    // of this node - so this stays false there, which is why it is a parameter rather than a
    // wait baked in unconditionally.
    wait_for_reset_ = declare_parameter<bool>("wait_for_reset", false);
    // Rate at which the actuator PD converts the policy's position target into torque. Must
    // be far higher than the policy rate - see run_pd_step(). Defaults to the rate real motor
    // firmware closes this loop at, which is also gz_ros2_control's controller_manager rate.
    const double pd_update_rate_hz = declare_parameter<double>("pd_update_rate_hz", 1000.0);
    if (pd_update_rate_hz <= 0.0) {
      throw std::runtime_error("pd_update_rate_hz must be positive");
    }

    n_ = joint_names_.size();
    for (size_t i = 0; i < n_; ++i) {
      joint_index_by_name_[joint_names_[i]] = i;
    }
    joint_vel_.assign(n_, 0.0f);
    last_raw_action_.assign(n_, 0.0f);

    const YAML::Node deploy = YAML::LoadFile(deploy_yaml_path);
    load_dynamics(deploy);
    // Start from default_joint_pos rather than zero. The URDF declares it as each joint's
    // initial_value, so it is genuinely where the robot spawns; it is overwritten with live
    // data on the first /joint_states message, which arrives before anything is commanded.
    joint_pos_ = default_joint_pos_;
    load_action_term(deploy);
    load_command_ranges(deploy);
    load_observation_terms(deploy);
    load_policy(policy_onnx_path);

    const double step_dt = deploy["step_dt"].as<double>();

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_states_topic, 10,
      std::bind(&PolicyBridgeNode::on_joint_state, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::SensorDataQoS(),
      std::bind(&PolicyBridgeNode::on_imu, this, std::placeholders::_1));
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic, 10, std::bind(&PolicyBridgeNode::on_cmd_vel, this, std::placeholders::_1));
    effort_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(effort_command_topic, 10);
    // Latched on the publisher side, so a reset that lands while this node is still loading
    // its ONNX session isn't missed.
    reset_sub_ = create_subscription<std_msgs::msg::Empty>(
      reset_topic, rclcpp::QoS(1).transient_local(),
      std::bind(&PolicyBridgeNode::on_reset, this, std::placeholders::_1));

    // A ROS-time timer, NOT create_wall_timer: with use_sim_time the policy must step once
    // per step_dt of *simulation* time, the way it did in training. A wall timer ignores
    // use_sim_time entirely, so whenever the simulator runs slower than real time (measured
    // at RTF ~0.17 here) it fires several times per trained control interval, and keeps
    // firing on unchanged observations if the sim stalls - which is what let last_action
    // diverge on frozen input. As a side effect this also removes the startup race: no
    // simulation time passes before Gazebo is up, so no policy step happens either.
    step_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(step_dt),
      std::bind(&PolicyBridgeNode::on_step, this));
    // Separate, much faster timer for the actuator PD - see run_pd_step(). deploy.yaml does
    // not export the training physics rate (it is env.yaml's `dt`, 0.005s for this H2
    // checkpoint), and matching it would not be enough anyway, so this is its own parameter
    // rather than something derived from step_dt.
    pd_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / pd_update_rate_hz),
      std::bind(&PolicyBridgeNode::run_pd_step, this));

    RCLCPP_INFO(
      get_logger(), "policy_bridge_node ready: %zu joints, step_dt=%.4fs, wait_for_reset=%s",
      n_, step_dt, wait_for_reset_ ? "true" : "false");
  }

private:
  // deploy.yaml is NOT uniformly ordered, and this is the one place that matters.
  //
  // export_deploy_cfg.py writes most arrays straight out of Isaac Lab's own tensors, so they
  // are in the environment's joint order - default_joint_pos, the action scale/offset/clip,
  // and the observation scales all are. But stiffness and damping are *scattered* into the
  // robot's SDK motor order first:
  //
  //     stiffness = np.zeros(len(joint_sdk_names))
  //     stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0]   # SDK-indexed
  //     cfg["default_joint_pos"] = asset.data.default_joint_pos[0]         # env-indexed
  //
  // Reading them positionally alongside everything else therefore hands each joint some other
  // joint's gains. On H2 that gave the right knee kp=40 instead of 200 and the waist kp=200
  // instead of 150 - a knee that soft cannot hold up a 75kg robot, which is why it collapsed
  // even under a plain PD hold with no policy involved. joint_ids_map[i] is the SDK index of
  // environment joint i, which is exactly the gather needed to put them back.
  void load_dynamics(const YAML::Node & deploy)
  {
    const auto joint_ids_map = deploy["joint_ids_map"].as<std::vector<int>>();
    const auto stiffness_sdk = as_float_vec(deploy["stiffness"]);
    const auto damping_sdk = as_float_vec(deploy["damping"]);
    default_joint_pos_ = as_float_vec(deploy["default_joint_pos"]);
    if (joint_ids_map.size() != n_ || default_joint_pos_.size() != n_) {
      throw std::runtime_error(
        "deploy.yaml's joint_ids_map/default_joint_pos length doesn't match joint_names");
    }

    stiffness_.resize(n_);
    damping_.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
      const auto sdk_index = static_cast<size_t>(joint_ids_map[i]);
      if (sdk_index >= stiffness_sdk.size() || sdk_index >= damping_sdk.size()) {
        throw std::runtime_error(
          "deploy.yaml's joint_ids_map indexes past the end of stiffness/damping");
      }
      stiffness_[i] = stiffness_sdk[sdk_index];
      damping_[i] = damping_sdk[sdk_index];
    }
  }

  // Only a single action term (JointPositionAction over all joints) is supported - true of
  // every exported manifest seen so far (H1, H1_2, G1, ...). See
  // unitree_rl_lab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h for the
  // scale/offset/clip semantics this mirrors.
  void load_action_term(const YAML::Node & deploy)
  {
    const YAML::Node actions = deploy["actions"];
    if (!actions || actions.size() != 1) {
      throw std::runtime_error("expected exactly one action term in deploy.yaml's 'actions'");
    }
    const YAML::Node action_cfg = actions.begin()->second;
    action_scale_ = as_float_vec(action_cfg["scale"]);
    action_offset_ = as_float_vec(action_cfg["offset"]);
    if (action_scale_.empty()) action_scale_.assign(n_, 1.0f);
    if (action_offset_.empty()) action_offset_.assign(n_, 0.0f);

    const YAML::Node clip = action_cfg["clip"];
    if (clip && !clip.IsNull()) {
      action_clip_.reserve(n_);
      for (const auto & pair : clip) {
        action_clip_.push_back({pair[0].as<float>(), pair[1].as<float>()});
      }
    }
  }

  void load_command_ranges(const YAML::Node & deploy)
  {
    const YAML::Node ranges = deploy["commands"]["base_velocity"]["ranges"];
    lin_vel_x_range_ = as_range_or_unbounded(ranges["lin_vel_x"]);
    lin_vel_y_range_ = as_range_or_unbounded(ranges["lin_vel_y"]);
    ang_vel_z_range_ = as_range_or_unbounded(ranges["ang_vel_z"]);
  }

  // Ports unitree_rl_lab's observation term registry
  // (deploy/include/isaaclab/envs/mdp/observations/observations.h) - only the terms that
  // appear in the exported manifests to date. Add a new `else if` here to support a new
  // term; that's the only code change a new observation type needs.
  void load_observation_terms(const YAML::Node & deploy)
  {
    const YAML::Node observations = deploy["observations"];
    for (auto it = observations.begin(); it != observations.end(); ++it) {
      const std::string term_name = it->first.as<std::string>();
      const YAML::Node term_cfg = it->second;

      ObservationTerm term;
      term.history_length = term_cfg["history_length"] ? term_cfg["history_length"].as<int>() : 1;
      term.scale = as_float_vec(term_cfg["scale"]);
      term.clip = as_float_vec(term_cfg["clip"]);
      term.compute = observation_term_fn(term_name);

      term.reset(term.compute());
      obs_terms_.emplace_back(term_name, std::move(term));
    }
  }

  std::function<std::vector<float>()> observation_term_fn(const std::string & term_name)
  {
    if (term_name == "base_ang_vel") {
      return [this]() {
        return std::vector<float>(root_ang_vel_b_.begin(), root_ang_vel_b_.end());
      };
    }
    if (term_name == "projected_gravity") {
      return [this]() {
        return std::vector<float>(projected_gravity_b_.begin(), projected_gravity_b_.end());
      };
    }
    if (term_name == "joint_pos_rel") {
      return [this]() {
        std::vector<float> out(n_);
        for (size_t i = 0; i < n_; ++i) out[i] = joint_pos_[i] - default_joint_pos_[i];
        return out;
      };
    }
    if (term_name == "joint_vel_rel") {
      return [this]() { return joint_vel_; };
    }
    if (term_name == "last_action") {
      return [this]() { return last_raw_action_; };
    }
    if (term_name == "velocity_commands") {
      return [this]() { return velocity_command_; };
    }
    throw std::runtime_error(
      "observation term '" + term_name + "' is not registered in policy_bridge_node "
      "(add it to observation_term_fn())");
  }

  void load_policy(const std::string & policy_onnx_path)
  {
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "unitree_policy_bridge");
    Ort::SessionOptions session_options;
    ort_session_ = std::make_unique<Ort::Session>(
      *ort_env_, policy_onnx_path.c_str(), session_options);
    // Copy out of the allocator-owned buffer immediately - Ort::AllocatedStringPtr isn't
    // default-constructible, so it can't be stored as a plain deferred-init member.
    input_name_ = ort_session_->GetInputNameAllocated(0, ort_allocator_).get();
    output_name_ = ort_session_->GetOutputNameAllocated(0, ort_allocator_).get();
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    ++joint_state_msg_count_;
    for (size_t k = 0; k < msg->name.size(); ++k) {
      const auto found = joint_index_by_name_.find(msg->name[k]);
      if (found == joint_index_by_name_.end()) continue;
      const size_t i = found->second;
      if (k < msg->position.size()) joint_pos_[i] = static_cast<float>(msg->position[k]);
      if (k < msg->velocity.size()) joint_vel_[i] = static_cast<float>(msg->velocity[k]);
    }
  }

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    ++imu_msg_count_;
    root_ang_vel_b_ = {
      static_cast<float>(msg->angular_velocity.x),
      static_cast<float>(msg->angular_velocity.y),
      static_cast<float>(msg->angular_velocity.z),
    };
    static constexpr std::array<float, 3> kGravityW{0.0f, 0.0f, -1.0f};
    const float w = static_cast<float>(msg->orientation.w);
    const float x = static_cast<float>(msg->orientation.x);
    const float y = static_cast<float>(msg->orientation.y);
    const float z = static_cast<float>(msg->orientation.z);
    // conjugate (w,-x,-y,-z) rotates the world-frame gravity vector into the body frame.
    const auto pg = rotate_vector(w, -x, -y, -z, kGravityW);
    projected_gravity_b_ = {pg[0], pg[1], pg[2]};
  }

  // The robot was just teleported (see unitree_gz_bringup/gz_reset_node.py). Everything this
  // node remembers about the robot's motion is now about a trajectory that no longer
  // happened, so drop back to HOLDING and re-seed. The re-seed itself is deferred: at this
  // instant joint_pos_/root_ang_vel_b_ still hold pre-reset values, and seeding the history
  // with those would hand the policy exactly the discontinuity this is meant to avoid. Wait
  // for genuinely post-reset messages on both topics instead.
  void on_reset(const std_msgs::msg::Empty::SharedPtr)
  {
    reset_received_ = true;
    reseed_pending_ = true;
    // The pre-teleport target describes a robot that, as far as the simulation is concerned,
    // was never there - tracking it would drive against the pose just established.
    have_target_ = false;
    joint_state_count_at_reset_ = joint_state_msg_count_;
    imu_count_at_reset_ = imu_msg_count_;
    RCLCPP_INFO(get_logger(), "Reset notification received - re-seeding on post-reset data.");
  }

  // Mirrors ObservationManager::reset()/ActionManager::reset() at episode start in the C++
  // hardware deploy reference: every term's history buffer is filled with the current state
  // rather than left holding whatever it accumulated before, and "no previous action" is the
  // only sensible prior for last_action.
  void reseed_observations()
  {
    for (auto & term_entry : obs_terms_) {
      term_entry.second.reset(term_entry.second.compute());
    }
    std::fill(last_raw_action_.begin(), last_raw_action_.end(), 0.0f);
  }

  void on_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    // Unlike the joystick-driven reference (which negates lx/rx for its physical remote's
    // axis wiring), a Twist's fields already follow ROS convention - no sign flip here.
    velocity_command_[0] = std::clamp(
      static_cast<float>(msg->linear.x), lin_vel_x_range_.first, lin_vel_x_range_.second);
    velocity_command_[1] = std::clamp(
      static_cast<float>(msg->linear.y), lin_vel_y_range_.first, lin_vel_y_range_.second);
    velocity_command_[2] = std::clamp(
      static_cast<float>(msg->angular.z), ang_vel_z_range_.first, ang_vel_z_range_.second);
  }

  // There is deliberately no stand-up/settle phase before the policy runs. The robot is put
  // into deploy.yaml's default_joint_pos directly - by the URDF's initial_value at spawn, and
  // by gz_reset_node's teleport before the policy starts - so the state a settle phase would
  // have been trying to reach is already established exactly. Running a PD toward that pose
  // from that pose achieves nothing and, at this node's step_dt, actively wrecks it (see the
  // step-rate note in run_policy_step()). Isaac Lab does not settle either: it resets into
  // this pose and the policy acts from the next step. A stand-up phase belongs on hardware,
  // where nothing can teleport the robot - it is not a substitute for this one.
  void on_step()
  {
    ++step_count_;

    if (reseed_pending_ &&
      joint_state_msg_count_ > joint_state_count_at_reset_ &&
      imu_msg_count_ > imu_count_at_reset_)
    {
      reseed_observations();
      reseed_pending_ = false;
    }

    // Until every precondition holds, command nothing rather than something approximate. The
    // world is paused or barely stepping for this whole window, so there is nothing to hold
    // the robot against; publishing a guessed torque here is what previously let a "hold"
    // satisfy its own exit criteria against an assumed state.
    if (!ready_to_run()) {
      return;
    }
    run_policy_step();
  }

  bool ready_to_run()
  {
    if (joint_state_msg_count_ == 0 || imu_msg_count_ == 0) {
      log_waiting(
        "waiting for /joint_states and /imu (joint_state_msgs=%zu imu_msgs=%zu)",
        joint_state_msg_count_, imu_msg_count_);
      return false;
    }
    // The trained policy must not see a single frame of the pre-reset trajectory: the reset
    // teleports the robot, so anything measured before it describes a robot that, as far as
    // the simulation is now concerned, was never there.
    if (wait_for_reset_ && !reset_received_) {
      log_waiting("waiting for the reset notification before starting the policy");
      return false;
    }
    if (reseed_pending_) {
      log_waiting("waiting for post-reset sensor data before re-seeding the observation history");
      return false;
    }
    if (!started_) {
      started_ = true;
      RCLCPP_INFO(get_logger(), "Starting the trained policy from the reset default pose.");
    }
    return true;
  }

  template<typename ... Args>
  void log_waiting(const char * format, Args ... args)
  {
    if (step_count_ % 50 == 1) {
      RCLCPP_INFO(get_logger(), format, args ...);
    }
  }

  void run_policy_step()
  {
    std::vector<float> obs;
    for (auto & [name, term] : obs_terms_) {
      term.add(term.compute());
      const auto frame = term.get();
      obs.insert(obs.end(), frame.begin(), frame.end());
    }

    const std::vector<float> raw_action = run_policy(obs);
    last_raw_action_ = raw_action;

    std::vector<float> processed(n_);
    for (size_t i = 0; i < n_; ++i) {
      processed[i] = raw_action[i] * action_scale_[i] + action_offset_[i];
      if (!action_clip_.empty()) {
        processed[i] = std::clamp(processed[i], action_clip_[i][0], action_clip_[i][1]);
      }
    }

    // The action is a joint position target, nothing more. Converting it to torque is the
    // actuator's job and happens in run_pd_step() at its own, much higher rate - see the
    // comment there for why that separation is not optional.
    joint_target_ = std::move(processed);
    have_target_ = true;

    // TEMPORARY diagnostic logging - once per second (step_dt is typically 0.02s, so ~50
    // steps) - to check whether /joint_states and /imu are actually being received, and what
    // values are flowing through the PD computation. Remove once behavior is confirmed good.
    if (step_count_ % 50 == 1) {
      RCLCPP_INFO(
        get_logger(),
        "diag: RUNNING joint_state_msgs=%zu imu_msgs=%zu | cmd=[%.2f,%.2f,%.2f] | "
        "joint_pos[0]=%.3f joint_vel[0]=%.3f "
        "| ang_vel_b=[%.3f,%.3f,%.3f] proj_grav_b=[%.3f,%.3f,%.3f] | "
        "raw_action[0]=%.3f target[0]=%.3f pd_steps=%zu",
        joint_state_msg_count_, imu_msg_count_,
        velocity_command_[0], velocity_command_[1], velocity_command_[2],
        joint_pos_[0], joint_vel_[0],
        root_ang_vel_b_[0], root_ang_vel_b_[1], root_ang_vel_b_[2],
        projected_gravity_b_[0], projected_gravity_b_[1], projected_gravity_b_[2],
        raw_action[0], joint_target_[0], pd_step_count_);
    }
  }

  // The PD that turns the policy's position target into torque, deliberately running far
  // faster than the policy itself.
  //
  // In training this is Isaac Lab's implicit actuator model: the same kp/kd, but solved
  // *inside* the physics solver, which makes the damping term unconditionally stable. An
  // explicit PD held constant across a whole control period is not, and the limit is set by
  // the lightest joint: stability needs roughly kd*dt/I < 2, and H2's wrists pair kd=1.0 with
  // an inertia on the order of 1e-3 kg*m^2. At the policy's own 20ms period that ratio is
  // ~20 - wildly unstable, which is exactly what was seen: wrist joints pinned at their
  // 37.7 rad/s limit and the robot shaking itself apart the moment the policy engaged.
  // Matching training's 200Hz physics rate is not enough either (~5); the fix is to run the
  // PD at the rate the real actuator does, which is also the rate controller_manager reads
  // and writes at.
  void run_pd_step()
  {
    if (!have_target_) {
      return;
    }
    ++pd_step_count_;
    std::vector<float> tau(n_);
    for (size_t i = 0; i < n_; ++i) {
      tau[i] = stiffness_[i] * (joint_target_[i] - joint_pos_[i]) - damping_[i] * joint_vel_[i];
    }
    publish_effort_checked(tau);
  }

  // Shared by both phases - the runaway-explosion guard applies regardless of whether the
  // torque came from the plain default-pose hold or the trained policy.
  void publish_effort_checked(const std::vector<float> & tau)
  {
    for (size_t i = 0; i < n_; ++i) {
      if (!std::isfinite(tau[i]) || std::abs(tau[i]) > kMaxSaneEffortNm) {
        // Not a real control decision - this is a runaway observation/action feedback loop
        // (e.g. last_action compounding on itself when fed bad/missing sensor input; see the
        // diag log's joint_state_msgs/imu_msgs counts) about to send a physically
        // nonsensical force into the simulator. Fail safe (zero effort) and stop loudly
        // rather than silently freezing/exploding the sim.
        std_msgs::msg::Float64MultiArray zero_msg;
        zero_msg.data.assign(n_, 0.0);
        effort_pub_->publish(zero_msg);
        RCLCPP_FATAL(
          get_logger(),
          "Refusing to command joint '%s' with tau=%.6g N*m (sane bound is +/-%.0f). "
          "Published zero effort and stopping. This is almost always caused by bad/missing "
          "sensor input, not a real policy decision - check /joint_states and /imu are "
          "actually publishing before re-running.",
          joint_names_[i].c_str(), static_cast<double>(tau[i]),
          static_cast<double>(kMaxSaneEffortNm));
        throw std::runtime_error("policy_bridge_node: commanded effort out of sane bounds");
      }
    }
    std_msgs::msg::Float64MultiArray effort_msg;
    effort_msg.data.assign(tau.begin(), tau.end());
    effort_pub_->publish(effort_msg);
  }

  std::vector<float> run_policy(const std::vector<float> & obs)
  {
    std::vector<int64_t> input_shape{1, static_cast<int64_t>(obs.size())};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, const_cast<float *>(obs.data()), obs.size(),
      input_shape.data(), input_shape.size());

    const char * input_names[] = {input_name_.c_str()};
    const char * output_names[] = {output_name_.c_str()};
    auto output_tensors = ort_session_->Run(
      Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    const float * data = output_tensors.front().GetTensorData<float>();
    const size_t count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<float>(data, data + count);
  }

  // Robot config
  std::vector<std::string> joint_names_;
  std::unordered_map<std::string, size_t> joint_index_by_name_;
  size_t n_ = 0;
  std::vector<float> stiffness_;
  std::vector<float> damping_;
  std::vector<float> default_joint_pos_;
  std::vector<float> action_scale_;
  std::vector<float> action_offset_;
  std::vector<std::array<float, 2>> action_clip_;
  std::pair<float, float> lin_vel_x_range_;
  std::pair<float, float> lin_vel_y_range_;
  std::pair<float, float> ang_vel_z_range_;

  // Live state
  std::vector<float> joint_pos_;
  std::vector<float> joint_vel_;
  std::array<float, 3> root_ang_vel_b_{0.0f, 0.0f, 0.0f};
  std::array<float, 3> projected_gravity_b_{0.0f, 0.0f, -1.0f};
  std::vector<float> velocity_command_{0.0f, 0.0f, 0.0f};
  std::vector<float> last_raw_action_;
  // The policy's latest joint position target, held between policy steps and tracked by
  // run_pd_step() in between. Nothing is commanded until the first policy step produces one.
  std::vector<float> joint_target_;
  bool have_target_ = false;
  size_t pd_step_count_ = 0;

  // joint_state_msg_count_/imu_msg_count_ gate on_step()'s "real data has arrived" check -
  // no longer just diagnostic. step_count_ is only used to throttle the TEMPORARY diag logs.
  size_t joint_state_msg_count_ = 0;
  size_t imu_msg_count_ = 0;
  size_t step_count_ = 0;

  bool started_ = false;

  // See on_reset(). reseed_pending_ bridges the gap between "the robot was teleported" and
  // "messages describing where it was teleported to have arrived".
  bool wait_for_reset_ = false;
  bool reset_received_ = false;
  bool reseed_pending_ = false;
  size_t joint_state_count_at_reset_ = 0;
  size_t imu_count_at_reset_ = 0;

  std::vector<std::pair<std::string, ObservationTerm>> obs_terms_;

  // ONNX Runtime
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::AllocatorWithDefaultOptions ort_allocator_;
  std::string input_name_;
  std::string output_name_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_pub_;
  rclcpp::TimerBase::SharedPtr step_timer_;
  rclcpp::TimerBase::SharedPtr pd_timer_;
};

}  // namespace unitree_policy_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<unitree_policy_bridge::PolicyBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
