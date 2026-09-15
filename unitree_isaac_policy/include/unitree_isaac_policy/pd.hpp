#pragma once

// Actuator model, deliberately free of ROS so a ros2_control controller can reuse it
// unchanged. torque = kp*(target_pos - joint_pos) - kd*joint_vel is what Isaac's implicit
// actuator, the motor firmware, and MuJoCo all compute; a PID's derivative-on-error term is
// not the same thing.
#include <cmath>
#include <stdexcept>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace unitree_isaac_policy
{

  struct Gains
  {
    std::vector<double> kp;
    std::vector<double> kd;
  };

  // deploy.yaml exports stiffness/damping in the robot's SDK motor order while everything
  // else is in Isaac's environment order; joint_ids_map[env_index] is the SDK index of
  // environment joint env_index. Reading them positionally hands each joint another joint's
  // gains.
  inline Gains load_gains(const YAML::Node & deploy, size_t joint_count)
  {
    const auto joint_ids_map = deploy["joint_ids_map"].as<std::vector<int>>();
    const auto stiffness_sdk_order = deploy["stiffness"].as<std::vector<double>>();
    const auto damping_sdk_order = deploy["damping"].as<std::vector<double>>();

    if (joint_ids_map.size() != joint_count) {
      throw std::runtime_error("deploy.yaml joint_ids_map length doesn't match joint_names");
    }

    Gains gains;
    for (size_t env_index = 0; env_index < joint_count; ++env_index) {
      const auto sdk_index = static_cast<size_t>(joint_ids_map[env_index]);

      if (sdk_index >= stiffness_sdk_order.size() || sdk_index >= damping_sdk_order.size()) {
        throw std::runtime_error("deploy.yaml joint_ids_map indexes past stiffness/damping");
      }

      gains.kp.push_back(stiffness_sdk_order[sdk_index]);
      gains.kd.push_back(damping_sdk_order[sdk_index]);
    }
    return gains;
  }

  // Far beyond any real actuator limit - only trips on a numerical blow-up, never on a
  // legitimate command.
  constexpr double kMaxSaneEffortNm = 1.0e6;

  inline bool sane(double torque)
  {
    return std::isfinite(torque) && std::abs(torque) <= kMaxSaneEffortNm;
  }

  inline void compute_effort(
    const Gains & gains,
    const std::vector<double> & target_pos,
    const std::vector<double> & joint_pos,
    const std::vector<double> & joint_vel,
    std::vector<double> & torque)
  {
    for (size_t joint_index = 0; joint_index < torque.size(); ++joint_index) {
      torque[joint_index] =
        gains.kp[joint_index] * (target_pos[joint_index] - joint_pos[joint_index]) -
          gains.kd[joint_index] * joint_vel[joint_index];
    }
  }

}  // namespace unitree_isaac_policy
