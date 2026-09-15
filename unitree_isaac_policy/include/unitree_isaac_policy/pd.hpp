#pragma once

// Actuator model, deliberately free of ROS so a ros2_control controller can reuse it
// unchanged. tau = kp*(q_des - q) - kd*dq is what Isaac's implicit actuator, the motor
// firmware, and MuJoCo all compute; a PID's derivative-on-error term is not the same thing.
#include <cmath>
#include <stdexcept>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace unitree_isaac_policy
{

  struct Gains
  {
    std::vector<double> kp, kd;
  };

  // deploy.yaml exports stiffness/damping in the robot's SDK motor order while everything else
  // is in Isaac's environment order; joint_ids_map[i] is the SDK index of environment joint i.
  // Reading them positionally hands each joint another joint's gains.
  inline Gains load_gains(const YAML::Node & deploy, size_t n)
  {
    const auto ids = deploy["joint_ids_map"].as<std::vector<int>>();
    const auto kp_sdk = deploy["stiffness"].as<std::vector<double>>();
    const auto kd_sdk = deploy["damping"].as<std::vector<double>>();
    if (ids.size() != n) {
      throw std::runtime_error("deploy.yaml joint_ids_map length doesn't match joint_names");
    }
    Gains g;
    for (size_t i = 0; i < n; ++i) {
      const auto s = static_cast<size_t>(ids[i]);
      if (s >= kp_sdk.size() || s >= kd_sdk.size()) {
        throw std::runtime_error("deploy.yaml joint_ids_map indexes past stiffness/damping");
      }
      g.kp.push_back(kp_sdk[s]);
      g.kd.push_back(kd_sdk[s]);
    }
    return g;
  }

  // Far beyond any real actuator limit - only trips on a numerical blow-up, never on a
  // legitimate command.
  constexpr double kMaxSaneEffortNm = 1.0e6;

  inline bool sane(double tau)
  {
    return std::isfinite(tau) && std::abs(tau) <= kMaxSaneEffortNm;
  }

  inline void compute_effort(
    const Gains & g, const std::vector<double> & target, const std::vector<double> & q,
    const std::vector<double> & dq, std::vector<double> & tau)
  {
    for (size_t i = 0; i < tau.size(); ++i) {
      tau[i] = g.kp[i] * (target[i] - q[i]) - g.kd[i] * dq[i];
    }
  }

}  // namespace unitree_isaac_policy
