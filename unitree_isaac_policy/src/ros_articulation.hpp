#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree_isaac_policy
{

  // The ROS counterpart of unitree_rl_lab's BaseArticulation (deploy/include/
  // unitree_articulation.h), which reads the same fields out of a DDS LowState.
  class RosArticulation : public isaaclab::Articulation
  {
    public:
      explicit RosArticulation(const size_t joint_count)
        : joint_pos_(joint_count, 0.0f), joint_vel_(joint_count, 0.0f)
        {}

      virtual ~RosArticulation() = default;

      void set_joint_pos(const size_t joint_index, const float position)
      {
        joint_pos_[joint_index] = position;
      }

      void set_joint_vel(const size_t joint_index, const float velocity)
      {
        joint_vel_[joint_index] = velocity;
      }

      // Taken together because they come from the same Imu message. Updating one without the
      // other would hand the policy an orientation from one instant and a rate from another.
      void set_imu(
        const Eigen::Quaternionf & orientation,
        const Eigen::Vector3f & angular_velocity)
      {
        orientation_ = orientation;
        angular_velocity_ = angular_velocity;
      }

      void update() override
      {
        for (size_t joint_index = 0; joint_index < joint_pos_.size(); ++joint_index) {
          data.joint_pos[static_cast<long>(joint_index)] = joint_pos_[joint_index];
          data.joint_vel[static_cast<long>(joint_index)] = joint_vel_[joint_index];
        }
        data.root_ang_vel_b = angular_velocity_;
        data.projected_gravity_b = orientation_.conjugate() * data.GRAVITY_VEC_W;
      }

    private:
      std::vector<float> joint_pos_;
      std::vector<float> joint_vel_;
      Eigen::Vector3f angular_velocity_{0.0f, 0.0f, 0.0f};
      Eigen::Quaternionf orientation_{1.0f, 0.0f, 0.0f, 0.0f};
  };

}  // namespace unitree_isaac_policy
