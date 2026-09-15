// Entry point for the policy_node executable. PolicyNode itself lives in policy_node.hpp.
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "policy_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<unitree_isaac_policy::PolicyNode>());
  rclcpp::shutdown();
  return 0;
}
