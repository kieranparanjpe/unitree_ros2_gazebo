# The original single-node bridge (observation building, inference and PD in one process),
# split out of sim.launch.py so it can be started against an already-running simulator.
# Mutually exclusive with unitree_isaac_policy's policy.launch.py - both drive
# /effort_controller/commands.
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from unitree_gz_description.generate_urdf import (
    derive_deploy_yaml_path,
    env_ordered_joint_names,
    load_deploy,
    load_robot_config,
)


def launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration('robot').perform(context)
    policy_onnx_path = LaunchConfiguration('policy_onnx_path').perform(context)

    robot_cfg = load_robot_config(get_package_share_directory('unitree_gz_description'), robot)
    deploy_yaml_path = derive_deploy_yaml_path(policy_onnx_path)
    joint_names = env_ordered_joint_names(load_deploy(deploy_yaml_path), robot_cfg)

    return [
        Node(
            package='unitree_policy_bridge',
            executable='policy_bridge_node',
            parameters=[{
                'deploy_yaml_path': deploy_yaml_path,
                'policy_onnx_path': policy_onnx_path,
                'joint_names': joint_names,
                'use_sim_time': True,
                'wait_for_reset': True,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            default_value='h2',
            description='Robot name - matches unitree_gz_description/robots/<name>.yaml',
        ),
        DeclareLaunchArgument(
            'policy_onnx_path',
            description=(
                'Path to the exported policy.onnx. deploy.yaml is derived as the sibling '
                "'params/' directory next to it."
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
