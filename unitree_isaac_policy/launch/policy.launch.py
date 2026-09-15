# Policy stack only - assumes a simulator is already up (unitree_gz_bringup's sim.launch.py).
# Both nodes derive the joint list the same way the simulator does, from deploy.yaml's
# joint_ids_map against robots/<name>.yaml's joint_sdk_names, so pd_node's effort array lines
# up with the effort_controller's `joints:` order by construction.
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

    common = {
        'deploy_yaml_path': deploy_yaml_path,
        'joint_names': joint_names,
        'use_sim_time': True,
    }

    return [
        Node(
            package='unitree_isaac_policy',
            executable='policy_node',
            parameters=[{**common, 'policy_onnx_path': policy_onnx_path, 'wait_for_reset': True}],
        ),
        Node(
            package='unitree_isaac_policy',
            executable='pd_node',
            parameters=[common],
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
