# Composes a policy launch file with the simulator launch in the order that actually works.
#
# The policy stack goes up first and the simulator follows after a short delay, which covers
# the ONNX session load - the slowest part of a policy node's startup. /policy_reset is latched,
# so a policy that is still loading when the reset fires will still pick it up.
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from unitree_gz_description.generate_urdf import derive_deploy_yaml_path


def _include(package, launch_file, arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)
        ),
        launch_arguments=arguments.items(),
    )


def _setup(context, policy_package, policy_launch_file):
    robot = LaunchConfiguration('robot').perform(context)
    policy_onnx_path = LaunchConfiguration('policy_onnx_path').perform(context)
    urdf_path = LaunchConfiguration('urdf_path').perform(context)
    sim_delay = LaunchConfiguration('sim_delay').perform(context)

    return [
        _include(policy_package, policy_launch_file, {
            'robot': robot,
            'policy_onnx_path': policy_onnx_path,
        }),
        TimerAction(period=float(sim_delay), actions=[
            _include('unitree_gz_bringup', 'sim.launch.py', {
                'robot': robot,
                'urdf_path': urdf_path,
                'deploy_yaml_path': derive_deploy_yaml_path(policy_onnx_path),
            }),
        ]),
    ]


def combined_launch_description(policy_package, policy_launch_file):
    """Full stack in one command: the given policy launch file, then the simulator."""
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
        DeclareLaunchArgument(
            'urdf_path',
            description="Path to the robot's bare URDF, with meshes/ alongside it",
        ),
        DeclareLaunchArgument(
            'sim_delay',
            default_value='5.0',
            description='Seconds to let the policy stack come up before starting the simulator',
        ),
        OpaqueFunction(function=_setup, kwargs={
            'policy_package': policy_package,
            'policy_launch_file': policy_launch_file,
        }),
    ])
