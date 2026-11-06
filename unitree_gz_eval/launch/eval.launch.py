# One command per evaluation: the normal unitree_isaac_policy stack (with the simulator's
# ground-truth instrumentation on), a bridge for that ground truth, and the runner. The whole
# launch shuts down when the runner finishes.
#
#   ros2 launch unitree_gz_eval eval.launch.py policy_onnx_path:=... urdf_path:=...
import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from unitree_gz_description.generate_urdf import (
    POSE_GZ_TOPIC_FMT,
    contact_gz_topic,
    load_robot_config,
)


def launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration('robot').perform(context)
    policy_onnx_path = os.path.abspath(LaunchConfiguration('policy_onnx_path').perform(context))
    urdf_path = os.path.abspath(LaunchConfiguration('urdf_path').perform(context))
    suite = LaunchConfiguration('suite').perform(context)
    output_dir = LaunchConfiguration('output_dir').perform(context)
    headless = LaunchConfiguration('headless').perform(context)

    share = get_package_share_directory('unitree_gz_eval')
    suite_path = suite if os.path.isabs(suite) else os.path.join(share, 'suites', f'{suite}.yaml')
    suite_name = os.path.basename(suite_path).removesuffix('.yaml')
    if not output_dir:
        policy_dir = os.path.dirname(os.path.dirname(policy_onnx_path))
        output_dir = os.path.join(
            policy_dir, 'eval', f'{suite_name}_{time.strftime("%Y%m%d-%H%M%S")}')

    robot_cfg = load_robot_config(get_package_share_directory('unitree_gz_description'), robot)
    foot_links = list(robot_cfg['foot_links'])
    pose_topic = POSE_GZ_TOPIC_FMT.format(model=robot)
    contact_topics = [contact_gz_topic(link) for link in foot_links]

    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('unitree_isaac_policy'), 'launch', 'all.launch.py')),
        launch_arguments={
            'robot': robot,
            'policy_onnx_path': policy_onnx_path,
            'urdf_path': urdf_path,
            'instrumentation': 'true',
            'headless': headless,
        }.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='eval_ground_truth_bridge',
        arguments=[f'{pose_topic}@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V']
        + [f'{t}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts' for t in contact_topics],
    )

    runner = Node(
        package='unitree_gz_eval',
        executable='eval_runner',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'suite_path': suite_path,
            'output_dir': output_dir,
            'policy_onnx_path': policy_onnx_path,
            'urdf_path': urdf_path,
            'model_name': robot,
            'foot_links': foot_links,
            'torso_link': robot_cfg['torso_link'],
            'contact_topics': contact_topics,
            'pose_topic': pose_topic,
        }],
    )

    shutdown_when_done = RegisterEventHandler(OnProcessExit(
        target_action=runner,
        on_exit=[EmitEvent(event=Shutdown(reason='evaluation finished'))],
    ))

    return [stack, bridge, runner, shutdown_when_done]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value='h2'),
        DeclareLaunchArgument('policy_onnx_path', description='<policy_dir>/exported/policy.onnx'),
        DeclareLaunchArgument('urdf_path', description="Robot's bare URDF, meshes/ alongside"),
        DeclareLaunchArgument(
            'suite', default_value='default',
            description='Suite name under share/unitree_gz_eval/suites, or an absolute path'),
        DeclareLaunchArgument(
            'output_dir', default_value='',
            description='Default: <policy_dir>/eval/<suite>_<timestamp>'),
        DeclareLaunchArgument('headless', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
