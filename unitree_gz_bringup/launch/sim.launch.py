import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from unitree_gz_description.generate_urdf import (
    IMU_GZ_TOPIC,
    build_controller_config,
    build_urdf,
    derive_deploy_yaml_path,
    env_ordered_joint_names,
    load_deploy,
    load_robot_config,
)


def launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration('robot').perform(context)
    urdf_path = LaunchConfiguration('urdf_path').perform(context)
    policy_onnx_path = LaunchConfiguration('policy_onnx_path').perform(context)

    description_share = get_package_share_directory('unitree_gz_description')
    robot_cfg = load_robot_config(description_share, robot)
    deploy_yaml_path = derive_deploy_yaml_path(policy_onnx_path)
    deploy = load_deploy(deploy_yaml_path)
    joint_names = env_ordered_joint_names(deploy, robot_cfg)

    controller_config = build_controller_config(robot_cfg, deploy)
    controller_config_path = os.path.join(tempfile.gettempdir(), f'{robot}_controllers.yaml')
    with open(controller_config_path, 'w') as f:
        yaml.safe_dump(controller_config, f)

    urdf_xml = build_urdf(robot_cfg, deploy, urdf_path, controller_config_path)

    # gz-sim's stock empty.sdf is deliberately minimal and doesn't load the gz-sim-imu-system
    # plugin, so a <sensor type="imu"> never actually produces data on it (confirmed via
    # `gz topic -i -t /imu` showing "No publishers"). This world is the same 4 plugins plus
    # that one.
    world_path = os.path.join(
        get_package_share_directory('unitree_gz_bringup'), 'worlds', 'default.sdf')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        # Deliberately no "-r": the world starts paused and gz_reset_node steps it forward
        # only as far as the controllers need to activate, then resets and un-pauses. Running
        # from the start instead gives the robot several seconds of falling before anything
        # can command it, and a teleport does not undo the base velocity it picks up doing so.
        launch_arguments={'gz_args': world_path}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': urdf_xml, 'use_sim_time': True}],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robot, '-topic', 'robot_description',
            '-z', str(robot_cfg['spawn_height']),
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )

    effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['effort_controller'],
    )

    # Bridges the gz-sim IMU sensor (attached to imu_link by build_urdf) to ROS2. The bare
    # "/imu" gz topic name (set via the sensor's <topic> tag) is confirmed not scoped/renamed
    # by gz-sim - verified with `gz topic -l` and `gz topic -i -t /imu`.
    #
    # /clock is bridged here too, and is not optional: every use_sim_time node below (the
    # policy bridge in particular) has no simulation clock without it, so its control loop
    # would fall back to wall time and run at a rate unrelated to how fast physics is actually
    # advancing. Its absence is also what produced controller_manager's constant "No clock
    # received" warning and the sec=0/nanosec=0 stamps on /joint_states.
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            f'/{IMU_GZ_TOPIC}@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        remappings=[(f'/{IMU_GZ_TOPIC}', '/imu')],
    )

    # Snaps the robot to deploy.yaml's default pose once the whole stack is live, then
    # notifies the policy bridge - see unitree_gz_bringup/gz_reset_node.py for why the
    # startup window can't simply be waited out.
    gz_reset = Node(
        package='unitree_gz_bringup',
        executable='gz_reset_node',
        parameters=[{
            'world_name': 'default',
            'model_name': robot,
            'joint_names': joint_names,
            'default_joint_pos': [float(v) for v in deploy['default_joint_pos']],
            'spawn_height': float(robot_cfg['spawn_height']),
        }],
    )

    policy_bridge = Node(
        package='unitree_policy_bridge',
        executable='policy_bridge_node',
        parameters=[{
            'deploy_yaml_path': deploy_yaml_path,
            'policy_onnx_path': policy_onnx_path,
            'joint_names': joint_names,
            'use_sim_time': True,
            'wait_for_reset': True,
        }],
    )

    return [
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster_spawner,
        effort_controller_spawner,
        imu_bridge,
        gz_reset,
        policy_bridge,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            default_value='h2',
            description='Robot name - matches unitree_gz_description/robots/<name>.yaml',
        ),
        DeclareLaunchArgument(
            'urdf_path',
            description=(
                "Path to the robot's bare URDF "
                '(e.g. unitree_ros/robots/h2_description/H2.urdf)'
            ),
        ),
        DeclareLaunchArgument(
            'policy_onnx_path',
            description=(
                'Path to the exported policy.onnx. deploy.yaml is derived as the sibling '
                "'params/' directory next to it (policy_dir/exported/policy.onnx and "
                'policy_dir/params/deploy.yaml).'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
