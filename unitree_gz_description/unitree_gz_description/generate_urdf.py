# Generic URDF -> Gazebo/ros2_control wiring, driven by explicit paths the user supplies at
# launch time (urdf_path, policy_onnx_path) plus a small per-robot metadata file for the
# handful of things that aren't paths and aren't derivable from either of those
# (joint_sdk_names, imu_link).
#
# Given those, this builds:
#   - the URDF augmented with a <ros2_control> block (one joint per deploy.yaml entry, in
#     deploy.yaml's own joint order) plus the gz_ros2_control Gazebo plugin block and an IMU
#     sensor
#   - the controller_manager YAML (joint_state_broadcaster + a raw effort passthrough
#     controller, listing joints in that same order)
#
# Adding a new robot means adding one robots/<name>.yaml file (joint_sdk_names + imu_link
# only) - no new code, and no assumption about where the user's URDF or trained policy live.
import os
import xml.etree.ElementTree as ET

import yaml

GZ_ROS2_CONTROL_PLUGIN_FILENAME = 'gz_ros2_control-system'
GZ_ROS2_CONTROL_PLUGIN_NAME = 'gz_ros2_control::GazeboSimROS2ControlPlugin'

# controller_manager's own update rate (physics/control loop), independent of the policy's
# step_dt decimation - the policy node runs its own slower timer on top of this.
CONTROLLER_MANAGER_UPDATE_RATE_HZ = 1000

# No <noise> block is configured below, so this reports the link's exact orientation/angular
# velocity - i.e. ground truth, matching what Isaac Lab's base_ang_vel/projected_gravity
# terms are computed from during training (see unitree_rl_lab/deploy/include/
# unitree_articulation.h - real hardware substitutes an actual IMU for that same ground
# truth at deploy time). Add a <noise> block per-axis under the <imu> element below if you
# want to rehearse real-sensor noise instead.
IMU_UPDATE_RATE_HZ = 500
IMU_GZ_TOPIC = 'imu'


def load_robot_config(share_dir: str, robot_name: str) -> dict:
    # Only non-path, non-derivable per-robot metadata lives here - see robots/h2.yaml.
    config_path = os.path.join(share_dir, 'robots', f'{robot_name}.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def derive_deploy_yaml_path(policy_onnx_path: str) -> str:
    # Isaac Lab's export convention (unitree_rl_lab's export_deploy_cfg.py, and every
    # deploy/robots/*/config/policy/.../ example in this repo): policy_dir/exported/
    # policy.onnx sits next to policy_dir/params/deploy.yaml.
    policy_dir = os.path.dirname(os.path.dirname(policy_onnx_path))
    deploy_yaml_path = os.path.join(policy_dir, 'params', 'deploy.yaml')
    if not os.path.exists(deploy_yaml_path):
        raise FileNotFoundError(
            f"Expected deploy.yaml at '{deploy_yaml_path}' (derived as the sibling 'params/' "
            f"directory next to policy_onnx_path='{policy_onnx_path}'). Isaac Lab's export "
            "convention is <policy_dir>/exported/policy.onnx next to "
            "<policy_dir>/params/deploy.yaml - point policy_onnx_path at the right file, or "
            "place deploy.yaml there."
        )
    return deploy_yaml_path


def load_deploy(deploy_yaml_path: str) -> dict:
    with open(deploy_yaml_path) as f:
        deploy = yaml.safe_load(f)
    return deploy


def env_ordered_joint_names(deploy: dict, robot_cfg: dict) -> list:
    # deploy.yaml's arrays (stiffness/damping/default_joint_pos/actions/...) are indexed in
    # this same order - joint_ids_map[i] is that joint's index into joint_sdk_names.
    sdk_names = robot_cfg['joint_sdk_names']
    return [sdk_names[sdk_idx] for sdk_idx in deploy['joint_ids_map']]


def _rename_colliding_joints(root: ET.Element, actuated_joint_names: list, urdf_path: str) -> None:
    # URDF allows a link and a joint to share a name (different element types, separate
    # namespaces); SDF (what Gazebo converts to) doesn't - it's one shared frame namespace,
    # so this collides with a "frame with name[...] already exists" error on spawn. Renaming
    # is only safe for the JOINT side: nothing in URDF/SDF references a joint by name from
    # outside itself, unlike links, which get referenced by other joints' parent/child and by
    # <gazebo reference="..."> blocks (including our own IMU one). If the collision instead
    # involves one of the policy's actuated joints, renaming it would break the
    # correspondence with deploy.yaml/the controller config, so fail loudly instead of
    # guessing.
    link_names = {link.get('name') for link in root.findall('link')}
    for joint in root.findall('joint'):
        name = joint.get('name')
        if name not in link_names:
            continue
        if name in actuated_joint_names:
            raise ValueError(
                f"URDF '{urdf_path}' has a link and joint both named '{name}', and that "
                "joint is one of the policy's actuated joints - renaming it would break the "
                "correspondence with deploy.yaml/the controller config. Fix the name "
                "collision in the source URDF directly."
            )
        joint.set('name', f'{name}_joint')


def build_urdf(robot_cfg: dict, deploy: dict, urdf_path: str, controller_config_path: str) -> str:
    joint_names = env_ordered_joint_names(deploy, robot_cfg)
    # The bare URDF's mesh filenames are relative ("meshes/xxx.stl"), resolved against
    # whatever directory the URDF itself is in - same convention unitree_ros' h1_description
    # catkin package uses (its CMakeLists.txt installs meshes/urdf/launch as siblings).
    mesh_dir = os.path.dirname(urdf_path)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    missing_meshes = []
    for mesh in root.iter('mesh'):
        rel = mesh.get('filename')
        if rel and '://' not in rel:
            resolved = os.path.join(mesh_dir, rel)
            if not os.path.exists(resolved):
                missing_meshes.append(resolved)
            mesh.set('filename', f'file://{resolved}')
    if missing_meshes:
        raise FileNotFoundError(
            f"URDF '{urdf_path}' references mesh files that don't exist next to it (meshes "
            f"resolved against the URDF's own directory, '{mesh_dir}'):\n  "
            + '\n  '.join(missing_meshes)
            + "\nMake sure the full description folder (URDF + meshes/) is intact, not just "
            "the URDF file on its own."
        )

    _rename_colliding_joints(root, joint_names, urdf_path)

    # Spawn already in deploy.yaml's default_joint_pos (the stance Isaac Lab resets episodes
    # into) rather than the URDF's all-zero configuration. gz_ros2_control turns this param
    # into a gz-sim JointPositionReset component at load, which the physics system applies as
    # a direct joint write - so /joint_states reports these angles in the URDF's own joint
    # frame, which is what deploy.yaml's default_joint_pos/joint_pos_rel/action offset are all
    # expressed in. It must be a nested <param>, not an attribute on <state_interface>;
    # ros2_control silently ignores unknown attributes.
    default_by_name = dict(zip(joint_names, deploy['default_joint_pos']))

    ros2_control = ET.SubElement(root, 'ros2_control', {'name': 'GazeboSystem', 'type': 'system'})
    hardware = ET.SubElement(ros2_control, 'hardware')
    ET.SubElement(hardware, 'plugin').text = 'gz_ros2_control/GazeboSimSystem'
    for name in joint_names:
        joint = ET.SubElement(ros2_control, 'joint', {'name': name})
        ET.SubElement(joint, 'command_interface', {'name': 'effort'})
        position = ET.SubElement(joint, 'state_interface', {'name': 'position'})
        ET.SubElement(position, 'param', {'name': 'initial_value'}).text = str(
            default_by_name[name])
        ET.SubElement(joint, 'state_interface', {'name': 'velocity'})
        ET.SubElement(joint, 'state_interface', {'name': 'effort'})

    gazebo = ET.SubElement(root, 'gazebo')
    plugin = ET.SubElement(gazebo, 'plugin', {
        'filename': GZ_ROS2_CONTROL_PLUGIN_FILENAME,
        'name': GZ_ROS2_CONTROL_PLUGIN_NAME,
    })
    ET.SubElement(plugin, 'parameters').text = controller_config_path

    imu_gazebo = ET.SubElement(root, 'gazebo', {'reference': robot_cfg['imu_link']})
    sensor = ET.SubElement(imu_gazebo, 'sensor', {'name': 'imu_sensor', 'type': 'imu'})
    ET.SubElement(sensor, 'always_on').text = 'true'
    ET.SubElement(sensor, 'update_rate').text = str(IMU_UPDATE_RATE_HZ)
    ET.SubElement(sensor, 'topic').text = IMU_GZ_TOPIC
    ET.SubElement(sensor, 'imu')

    return ET.tostring(root, encoding='unicode')


def build_controller_config(robot_cfg: dict, deploy: dict) -> dict:
    joint_names = env_ordered_joint_names(deploy, robot_cfg)
    return {
        'controller_manager': {
            'ros__parameters': {
                'update_rate': CONTROLLER_MANAGER_UPDATE_RATE_HZ,
                'joint_state_broadcaster': {
                    'type': 'joint_state_broadcaster/JointStateBroadcaster',
                },
                'effort_controller': {
                    'type': 'forward_command_controller/ForwardCommandController',
                },
            },
        },
        'effort_controller': {
            'ros__parameters': {
                'joints': joint_names,
                'interface_name': 'effort',
            },
        },
    }
