# Runs every test of a suite against whatever policy is listening on /cmd_vel. Per test:
#   reset (gz_reset_node's service) -> check that the first pose after it is at the origin and
#   upright, and that the base isn't moving over the next 0.1 s -> record: publish the scheduled
#   /cmd_vel, fire pushes, stop early on a fall -> save base, torso and foot poses to
#   <NN>_<name>_r<repeat>.npz. The command schedule and pushes go to meta.json.
# meta.json is rewritten after every test, so an interrupted run still leaves usable results.
#
# Callbacks only store messages; they are turned into arrays once, when a test is saved. All
# timing is simulation time. Nothing here reads the policy's deploy.yaml.
import json
import math
import os
import shutil
import time
import xml.etree.ElementTree as ElementTree
from enum import Enum, auto

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from unitree_gz_eval.gz_push import Pusher
from unitree_gz_eval.suite import load_suite


class Phase(Enum):
    STARTING = auto()
    RESETTING = auto()
    CHECKING = auto()
    RUNNING = auto()


def _stamp_seconds(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _tilt_deg(orientation) -> float:
    cos_tilt = 1.0 - 2.0 * (orientation.x * orientation.x + orientation.y * orientation.y)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


def _yaw_deg(orientation) -> float:
    sin_yaw = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cos_yaw = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
    return math.degrees(math.atan2(sin_yaw, cos_yaw))


class EvalRunner(Node):
    def __init__(self):
        super().__init__('eval_runner')
        suite_path = self.declare_parameter('suite_path', '').value
        self._output_dir = self.declare_parameter('output_dir', '').value
        policy_onnx_path = self.declare_parameter('policy_onnx_path', '').value
        urdf_path = self.declare_parameter('urdf_path', '').value
        self._world_name = self.declare_parameter('world_name', 'default').value
        self._model_name = self.declare_parameter('model_name', 'h2').value
        self._foot_links = self.declare_parameter('foot_links', Parameter.Type.STRING_ARRAY).value
        self._torso_link = self.declare_parameter('torso_link', 'torso_link').value
        pose_topic = self.declare_parameter('pose_topic', '/model/h2/pose').value
        reset_service = self.declare_parameter('reset_service', '/gz_reset_node/reset').value

        self._suite = load_suite(suite_path)
        # Whole suite, then whole suite again: anything that drifts over a session (CPU load)
        # spreads evenly over the tests.
        self._scheduled_runs = [
            (repeat_index, test_index, test)
            for repeat_index in range(self._suite.repeats)
            for test_index, test in enumerate(self._suite.tests)]
        os.makedirs(self._output_dir, exist_ok=True)
        shutil.copy(suite_path, os.path.join(self._output_dir, 'suite.yaml'))

        urdf = ElementTree.parse(urdf_path).getroot()
        self._robot_mass = sum(float(mass.get('value')) for mass in urdf.iter('mass'))
        # The URDF root link (the pelvis) is the model's canonical link, and its frame is the
        # model frame (checked: identity over a whole run), so the model pose Gazebo publishes
        # is the pelvis pose.
        child_links = {joint.find('child').get('link') for joint in urdf.findall('joint')}
        self._base_link = next(link.get('name') for link in urdf.findall('link')
                               if link.get('name') not in child_links)
        world_sdf_path = os.path.join(
            get_package_share_directory('unitree_gz_bringup'), 'worlds', f'{self._world_name}.sdf')
        self._physics_step_seconds = float(
            ElementTree.parse(world_sdf_path).getroot().find('.//physics/max_step_size').text)

        policy_dir = os.path.dirname(os.path.dirname(policy_onnx_path))
        self._meta = {
            'suite': {'name': self._suite.name, 'hash': self._suite.hash},
            'policy': {'name': os.path.basename(policy_dir), 'dir': policy_dir},
            'robot': {'model': self._model_name, 'mass': self._robot_mass,
                      'base_link': self._base_link, 'torso_link': self._torso_link,
                      'foot_links': list(self._foot_links)},
            'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'finished': False,
            'tests': [],
        }

        self._cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self._reset_client = self.create_client(Trigger, reset_service)
        self.create_subscription(TFMessage, pose_topic, self._on_poses, 50)
        # gz_reset_node's startup reset (latched): the stack is usable once it has fired.
        self.create_subscription(
            Empty, '/policy_reset', self._on_policy_reset,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._latest_poses = None
        self._startup_reset_seen = False
        self._recording = False
        self._recorded = None
        self._reset_report = None
        self._pusher = None
        self._phase = Phase.STARTING
        self._run_index = 0
        self.create_timer(1.0 / self._suite.cmd_rate, self._tick)
        self.get_logger().info(
            f'suite "{self._suite.name}" ({self._suite.hash}): {len(self._suite.tests)} tests x '
            f'{self._suite.repeats} repeats -> {self._output_dir}')

    # --- callbacks: store, nothing else -------------------------------------------------------

    def _on_poses(self, msg):
        self._latest_poses = msg
        if self._recording:
            self._recorded['poses'].append(msg)

    def _on_policy_reset(self, _msg):
        self._startup_reset_seen = True

    # --- test sequence ------------------------------------------------------------------------

    def _sim_time(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _base_pose(self):
        """(stamp, position, orientation) of the pelvis, from the latest poses."""
        for transform in self._latest_poses.transforms:
            if transform.child_frame_id == self._model_name:
                return (_stamp_seconds(transform.header), transform.transform.translation,
                        transform.transform.rotation)
        return None

    def _publish_cmd_vel(self, command):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = command
        self._cmd_vel_publisher.publish(msg)

    def _tick(self):
        sim_time = self._sim_time()
        test = self._scheduled_runs[self._run_index][2]

        if self._phase == Phase.STARTING:
            if (self._startup_reset_seen and self._latest_poses
                    and self._reset_client.service_is_ready()):
                self._pusher = Pusher(self._world_name, self._base_link, self._physics_step_seconds)
                self._suite_start_sim_time, self._suite_start_wall_time = sim_time, time.time()
                self._start_reset()

        elif self._phase == Phase.RESETTING:
            self._publish_cmd_vel((0.0, 0.0, 0.0))
            if self._reset_future.done():
                result = self._reset_future.result()
                if not result.success:
                    self._finish(test, 'aborted', reason=f'reset failed: {result.message}')
                else:
                    self._first_pose_after_reset = None
                    self._reset_done_sim_time = sim_time
                    self._phase = Phase.CHECKING

        elif self._phase == Phase.CHECKING:
            # Measures the teleport, not the policy: shortly after the reset, using only poses
            # stamped after it (older ones may still be queued).
            self._publish_cmd_vel((0.0, 0.0, 0.0))
            pose = self._base_pose()
            if pose is None or pose[0] < self._reset_done_sim_time:
                return
            if self._first_pose_after_reset is None:
                self._first_pose_after_reset = pose
            if pose[0] - self._first_pose_after_reset[0] < self._suite.reset_check['after']:
                return
            self._reset_report = self._check_reset(self._first_pose_after_reset, pose)
            if not self._reset_report['ok']:
                self._finish(test, 'aborted', reason='reset check failed')
                return
            self._recorded = {'poses': []}
            self._recording = True
            self._test_start_sim_time, self._test_start_wall_time = sim_time, time.time()
            self._pending_pushes = list(test.pushes)
            self._phase = Phase.RUNNING

        elif self._phase == Phase.RUNNING:
            time_into_test = sim_time - self._test_start_sim_time
            if time_into_test >= test.duration:
                self._finish(test, 'ok')
                return
            self._publish_cmd_vel(test.cmd_at(time_into_test))
            while self._pending_pushes and self._pending_pushes[0].t <= time_into_test:
                push = self._pending_pushes.pop(0)
                # Impulse mass * dv changes the robot's centre-of-mass velocity by dv, as Isaac's
                # push_by_setting_velocity does by adding dv to the root velocity.
                self._pusher.push([self._robot_mass * push.dv[0],
                                   self._robot_mass * push.dv[1], 0.0])
            _, position, orientation = self._base_pose()
            fall_limits = self._suite.fall
            if (position.z < fall_limits['min_pelvis_height']
                    or _tilt_deg(orientation) > fall_limits['max_tilt_deg']):
                self._finish(test, 'fell', fall_time=time_into_test)

    def _start_reset(self):
        self._reset_future = self._reset_client.call_async(Trigger.Request())
        self._phase = Phase.RESETTING

    def _check_reset(self, first_pose, later_pose):
        # Pose is judged on the first sample after the reset - that is the teleport. By `after`
        # the robot has dropped ~1 cm onto its feet and the policy has turned it 0.4-2.4 deg
        # doing so. Speed spans both samples, so leftover momentum still shows.
        first_stamp, first_position, first_orientation = first_pose
        later_stamp, later_position, _ = later_pose
        limits = self._suite.reset_check
        distance_moved = math.dist(
            (later_position.x, later_position.y, later_position.z),
            (first_position.x, first_position.y, first_position.z))
        report = {
            'distance_from_origin': math.hypot(first_position.x, first_position.y),
            'yaw_deg': _yaw_deg(first_orientation),
            'tilt_deg': _tilt_deg(first_orientation),
            'pelvis_height': first_position.z,
            'speed': distance_moved / max(later_stamp - first_stamp, 1e-6),
        }
        report['ok'] = (report['distance_from_origin'] <= limits['max_xy']
                        and abs(report['yaw_deg']) <= limits['max_yaw_deg']
                        and report['tilt_deg'] <= limits['max_tilt_deg']
                        and report['pelvis_height'] >= limits['min_pelvis_height']
                        and report['speed'] <= limits['max_speed'])
        return report

    def _finish(self, test, status, reason=None, fall_time=None):
        self._recording = False
        self._publish_cmd_vel((0.0, 0.0, 0.0))
        repeat_index, test_index, _ = self._scheduled_runs[self._run_index]
        entry = {
            'name': test.name,
            'repeat': repeat_index + 1,
            'status': status,
            'reason': reason,
            'fall_time': fall_time,
            'duration': test.duration,
            'segments': [{'start_time': segment.t0, 'end_time': segment.t1,
                          'command': list(segment.cmd)} for segment in test.segments],
            'pushes': [{'time': push.t, 'velocity_change': list(push.dv)}
                       for push in test.pushes],
            'reset_check': self._reset_report,
        }
        if status != 'aborted':
            entry['file'] = f'{test_index + 1:02d}_{test.name}_r{repeat_index + 1}.npz'
            entry['real_time_factor'] = ((self._sim_time() - self._test_start_sim_time)
                                         / max(time.time() - self._test_start_wall_time, 1e-6))
            self._save(os.path.join(self._output_dir, entry['file']))
        self._reset_report = None
        self._meta['tests'].append(entry)
        self._write_meta()
        self.get_logger().info(
            f'[{self._run_index + 1}/{len(self._scheduled_runs)}] {test.name} '
            f'(repeat {repeat_index + 1}): {status}'
            + (f' ({reason})' if reason else '')
            + (f' at {fall_time:.2f}s' if fall_time is not None else ''))

        self._run_index += 1
        if self._run_index < len(self._scheduled_runs):
            self._start_reset()
        else:
            self._meta['finished'] = True
            self._write_meta()
            raise SystemExit(0)

    # --- messages -> arrays, once per test ----------------------------------------------------

    def _save(self, path):
        # Poses as published: the model's world pose (the model frame is the pelvis frame), and
        # the torso's and feet's poses in the model frame. ros_gz_bridge names links
        # "<model>/<link>". Pose rows are (x, y, z, qx, qy, qz, qw).
        pose_msgs = self._recorded['poses']
        frame_names = [self._model_name] + [
            f'{self._model_name}/{link}' for link in (self._torso_link, *self._foot_links)]
        pose_times = np.zeros(len(pose_msgs))
        poses = np.zeros((len(pose_msgs), len(frame_names), 7))
        for sample_index, msg in enumerate(pose_msgs):
            transform_by_frame = {transform.child_frame_id: transform
                                  for transform in msg.transforms}
            pose_times[sample_index] = _stamp_seconds(transform_by_frame[self._model_name].header)
            for frame_index, frame_name in enumerate(frame_names):
                translation = transform_by_frame[frame_name].transform.translation
                rotation = transform_by_frame[frame_name].transform.rotation
                poses[sample_index, frame_index] = (
                    translation.x, translation.y, translation.z,
                    rotation.x, rotation.y, rotation.z, rotation.w)
        np.savez_compressed(path,
                            pose_times=pose_times - self._test_start_sim_time,
                            base_poses=poses[:, 0],
                            torso_poses_in_base=poses[:, 1],
                            foot_poses_in_base=poses[:, 2:])

    def _write_meta(self):
        sim_seconds = self._sim_time() - self._suite_start_sim_time
        wall_seconds = time.time() - self._suite_start_wall_time
        self._meta.update({'sim_seconds': sim_seconds, 'wall_seconds': wall_seconds,
                           'real_time_factor': sim_seconds / max(wall_seconds, 1e-6)})
        temporary_path = os.path.join(self._output_dir, 'meta.json.tmp')
        with open(temporary_path, 'w') as meta_file:
            json.dump(self._meta, meta_file, indent=2)
        os.replace(temporary_path, os.path.join(self._output_dir, 'meta.json'))


def main():
    rclpy.init()
    node = EvalRunner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
