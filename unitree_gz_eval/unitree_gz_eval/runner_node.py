# Runs every test of a suite against whatever policy is listening on /cmd_vel. Per test:
#   reset (gz_reset_node's service) -> check, 0.1 s later, that the base is back at the origin,
#   upright and not moving -> record: publish the scheduled /cmd_vel, fire pushes, stop early on a
#   fall -> save <NN>_<name>.npz.
# meta.json is rewritten after every test, so an interrupted run still leaves usable results.
#
# Callbacks only store messages; they are turned into arrays once, when a test is saved. All
# timing is simulation time. Nothing here reads the policy's deploy.yaml.
import json
import math
import os
import shutil
import time
import xml.etree.ElementTree as ET
from enum import Enum, auto

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState
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


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _tilt_deg(q) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q.x * q.x + q.y * q.y)))))


def _yaw_deg(q) -> float:
    return math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class EvalRunner(Node):
    def __init__(self):
        super().__init__('eval_runner')
        suite_path = self.declare_parameter('suite_path', '').value
        self._out = self.declare_parameter('output_dir', '').value
        policy_onnx = self.declare_parameter('policy_onnx_path', '').value
        urdf_path = self.declare_parameter('urdf_path', '').value
        self._world = self.declare_parameter('world_name', 'default').value
        self._model = self.declare_parameter('model_name', 'h2').value
        self._foot_links = self.declare_parameter('foot_links', Parameter.Type.STRING_ARRAY).value
        self._torso_link = self.declare_parameter('torso_link', 'torso_link').value
        contact_topics = self.declare_parameter(
            'contact_topics', Parameter.Type.STRING_ARRAY).value
        pose_topic = self.declare_parameter('pose_topic', '/model/h2/pose').value
        reset_service = self.declare_parameter('reset_service', '/gz_reset_node/reset').value

        self._suite = load_suite(suite_path)
        os.makedirs(self._out, exist_ok=True)
        shutil.copy(suite_path, os.path.join(self._out, 'suite.yaml'))

        urdf = ET.parse(urdf_path).getroot()
        self._mass = sum(float(m.get('value')) for m in urdf.iter('mass'))
        # The URDF root link (the pelvis) is the model's canonical link, and its frame is the
        # model frame (checked: identity over a whole run), so the model pose Gazebo publishes
        # is the pelvis pose.
        children = {j.find('child').get('link') for j in urdf.findall('joint')}
        self._base_link = next(l.get('name') for l in urdf.findall('link')
                               if l.get('name') not in children)
        world_sdf = os.path.join(
            get_package_share_directory('unitree_gz_bringup'), 'worlds', f'{self._world}.sdf')
        self._physics_step = float(
            ET.parse(world_sdf).getroot().find('.//physics/max_step_size').text)

        policy_dir = os.path.dirname(os.path.dirname(policy_onnx))
        self._meta = {
            'suite': {'name': self._suite.name, 'hash': self._suite.hash},
            'policy': {'name': os.path.basename(policy_dir), 'dir': policy_dir},
            'robot': {'model': self._model, 'mass': self._mass, 'base_link': self._base_link,
                      'foot_links': list(self._foot_links)},
            'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'finished': False,
            'tests': [],
        }

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._reset_client = self.create_client(Trigger, reset_service)
        self.create_subscription(TFMessage, pose_topic, self._on_poses, 50)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 200)
        for i, topic in enumerate(contact_topics):
            self.create_subscription(
                Contacts, topic, lambda msg, i=i: self._on_contacts(i, msg), 200)
        # gz_reset_node's startup reset (latched): the stack is usable once it has fired.
        self.create_subscription(
            Empty, '/policy_reset', self._on_policy_reset,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._last_pose = None
        self._startup_reset_seen = False
        self._recording = False
        self._buf = None
        self._reset = None
        self._pusher = None
        self._phase = Phase.STARTING
        self._idx = 0
        self.create_timer(1.0 / self._suite.cmd_rate, self._tick)
        self.get_logger().info(
            f'suite "{self._suite.name}" ({self._suite.hash}): {len(self._suite.tests)} tests '
            f'-> {self._out}')

    # --- callbacks: store, nothing else -------------------------------------------------------

    def _on_poses(self, msg):
        self._last_pose = msg
        if self._recording:
            self._buf['poses'].append(msg)

    def _on_joints(self, msg):
        if self._recording:
            self._buf['joints'].append(msg)

    def _on_contacts(self, i, msg):
        if self._recording:
            self._buf['contacts'][i].append(msg)

    def _on_policy_reset(self, _msg):
        self._startup_reset_seen = True

    # --- test sequence ------------------------------------------------------------------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _base_pose(self):
        """(t, position, orientation) of the pelvis, from the latest poses."""
        for tf in self._last_pose.transforms:
            if tf.child_frame_id == self._model:
                return _stamp(tf.header), tf.transform.translation, tf.transform.rotation
        return None

    def _publish_cmd(self, cmd, now):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = cmd
        self._cmd_pub.publish(msg)
        if self._recording:
            self._buf['cmd'].append((now, cmd))

    def _tick(self):
        now = self._now()
        test = self._suite.tests[self._idx]

        if self._phase == Phase.STARTING:
            if (self._startup_reset_seen and self._last_pose
                    and self._reset_client.service_is_ready()):
                self._pusher = Pusher(self._world, self._base_link, self._physics_step)
                self._sim_start, self._wall_start = now, time.time()
                self._start_reset()

        elif self._phase == Phase.RESETTING:
            self._publish_cmd((0.0, 0.0, 0.0), now)
            if self._reset_future.done():
                result = self._reset_future.result()
                if not result.success:
                    self._finish(test, 'aborted', reason=f'reset failed: {result.message}')
                else:
                    self._check_start = None
                    self._check_from = now
                    self._phase = Phase.CHECKING

        elif self._phase == Phase.CHECKING:
            # Measures the teleport, not the policy: shortly after the reset, using only poses
            # stamped after it (older ones may still be queued).
            self._publish_cmd((0.0, 0.0, 0.0), now)
            pose = self._base_pose()
            if pose is None or pose[0] < self._check_from:
                return
            if self._check_start is None:
                self._check_start = (pose[0], pose[1].x, pose[1].y, pose[1].z)
            if pose[0] - self._check_start[0] < self._suite.reset_check['after']:
                return
            self._reset = self._reset_report(pose)
            if not self._reset['ok']:
                self._finish(test, 'aborted', reason='reset check failed')
                return
            self._buf = {'poses': [], 'joints': [], 'cmd': [], 'pushes': [],
                         'contacts': [[] for _ in self._foot_links]}
            self._recording = True
            self._t0, self._wall_t0 = now, time.time()
            self._pushes_left = list(test.pushes)
            self._phase = Phase.RUNNING

        elif self._phase == Phase.RUNNING:
            elapsed = now - self._t0
            if elapsed >= test.duration:
                self._finish(test, 'ok')
                return
            self._publish_cmd(test.cmd_at(elapsed), now)
            while self._pushes_left and self._pushes_left[0].t <= elapsed:
                push = self._pushes_left.pop(0)
                # Impulse mass * dv changes the robot's centre-of-mass velocity by dv, as Isaac's
                # push_by_setting_velocity does by adding dv to the root velocity.
                self._pusher.push([self._mass * push.dv[0], self._mass * push.dv[1], 0.0])
                self._buf['pushes'].append((now, push.dv))
            _, pos, quat = self._base_pose()
            fall = self._suite.fall
            if pos.z < fall['min_pelvis_height'] or _tilt_deg(quat) > fall['max_tilt_deg']:
                self._finish(test, 'fell', fall_t=elapsed)

    def _start_reset(self):
        self._reset_future = self._reset_client.call_async(Trigger.Request())
        self._phase = Phase.RESETTING

    def _reset_report(self, pose):
        t, p, q = pose
        t0, x0, y0, z0 = self._check_start
        lim = self._suite.reset_check
        report = {
            'xy': math.hypot(p.x, p.y),
            'yaw_deg': _yaw_deg(q),
            'tilt_deg': _tilt_deg(q),
            'pelvis_height': p.z,
            'speed': math.dist((p.x, p.y, p.z), (x0, y0, z0)) / max(t - t0, 1e-6),
        }
        report['ok'] = (report['xy'] <= lim['max_xy']
                        and abs(report['yaw_deg']) <= lim['max_yaw_deg']
                        and report['tilt_deg'] <= lim['max_tilt_deg']
                        and report['pelvis_height'] >= lim['min_pelvis_height']
                        and report['speed'] <= lim['max_speed'])
        return report

    def _finish(self, test, status, reason=None, fall_t=None):
        self._recording = False
        self._publish_cmd((0.0, 0.0, 0.0), self._now())
        entry = {
            'name': test.name, 'status': status, 'reason': reason, 'fall_t': fall_t,
            'duration': test.duration,
            'segments': [{'t0': s.t0, 't1': s.t1, 'cmd': list(s.cmd)} for s in test.segments],
            'pushes': [{'t': p.t, 'dv': list(p.dv)} for p in test.pushes],
            'reset_check': self._reset,
        }
        if status != 'aborted':
            entry['file'] = f'{self._idx + 1:02d}_{test.name}.npz'
            entry['rtf'] = (self._now() - self._t0) / max(time.time() - self._wall_t0, 1e-6)
            self._save(os.path.join(self._out, entry['file']))
        self._reset = None
        self._meta['tests'].append(entry)
        self._write_meta()
        self.get_logger().info(
            f'[{self._idx + 1}/{len(self._suite.tests)}] {test.name}: {status}'
            + (f' ({reason})' if reason else '')
            + (f' at {fall_t:.2f}s' if fall_t is not None else ''))

        self._idx += 1
        if self._idx < len(self._suite.tests):
            self._start_reset()
        else:
            self._meta['finished'] = True
            self._write_meta()
            raise SystemExit(0)

    # --- messages -> arrays, once per test ----------------------------------------------------

    def _save(self, path):
        b, t0 = self._buf, self._t0
        arrays = {
            'cmd_t': np.array([c[0] for c in b['cmd']]) - t0,
            'cmd': np.array([c[1] for c in b['cmd']]).reshape(-1, 3),
            'push_t': np.array([p[0] for p in b['pushes']]) - t0,
            'push_dv': np.array([p[1] for p in b['pushes']]).reshape(-1, 2),
            'base_link': np.array(self._base_link),
            'foot_links': np.array(self._foot_links),
            'torso_link': np.array(self._torso_link),
        }
        arrays.update(self._poses_to_arrays(b['poses'], t0))
        arrays.update(self._joints_to_arrays(b['joints'], t0))
        for i, msgs in enumerate(b['contacts']):
            arrays[f'contact{i}_t'] = np.array([_stamp(m.header) for m in msgs]) - t0
            arrays[f'contact{i}_fz'] = np.array(
                [self._foot_fz(m, self._foot_links[i]) for m in msgs], dtype=np.float32)
        np.savez_compressed(path, **arrays)

    def _poses_to_arrays(self, msgs, t0):
        # As published: the model's (= pelvis) world pose, and the torso's and feet's poses in
        # the model frame - the only links the analysis uses. ros_gz_bridge names links
        # "<model>/<link>".
        prefix = f'{self._model}/'
        names = [self._torso_link, *self._foot_links]
        index = {n: k for k, n in enumerate(names)}
        t = np.zeros(len(msgs))
        model = np.zeros((len(msgs), 7))
        links = np.zeros((len(msgs), len(names), 7))
        for row, msg in enumerate(msgs):
            for tf in msg.transforms:
                tr, q = tf.transform.translation, tf.transform.rotation
                pose = (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)
                if tf.child_frame_id == self._model:
                    t[row] = _stamp(tf.header)
                    model[row] = pose
                elif tf.child_frame_id[len(prefix):] in index:
                    links[row, index[tf.child_frame_id[len(prefix):]]] = pose
        return {'pose_t': t - t0, 'link_names': np.array(names),
                'model_pose': model.astype(np.float32),
                'link_pose_in_model': links.astype(np.float32)}

    @staticmethod
    def _joints_to_arrays(msgs, t0, period=0.005):
        # 200 Hz is plenty for power/torque/jerk; /joint_states arrives at up to 1 kHz.
        kept, last = [], -1.0
        for m in msgs:
            if _stamp(m.header) >= last + period - 1e-6:
                kept.append(m)
                last = _stamp(m.header)
        msgs = kept
        return {
            'joint_t': np.array([_stamp(m.header) for m in msgs]) - t0,
            'joint_names': np.array(msgs[0].name),
            'joint_vel': np.array([m.velocity for m in msgs], dtype=np.float32),
            'joint_effort': np.array([m.effort for m in msgs], dtype=np.float32),
        }

    @staticmethod
    def _foot_fz(msg, link):
        # Vertical ground force on the foot, summed over its contact points. Each contact pair
        # reports the wrench on both bodies; take the one on the foot's side.
        fz = 0.0
        for contact in msg.contacts:
            on_body_1 = f'/{link}/' in contact.collision1.name.replace('::', '/')
            for w in contact.wrenches:
                fz += (w.body_1_wrench if on_body_1 else w.body_2_wrench).force.z
        return abs(fz)

    def _write_meta(self):
        sim, wall = self._now() - self._sim_start, time.time() - self._wall_start
        self._meta.update({'sim_seconds': sim, 'wall_seconds': wall, 'rtf': sim / max(wall, 1e-6)})
        tmp = os.path.join(self._out, 'meta.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(self._meta, f, indent=2)
        os.replace(tmp, os.path.join(self._out, 'meta.json'))


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
