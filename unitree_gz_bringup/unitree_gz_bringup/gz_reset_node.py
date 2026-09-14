# Sim-only reset primitive: writes the robot's joints (and base pose) straight into gz-sim's
# Entity Component Manager, bypassing dynamics.
#
# Why this exists: ROS2 nodes come up several seconds before Gazebo has a robot, so there is
# no way to have the controllers holding the robot from the instant physics starts - it sags
# before anything can command it. The launch therefore starts the world paused and this node
# drives startup: step just far enough for the controllers to activate, snap the robot to
# deploy.yaml's default_joint_pos, tell unitree_policy_bridge to re-seed, then un-pause. The
# same entry point doubles as an episode reset (the ~/reset service), which is what Isaac Lab
# does between episodes too.
#
# Mechanism: gz-sim state is entities + components, and a teleport is just a component the
# Physics system consumes on its next update - JointPositionReset/JointVelocityReset for the
# joints (gz/sim/components/JointPositionReset.hh), WorldLinearVelocityCmd/
# WorldAngularVelocityCmd for the base, plus the /world/<world>/set_pose service for the base
# pose. /world/<world>/control/state accepts an arbitrary serialized component write, so no
# custom Gazebo system plugin is needed. This is Harmonic's equivalent of Gazebo Classic's
# /gazebo/set_model_configuration, which was never ported as a named service.
#
# Resetting the velocities is not optional: a teleport rewrites position, not momentum, so a
# robot that has been falling gets snapped upright and immediately continues downwards at the
# speed it was already doing - which looks exactly like the reset never happened. The world is
# also started paused so the robot has barely moved by the time any of this runs.
#
# This is deliberately NOT part of unitree_policy_bridge: that node is meant to stay
# transport-agnostic so the same observation/action path can drive real hardware over DDS,
# where no such teleport exists (there, getting to the default pose is the controller FSM's
# stand-up state, upstream of that node). Keeping the Gazebo-specific write here preserves
# that boundary.
import gz.transport13 as gz_transport
import rclpy
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty as GzEmpty
from gz.msgs10.pose_pb2 import Pose as GzPose
from gz.msgs10.serialized_map_pb2 import SerializedStepMap
from gz.msgs10.double_v_pb2 import Double_V
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.world_control_state_pb2 import WorldControlState
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from std_srvs.srv import Trigger


def _hash64(name: str) -> int:
    # gz-sim derives a component's numeric type id as gz::common::hash64 of its registered
    # type name (components/Factory.hh: `auto typeHash = gz::common::hash64(_type);
    # ComponentTypeT::typeId = typeHash;`). That's FNV-1a/64. C++ callers never do this - they
    # reference components::JointPositionReset::typeId directly - so this reimplementation is
    # the one part of this file that leans on an internal detail rather than a public API.
    # _verify_type_ids() below checks it against components we can independently identify,
    # because the failure mode is otherwise silent: an unrecognised type id writes a component
    # nothing reads, and the service still reports success.
    h = 0xCBF29CE484222325
    for byte in name.encode():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


TYPE_NAME = _hash64('gz_sim_components.Name')
TYPE_POSE = _hash64('gz_sim_components.Pose')
TYPE_PARENT_ENTITY = _hash64('gz_sim_components.ParentEntity')
TYPE_JOINT = _hash64('gz_sim_components.Joint')
TYPE_JOINT_POSITION_RESET = _hash64('gz_sim_components.JointPositionReset')
TYPE_JOINT_VELOCITY_RESET = _hash64('gz_sim_components.JointVelocityReset')
TYPE_MODEL = _hash64('gz_sim_components.Model')
TYPE_WORLD_LINEAR_VELOCITY_CMD = _hash64('gz_sim_components.WorldLinearVelocityCmd')
TYPE_WORLD_ANGULAR_VELOCITY_CMD = _hash64('gz_sim_components.WorldAngularVelocityCmd')

# See GzResetNode._nudge(). Measured: both controllers activate within ~40-60 steps, by which
# point an uncommanded H2 has drifted about 0.045 rad. The budget is generous enough to absorb
# a slow start but small enough that hitting it means something is wrong, not just slow.
NUDGE_STEP_BURST = 10
MAX_NUDGE_SIM_SECONDS = 0.6
MAX_RESUME_ATTEMPTS = 20
# Consecutive 0.25s checks showing the world stepping before it counts as running.
RESUME_CONFIRMATIONS = 6
# 0.1s ticks to wait for the IMU bridge process to appear before resetting without it.
MAX_IMU_WAIT_TICKS = 100


class GzResetNode(Node):
    def __init__(self):
        super().__init__('gz_reset_node')
        self._world = self.declare_parameter('world_name', 'default').value
        self._model = self.declare_parameter('model_name', 'h2').value
        # Declared by type, not by an empty-list default: rclpy infers BYTE_ARRAY from [] and
        # then rejects the real string/double arrays the launch passes.
        self._joint_names = self.declare_parameter(
            'joint_names', Parameter.Type.STRING_ARRAY).value
        self._default_joint_pos = self.declare_parameter(
            'default_joint_pos', Parameter.Type.DOUBLE_ARRAY).value
        self._spawn_height = self.declare_parameter('spawn_height', 1.0).value
        reset_topic = self.declare_parameter('reset_topic', '/policy_reset').value
        self._imu_topic = self.declare_parameter('imu_topic', '/imu').value

        if len(self._joint_names) != len(self._default_joint_pos):
            raise RuntimeError(
                'joint_names and default_joint_pos must be the same length '
                f'({len(self._joint_names)} vs {len(self._default_joint_pos)})')

        self._gz = gz_transport.Node()
        self._joint_state_count = 0
        self._imu_wait_ticks = 0
        self._nudge_exhausted = False
        # Simulation time, from /clock - our only feedback about steps actually executed
        # rather than merely requested. See _nudge().
        self._sim_time = None
        self._last_nudge_sim_time = None
        self._last_joint_state = None
        self._error_before_reset = None

        # Latched: unitree_policy_bridge must not miss the notification if it finishes loading
        # its ONNX session after the reset has already happened.
        self._reset_pub = self.create_publisher(
            Empty, reset_topic,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.create_service(Trigger, '~/reset', self._on_reset_request)
        # Stepping has to be closed-loop against what the simulator has actually executed.
        # Requests queue on the server while it is paused, so counting the steps *asked for*
        # over-counts badly, and each queued multi_step re-pauses the world as it completes.
        # Simulation time is that feedback. It comes from ROS rather than gz's own /stats topic
        # because a gz-transport subscription on this node's transport makes its blocking
        # service requests start timing out.
        self.create_subscription(Clock, '/clock', self._on_clock, 10)

        self._startup_timer = self.create_timer(0.1, self._try_startup_reset)
        self.get_logger().info(
            f'gz_reset_node waiting for controllers and {self._imu_topic} before resetting '
            f'{len(self._joint_names)} joints of model "{self._model}"')

    def _on_clock(self, msg):
        self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_joint_state(self, msg):
        self._joint_state_count += 1
        self._last_joint_state = msg

    def _pose_error(self):
        # Largest |measured - default| over the policy's joints, or None if nothing measured
        # yet. Reported either side of the reset: a teleport that reports success but changes
        # nothing looks identical in the logs otherwise, and "no visible snap" is ambiguous
        # between "the robot had not drifted" and "the write did nothing".
        msg = self._last_joint_state
        if msg is None:
            return None
        measured = dict(zip(msg.name, msg.position))
        errors = [
            abs(measured[name] - target)
            for name, target in zip(self._joint_names, self._default_joint_pos)
            if name in measured
        ]
        return max(errors) if errors else None

    @staticmethod
    def _fmt(error):
        return 'n/a' if error is None else f'{error:.4f} rad'

    def _try_startup_reset(self):
        # Two preconditions, waited on differently, because only one of them costs simulated
        # time. Controller activation is serviced by the controller_manager update loop, so it
        # genuinely needs steps - /joint_states publishing is the signal it is done. The IMU
        # bridge needs only wall time to start, so waiting for it is done with the world still
        # paused, by checking for a publisher rather than for a message. Stepping while waiting
        # for that bridge is what previously ran the nudge out to 400 steps: the step rate was
        # tied to wall time, so it scaled with process startup latency instead of with what
        # the simulation actually needed. The robot sags visibly over 400 steps.
        if self._joint_state_count == 0:
            self._nudge()
            return
        if self.count_publishers(self._imu_topic) == 0:
            self._imu_wait_ticks += 1
            if self._imu_wait_ticks == MAX_IMU_WAIT_TICKS:
                self.get_logger().error(
                    f'No publisher on {self._imu_topic} after '
                    f'{MAX_IMU_WAIT_TICKS * 0.1:.0f}s - resetting anyway; the policy bridge '
                    'will wait for IMU data on its own.')
            elif self._imu_wait_ticks > MAX_IMU_WAIT_TICKS:
                pass
            else:
                return
        self._startup_timer.cancel()
        self._error_before_reset = self._pose_error()
        ok, detail = self._reset()
        if ok:
            self.get_logger().info(
                f'Startup reset complete ({detail}) after {self._sim_time:.3f}s of simulation; '
                f'pose error before reset {self._fmt(self._error_before_reset)}; '
                'notifying policy bridge')
            self._reset_pub.publish(Empty())
            self._start_resume()
        else:
            self.get_logger().error(f'Startup reset failed: {detail}')

    def _start_resume(self):
        # Un-pausing is retried rather than fired once, because a single request is not
        # reliable here: the server drains queued world-control requests in one pass, and
        # multi_step's "pause once the steps are done" is applied after whatever else is in
        # that batch. An un-pause that arrives alongside a still-pending step burst is
        # therefore silently undone - and a world left paused looks entirely successful from
        # here (every service call returns true) while simulation time stops dead, taking
        # every use_sim_time control loop with it.
        #
        # The check is the simulator's own iteration count advancing while not paused -
        # i.e. steps actually being executed, not merely a service call returning true.
        self._resume_attempts = 0
        self._resume_confirmations = 0
        self._resume_mark = self._sim_time
        self._resume_timer = self.create_timer(0.25, self._try_resume)

    def _try_resume(self):
        # Keeps watching after the first success rather than stopping there. The step bursts
        # issued while waiting for the controllers queue up on the server, and each one
        # re-pauses the world as it completes - so the world can start stepping, then stop
        # again a moment later as the backlog drains. Requiring several consecutive advancing
        # checks rides that out; a single one does not.
        advanced = self._sim_time != self._resume_mark
        self._resume_mark = self._sim_time
        if advanced:
            self._resume_confirmations += 1
            if self._resume_confirmations >= RESUME_CONFIRMATIONS:
                self._resume_timer.cancel()
                self.get_logger().info(
                    f'Simulation resumed after reset. Pose error before={self._fmt(self._error_before_reset)} '
                    f'after={self._fmt(self._pose_error())} - if these are both small the '
                    'teleport had nothing visible to correct; if "after" is large the write '
                    'did not take effect.')
            return

        self._resume_confirmations = 0
        if self._resume_attempts >= MAX_RESUME_ATTEMPTS:
            self._resume_timer.cancel()
            self.get_logger().error(
                f'World "{self._world}" is still not stepping after '
                f'{MAX_RESUME_ATTEMPTS} un-pause attempts - simulation time is not '
                'advancing, so every use_sim_time control loop is stalled.')
            return
        self._resume_attempts += 1
        try:
            self._world_control(pause=False)
        except RuntimeError as exc:
            self.get_logger().warn(f'un-pause request failed: {exc}')

    def _nudge(self):
        # The world is launched paused, but controller activation is itself serviced by the
        # controller_manager update loop, which only runs on simulation steps - a world left
        # fully paused never finishes activating and the spawners give up. So step it forward
        # in small bursts. Measured cost: both controllers activate within ~40-60 steps
        # (0.04-0.06s), by which point an uncommanded H2 has drifted 0.045 rad - less than the
        # policy bridge's own hold tolerance. Bursts rather than one large step so we overrun
        # that by as little as possible.
        if self._sim_time is None:
            return  # no /clock yet, so the simulator isn't up to be stepped
        if self._sim_time >= MAX_NUDGE_SIM_SECONDS:
            if not self._nudge_exhausted:
                self._nudge_exhausted = True
                self.get_logger().error(
                    f'Stepped {self._sim_time:.3f}s of simulation without /joint_states '
                    'publishing - the controllers never activated, so no reset was performed '
                    'and the world is still paused.')
            return
        # Only ask for more once the previous burst has actually executed. Without this the
        # requests pile up on a wall clock while the simulator drains them at its own pace, so
        # the robot ends up simulating far more than intended (measured: 400+ steps of sag when
        # ~50 were wanted) and the leftovers keep re-pausing the world afterwards.
        if self._sim_time == self._last_nudge_sim_time:
            return
        self._last_nudge_sim_time = self._sim_time
        try:
            self._world_control(pause=True, multi_step=NUDGE_STEP_BURST)
        except RuntimeError as exc:
            self.get_logger().warn(f'step request failed: {exc}')

    def _on_reset_request(self, _request, response):
        response.success, response.message = self._reset()
        if response.success:
            self._reset_pub.publish(Empty())
            self._start_resume()
        return response

    def _read_state(self):
        ok, rep = self._gz.request(
            f'/world/{self._world}/state', GzEmpty(), GzEmpty, SerializedStepMap, 15000)
        if not ok:
            raise RuntimeError(f'/world/{self._world}/state request failed')
        return rep.state

    def _verify_type_ids(self, state):
        # Name/Pose/ParentEntity are on essentially every entity, so if our hash matches the
        # ids actually present, the derivation is still correct for this gz-sim build - and
        # therefore so is JointPositionReset's, which is produced the same way.
        present = {c.type for ent in state.entities.values() for c in ent.components.values()}
        missing = [
            n for n, t in (('Name', TYPE_NAME), ('Pose', TYPE_POSE),
                           ('ParentEntity', TYPE_PARENT_ENTITY))
            if t not in present
        ]
        if missing:
            raise RuntimeError(
                'gz-sim component type ids no longer match hash64(type name) for '
                f'{", ".join(missing)} - this build derives them differently, so a '
                'JointPositionReset write would be silently ignored. See _hash64().')

    def _joint_entities(self, state):
        entities = {}
        wanted = set(self._joint_names)
        for entity_id, ent in state.entities.items():
            comps = {c.type for c in ent.components.values()}
            if TYPE_JOINT not in comps:
                continue
            for comp in ent.components.values():
                if comp.type == TYPE_NAME:
                    name = comp.component.decode(errors='replace')
                    if name in wanted:
                        entities[name] = entity_id
        missing = [j for j in self._joint_names if j not in entities]
        if missing:
            raise RuntimeError(
                f'{len(missing)} of {len(self._joint_names)} joints not found as joint '
                f'entities in the ECM (first few: {missing[:5]})')
        return entities

    def _model_entity(self, state):
        for entity_id, ent in state.entities.items():
            comps = {c.type for c in ent.components.values()}
            if TYPE_MODEL not in comps:
                continue
            for comp in ent.components.values():
                if comp.type == TYPE_NAME and comp.component.decode(errors='replace') == self._model:
                    return entity_id
        raise RuntimeError(f'no model entity named "{self._model}" in the ECM')

    def _world_control(self, **fields):
        request = WorldControl(**fields)
        ok, rep = self._gz.request(
            f'/world/{self._world}/control', request, WorldControl, Boolean, 5000)
        if not ok or not rep.data:
            raise RuntimeError(f'/world/{self._world}/control rejected {fields}')

    def _reset(self):
        try:
            state = self._read_state()
            self._verify_type_ids(state)
            entities = self._joint_entities(state)
            model_entity = self._model_entity(state)

            # Pause first: the writes below land in the ECM and are applied by the Physics
            # system on its next update, so stepping in between would let the robot fall
            # partway through a reset that is meant to be atomic.
            self._world_control(pause=True)

            request = WorldControlState()
            for name, target in zip(self._joint_names, self._default_joint_pos):
                entity = request.state.entities.add()
                entity.id = entities[name]
                position = entity.components.add()
                position.type = TYPE_JOINT_POSITION_RESET
                position.component = Double_V(data=[float(target)]).SerializeToString()
                # Position without velocity would leave whatever momentum the robot had
                # before the teleport, which the policy would then see as a real motion.
                velocity = entity.components.add()
                velocity.type = TYPE_JOINT_VELOCITY_RESET
                velocity.component = Double_V(data=[0.0]).SerializeToString()
            ok, rep = self._gz.request(
                f'/world/{self._world}/control/state', request,
                WorldControlState, Boolean, 10000)
            if not ok or not rep.data:
                raise RuntimeError(f'/world/{self._world}/control/state write rejected')

            # The joint write alone would leave a collapsed robot lying on the floor with its
            # limbs snapped to the standing pose - and teleporting joints while links are
            # interpenetrating the ground spikes the contact solver. Lifting the base back to
            # its spawn pose puts it in free space first.
            base = GzPose()
            base.name = self._model
            base.position.z = float(self._spawn_height)
            base.orientation.w = 1.0
            ok, rep = self._gz.request(
                f'/world/{self._world}/set_pose', base, GzPose, Boolean, 5000)
            if not ok or not rep.data:
                raise RuntimeError(f'/world/{self._world}/set_pose rejected')

            # Pose is not momentum. Without this the base keeps whatever velocity it had while
            # falling, so it gets teleported upright and immediately continues downwards at the
            # speed it was already doing - which looks exactly like the reset never happened.
            # These are one-shot commands, verified: the robot falls normally afterwards rather
            # than being pinned at zero velocity. Unlike the joint resets these components have
            # no explicit serializer, so gz-sim's default one applies and the payload is the
            # ASCII form of a math::Vector3d, not a protobuf message.
            velocity_request = WorldControlState()
            base_entity = velocity_request.state.entities.add()
            base_entity.id = model_entity
            for type_id in (TYPE_WORLD_LINEAR_VELOCITY_CMD, TYPE_WORLD_ANGULAR_VELOCITY_CMD):
                component = base_entity.components.add()
                component.type = type_id
                component.component = b'0 0 0'
            ok, rep = self._gz.request(
                f'/world/{self._world}/control/state', velocity_request,
                WorldControlState, Boolean, 10000)
            if not ok or not rep.data:
                raise RuntimeError('base velocity write rejected')

            # The components just written sit in the ECM until Physics consumes them on the
            # first step after the world resumes, and the policy bridge waits for post-reset
            # messages before re-seeding, so nothing here needs to observe the stepped result.
            # Resuming is deliberately left to _start_resume()'s retry loop - see why there.
            return True, f'{len(entities)} joints reset to default pose'
        except Exception as exc:  # noqa: BLE001 - reported to the caller either way
            # Never leave the world paused because a reset failed partway.
            try:
                self._world_control(pause=False)
            except Exception:  # noqa: BLE001
                pass
            return False, str(exc)


def main():
    rclpy.init()
    node = GzResetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
