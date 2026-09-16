# Sim-only reset primitive: the world runs from the start, the robot falls (nothing commands
# it until the policy engages), and once it has settled this node teleports it into
# deploy.yaml's default pose and tells the policy to re-seed. The same entry point doubles as
# an episode reset via the ~/reset service, which is what Isaac Lab does between episodes too.
#
# The world cannot be started paused instead: controller activation is serviced by
# controller_manager's update loop, which under gz_ros2_control only runs on simulation steps,
# so a paused world never finishes activating and the spawners time out.
#
# This is deliberately NOT part of the policy nodes: those stay transport-agnostic so the same
# observation/action path can drive real hardware over DDS, where no such teleport exists
# (there, reaching the default pose is the controller FSM's stand-up state, upstream of them).
import time

import gz.transport13 as gz_transport
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from std_srvs.srv import Trigger

from unitree_gz_bringup import gz_ecm

TICK_SECONDS = 0.2
# Wall time to let a queued single step actually execute before issuing the next write.
STEP_SETTLE_SECONDS = 0.3


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
        # How long to let the robot fall before resetting - long enough that it is lying still
        # rather than mid-air.
        self._settle_seconds = self.declare_parameter('settle_seconds', 3.0).value

        if len(self._joint_names) != len(self._default_joint_pos):
            raise RuntimeError(
                'joint_names and default_joint_pos must be the same length '
                f'({len(self._joint_names)} vs {len(self._default_joint_pos)})')

        self._gz = gz_transport.Node()
        self._joint_state_count = 0
        self._settle_ticks = 0

        # Latched: policy_node must not miss the notification if it finishes loading its ONNX
        # session after the reset has already happened.
        self._reset_pub = self.create_publisher(
            Empty, reset_topic,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.create_service(Trigger, '~/reset', self._on_reset_request)

        # Wall time, not sim time: this has to tick regardless of what the simulator is doing.
        self._timer = self.create_timer(TICK_SECONDS, self._try_startup_reset)
        self.get_logger().info(
            f'gz_reset_node waiting for /joint_states, then {self._settle_seconds:.1f}s of '
            f'settling, before resetting {len(self._joint_names)} joints of '
            f'model "{self._model}"')

    def _on_joint_state(self, msg):
        self._joint_state_count += 1
        self._last_joint_state = msg

    def _try_startup_reset(self):
        # /joint_states publishing is the signal that joint_state_broadcaster activated, which
        # is the only thing worth waiting for. The IMU bridge is not waited on: policy_node
        # will not leave Phase::ResetTriggered until it has seen both /joint_states and /imu,
        # so a reset that lands before the bridge is up costs nothing.
        if self._joint_state_count == 0:
            return
        self._settle_ticks += 1
        if self._settle_ticks * TICK_SECONDS < self._settle_seconds:
            return

        self._timer.cancel()
        ok, detail = self._reset()
        if ok:
            self.get_logger().info(f'Startup reset complete ({detail}); notifying policy node')
            self._reset_pub.publish(Empty())
        else:
            self.get_logger().error(f'Startup reset failed: {detail}')

    def _on_reset_request(self, _request, response):
        response.success, response.message = self._reset()
        if response.success:
            self._reset_pub.publish(Empty())
        return response

    def _step_once(self):
        # Advances the world exactly one step so Physics consumes the components just written,
        # leaving it paused again. The sleep is what makes it a step rather than a request:
        # gz queues world-control requests and drains them at its own pace, so issuing the
        # next write immediately would put it in the same batch as this step.
        gz_ecm.world_control(self._gz, self._world, pause=True, multi_step=1)
        time.sleep(STEP_SETTLE_SECONDS)

    def _reset(self):
        try:
            state = gz_ecm.read_state(self._gz, self._world)
            gz_ecm.verify_type_ids(state)
            entities = gz_ecm.joint_entities(state, self._joint_names)
            model_entity_id = gz_ecm.model_entity(state, self._model)

            # Each write gets its own step. Measured: a JointPositionReset write applied on its
            # own takes a robot from 3.025 rad of error to 0.007, but the same write followed
            # by a set_pose or a second /control/state call before the next step lands at
            # 0.8-1.2 rad - i.e. it is silently discarded. The base writes go first so the
            # joints are the last word, and each is consumed by Physics before the next is
            # issued.
            gz_ecm.world_control(self._gz, self._world, pause=True)

            gz_ecm.write_base_pose(self._gz, self._world, self._model, self._spawn_height)
            gz_ecm.write_base_velocity_zero(self._gz, self._world, model_entity_id)
            self._step_once()

            gz_ecm.write_joint_reset(
                self._gz, self._world, entities, self._joint_names, self._default_joint_pos)
            self._step_once()

            gz_ecm.world_control(self._gz, self._world, pause=False)

            return True, f'{len(entities)} joints reset to default pose'
        except Exception as exc:  # noqa: BLE001 - reported to the caller either way
            # Never leave the world paused because a reset failed partway.
            try:
                gz_ecm.world_control(self._gz, self._world, pause=False)
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
