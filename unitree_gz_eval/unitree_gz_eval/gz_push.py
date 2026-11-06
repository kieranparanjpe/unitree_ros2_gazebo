# Pushes a link by writing a one-step force into Gazebo's ECM. Kept apart from
# unitree_gz_bringup's gz_ecm on purpose: the evaluation's push must not change when the reset
# code does.
#
# ExternalWorldWrenchCmd is applied for exactly one physics step and then cleared (gz-sim 8
# Physics.cc), so force = impulse / step is a clean one-shot impulse.
import gz.transport13 as gz_transport
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.serialized_map_pb2 import SerializedStepMap
from gz.msgs10.vector3d_pb2 import Vector3d
from gz.msgs10.world_control_state_pb2 import WorldControlState
from gz.msgs10.wrench_pb2 import Wrench


def _hash64(name: str) -> int:
    # gz-sim component type id = FNV-1a/64 of the registered type name.
    h = 0xCBF29CE484222325
    for byte in name.encode():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


_TYPE_NAME = _hash64('gz_sim_components.Name')
_TYPE_LINK = _hash64('gz_sim_components.Link')
_TYPE_WRENCH_CMD = _hash64('gz_sim_components.ExternalWorldWrenchCmd')


class Pusher:
    def __init__(self, world: str, link: str, physics_step: float):
        self._gz = gz_transport.Node()
        self._world = world
        self._step = physics_step
        self._link = self._find_link(link)

    def _find_link(self, link):
        ok, rep = self._gz.request(
            f'/world/{self._world}/state', Empty(), Empty, SerializedStepMap, 15000)
        if not ok:
            raise RuntimeError(f'/world/{self._world}/state request failed')
        found = [
            eid for eid, ent in rep.state.entities.items()
            if any(c.type == _TYPE_LINK for c in ent.components.values())
            and any(c.type == _TYPE_NAME and c.component.decode(errors='replace') == link
                    for c in ent.components.values())
        ]
        if len(found) != 1:
            raise RuntimeError(f'expected one link entity named "{link}", found {len(found)}')
        return found[0]

    def push(self, impulse):
        """Apply a world-frame impulse (N*s) to the link over one physics step."""
        f = [v / self._step for v in impulse]
        request = WorldControlState()
        ent = request.state.entities.add()
        ent.id = self._link
        comp = ent.components.add()
        comp.type = _TYPE_WRENCH_CMD
        comp.component = Wrench(force=Vector3d(x=f[0], y=f[1], z=f[2])).SerializeToString()
        ok, rep = self._gz.request(
            f'/world/{self._world}/control/state', request, WorldControlState, Boolean, 5000)
        if not ok or not rep.data:
            raise RuntimeError('push wrench write rejected')
