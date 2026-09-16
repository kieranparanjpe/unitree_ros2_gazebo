# The Gazebo-Harmonic side of a reset, as plain functions: find the entities, write the
# components. No policy about *when* to do it - that is the reset node's job, and it is the
# only thing the two reset strategies disagree about.
#
# Mechanism: gz-sim state is entities + components, and a teleport is just a component the
# Physics system consumes on its next update - JointPositionReset/JointVelocityReset for the
# joints, WorldLinearVelocityCmd/WorldAngularVelocityCmd for the base, plus the
# /world/<world>/set_pose service for the base pose. /world/<world>/control/state accepts an
# arbitrary serialized component write, so no custom Gazebo system plugin is needed. This is
# Harmonic's equivalent of Gazebo Classic's /gazebo/set_model_configuration, which was never
# ported as a named service.
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.double_v_pb2 import Double_V
from gz.msgs10.empty_pb2 import Empty as GzEmpty
from gz.msgs10.pose_pb2 import Pose as GzPose
from gz.msgs10.serialized_map_pb2 import SerializedStepMap
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.world_control_state_pb2 import WorldControlState


def hash64(name: str) -> int:
    # gz-sim derives a component's numeric type id as gz::common::hash64 of its registered
    # type name (components/Factory.hh: `auto typeHash = gz::common::hash64(_type);
    # ComponentTypeT::typeId = typeHash;`). That's FNV-1a/64. C++ callers never do this - they
    # reference components::JointPositionReset::typeId directly - so this reimplementation is
    # the one part of this file that leans on an internal detail rather than a public API.
    # verify_type_ids() below checks it against components we can independently identify,
    # because the failure mode is otherwise silent: an unrecognised type id writes a component
    # nothing reads, and the service still reports success.
    h = 0xCBF29CE484222325
    for byte in name.encode():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


TYPE_NAME = hash64('gz_sim_components.Name')
TYPE_POSE = hash64('gz_sim_components.Pose')
TYPE_PARENT_ENTITY = hash64('gz_sim_components.ParentEntity')
TYPE_JOINT = hash64('gz_sim_components.Joint')
TYPE_JOINT_POSITION_RESET = hash64('gz_sim_components.JointPositionReset')
TYPE_JOINT_VELOCITY_RESET = hash64('gz_sim_components.JointVelocityReset')
TYPE_MODEL = hash64('gz_sim_components.Model')
TYPE_WORLD_LINEAR_VELOCITY_CMD = hash64('gz_sim_components.WorldLinearVelocityCmd')
TYPE_WORLD_ANGULAR_VELOCITY_CMD = hash64('gz_sim_components.WorldAngularVelocityCmd')


def read_state(gz, world):
    ok, rep = gz.request(
        f'/world/{world}/state', GzEmpty(), GzEmpty, SerializedStepMap, 15000)
    if not ok:
        raise RuntimeError(f'/world/{world}/state request failed')
    return rep.state


def verify_type_ids(state):
    # Name/Pose/ParentEntity are on essentially every entity, so if our hash matches the ids
    # actually present, the derivation is still correct for this gz-sim build - and therefore
    # so is JointPositionReset's, which is produced the same way.
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
            'JointPositionReset write would be silently ignored. See hash64().')


def joint_entities(state, joint_names):
    entities = {}
    wanted = set(joint_names)
    for entity_id, ent in state.entities.items():
        comps = {c.type for c in ent.components.values()}
        if TYPE_JOINT not in comps:
            continue
        for comp in ent.components.values():
            if comp.type == TYPE_NAME:
                name = comp.component.decode(errors='replace')
                if name in wanted:
                    entities[name] = entity_id
    missing = [j for j in joint_names if j not in entities]
    if missing:
        raise RuntimeError(
            f'{len(missing)} of {len(joint_names)} joints not found as joint entities in '
            f'the ECM (first few: {missing[:5]})')
    return entities


def model_entity(state, model_name):
    for entity_id, ent in state.entities.items():
        comps = {c.type for c in ent.components.values()}
        if TYPE_MODEL not in comps:
            continue
        for comp in ent.components.values():
            if comp.type == TYPE_NAME and comp.component.decode(errors='replace') == model_name:
                return entity_id
    raise RuntimeError(f'no model entity named "{model_name}" in the ECM')


def world_control(gz, world, **fields):
    request = WorldControl(**fields)
    ok, rep = gz.request(f'/world/{world}/control', request, WorldControl, Boolean, 5000)
    if not ok or not rep.data:
        raise RuntimeError(f'/world/{world}/control rejected {fields}')


def write_joint_reset(gz, world, entities, joint_names, default_joint_pos):
    request = WorldControlState()
    for name, target in zip(joint_names, default_joint_pos):
        entity = request.state.entities.add()
        entity.id = entities[name]
        position = entity.components.add()
        position.type = TYPE_JOINT_POSITION_RESET
        position.component = Double_V(data=[float(target)]).SerializeToString()
        # Position without velocity would leave whatever momentum the robot had before the
        # teleport, which the policy would then see as a real motion.
        velocity = entity.components.add()
        velocity.type = TYPE_JOINT_VELOCITY_RESET
        velocity.component = Double_V(data=[0.0]).SerializeToString()
    ok, rep = gz.request(
        f'/world/{world}/control/state', request, WorldControlState, Boolean, 10000)
    if not ok or not rep.data:
        raise RuntimeError(f'/world/{world}/control/state write rejected')


def write_base_pose(gz, world, model_name, spawn_height):
    # The joint write alone would leave a collapsed robot lying on the floor with its limbs
    # snapped to the standing pose - and teleporting joints while links are interpenetrating
    # the ground spikes the contact solver. Lifting the base back to its spawn pose puts it in
    # free space first.
    base = GzPose()
    base.name = model_name
    base.position.z = float(spawn_height)
    base.orientation.w = 1.0
    ok, rep = gz.request(f'/world/{world}/set_pose', base, GzPose, Boolean, 5000)
    if not ok or not rep.data:
        raise RuntimeError(f'/world/{world}/set_pose rejected')


def write_base_velocity_zero(gz, world, model_entity_id):
    # Pose is not momentum. Without this the base keeps whatever velocity it had while falling,
    # so it gets teleported upright and immediately continues downwards at the speed it was
    # already doing - which looks exactly like the reset never happened. These are one-shot
    # commands, verified: the robot falls normally afterwards rather than being pinned at zero
    # velocity. Unlike the joint resets these components have no explicit serializer, so
    # gz-sim's default one applies and the payload is the ASCII form of a math::Vector3d, not a
    # protobuf message.
    request = WorldControlState()
    entity = request.state.entities.add()
    entity.id = model_entity_id
    for type_id in (TYPE_WORLD_LINEAR_VELOCITY_CMD, TYPE_WORLD_ANGULAR_VELOCITY_CMD):
        component = entity.components.add()
        component.type = type_id
        component.component = b'0 0 0'
    ok, rep = gz.request(
        f'/world/{world}/control/state', request, WorldControlState, Boolean, 10000)
    if not ok or not rep.data:
        raise RuntimeError('base velocity write rejected')
