# Synthetic recordings with known answers: uv run pytest
import numpy as np

import metrics


def _quat_yaw(psi):
    return np.stack([np.zeros_like(psi), np.zeros_like(psi), np.sin(psi / 2), np.cos(psi / 2)], -1)


def _recording(duration=10.0, v=0.5, wz=0.0, cmd=(0.5, 0.0, 0.0), rate=200.0):
    """Pelvis gliding at body-frame speed v (turning at wz), feet alternating 0.4 s stances."""
    t = np.arange(0, duration, 1 / rate)
    psi = wz * t
    if wz:
        x = v / wz * np.sin(psi)
        y = v / wz * (1 - np.cos(psi))
    else:
        x, y = v * t, np.zeros_like(t)
    pelvis = np.stack([x, y, np.full_like(t, 1.0)], -1)
    q = _quat_yaw(psi)
    side = np.stack([-np.sin(psi), np.cos(psi), np.zeros_like(t)], -1)
    left_on = (t % 0.8) < 0.4
    feet = []
    for sign, on in ((1, left_on), (-1, ~left_on)):
        p = pelvis + sign * 0.1 * side
        p[:, 2] = np.where(on, 0.05, 0.10)
        feet.append((p, on))
    names = np.array(['left_foot', 'pelvis', 'right_foot', 'torso_link'])
    world_pos = np.stack([feet[0][0], pelvis, feet[1][0], pelvis + [0, 0, 0.3]], 1)
    # Stored as published: model (= pelvis) world pose + link poses in the model frame.
    q_inv = q * np.array([-1.0, -1.0, -1.0, 1.0])
    rel_pos = metrics.quat_rotate(q_inv[:, None, :].repeat(4, 1), world_pos - pelvis[:, None, :])
    rel_quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 4, 1))
    d = {
        'pose_t': t, 'link_names': names,
        'model_pose': np.concatenate([pelvis, q], -1),
        'link_pose_in_model': np.concatenate([rel_pos, rel_quat], -1),
        'base_link': np.array('pelvis'), 'torso_link': np.array('torso_link'), 'foot_links': np.array(['left_foot', 'right_foot']),
        'cmd_t': t[::4], 'cmd': np.tile(cmd, (len(t[::4]), 1)),
        'joint_t': t, 'joint_vel': np.ones((len(t), 2)), 'joint_effort': np.ones((len(t), 2)),
    }
    for i, (_, on) in enumerate(feet):
        tc = np.arange(0, duration, 0.001)
        d[f'contact{i}_t'] = tc
        d[f'contact{i}_fz'] = np.where(np.interp(tc, t, on) > 0.5, 400.0, 0.0)
    entry = {'status': 'ok', 'duration': duration, 'pushes': [],
             'segments': [{'t0': 0.0, 't1': duration, 'cmd': list(cmd)}]}
    return d, entry


def test_perfect_forward_tracking():
    m, _ = metrics.analyze_test(*_recording(), mass=80.0)
    assert m['err_vx_steady'] < 1e-3
    assert m['err_vy_steady'] < 1e-3
    assert abs(m['ratio_vx'] - 1) < 1e-3
    assert m['heading_drift_deg_per_m'] < 1e-6
    assert m['lateral_drift_cm_per_m'] < 1e-6


def test_body_frame_velocity_while_turning():
    m, _ = metrics.analyze_test(*_recording(v=0.5, wz=0.4, cmd=(0.5, 0.0, 0.4)), mass=80.0)
    assert m['err_vx_steady'] < 1e-3      # forward stays 0.5 even though world x/y rotate
    assert m['err_vy_steady'] < 1e-3
    assert m['err_wz_steady'] < 1e-3


def test_footsteps():
    m, _ = metrics.analyze_test(*_recording(), mass=80.0)
    assert abs(m['stance_width'] - 0.2) < 1e-3
    assert abs(m['cadence_steps_per_s'] - 2.5) < 0.2      # one touchdown per 0.4 s
    assert abs(m['stride_length'] - 0.4) < 0.01            # 0.8 s at 0.5 m/s
    assert abs(m['swing_height_cm'] - 5.0) < 0.1
    assert abs(m['duty_factor'] - 0.5) < 0.02
    assert m['stance_asymmetry_pct'] < 2


def test_slip_measures_stance_motion():
    # The synthetic feet ride along with the pelvis even in stance: 0.5 m/s for ~0.38 s.
    m, _ = metrics.analyze_test(*_recording(), mass=80.0)
    assert abs(m['foot_slip_cm_per_step'] - 0.5 * 0.38 * 100) < 1.5


def test_contact_hysteresis_merges_blips():
    t = np.arange(0, 1, 0.005)
    fz = np.where((t > 0.2) & (t < 0.8), 500.0, 0.0)
    fz[(t > 0.5) & (t < 0.52)] = 0.0            # 20 ms dropout inside the stance
    state = metrics.contact_state(t, fz, t, body_weight=800.0)
    touchdowns, liftoffs = metrics.edges(state)
    assert len(touchdowns) == 1 and len(liftoffs) == 1


def test_world_link_poses_composes_model_and_link():
    yaw90 = np.array([[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    model = np.concatenate([[[1.0, 2.0, 0.0]], yaw90], -1)
    link = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])     # 1 m ahead of the model
    pos, quat = metrics.world_link_poses(model, link)
    assert np.allclose(pos[0, 0], [1.0, 3.0, 0.0])                # ahead = +y after 90 deg yaw
    assert np.allclose(quat[0, 0], yaw90[0])
