# Metrics for one recorded test, from simulator ground truth only (link poses, foot contact
# forces, joint states). Pure numpy - no ROS - so it runs in whatever venv has wandb.
#
# Conventions: world frame z up; the robot's heading psi is the pelvis yaw. "forward" and
# "sideways" are the pelvis's planar world velocity rotated into the heading frame, "turn" is
# d(psi)/dt - the same three quantities a /cmd_vel Twist commands.
import numpy as np

G = 9.81
CYCLE = 0.8           # s, gait period the tracking averages over (training's gait_phase period)
TRANSIENT = 1.5       # s after a command change that counts as transient, not steady
CONTACT_ON = 0.05     # fraction of body weight on a foot to count as contact ...
CONTACT_OFF = 0.02    # ... and below which it is released (hysteresis)
MIN_PHASE = 0.05      # s; shorter contact/swing blips are merged away
STOMP_WINDOW = 0.05   # s after touchdown the peak force is taken over ...
STOMP_SMOOTH = 0.01   # ... of the force averaged over this (one 1 kHz sample is a solver spike)


# --- rotation helpers (quaternions [x, y, z, w]) -------------------------------------------------

def yaw(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_rotate(q, v):
    qv = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], axis=-1)
    return quat_mul(quat_mul(q, qv), q * np.array([-1.0, -1.0, -1.0, 1.0]))[..., :3]


def world_link_poses(model_pose, link_pose_in_model):
    """(T, 7) model world pose + (T, L, 7) link poses in the model frame -> world pos, quat."""
    mp = model_pose[:, None, :3]
    mq = np.broadcast_to(model_pose[:, None, 3:], link_pose_in_model[..., 3:].shape)
    return (mp + quat_rotate(mq, link_pose_in_model[..., :3]),
            quat_mul(mq, link_pose_in_model[..., 3:]))


def roll_pitch(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


# --- signal helpers ------------------------------------------------------------------------------

def moving_average(t, x, window):
    # Centred average over +-window/2 of (possibly non-uniform) samples.
    x = np.asarray(x, dtype=float)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    lo = np.searchsorted(t, t - window / 2, side='left')
    hi = np.searchsorted(t, t + window / 2, side='right')
    n = (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (c[hi] - c[lo]) / n


def hold(t_src, x_src, t_new):
    # Zero-order hold: the latest source sample at or before each new time.
    idx = np.clip(np.searchsorted(t_src, t_new, side='right') - 1, 0, len(t_src) - 1)
    return np.asarray(x_src)[idx]


def planar_velocity(t, pos, psi):
    vx = np.gradient(pos[:, 0], t)
    vy = np.gradient(pos[:, 1], t)
    forward = vx * np.cos(psi) + vy * np.sin(psi)
    sideways = -vx * np.sin(psi) + vy * np.cos(psi)
    return forward, sideways


def contact_state(t_src, fz, t_grid, body_weight, max_gap=0.003):
    # Force on the grid (0 where no message is within max_gap - the sensor stops publishing
    # when nothing touches), then hysteresis, then merge blips shorter than MIN_PHASE.
    f = np.zeros(len(t_grid))
    if len(t_src):
        i = np.clip(np.searchsorted(t_src, t_grid), 1, len(t_src) - 1)
        nearest = np.where(np.abs(t_src[i - 1] - t_grid) < np.abs(t_src[i] - t_grid), i - 1, i)
        ok = np.abs(t_src[nearest] - t_grid) <= max_gap
        f[ok] = np.abs(fz[nearest[ok]])
    state = np.zeros(len(t_grid), dtype=bool)
    on = False
    for k, v in enumerate(f):
        on = v > CONTACT_ON * body_weight if not on else v > CONTACT_OFF * body_weight
        state[k] = on
    return _merge_blips(t_grid, state)


def _merge_blips(t, state):
    state = state.copy()
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(np.diff(state.astype(int))) + 1
        bounds = np.concatenate([[0], edges, [len(state)]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            interior = a > 0 and b < len(state)
            if interior and t[b - 1] - t[a] < MIN_PHASE:
                state[a:b] = not state[a]
                changed = True
                break
    return state


def edges(state):
    d = np.diff(state.astype(int))
    return np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1   # touchdowns, liftoffs


def _nanmean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float('nan')


# --- one test ------------------------------------------------------------------------------------

class TestData:
    """Everything a metric needs, on the pose time grid (200 Hz), cut at a fall."""

    def __init__(self, d, entry, mass):
        self.mass = mass
        self.weight = mass * G
        self.segments = entry['segments']
        self.pushes = entry['pushes']
        self.fell = entry['status'] == 'fell'
        self.fall_t = entry.get('fall_t')
        end = self.fall_t if self.fell else float('inf')

        t = d['pose_t']
        keep = t <= end
        self.t = t = t[keep]
        names = [str(n) for n in d['link_names']]
        model = d['model_pose'][keep].astype(float)
        pos, quat = world_link_poses(model, d['link_pose_in_model'][keep].astype(float))
        # The model frame is the base link's (pelvis) frame.
        self.pelvis = model[:, :3]
        self.psi = np.unwrap(yaw(model[:, 3:]))
        self.torso_roll, self.torso_pitch = roll_pitch(quat[:, names.index(str(d['torso_link']))])
        self.forward, self.sideways = planar_velocity(t, self.pelvis, self.psi)
        self.turn = np.gradient(self.psi, t)
        self.achieved = np.stack([self.forward, self.sideways, self.turn], axis=1)
        self.achieved_cycle = moving_average(t, self.achieved, CYCLE)
        self.achieved_short = moving_average(t, self.achieved, 0.2)

        # What was actually published, held between messages.
        self.cmd = hold(d['cmd_t'], d['cmd'].astype(float), t)

        self.feet = []
        for i, link in enumerate(str(n) for n in d['foot_links']):
            ct, fz = d[f'contact{i}_t'], d[f'contact{i}_fz'].astype(float)
            self.feet.append({
                'name': link,
                'pos': pos[:, names.index(link)],
                'state': contact_state(ct, fz, t, self.weight),
                'raw_t': ct,
                'raw_fz': moving_average(ct, fz, STOMP_SMOOTH) if len(ct) else ct,
            })

        jt = d['joint_t']
        jkeep = jt <= end
        self.joint_t = jt[jkeep]
        self.joint_vel = d['joint_vel'][jkeep].astype(float)
        self.joint_effort = d['joint_effort'][jkeep].astype(float)

    # Masks over self.t.
    def segment_mask(self, seg, start=0.0, stop=0.0):
        return (self.t >= seg['t0'] + start) & (self.t < seg['t1'] - stop)

    def moving_segments(self):
        return [s for s in self.segments if any(abs(c) > 1e-9 for c in s['cmd'])]

    def standing_segments(self):
        return [s for s in self.segments if all(abs(c) <= 1e-9 for c in s['cmd'])]

    def steady_mask(self):
        m = np.zeros(len(self.t), dtype=bool)
        for s in self.moving_segments():
            m |= self.segment_mask(s, TRANSIENT, CYCLE / 2)
        return m

    def transient_mask(self):
        m = np.zeros(len(self.t), dtype=bool)
        for s in self.moving_segments():
            m |= self.segment_mask(s) & (self.t < s['t0'] + TRANSIENT)
        return m

    def moving_mask(self):
        m = np.zeros(len(self.t), dtype=bool)
        for s in self.moving_segments():
            m |= self.segment_mask(s)
        return m


AXES = ('vx', 'vy', 'wz')


def tracking(td):
    out = {}
    err = np.abs(td.achieved_cycle - td.cmd)
    for label, mask in (('steady', td.steady_mask()), ('transient', td.transient_mask())):
        for k, axis in enumerate(AXES):
            out[f'err_{axis}_{label}'] = float(err[mask, k].mean()) if mask.any() else float('nan')

    # Achieved / commanded along each commanded axis, steady only (1 = exact, 0.5 = half speed).
    ratios = {a: [] for a in AXES}
    steady = td.steady_mask()
    for s in td.moving_segments():
        m = td.segment_mask(s, TRANSIENT, CYCLE / 2) & steady
        if not m.any():
            continue
        for k, axis in enumerate(AXES):
            if abs(s['cmd'][k]) > 1e-9:
                ratios[axis].append(td.achieved_cycle[m, k].mean() / s['cmd'][k])
    for axis in AXES:
        out[f'ratio_{axis}'] = _nanmean(ratios[axis])

    # Response time: after a command change, how long until the short-averaged velocity is
    # within 25% of the change on every axis that changed.
    times = []
    for prev, s in zip(td.segments[:-1], td.segments[1:]):
        delta = np.subtract(s['cmd'], prev['cmd'])
        axes = np.flatnonzero(np.abs(delta) > 0.05)
        m = td.segment_mask(s)
        if not len(axes) or not m.any():
            continue
        e = np.abs(td.achieved_short[m][:, axes] - np.asarray(s['cmd'])[axes])
        ok = np.all(e <= 0.25 * np.abs(delta[axes]), axis=1)
        times.append(td.t[m][np.argmax(ok)] - s['t0'] if ok.any() else s['t1'] - s['t0'])
    out['response_time'] = _nanmean(times)
    return out


def drift(td):
    # Pure-forward steady stretches: how much heading and sideways position wander per metre.
    heading, lateral = [], []
    for s in td.moving_segments():
        vx, vy, wz = s['cmd']
        if abs(vx) < 1e-9 or abs(vy) > 1e-9 or abs(wz) > 1e-9:
            continue
        m = td.segment_mask(s, TRANSIENT, 0.0)
        if m.sum() < 2:
            continue
        p, psi = td.pelvis[m, :2], td.psi[m]
        dist = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        if dist < 0.2:
            continue
        heading.append(abs(np.degrees(psi[-1] - psi[0])) / dist)
        side = np.array([-np.sin(psi[0]), np.cos(psi[0])])
        lateral.append(abs((p[-1] - p[0]) @ side) * 100 / dist)
    return {'heading_drift_deg_per_m': _nanmean(heading),
            'lateral_drift_cm_per_m': _nanmean(lateral)}


def standing(td):
    out = {}
    # Long zero-command stretches without pushes (the stand test).
    drift_, sway, steps = [], [], []
    for s in td.standing_segments():
        if s['t1'] - s['t0'] < 2.0 or any(s['t0'] <= p['t'] < s['t1'] for p in td.pushes):
            continue
        m = td.segment_mask(s)
        if m.sum() < 2:
            continue
        p = td.pelvis[m, :2]
        dur = td.t[m][-1] - td.t[m][0]
        drift_.append(np.linalg.norm(p[-1] - p[0]) * 100 / dur)
        sway.append(np.linalg.norm(td.achieved_short[m, :2], axis=1).mean() * 100)
        late = td.segment_mask(s, 1.0)
        lifts = sum(int(np.sum(np.isin(edges(f['state'])[1], np.flatnonzero(late))))
                    for f in td.feet)
        steps.append(lifts / max(late.sum() * np.median(np.diff(td.t)), 1e-6))
    out['stand_drift_cm_per_s'] = _nanmean(drift_)
    out['stand_sway_cm_per_s'] = _nanmean(sway)
    out['stand_steps_per_s'] = _nanmean(steps)

    # Stops after moving: steps taken after the zero command, and when the last one lifted.
    stop_steps, settle = [], []
    for prev, s in zip(td.segments[:-1], td.segments[1:]):
        if s not in td.standing_segments() or prev in td.standing_segments():
            continue
        m = td.segment_mask(s)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        lifts = [td.t[k] for f in td.feet for k in edges(f['state'])[1] if idx[0] <= k <= idx[-1]]
        stop_steps.append(len(lifts))
        settle.append(max(lifts) - s['t0'] if lifts else 0.0)
    out['stop_steps'] = _nanmean(stop_steps)
    out['stop_settle_time'] = _nanmean(settle)
    return out


def posture(td):
    steady = td.steady_mask()
    out = {
        'torso_pitch_deg': float(np.degrees(td.torso_pitch).mean()),
        'torso_pitch_std_deg': float(np.degrees(td.torso_pitch).std()),
        'torso_roll_deg': float(np.degrees(td.torso_roll).mean()),
        'torso_roll_std_deg': float(np.degrees(td.torso_roll).std()),
        'pelvis_height': float(td.pelvis[:, 2].mean()),
    }
    out['pelvis_bounce_cm'] = (float(td.pelvis[steady, 2].std() * 100)
                               if steady.any() else float('nan'))
    return out


def footsteps(td):
    steady = td.steady_mask()
    out = {}
    if not steady.any() or len(td.feet) != 2:
        return out
    dt = np.median(np.diff(td.t))
    left, right = td.feet
    sl, sr = left['state'], right['state']
    out['duty_factor'] = float(np.mean([sl[steady].mean(), sr[steady].mean()]))
    out['double_support'] = float((sl & sr)[steady].mean())
    out['flight'] = float((~sl & ~sr)[steady].mean())

    strides, widths, swings, slips, stomps, stance_dur = [], [], [], [], [], {0: [], 1: []}
    n_touchdowns = 0
    for fi, (foot, other) in enumerate(((left, right), (right, left))):
        td_idx, lo_idx = edges(foot['state'])
        td_idx = td_idx[steady[td_idx]]
        n_touchdowns += len(td_idx)
        pos = foot['pos']
        for a, b in zip(td_idx[:-1], td_idx[1:]):
            strides.append(np.linalg.norm(pos[b, :2] - pos[a, :2]))
        for k in td_idx:
            psi = td.psi[k]
            side = np.array([-np.sin(psi), np.cos(psi)])
            widths.append(abs((pos[k, :2] - other['pos'][k, :2]) @ side))
            w = (foot['raw_t'] >= td.t[k]) & (foot['raw_t'] <= td.t[k] + STOMP_WINDOW)
            if w.any():
                stomps.append(foot['raw_fz'][w].max() / td.weight)
        # Swing: liftoff -> next touchdown; stance: touchdown -> next liftoff.
        for lo in lo_idx[steady[lo_idx]]:
            nxt = td_idx[td_idx > lo]
            if len(nxt):
                swings.append((pos[lo:nxt[0], 2].max() - pos[lo - 1, 2]) * 100)
        for k in td_idx:
            nxt = lo_idx[lo_idx > k]
            if len(nxt):
                stance_dur[fi].append((nxt[0] - k) * dt)
                start = k + int(round(0.02 / dt))
                seg = pos[start:nxt[0], :2]
                if len(seg) > 1:
                    slips.append(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)) * 100)

    steady_time = steady.sum() * dt
    out['cadence_steps_per_s'] = n_touchdowns / steady_time
    out['stride_length'] = _nanmean(strides)
    out['stance_width'] = _nanmean(widths)
    out['swing_height_cm'] = _nanmean(swings)
    out['foot_slip_cm_per_step'] = _nanmean(slips)
    out['stomp_body_weights'] = _nanmean(stomps)
    if stance_dur[0] and stance_dur[1]:
        a, b = np.mean(stance_dur[0]), np.mean(stance_dur[1])
        out['stance_asymmetry_pct'] = float(abs(a - b) / ((a + b) / 2) * 100)
    return out


def effort(td):
    out = {}
    jt = td.joint_t
    if len(jt) < 3:
        return out
    power = np.abs(td.joint_effort * td.joint_vel).sum(axis=1)
    dt = np.diff(jt, prepend=jt[0])
    moving = hold(td.t, td.moving_mask().astype(int), jt).astype(bool) & (jt <= td.t[-1])
    if moving.any():
        energy = float(np.sum(power[moving] * dt[moving]))
        m = td.moving_mask()
        speed = np.linalg.norm(td.achieved_cycle[m, :2], axis=1)
        dist = float(np.sum(speed * np.median(np.diff(td.t))))
        out['cost_of_transport'] = energy / (td.weight * dist) if dist > 0.1 else float('nan')
    out['torque_rms'] = float(np.sqrt(np.mean(td.joint_effort ** 2)))
    out['mech_power_w'] = float(np.mean(power))

    # Smoothness from measured joint motion at the policy rate (50 Hz).
    grid = np.arange(jt[0], jt[-1], 0.02)
    if len(grid) > 4:
        v = hold(jt, moving_average(jt, td.joint_vel, 0.02), grid)
        acc = np.diff(v, axis=0) / 0.02
        jerk = np.diff(acc, axis=0) / 0.02
        out['joint_acc_rms'] = float(np.sqrt(np.mean(acc ** 2)))
        out['joint_jerk_rms'] = float(np.sqrt(np.mean(jerk ** 2)))
    return out


def falls(td, duration):
    out = {'fell': float(td.fell), 'time_to_fall': td.fall_t if td.fell else float('nan')}
    if td.pushes:
        end = td.fall_t if td.fell else duration
        out['pushes_survived'] = float(sum(1 for p in td.pushes if p['t'] + 2.0 <= end))
        out['pushes_total'] = float(len(td.pushes))
    return out


def path(td):
    # Where the commands say the robot should be vs where it went, integrated from where it
    # actually started (so only its own errors accumulate).
    t = td.t
    dt = np.diff(t, prepend=t[0])
    psi_c = td.psi[0] + np.cumsum(td.cmd[:, 2] * dt)
    vx = td.cmd[:, 0] * np.cos(psi_c) - td.cmd[:, 1] * np.sin(psi_c)
    vy = td.cmd[:, 0] * np.sin(psi_c) + td.cmd[:, 1] * np.cos(psi_c)
    p_c = td.pelvis[0, :2] + np.stack([np.cumsum(vx * dt), np.cumsum(vy * dt)], axis=1)
    err = np.linalg.norm(td.pelvis[:, :2] - p_c, axis=1)
    return {'path_final_err': float(err[-1]), 'path_mean_err': float(err.mean()),
            'heading_final_err_deg': float(abs(np.degrees(td.psi[-1] - psi_c[-1])))}


def analyze_test(d, entry, mass):
    td = TestData(d, entry, mass)
    out = {}
    for fn in (tracking, drift, standing, posture, footsteps, effort, path):
        out.update(fn(td))
    out.update(falls(td, entry['duration']))
    return out, td


def timeseries(td, rate=25.0):
    """Commanded vs achieved, downsampled for plotting."""
    grid = np.arange(td.t[0], td.t[-1], 1.0 / rate)
    idx = np.clip(np.searchsorted(td.t, grid), 0, len(td.t) - 1)
    return {
        't': grid,
        'cmd_vx': td.cmd[idx, 0], 'cmd_vy': td.cmd[idx, 1], 'cmd_wz': td.cmd[idx, 2],
        'vx': td.achieved_cycle[idx, 0], 'vy': td.achieved_cycle[idx, 1],
        'wz': td.achieved_cycle[idx, 2],
        'pelvis_height': td.pelvis[idx, 2],
        'torso_pitch_deg': np.degrees(td.torso_pitch[idx]),
    }


def overall(per_test):
    """Mean of each metric over the tests that have it (each test weighted equally)."""
    keys = sorted({k for m in per_test.values() for k in m})
    out = {k: _nanmean([m.get(k) for m in per_test.values()]) for k in keys}
    out['falls'] = float(sum(m.get('fell', 0) for m in per_test.values()))
    return out
