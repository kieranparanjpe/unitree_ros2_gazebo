"""Score one evaluation directory and upload it to wandb.

    uv run analyze.py <eval_dir>             # -> <eval_dir>/results.json, then wandb
    uv run analyze.py <eval_dir> --no-wandb

Every metric function returns two things for one repeat of one test:
  summary - {metric: one number}, the value averaged over time / segments / strides;
  series  - {metric: (times, values)}, the same quantity before that averaging.
Summaries are collected over repeats into mean and std (results.json). In wandb, each evaluation
is one group of runs: one per repeat (job_type 'repeat': that repeat's values and series) and one
aggregate (job_type 'aggregate': mean and std over repeats, series averaged over repeats). All
log the same series keys, so charts overlay them; compare policies on the aggregate runs.

A segment is a stretch of constant command; its "full" window is the whole segment, its
"steady" window drops the first STEADY_WINDOW_SKIP_SECONDS. Anything after a fall is ignored:
only segments that ended before it count.
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

WANDB_ENTITY = 'kieranparanjpe-mcgill-university'
WANDB_PROJECT = 'humanoidx-gz-eval'

STEADY_WINDOW_SKIP_SECONDS = 1.0         # dropped from the start of a segment for its steady window
FOOT_DOWN_HEIGHT_MARGIN = 0.015          # m: a foot is down within this of its lowest height
MINIMUM_SWING_SECONDS = 0.1              # foot up for at least this before a touchdown counts
PUSH_FALL_WINDOW_SECONDS = 3.0           # after a push, in which a fall counts against it
PLOT_RATE_HZ = 50.0                      # of the time series sent to wandb

# The command's three entries, in order, and the measured quantity each is compared with.
COMMANDED_QUANTITIES = ('forward_velocity', 'sideways_velocity', 'turn_rate')


# --- quaternions, stored as (x, y, z, w) --------------------------------------------------------

def quaternion_multiply(first, second):
    first_x, first_y, first_z, first_w = np.moveaxis(first, -1, 0)
    second_x, second_y, second_z, second_w = np.moveaxis(second, -1, 0)
    return np.stack([
        first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
        first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
        first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w,
        first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
    ], axis=-1)


def rotate_vector(quaternion, vector):
    vector_part, scalar_part = quaternion[..., :3], quaternion[..., 3:]
    twice_cross = 2 * np.cross(vector_part, vector)
    return vector + scalar_part * twice_cross + np.cross(vector_part, twice_cross)


def roll_pitch_yaw(quaternion):
    quaternion_x, quaternion_y, quaternion_z, quaternion_w = np.moveaxis(quaternion, -1, 0)
    roll = np.arctan2(2 * (quaternion_w * quaternion_x + quaternion_y * quaternion_z),
                      1 - 2 * (quaternion_x ** 2 + quaternion_y ** 2))
    pitch = np.arcsin(np.clip(2 * (quaternion_w * quaternion_y - quaternion_z * quaternion_x),
                              -1, 1))
    yaw = np.arctan2(2 * (quaternion_w * quaternion_z + quaternion_x * quaternion_y),
                     1 - 2 * (quaternion_y ** 2 + quaternion_z ** 2))
    return roll, pitch, yaw


# --- one recorded test -> world-frame quantities ------------------------------------------------

def find_touchdown_times(times, foot_height):
    """Times a foot lands. The foot is down when its height (of the foot link's frame) is within
    FOOT_DOWN_HEIGHT_MARGIN of its lowest height in the test; a landing is a down sample after
    at least MINIMUM_SWING_SECONDS without one. A foot already down at t=0 has not landed."""
    times_down = times[foot_height < foot_height.min() + FOOT_DOWN_HEIGHT_MARGIN]
    time_since_previous_down = np.diff(times_down, prepend=0.0)
    return times_down[time_since_previous_down >= MINIMUM_SWING_SECONDS]


def load_test(path):
    with np.load(path) as recording:
        times = recording['pose_times']
        base_poses = recording['base_poses']
        torso_poses_in_base = recording['torso_poses_in_base']
        foot_poses_in_base = recording['foot_poses_in_base']

    base_position, base_orientation = base_poses[:, :3], base_poses[:, 3:]
    base_yaw = np.unwrap(roll_pitch_yaw(base_orientation)[2])
    world_velocity_x = np.gradient(base_position[:, 0], times)  # central differences
    world_velocity_y = np.gradient(base_position[:, 1], times)
    torso_roll, torso_pitch, _ = roll_pitch_yaw(
        quaternion_multiply(base_orientation, torso_poses_in_base[:, 3:]))
    foot_positions = base_position[:, None] + rotate_vector(
        base_orientation[:, None], foot_poses_in_base[..., :3])

    return {
        'time': times,
        'base_x': base_position[:, 0],
        'base_y': base_position[:, 1],
        'base_height': base_position[:, 2],
        'base_yaw': base_yaw,
        'forward_velocity': np.cos(base_yaw) * world_velocity_x + np.sin(base_yaw) * world_velocity_y,
        'sideways_velocity': -np.sin(base_yaw) * world_velocity_x + np.cos(base_yaw) * world_velocity_y,
        'turn_rate': np.gradient(base_yaw, times),
        'torso_roll': torso_roll,
        'torso_pitch': torso_pitch,
        'foot_positions_xy': foot_positions[..., :2],          # (sample, foot, xy)
        'touchdown_times': [find_touchdown_times(times, foot_positions[:, foot_index, 2])
                            for foot_index in range(foot_positions.shape[1])],
    }


def samples_between(test_data, start_time, end_time):
    return (test_data['time'] >= start_time) & (test_data['time'] < end_time)


def segments_before_fall(test_entry):
    fall_time = test_entry['fall_time'] if test_entry['status'] == 'fell' else math.inf
    return [segment for segment in test_entry['segments'] if segment['end_time'] <= fall_time]


def is_standing_command(segment):
    return not any(segment['command'])


def mean(values):
    return float(np.mean(values))


def rms(values):
    """Root mean square: sqrt(mean(value^2)). Never cancels, unlike a plain mean."""
    return float(np.sqrt(np.mean(np.square(values))))


def commanded_at_each_sample(test_data, segments):
    """(samples, 3) command in force at each sample; NaN where no segment covers it (after a
    fall)."""
    commanded = np.full((len(test_data['time']), 3), np.nan)
    for segment in segments:
        commanded[samples_between(test_data, segment['start_time'], segment['end_time'])] = (
            segment['command'])
    return commanded


# --- metrics: each returns (summary, series) for one repeat of one test -------------------------

def tracking_metrics(test_data, segments):
    """1, 2. Series, every sample (signed):  error = measured - commanded  for forward velocity,
    sideways velocity and turn rate;  heading error = heading change since the segment started -
    commanded heading change since then, in degrees.
    Summary, per segment:  rms error = sqrt(mean(error^2)) over the full and steady windows;
    heading rms error = sqrt(mean(heading error^2)) over the segment. Each averaged over
    segments."""
    times = test_data['time']
    commanded = commanded_at_each_sample(test_data, segments)
    errors = {f'{quantity}_error': test_data[quantity] - commanded[:, command_index]
              for command_index, quantity in enumerate(COMMANDED_QUANTITIES)}
    # Running heading error since the test started. Samples no segment covers (e.g. one stamped
    # just before the test started) add no commanded turn.
    commanded_turn_rate = np.nan_to_num(commanded[:, 2])
    commanded_heading_change = np.concatenate(
        [[0.0], np.cumsum(commanded_turn_rate[:-1] * np.diff(times))])
    running_heading_error = np.degrees(
        test_data['base_yaw'] - test_data['base_yaw'][0] - commanded_heading_change)

    summary = defaultdict(list)
    heading_error = np.full(len(times), np.nan)   # since the start of each segment
    for segment in segments:
        for window_name, skipped_seconds in (('full', 0.0), ('steady', STEADY_WINDOW_SKIP_SECONDS)):
            in_window = samples_between(
                test_data, segment['start_time'] + skipped_seconds, segment['end_time'])
            if not in_window.any():
                continue
            for name, error in errors.items():
                quantity = name.removesuffix('_error')
                summary[f'{quantity}_rms_error_{window_name}'].append(rms(error[in_window]))
        in_segment = samples_between(test_data, segment['start_time'], segment['end_time'])
        heading_error[in_segment] = (running_heading_error[in_segment]
                                     - running_heading_error[in_segment][0])
        summary['heading_rms_error_deg'].append(rms(heading_error[in_segment]))

    covered = ~np.isnan(commanded[:, 0])
    series = {name: (times[covered], error[covered])
              for name, error in {**errors, 'heading_error_deg': heading_error}.items()}
    return {name: mean(values) for name, values in summary.items()}, series


def standing_metrics(test_data, segments, pushes):
    """4. Zero-command segments with no push in them, whole segment.
    Series:   distance from where the segment started (m), yaw change since it started (deg).
    Summary:  rms drift = sqrt(mean(distance^2)) (m), rms yaw drift = sqrt(mean(yaw change^2))
              (deg), per segment, averaged over segments."""
    position_drifts, yaw_drifts = [], []
    series_times, distances_from_start, yaw_changes_deg = [], [], []
    for segment in segments:
        pushed = any(segment['start_time'] <= push['time'] < segment['end_time'] for push in pushes)
        if not is_standing_command(segment) or pushed:
            continue
        in_segment = samples_between(test_data, segment['start_time'], segment['end_time'])
        if in_segment.sum() < 2:
            continue
        times = test_data['time'][in_segment]
        distance_from_start = np.hypot(
            test_data['base_x'][in_segment] - test_data['base_x'][in_segment][0],
            test_data['base_y'][in_segment] - test_data['base_y'][in_segment][0])
        yaw_change_deg = np.degrees(
            test_data['base_yaw'][in_segment] - test_data['base_yaw'][in_segment][0])
        position_drifts.append(rms(distance_from_start))
        yaw_drifts.append(rms(yaw_change_deg))
        series_times.append(times)
        distances_from_start.append(distance_from_start)
        yaw_changes_deg.append(yaw_change_deg)
    if not position_drifts:
        return {}, {}
    summary = {'standing_rms_drift_m': mean(position_drifts),
               'standing_rms_yaw_drift_deg': mean(yaw_drifts)}
    series_times = np.concatenate(series_times)
    series = {'standing_distance_from_start': (series_times, np.concatenate(distances_from_start)),
              'standing_yaw_change_deg': (series_times, np.concatenate(yaw_changes_deg))}
    return summary, series


def fall_metrics(test_entry):
    """3. fell = 1 if pelvis height < 0.5 m or tilt > 70 deg at any time (the runner checks this
    and stops the test). 5. push survival rate = pushes not followed by a fall within
    PUSH_FALL_WINDOW_SECONDS / pushes applied (pushes after a fall never happen).
    One event per test, so no series."""
    fell = test_entry['status'] == 'fell'
    fall_time = test_entry['fall_time']
    summary = {'fell': float(fell)}
    if fell:
        summary['fall_time'] = fall_time
    applied_push_times = [push['time'] for push in test_entry['pushes']
                          if not fell or push['time'] <= fall_time]
    if applied_push_times:
        summary['push_survival_rate'] = mean([
            not (fell and push_time <= fall_time <= push_time + PUSH_FALL_WINDOW_SECONDS)
            for push_time in applied_push_times])
    return summary, {}


def upright_metrics(test_data):
    """6. Series: torso pitch and roll at every sample (signed), in degrees.
    Summary: sqrt(mean(angle^2)) of each over the whole test - counts both a steady lean and the
    rocking back and forth."""
    pitch = np.degrees(test_data['torso_pitch'])
    roll = np.degrees(test_data['torso_roll'])
    summary = {'torso_pitch_rms_deg': rms(pitch), 'torso_roll_rms_deg': rms(roll)}
    series = {'torso_pitch_deg': (test_data['time'], pitch),
              'torso_roll_deg': (test_data['time'], roll)}
    return summary, series


def gait_metrics(test_data, segments):
    """7. Walking (non-zero command) segments, steady window. Per foot, consecutive touchdowns
    n, n+1:  stride length = |position(n+1) - position(n)|,  stride time = time(n+1) - time(n).
    At each touchdown of either foot:  step width = sideways distance between the feet in the
    base frame.
    Series: each stride at the time of its second touchdown, each step width at its touchdown.
    Summary: mean over both feet's strides (touchdowns) per segment, then over segments."""
    foot_positions = test_data['foot_positions_xy']
    # metric -> list of (times, values) arrays, one per foot per segment
    pieces = {'stride_length': [], 'stride_time': [], 'step_width': []}
    per_segment_means = defaultdict(list)
    for segment in segments:
        if is_standing_command(segment):
            continue
        steady_start = segment['start_time'] + STEADY_WINDOW_SKIP_SECONDS
        segment_values = defaultdict(list)
        for foot_index, touchdown_times in enumerate(test_data['touchdown_times']):
            touchdown_times = touchdown_times[
                (touchdown_times >= steady_start) & (touchdown_times < segment['end_time'])]
            # First pose sample at or after each touchdown.
            touchdown_samples = np.minimum(np.searchsorted(test_data['time'], touchdown_times),
                                           len(test_data['time']) - 1)
            landing_positions = foot_positions[touchdown_samples, foot_index]
            left_minus_right = (foot_positions[touchdown_samples, 0]
                                - foot_positions[touchdown_samples, 1])
            yaw = test_data['base_yaw'][touchdown_samples]
            foot_values = {
                'stride_length': (touchdown_times[1:],
                                  np.linalg.norm(np.diff(landing_positions, axis=0), axis=1)),
                'stride_time': (touchdown_times[1:], np.diff(touchdown_times)),
                'step_width': (touchdown_times, np.abs(-np.sin(yaw) * left_minus_right[:, 0]
                                                       + np.cos(yaw) * left_minus_right[:, 1])),
            }
            for name, (times, values) in foot_values.items():
                pieces[name].append((times, values))
                segment_values[name] += list(values)
        for name, values in segment_values.items():
            if values:
                per_segment_means[name].append(mean(values))

    summary = {name: mean(values) for name, values in per_segment_means.items()}
    series = {}
    for name in summary:
        times = np.concatenate([times for times, _ in pieces[name]])
        values = np.concatenate([values for _, values in pieces[name]])
        time_order = np.argsort(times)
        series[name] = (times[time_order], values[time_order])
    return summary, series


def score_test(test_data, test_entry):
    segments = segments_before_fall(test_entry)
    summary, series = {}, {}
    for metric_summary, metric_series in (
            tracking_metrics(test_data, segments),
            standing_metrics(test_data, segments, test_entry['pushes']),
            fall_metrics(test_entry),
            upright_metrics(test_data),
            gait_metrics(test_data, segments)):
        summary.update(metric_summary)
        series.update(metric_series)
    return summary, series


# --- whole evaluation ---------------------------------------------------------------------------

def mean_and_std_over_repeats(values_per_repeat):
    """Mean and std over the repeats that have a value; no std with fewer than two."""
    present_values = [value for value in values_per_repeat if value is not None]
    statistics = {'mean': mean(present_values)}
    if len(present_values) > 1:
        statistics['std'] = float(np.std(present_values, ddof=1))
    return statistics


def per_repeat_values(values_by_test, overall_values):
    """{metric: {'r1': {'overall': value, test name: value, ...}, 'r2': ...}}, leaving out
    repeats and tests without a value."""
    by_metric = defaultdict(lambda: defaultdict(dict))
    for source, values_by_metric in {'overall': overall_values, **values_by_test}.items():
        for metric, values_per_repeat in values_by_metric.items():
            for repeat_number, value in enumerate(values_per_repeat, 1):
                if value is not None:
                    by_metric[metric][f'r{repeat_number}'][source] = value
    return {metric: dict(by_repeat) for metric, by_repeat in by_metric.items()}


def overall_per_repeat(values_by_test, repeat_count):
    """{metric: value per repeat}: in each repeat, the metric's plain mean over the tests that
    have it in that repeat. Every test counts equally, whatever its length."""
    overall_values = defaultdict(lambda: [None] * repeat_count)
    for repeat_index in range(repeat_count):
        values_across_tests = defaultdict(list)
        for values_by_metric in values_by_test.values():
            for metric, values_per_repeat in values_by_metric.items():
                if values_per_repeat[repeat_index] is not None:
                    values_across_tests[metric].append(values_per_repeat[repeat_index])
        for metric, values in values_across_tests.items():
            overall_values[metric][repeat_index] = mean(values)
    return overall_values


def analyze(eval_dir):
    with open(os.path.join(eval_dir, 'meta.json')) as meta_file:
        meta = json.load(meta_file)
    repeat_count = max(test_entry['repeat'] for test_entry in meta['tests'])
    status_by_test = defaultdict(lambda: [None] * repeat_count)
    # test name -> metric name -> value per repeat (None where the repeat has no value)
    values_by_test = defaultdict(lambda: defaultdict(lambda: [None] * repeat_count))
    # test name -> repeat number -> {metric: (times, values)}
    series_by_test = defaultdict(dict)
    for test_entry in meta['tests']:
        test_name, repeat_index = test_entry['name'], test_entry['repeat'] - 1
        status_by_test[test_name][repeat_index] = test_entry['status']
        series_by_test.setdefault(test_name, {})   # keeps tests in suite order
        if test_entry['status'] == 'aborted':
            continue
        test_data = load_test(os.path.join(eval_dir, test_entry['file']))
        summary, series = score_test(test_data, test_entry)
        for metric, value in summary.items():
            values_by_test[test_name][metric][repeat_index] = value
        series_by_test[test_name][repeat_index + 1] = series

    overall_values = overall_per_repeat(values_by_test, repeat_count)
    results = {
        'policy': meta['policy']['name'], 'suite': meta['suite'], 'repeats': repeat_count,
        'status': dict(status_by_test),
        # (1) over all tests and repeats
        'overall': {metric: mean_and_std_over_repeats(values_per_repeat)
                    for metric, values_per_repeat in overall_values.items()},
        # (2) over repeats, per test
        'by_test': {test_name: {metric: mean_and_std_over_repeats(values_per_repeat)
                                for metric, values_per_repeat in values_by_metric.items()}
                    for test_name, values_by_metric in values_by_test.items()},
        # (3) every repeat's own value, by metric then repeat
        'per_repeat': per_repeat_values(values_by_test, overall_values),
    }
    with open(os.path.join(eval_dir, 'results.json'), 'w') as results_file:
        json.dump(results, results_file, indent=2)
    return meta, results, series_by_test


def print_summary(results):
    print(f"{results['policy']}  suite {results['suite']['name']} ({results['suite']['hash']}), "
          f"{results['repeats']} repeats (mean ± std over repeats)")
    sections = [(f'{test_name}: {", ".join(str(status) for status in statuses)}',
                 results['by_test'].get(test_name, {}))
                for test_name, statuses in results['status'].items()]
    sections.append(('overall (mean over tests)', results['overall']))
    for heading, stats_by_metric in sections:
        print(f'  {heading}')
        for metric, stats in stats_by_metric.items():
            spread = f" ± {stats['std']:.3g}" if 'std' in stats else ''
            print(f"      {metric:36s} {stats['mean']:.3g}{spread}")


# --- wandb --------------------------------------------------------------------------------------

def values_at_ticks(times, values):
    """{tick number: value}: the series downsampled to PLOT_RATE_HZ, keeping the first sample
    that falls in each tick."""
    ticks = np.round(np.asarray(times) * PLOT_RATE_HZ).astype(int)
    unique_ticks, first_sample_in_tick = np.unique(ticks, return_index=True)
    return dict(zip(unique_ticks.tolist(), np.asarray(values)[first_sample_in_tick].tolist()))


def mean_series_over_repeats(series_by_repeat):
    """{metric: (times, values)}: at each PLOT_RATE_HZ tick, the mean over the repeats that have a
    value there. Per-stride gait values rarely share a tick, so those are mostly single repeats."""
    values_by_metric = defaultdict(lambda: defaultdict(list))   # metric -> tick -> [values]
    for series in series_by_repeat.values():
        for metric, (times, values) in series.items():
            for tick, value in values_at_ticks(times, values).items():
                values_by_metric[metric][tick].append(value)
    mean_series = {}
    for metric, values_by_tick in values_by_metric.items():
        ticks = sorted(values_by_tick)
        mean_series[metric] = (np.array(ticks) / PLOT_RATE_HZ,
                               np.array([np.mean(values_by_tick[tick]) for tick in ticks]))
    return mean_series


def history_rows(test_name, series):
    """Rows for run.log, in time order: one per PLOT_RATE_HZ tick holding each metric's value
    there, against the test's own time axis {test}/time."""
    rows = defaultdict(dict)   # tick number -> {key: value}
    for metric, (times, values) in series.items():
        for tick, value in values_at_ticks(times, values).items():
            rows[tick][f'{test_name}/{metric}'] = float(value)
    return [{f'{test_name}/time': tick / PLOT_RATE_HZ, **rows[tick]} for tick in sorted(rows)]


def repeat_summary(results, repeat_number):
    """This repeat's own values: overall/<metric> and by_test/<test>/<metric>."""
    summary = {}
    for metric, values_by_repeat in results['per_repeat'].items():
        for source, value in values_by_repeat.get(f'r{repeat_number}', {}).items():
            key = f'overall/{metric}' if source == 'overall' else f'by_test/{source}/{metric}'
            summary[key] = value
    return summary


def aggregate_summary(results):
    """Mean and std over repeats: overall/<metric>/<statistic> and
    by_test/<test>/<metric>/<statistic>."""
    sections = {'overall': results['overall'],
                **{f'by_test/{test_name}': statistics_by_metric
                   for test_name, statistics_by_metric in results['by_test'].items()}}
    return {f'{prefix}/{metric}/{statistic}': value
            for prefix, statistics_by_metric in sections.items()
            for metric, statistics in statistics_by_metric.items()
            for statistic, value in statistics.items()}


def log_run(eval_dir, entity, project, run_settings, summary, series_by_test):
    """One wandb run: summary numbers, then each test's series against {test}/time."""
    import wandb

    run = wandb.init(entity=entity, project=project, **run_settings)
    run.summary.update(summary)
    for test_name, series in series_by_test.items():
        # summary='none': the last sample of a series is not a meaningful run summary.
        run.define_metric(f'{test_name}/time', summary='none')
        run.define_metric(f'{test_name}/*', step_metric=f'{test_name}/time', summary='none')
        for row in history_rows(test_name, series):
            run.log(row)
    run.save(os.path.join(eval_dir, 'results.json'), base_path=eval_dir, policy='now')
    run.save(os.path.join(eval_dir, 'meta.json'), base_path=eval_dir, policy='now')
    run.finish()
    return run.url


def upload(eval_dir, meta, results, series_by_test, entity, project):
    """Per evaluation, in one wandb group: one run per repeat (job_type 'repeat') with that
    repeat's values and series, and one run (job_type 'aggregate') with the mean and std over
    repeats and the series averaged over repeats. Compare policies on the aggregate runs."""
    policy, suite = results['policy'], meta['suite']
    group = f"{policy}__{os.path.basename(os.path.normpath(eval_dir))}"
    config = {'policy': policy, 'policy_dir': meta['policy']['dir'], 'suite_name': suite['name'],
              'suite_hash': suite['hash'], 'evaluation': group, 'robot': meta['robot']}
    suite_tag = f"suite:{suite['hash']}"

    run_urls = []
    for repeat_number in range(1, results['repeats'] + 1):
        status_by_test = {test_name: statuses[repeat_number - 1]
                          for test_name, statuses in results['status'].items()}
        run_settings = {
            'job_type': 'repeat', 'group': group,
            'name': f"{policy}__{suite['name']}__r{repeat_number}",
            'tags': [suite_tag] + [f'{status}:{test_name}'
                                   for test_name, status in status_by_test.items()
                                   if status in ('fell', 'aborted')],
            'config': {**config, 'repeat': repeat_number, 'test_status': status_by_test},
        }
        repeat_series = {test_name: series_by_repeat[repeat_number]
                         for test_name, series_by_repeat in series_by_test.items()
                         if repeat_number in series_by_repeat}   # absent if aborted
        run_urls.append(log_run(eval_dir, entity, project, run_settings,
                                repeat_summary(results, repeat_number), repeat_series))

    run_settings = {
        'job_type': 'aggregate', 'group': group,
        'name': f"{policy}__{suite['name']}",
        'tags': [suite_tag] + [f'{status}:{test_name}'
                               for test_name, statuses in results['status'].items()
                               for status in sorted(set(statuses))
                               if status in ('fell', 'aborted')],
        'config': {**config, 'repeats': results['repeats'], 'test_status': results['status']},
    }
    aggregate_series = {test_name: mean_series_over_repeats(series_by_repeat)
                        for test_name, series_by_repeat in series_by_test.items()
                        if series_by_repeat}
    run_urls.append(log_run(eval_dir, entity, project, run_settings,
                            aggregate_summary(results), aggregate_series))
    return run_urls


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('eval_dir')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--entity', default=WANDB_ENTITY)
    parser.add_argument('--project', default=WANDB_PROJECT)
    args = parser.parse_args()

    meta, results, series_by_test = analyze(args.eval_dir)
    print_summary(results)
    print(f"-> {os.path.join(args.eval_dir, 'results.json')}")
    if not args.no_wandb:
        for run_url in upload(args.eval_dir, meta, results, series_by_test,
                              args.entity, args.project):
            print(f'wandb: {run_url}')


if __name__ == '__main__':
    main()
