"""Synthetic recordings in the runner's file format, with answers known by hand."""
import math

import numpy as np
import pytest

import analyze

SAMPLE_PERIOD = 0.005
BASE_HEIGHT = 1.0
FOOT_DOWN_HEIGHT = 0.05
FOOT_UP_HEIGHT = 0.15


def foot_heights_with_stances(times, stances):
    """Foot height: FOOT_DOWN_HEIGHT during each (start, end) stance, FOOT_UP_HEIGHT otherwise."""
    heights = np.full_like(times, FOOT_UP_HEIGHT)
    for start, end in stances:
        heights[(times >= start - 1e-9) & (times < end - 1e-9)] = FOOT_DOWN_HEIGHT
    return heights


def yaw_quaternion(yaw):
    yaw = np.asarray(yaw, float)
    zeros = np.zeros_like(yaw)
    return np.stack([zeros, zeros, np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1)


def record(tmp_path, times, base_x, base_y, base_yaw, torso_orientation_in_base=None,
           foot_positions_in_base=None):
    """Write a recording as the runner does and load it back through the analysis."""
    sample_count = len(times)
    base_poses = np.zeros((sample_count, 7))
    base_poses[:, 0], base_poses[:, 1], base_poses[:, 2] = base_x, base_y, BASE_HEIGHT
    base_poses[:, 3:] = yaw_quaternion(base_yaw)
    torso_poses_in_base = np.zeros((sample_count, 7))
    torso_poses_in_base[:, 3:] = (
        [0, 0, 0, 1] if torso_orientation_in_base is None else torso_orientation_in_base)
    foot_poses_in_base = np.zeros((sample_count, 2, 7))
    foot_poses_in_base[..., 6] = 1
    if foot_positions_in_base is not None:
        foot_poses_in_base[..., :3] = foot_positions_in_base
    path = tmp_path / 'test.npz'
    np.savez(path, pose_times=times, base_poses=base_poses,
             torso_poses_in_base=torso_poses_in_base, foot_poses_in_base=foot_poses_in_base)
    return analyze.load_test(path)


def make_test_entry(segments, status='ok', fall_time=None, push_times=()):
    return {'segments': [{'start_time': start, 'end_time': end, 'command': list(command)}
                         for start, end, command in segments],
            'status': status, 'fall_time': fall_time,
            'pushes': [{'time': push_time} for push_time in push_times]}


def test_straight_walk_at_an_angle_tracks_perfectly(tmp_path):
    times = np.arange(0, 4, SAMPLE_PERIOD)
    heading = math.radians(30)
    test_data = record(tmp_path, times, 0.5 * math.cos(heading) * times,
                       0.5 * math.sin(heading) * times, np.full_like(times, heading))
    metrics, series = analyze.tracking_metrics(
        test_data, analyze.segments_before_fall(make_test_entry([(0, 4, (0.5, 0, 0))])))
    for metric in ('forward_velocity_error_full', 'forward_velocity_error_steady',
                   'sideways_velocity_error_full', 'turn_rate_error_full', 'heading_error_deg'):
        assert metrics[metric] == pytest.approx(0, abs=1e-6)


def test_turning_too_fast(tmp_path):
    # Commanded 0.4 rad/s, turns at 0.5 rad/s: 0.1 rad/s error, 0.1 * duration heading error.
    duration = 4 + SAMPLE_PERIOD
    times = np.arange(0, duration - SAMPLE_PERIOD / 2, SAMPLE_PERIOD)
    test_data = record(tmp_path, times, 0 * times, 0 * times, 0.5 * times)
    metrics, series = analyze.tracking_metrics(
        test_data, analyze.segments_before_fall(make_test_entry([(0, duration, (0, 0, 0.4))])))
    assert metrics['turn_rate_error_steady'] == pytest.approx(0.1, abs=1e-6)
    # Across the segment's samples, first to last.
    assert metrics['heading_error_deg'] == pytest.approx(
        math.degrees(0.1 * (times[-1] - times[0])), abs=1e-6)
    # Series: the signed error every sample, and the heading error growing at 0.1 rad/s.
    error_times, turn_rate_errors = series['turn_rate_error']
    assert turn_rate_errors == pytest.approx(np.full_like(error_times, 0.1), abs=1e-6)
    heading_times, heading_errors = series['heading_error_deg']
    assert heading_errors == pytest.approx(np.degrees(0.1 * heading_times), abs=1e-6)


def test_steady_window_ignores_first_second(tmp_path):
    # Stands still for 1 s, then walks at the commanded 0.5 m/s.
    times = np.arange(0, 4, SAMPLE_PERIOD)
    test_data = record(tmp_path, times, 0.5 * np.maximum(times - 1, 0), 0 * times, 0 * times)
    metrics, series = analyze.tracking_metrics(
        test_data, analyze.segments_before_fall(make_test_entry([(0, 4, (0.5, 0, 0))])))
    assert metrics['forward_velocity_error_steady'] == pytest.approx(0, abs=1e-3)
    assert metrics['forward_velocity_error_full'] == pytest.approx(0.125, abs=1e-2)
    # The signed series averages back to the summary: |mean(series)| == the full-window error.
    _, forward_errors = series['forward_velocity_error']
    assert abs(forward_errors.mean()) == pytest.approx(metrics['forward_velocity_error_full'])


def test_standing_drift_and_pushed_segments_skipped(tmp_path):
    times = np.arange(0, 8, SAMPLE_PERIOD)
    test_data = record(tmp_path, times, 0.025 * times, 0 * times, np.radians(1.0) * times)
    entry = make_test_entry([(0, 4, (0, 0, 0)), (4, 8, (0, 0, 0))], push_times=(5.0,))
    metrics, series = analyze.standing_metrics(
        test_data, analyze.segments_before_fall(entry), entry['pushes'])
    assert metrics['standing_drift_m_per_s'] == pytest.approx(0.025, rel=1e-6)
    assert metrics['standing_yaw_drift_deg_per_s'] == pytest.approx(1.0, rel=1e-6)
    # Only the unpushed first segment appears in the series.
    distance_times, distances = series['standing_distance_from_start']
    assert distance_times.max() < 4
    assert distances[-1] == pytest.approx(0.025 * distance_times[-1])


def test_falls_and_push_survival():
    entry = make_test_entry([(0, 6, (0, 0, 0))], status='fell', fall_time=4.0,
                       push_times=(0.5, 2.0, 5.0))
    # Push at 0.5 survived (the fall is 3.5 s later), push at 2.0 did not, push at 5.0 never
    # happened.
    assert analyze.fall_metrics(entry) == ({'fell': 1.0, 'fall_time': 4.0,
                                            'push_survival_rate': 0.5}, {})


def test_segments_after_a_fall_are_dropped():
    entry = make_test_entry([(0, 2, (0.5, 0, 0)), (2, 4, (0, 0, 0))], status='fell', fall_time=3.0)
    assert [segment['end_time'] for segment in analyze.segments_before_fall(entry)] == [2]


def test_torso_pitch_is_in_the_world_frame(tmp_path):
    # Base turned 90 deg, torso pitched 10 deg relative to it: still 10 deg pitch, 0 roll.
    times = np.arange(0, 1, SAMPLE_PERIOD)
    pitch = math.radians(10)
    test_data = record(tmp_path, times, 0 * times, 0 * times, np.full_like(times, math.pi / 2),
                       torso_orientation_in_base=[0, math.sin(pitch / 2), 0, math.cos(pitch / 2)])
    metrics, series = analyze.upright_metrics(test_data)
    assert metrics['torso_pitch_deg'] == pytest.approx(10, abs=1e-6)
    assert metrics['torso_roll_deg'] == pytest.approx(0, abs=1e-6)


def test_touchdowns_ignore_brief_lifts_and_a_foot_already_down():
    # Stance 0-0.5 s (already down at the start), 0.8-1.3 s with a one-sample lift in the
    # middle, 1.6-2.1 s.
    times = np.arange(0, 2.5, SAMPLE_PERIOD)
    heights = foot_heights_with_stances(times, [(0, 0.5), (0.8, 1.0), (1.005, 1.3), (1.6, 2.1)])
    assert analyze.find_touchdown_times(times, heights) == pytest.approx([0.8, 1.6])


def test_gait(tmp_path):
    # Base still at the origin facing +y. Feet 0.3 m apart sideways; each foot lands every
    # 0.8 s (left at 0.8k, right at 0.8k + 0.4) and moves 0.4 m forward per landing.
    stride_time, stride_length, half_width = 0.8, 0.4, 0.15
    times = np.arange(0, 6, SAMPLE_PERIOD)
    foot_positions_in_base = np.zeros((len(times), 2, 3))
    for foot_index, (first_landing, sideways_offset) in enumerate(((0.0, half_width),
                                                                   (0.4, -half_width))):
        landing_times = np.arange(first_landing, 6, stride_time)
        landings_so_far = np.searchsorted(landing_times, times + 1e-9, side='right')
        foot_positions_in_base[:, foot_index, 0] = stride_length * landings_so_far
        foot_positions_in_base[:, foot_index, 1] = sideways_offset
        foot_positions_in_base[:, foot_index, 2] = foot_heights_with_stances(
            times, [(landing, landing + 0.5) for landing in landing_times]) - BASE_HEIGHT
    test_data = record(tmp_path, times, 0 * times, 0 * times, np.full_like(times, math.pi / 2),
                       foot_positions_in_base=foot_positions_in_base)
    metrics, series = analyze.gait_metrics(
        test_data, analyze.segments_before_fall(make_test_entry([(0, 6, (0.5, 0, 0))])))
    assert metrics['stride_length'] == pytest.approx(stride_length, abs=1e-6)
    assert metrics['stride_time'] == pytest.approx(stride_time, abs=1e-6)
    assert metrics['step_width'] == pytest.approx(2 * half_width, abs=1e-6)
    # One series value per stride, in time order: both feet's strides interleaved.
    stride_end_times, stride_lengths = series['stride_length']
    assert np.all(np.diff(stride_end_times) > 0)
    assert stride_lengths == pytest.approx(np.full_like(stride_lengths, stride_length))


def test_standing_segments_have_no_gait(tmp_path):
    times = np.arange(0, 4, SAMPLE_PERIOD)
    test_data = record(tmp_path, times, 0 * times, 0 * times, 0 * times)
    assert analyze.gait_metrics(
        test_data, analyze.segments_before_fall(make_test_entry([(0, 4, (0, 0, 0))]))) == ({}, {})


def test_history_rows_downsample_and_merge_sparse_series():
    # A 200 Hz series and a sparse per-stride series, merged onto one 50 Hz time axis.
    times = np.arange(0, 0.09, SAMPLE_PERIOD)   # last tick 0.08
    series = {'forward_velocity_error': (times, np.ones_like(times)),
              'stride_length': (np.array([0.041]), np.array([0.4]))}
    rows = analyze.history_rows('walk', series)
    assert [row['walk/time'] for row in rows] == pytest.approx(np.arange(0, 0.1, 0.02))
    assert all(row['walk/forward_velocity_error'] == 1 for row in rows)
    assert [row.get('walk/stride_length') for row in rows] == [None, None, 0.4, None, None]


def test_overall_is_the_plain_mean_over_tests_per_repeat():
    values_by_test = {
        'short_test': {'stride_length': [0.2, 0.3], 'fell': [0.0, 0.0]},
        'long_test': {'stride_length': [0.4, None], 'fell': [1.0, 0.0]},   # aborted in repeat 2
    }
    overall = analyze.overall_per_repeat(values_by_test, repeat_count=2)
    assert overall['stride_length'] == pytest.approx([0.3, 0.3])   # repeat 2: only short_test
    assert overall['fell'] == pytest.approx([0.5, 0.0])
    stats = analyze.mean_and_std_over_repeats(overall['fell'])
    assert stats['mean'] == pytest.approx(0.25)
    assert stats['std'] == pytest.approx(np.std([0.5, 0.0], ddof=1))
    assert analyze.mean_and_std_over_repeats([0.4, None]) == {'mean': 0.4}   # no std from one

    per_repeat = analyze.per_repeat_values(values_by_test, overall)
    assert per_repeat['stride_length'] == {
        'r1': {'overall': pytest.approx(0.3), 'short_test': 0.2, 'long_test': 0.4},
        'r2': {'overall': pytest.approx(0.3), 'short_test': 0.3}}


def test_mean_series_over_repeats_averages_values_sharing_a_tick():
    times = np.arange(0, 0.1, 0.02)
    series_by_repeat = {1: {'error': (times, np.full_like(times, 1.0))},
                        2: {'error': (times + 0.001, np.full_like(times, 3.0))},
                        3: {'error': (times[:2], np.full_like(times[:2], 5.0))}}   # shorter
    mean_times, mean_values = analyze.mean_series_over_repeats(series_by_repeat)['error']
    assert mean_times == pytest.approx(times)
    assert mean_values == pytest.approx([3.0, 3.0, 2.0, 2.0, 2.0])


def test_wandb_summaries_per_run_kind():
    results = {
        'overall': {'fell': {'mean': 0.25, 'std': 0.1}, 'fall_time': {'mean': 4.0}},
        'by_test': {'walk': {'fell': {'mean': 0.5, 'std': 0.2}}},
        'per_repeat': {'fell': {'r1': {'overall': 0.5, 'walk': 1.0}, 'r2': {'overall': 0.0}}},
    }
    assert analyze.repeat_summary(results, 1) == {'overall/fell': 0.5, 'by_test/walk/fell': 1.0}
    assert analyze.repeat_summary(results, 2) == {'overall/fell': 0.0}
    assert analyze.aggregate_summary(results) == {
        'overall/fell/mean': 0.25, 'overall/fell/std': 0.1, 'overall/fall_time/mean': 4.0,
        'by_test/walk/fell/mean': 0.5, 'by_test/walk/fell/std': 0.2}
