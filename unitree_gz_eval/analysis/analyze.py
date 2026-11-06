"""Score one evaluation directory and (optionally) upload it to wandb.

    uv run analyze.py <eval_dir>             # metrics -> <eval_dir>/results.json, then wandb
    uv run analyze.py <eval_dir> --no-wandb  # metrics only

Every test is run `repeats` times. Each metric is reported as the mean over repeats, with its
std and standard error (std / sqrt(n)); a difference between two policies is only meaningful
if it is well above their combined standard error.

One wandb run per evaluation, named <policy>__<suite>, grouped by suite hash so only runs of
identical tests (and repeat counts) sit together.
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

import metrics

WANDB_ENTITY = 'kieranparanjpe-mcgill-university'
WANDB_PROJECT = 'humanoidx-gz-eval'

HEADLINE = (
    'err_vx_steady', 'err_vy_steady', 'err_wz_steady', 'response_time',
    'heading_drift_deg_per_m', 'stand_sway_cm_per_s', 'torso_pitch_deg', 'stance_width',
    'foot_slip_cm_per_step', 'cost_of_transport', 'joint_jerk_rms', 'falls',
)


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _stats(samples):
    """{metric: [value per repeat]} -> {metric: {mean, std, sem, n}}."""
    out = {}
    for k, vals in samples.items():
        vals = [v for v in vals if _finite(v)]
        if not vals:
            continue
        n = len(vals)
        std = float(np.std(vals, ddof=1)) if n > 1 else float('nan')
        out[k] = {'mean': float(np.mean(vals)), 'std': std,
                  'sem': std / math.sqrt(n) if n > 1 else float('nan'), 'n': n}
    return out


def _mean_series(series_list):
    """Average per-repeat time series on their shared 25 Hz grid (shorter ones, e.g. after a
    fall, simply stop contributing)."""
    longest = max(series_list, key=lambda s: len(s['t']))
    out = {'t': longest['t']}
    for k in longest:
        if k == 't':
            continue
        stack = np.full((len(series_list), len(longest['t'])), np.nan)
        for i, s in enumerate(series_list):
            stack[i, :len(s[k])] = s[k]
        out[k] = np.nanmean(stack, axis=0)
    return out


def analyze(eval_dir):
    with open(os.path.join(eval_dir, 'meta.json')) as f:
        meta = json.load(f)
    mass = meta['robot']['mass']

    per_rep = defaultdict(dict)          # repeat -> test -> metrics
    series = defaultdict(list)           # test -> [timeseries per repeat]
    status = defaultdict(list)           # test -> [status per repeat]
    for entry in meta['tests']:
        name, rep = entry['name'], entry.get('repeat', 1)
        status[name].append(entry['status'])
        if entry['status'] == 'aborted':
            continue
        with np.load(os.path.join(eval_dir, entry['file'])) as d:
            per_rep[rep][name], td = metrics.analyze_test(d, entry, mass)
        series[name].append(metrics.timeseries(td))

    tests = {}
    for name in status:
        samples = defaultdict(list)
        for rep in per_rep.values():
            for k, v in rep.get(name, {}).items():
                samples[k].append(v)
        tests[name] = _stats(samples)

    # Overall score per repeat, then its spread over repeats.
    overall_samples = defaultdict(list)
    for rep in per_rep.values():
        for k, v in metrics.overall(rep).items():
            overall_samples[k].append(v)

    results = {
        'policy': meta['policy']['name'],
        'suite': meta['suite'],
        'repeats': len(per_rep),
        'status': dict(status),
        'tests': tests,
        'overall': _stats(overall_samples),
    }
    with open(os.path.join(eval_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return meta, results, {k: _mean_series(v) for k, v in series.items()}


def _fmt(s):
    return f"{s['mean']:.3g} ± {s['sem']:.2g}" if _finite(s['sem']) else f"{s['mean']:.3g}"


def print_summary(results):
    print(f"{results['policy']}  suite {results['suite']['name']} ({results['suite']['hash']}), "
          f"{results['repeats']} repeats  (mean ± standard error)")
    for name, st in results['status'].items():
        m = results['tests'][name]
        bits = ', '.join(f'{k}={_fmt(m[k])}' for k in ('err_vx_steady', 'err_vy_steady',
                                                        'err_wz_steady') if k in m)
        fell = st.count('fell')
        print(f'  {name:14s} {"fell " + str(fell) + "/" + str(len(st)) if fell else "ok":9s} {bits}')
    print('  overall:')
    for k in HEADLINE:
        if k in results['overall']:
            print(f"    {k:26s} {_fmt(results['overall'][k])}")


def upload(eval_dir, meta, results, series, entity, project):
    import wandb

    suite = meta['suite']
    aborted = sorted({n for n, st in results['status'].items() if 'aborted' in st})
    fell = sorted({n for n, st in results['status'].items() if 'fell' in st})
    config = {k: v for k, v in meta.items() if k != 'tests'}
    config['repeats'] = results['repeats']
    config['test_status'] = results['status']

    run = wandb.init(
        entity=entity, project=project,
        name=f"{results['policy']}__{suite['name']}",
        group=suite['hash'],
        job_type='gz-eval',
        tags=(['valid'] if not aborted and meta.get('finished') else ['invalid'])
        + [f'aborted:{n}' for n in aborted] + [f'fell:{n}' for n in fell],
        config=config,
    )
    summary = {}
    for prefix, block in [*((n, m) for n, m in results['tests'].items()),
                          ('overall', results['overall'])]:
        for k, s in block.items():
            summary[f'{prefix}/{k}'] = s['mean']
            if _finite(s['std']):
                summary[f'{prefix}/{k}_std'] = s['std']
    run.summary.update(summary)

    # Per-test time series (mean over repeats), each on its own time axis.
    for name, s in series.items():
        run.define_metric(f'{name}/t')
        run.define_metric(f'{name}/ts/*', step_metric=f'{name}/t')
        keys = [k for k in s if k != 't']
        for i, t in enumerate(s['t']):
            run.log({f'{name}/t': float(t), **{f'{name}/ts/{k}': float(s[k][i]) for k in keys}})

    run.save(os.path.join(eval_dir, 'results.json'), base_path=eval_dir, policy='now')
    run.save(os.path.join(eval_dir, 'meta.json'), base_path=eval_dir, policy='now')
    run.finish()
    return run.url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('eval_dir')
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--entity', default=WANDB_ENTITY)
    ap.add_argument('--project', default=WANDB_PROJECT)
    args = ap.parse_args()

    meta, results, series = analyze(args.eval_dir)
    print_summary(results)
    print(f"-> {os.path.join(args.eval_dir, 'results.json')}")
    if not args.no_wandb:
        print(f'wandb: {upload(args.eval_dir, meta, results, series, args.entity, args.project)}')


if __name__ == '__main__':
    main()
