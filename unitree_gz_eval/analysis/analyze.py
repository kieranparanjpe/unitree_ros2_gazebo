"""Score one evaluation directory and (optionally) upload it to wandb.

    uv run analyze.py <eval_dir>             # metrics -> <eval_dir>/results.json, then wandb
    uv run analyze.py <eval_dir> --no-wandb  # metrics only

One wandb run per evaluation, named <policy>__<suite>, grouped by suite hash so only runs of
identical tests sit together.
"""
import argparse
import json
import math
import os

import numpy as np

import metrics

WANDB_ENTITY = 'kieranparanjpe-mcgill-university'
WANDB_PROJECT = 'humanoidx-gz-eval'

HEADLINE = (
    'err_vx_steady', 'err_vy_steady', 'err_wz_steady', 'response_time',
    'heading_drift_deg_per_m', 'stand_sway_cm_per_s', 'torso_pitch_deg', 'stance_width',
    'foot_slip_cm_per_step', 'cost_of_transport', 'joint_jerk_rms', 'falls',
)


def analyze(eval_dir):
    with open(os.path.join(eval_dir, 'meta.json')) as f:
        meta = json.load(f)
    mass = meta['robot']['mass']
    per_test, series = {}, {}
    for entry in meta['tests']:
        name = entry['name']
        if entry['status'] == 'aborted':
            per_test[name] = {}
            continue
        with np.load(os.path.join(eval_dir, entry['file'])) as d:
            per_test[name], td = metrics.analyze_test(d, entry, mass)
        series[name] = metrics.timeseries(td)
    results = {
        'policy': meta['policy']['name'],
        'suite': meta['suite'],
        'status': {e['name']: e['status'] for e in meta['tests']},
        'tests': per_test,
        'overall': metrics.overall({k: v for k, v in per_test.items() if v}),
    }
    with open(os.path.join(eval_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_json_float)
    return meta, results, series


def _json_float(x):
    return float(x)


def print_summary(results):
    print(f"{results['policy']}  suite {results['suite']['name']} ({results['suite']['hash']})")
    for name, status in results['status'].items():
        m = results['tests'][name]
        bits = ', '.join(f'{k}={m[k]:.3g}' for k in ('err_vx_steady', 'err_vy_steady',
                                                       'err_wz_steady') if _finite(m.get(k)))
        print(f'  {name:14s} {status:8s} {bits}')
    print('  overall:')
    for k in HEADLINE:
        v = results['overall'].get(k)
        if _finite(v):
            print(f'    {k:26s} {v:.3g}')


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def upload(eval_dir, meta, results, series, entity, project):
    import wandb

    suite = meta['suite']
    aborted = [n for n, s in results['status'].items() if s == 'aborted']
    fell = [n for n, s in results['status'].items() if s == 'fell']
    config = {k: v for k, v in meta.items() if k != 'tests'}
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
    for name, m in results['tests'].items():
        summary.update({f'{name}/{k}': v for k, v in m.items() if _finite(v)})
    summary.update({f'overall/{k}': v for k, v in results['overall'].items() if _finite(v)})
    run.summary.update(summary)

    # Per-test time series, each on its own time axis.
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
