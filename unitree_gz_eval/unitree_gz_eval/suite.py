# Suite file -> an absolute per-test schedule. No ROS imports, so analysis can reuse the same
# definition of what was commanded when.
import hashlib
from dataclasses import dataclass, field

import yaml

_TOP_KEYS = {'lead', 'tail', 'cmd_rate', 'fall', 'reset_check', 'tests'}
_TEST_KEYS = {'name', 'lead', 'tail', 'steps'}
_STEP_KEYS = {'cmd', 'for', 'pushes'}
_PUSH_KEYS = {'at', 'dv'}


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    cmd: tuple


@dataclass(frozen=True)
class Push:
    t: float
    dv: tuple


@dataclass
class Test:
    name: str
    segments: list = field(default_factory=list)   # contiguous from 0 to duration
    pushes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.segments[-1].t1

    def cmd_at(self, t: float) -> tuple:
        for seg in self.segments:
            if t < seg.t1:
                return seg.cmd
        return self.segments[-1].cmd


@dataclass
class Suite:
    name: str
    hash: str
    cmd_rate: float
    fall: dict
    reset_check: dict
    tests: list


def _check_keys(where: str, d: dict, allowed: set) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f'{where}: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})')


def suite_hash(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _build_test(raw: dict, lead: float, tail: float) -> Test:
    _check_keys(f"test {raw.get('name')}", raw, _TEST_KEYS)
    test = Test(name=raw['name'])
    lead = raw.get('lead', lead)
    tail = raw.get('tail', tail)
    zero = (0.0, 0.0, 0.0)

    t = 0.0
    if lead > 0:
        test.segments.append(Segment(0.0, lead, zero))
        t = lead
    for i, step in enumerate(raw['steps']):
        where = f"test {test.name} step {i}"
        _check_keys(where, step, _STEP_KEYS)
        cmd = tuple(float(v) for v in step['cmd'])
        duration = float(step['for'])
        if len(cmd) != 3 or duration <= 0:
            raise ValueError(f'{where}: cmd needs 3 values and for must be > 0')
        for push in step.get('pushes', []):
            _check_keys(where, push, _PUSH_KEYS)
            dv = tuple(float(v) for v in push['dv'])
            if len(dv) != 2 or not 0 <= push['at'] < duration:
                raise ValueError(f'{where}: push needs dv [x, y] and 0 <= at < for')
            test.pushes.append(Push(t + float(push['at']), dv))
        test.segments.append(Segment(t, t + duration, cmd))
        t += duration
    if tail > 0:
        test.segments.append(Segment(t, t + tail, zero))
    return test


def load_suite(path: str) -> Suite:
    with open(path) as f:
        raw = yaml.safe_load(f)
    _check_keys('suite', raw, _TOP_KEYS)
    tests = [_build_test(t, float(raw['lead']), float(raw['tail'])) for t in raw['tests']]
    names = [t.name for t in tests]
    if len(set(names)) != len(names):
        raise ValueError('test names must be unique')
    return Suite(
        name=path.rsplit('/', 1)[-1].removesuffix('.yaml'),
        hash=suite_hash(path),
        cmd_rate=float(raw['cmd_rate']),
        fall=raw['fall'],
        reset_check=raw['reset_check'],
        tests=tests,
    )
