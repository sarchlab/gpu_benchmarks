#!/usr/bin/env python3
"""Verify Barnes-Hut's actual CUDA entry and complete timestep order with nsys.

Requires an NVIDIA GPU, Nsight Systems, a built binary and a new output folder.
Profiled times are functional evidence, not performance measurements.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def check_capture(database, stdout, bodies, steps, warmer):
    with sqlite3.connect(database) as db:
        kernels = db.execute(
            'SELECT s.value,k.start,k.end,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k '
            'JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
        runtime = db.execute(
            'SELECT s.value,r.start,r.end,r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
            'JOIN StringIds s ON s.id=r.nameId ORDER BY r.start').fetchall()
        copies = db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start').fetchall()
    names = [k[0].split('(')[0] for k in kernels]
    first = names.index('bh_force_kernel')
    assert all(n == 'bench_warmer_kernel' for n in names[:first]), names
    assert bool(first) == warmer, names
    # The existing verifier separately relaunches the initial force calculation.
    assert names[first:] == ['bh_force_kernel', 'integrate_kernel'] * steps + ['bh_force_kernel'], names
    timed = kernels[first:first + 2 * steps]
    launches = {r[3]: r for r in runtime}
    first_launch = launches[timed[0][3]]
    entry_copies = [c for c in copies if c[0] < timed[0][1]]
    # Five initial body uploads, two position downloads and five tree uploads.
    assert len(entry_copies) == 12, entry_copies
    if warmer:
        assert all(c[1] <= kernels[0][1] for c in entry_copies), 'initial body/tree transfers follow the warmer'
        assert kernels[first - 1][2] <= first_launch[1], 'warmer not complete before first launch'
    assert all(c[1] <= timed[0][1] for c in entry_copies), 'initial upload incomplete at force entry'
    events = [r for r in runtime if r[0].startswith('cudaEventRecord_')]
    syncs = [r for r in runtime if r[0].startswith('cudaEventSynchronize_')]
    assert len(events) == 4 * steps and len(syncs) == 2 * steps, (events, syncs)
    for i, kernel in enumerate(timed):
        assert events[2 * i][1] < launches[kernel[3]][1] < events[2 * i + 1][1], (i, kernel)
        assert syncs[i][2] >= kernel[2], (i, kernel)
        if i:
            assert timed[i - 1][2] <= events[2 * i][1], 'previous stage incomplete at next start event'
    events_out = [json.loads(l) for l in stdout.splitlines() if l.startswith('{"type":')]
    assert [e['type'] for e in events_out] == ['kernel', 'kernel', 'summary'], events_out
    assert [e['name'] for e in events_out[:2]] == ['bh_force_kernel', 'integrate_kernel'], events_out
    for event in events_out[:2]:
        assert event['iter'] == 0 and event['params'] == dict(bodies=bodies, timesteps=steps, theta=0.5), event
        assert all(math.isfinite(event[k]) and event[k] >= 0 for k in ('time_ms', 'host_time_ms')), event
    assert abs(events_out[-1]['total_time_ms'] - sum(e['time_ms'] for e in events_out[:2])) < 2e-6, events_out
    return dict(kernels=kernels, initial_copies=entry_copies, event_records=events,
                timed_launches=len(timed), verification_launches=1)


def main():
    if not __debug__:
        raise SystemExit('Run without -O/PYTHONOPTIMIZE; this check requires assertions.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error('binary does not exist')
    args.work_dir.mkdir(parents=True, exist_ok=False)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    results = []
    for bodies, steps in ((256, 1), (32768, 5)):
        for warmer in (False, True):
            label = f'{bodies}-{steps}-warmer-{str(warmer).lower()}'
            prefix = args.work_dir.resolve() / label
            env = {**base_env, 'BENCH_PARAM_bodies': str(bodies), 'BENCH_PARAM_timesteps': str(steps),
                   'BENCH_PARAM_theta': '0.5', 'BENCH_PARAM_vector_add_warmup': str(warmer).lower()}
            try:
                run = subprocess.run(['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                                      '--output=' + str(prefix), str(binary)], env=env,
                                     capture_output=True, text=True, timeout=180)
                prefix.with_suffix('.stdout').write_text(run.stdout)
                prefix.with_suffix('.stderr').write_text(run.stderr)
                assert run.returncode == 0, (label, run.returncode, run.stderr)
                database = prefix.with_suffix('.sqlite')
                export = subprocess.run(['nsys', 'export', '--type=sqlite', '--output=' + str(database),
                                         str(prefix.with_suffix('.nsys-rep'))], capture_output=True,
                                        text=True, timeout=60)
                prefix.with_suffix('.export.log').write_text(export.stdout + export.stderr)
                assert export.returncode == 0, (label, export.returncode, export.stderr)
                result = check_capture(database, run.stdout, bodies, steps, warmer)
                result.update(case=label, passed=True)
            except (AssertionError, KeyError, ValueError, IndexError, sqlite3.Error) as error:
                result = dict(case=label, passed=False, error=str(error))
            results.append(result)
            (args.work_dir / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
            print(label + ': ' + ('PASS' if result['passed'] else 'FAIL ' + result['error']), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
