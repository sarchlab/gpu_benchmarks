#!/usr/bin/env python3
"""Check actual Gram-Schmidt/MG reset launches and event order with NVIDIA nsys.

Requires both built binaries and a new work directory. Retains capture evidence;
profiled times are functional evidence, not performance measurements.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def expected_kernels(kind, params):
    sequence = []
    if kind == 'gram':
        for k in range(params['n']):
            sequence += ['gram_norm_zero', 'gram_norm', 'gram_norm_finish', 'gram_normalize']
            if k + 1 < params['n']:
                sequence.append('gram_project')
    else:
        levels = params['grid_size'].bit_length() - 2  # N, N/2, ..., 4
        smooth = ['smooth_kernel'] * params['num_smooth_steps']
        cycle = (smooth + ['residual_kernel', 'restrict_kernel', 'coarse_zero']) * (levels - 1)
        cycle += smooth * 2
        cycle += (['prolong_kernel'] + smooth) * (levels - 1)
        sequence = cycle * params['num_vcycles']
    return sequence


def check_capture(database, stdout, kind, params, warmer):
    with sqlite3.connect(database) as db:
        kernels = db.execute(
            'SELECT s.value,k.start,k.end,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k '
            'JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
        runtime = db.execute(
            'SELECT s.value,r.start,r.end,r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
            'JOIN StringIds s ON s.id=r.nameId ORDER BY r.start').fetchall()
    expected = expected_kernels(kind, params)
    names = [k[0].split('(')[0] for k in kernels]
    first = names.index(expected[0])
    assert all(n == 'bench_warmer_kernel' for n in names[:first]), names
    assert bool(first) == warmer, names
    verification = [] if kind == 'gram' else ['residual_kernel', 'norm_sq_kernel', 'norm_sq_kernel']
    assert names[first:] == expected + verification, names
    timed = kernels[first:first + len(expected)]
    launches = {r[3]: r for r in runtime}
    events = [r for r in runtime if r[0].startswith('cudaEventRecord_')]
    syncs = [r for r in runtime if r[0].startswith('cudaEventSynchronize_')]
    assert len(events) == 2 * len(timed) and len(syncs) == len(timed), (len(events), len(syncs))
    if warmer:
        assert kernels[first - 1][2] <= events[0][1], 'warmer incomplete at measurement entry'
    for i, kernel in enumerate(timed):
        assert events[2 * i][1] < launches[kernel[3]][1] < events[2 * i + 1][1], (i, kernel)
        assert syncs[i][2] >= kernel[2], (i, kernel)
        if i:
            assert timed[i - 1][2] <= events[2 * i][1], 'previous stage incomplete'
    # Initial setup memsets remain outside the pass; no runtime reset/copy may
    # replace or supplement the explicit algorithm kernels between event pairs.
    transfers = [r for r in runtime if 'cudaMemset' in r[0] or 'cudaMemcpy' in r[0]]
    assert not any(events[0][1] <= r[1] <= syncs[-1][2] for r in transfers), transfers
    output = [json.loads(line) for line in stdout.splitlines() if line.startswith('{"type":')]
    records = output[:-1]
    assert output[-1]['type'] == 'summary', output
    # gram_project has a zero accumulator record even for the manual N=1 case.
    record_names = (['gram_norm_zero', 'gram_norm', 'gram_norm_finish', 'gram_normalize', 'gram_project']
                    if kind == 'gram' else
                    ['smooth_kernel', 'restrict_kernel', 'prolong_kernel', 'residual_kernel', 'coarse_zero'])
    assert [r['name'] for r in records] == record_names, records
    for r in records:
        assert r['type'] == 'kernel' and r['iter'] == 0 and r['params'] == params, r
        assert all(math.isfinite(r[k]) and r[k] >= 0 for k in ('time_ms', 'host_time_ms')), r
    total = output[-1]['total_time_ms']
    assert math.isfinite(total) and abs(total - sum(r['time_ms'] for r in records)) <= 3.1e-6, output
    return dict(counts=dict(Counter(expected)), event_records=len(events),
                verification_launches=len(verification), timed_launches=len(timed))


def main():
    if not __debug__:
        raise SystemExit('Run without -O/PYTHONOPTIMIZE; this check requires assertions.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gram-binary', type=Path, required=True)
    parser.add_argument('--mg-binary', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    binaries = dict(gram=args.gram_binary.resolve(), mg=args.mg_binary.resolve())
    if not all(p.is_file() for p in binaries.values()):
        parser.error('both binaries must exist')
    args.work_dir.mkdir(parents=True, exist_ok=False)
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    cases = [('gram', dict(m=17, n=1)), ('gram', dict(m=512, n=64)),
             ('mg', dict(grid_size=8, block_size=256, num_vcycles=2, num_smooth_steps=1)),
             ('mg', dict(grid_size=64, block_size=256, num_vcycles=5, num_smooth_steps=3))]
    results = []
    for kind, params in cases:
        for warmer in (False, True):
            label = f'{kind}-{next(iter(params.values()))}-warmer-{str(warmer).lower()}'
            prefix = args.work_dir.resolve() / label
            env = {**clean_env, **{'BENCH_PARAM_' + k: str(v) for k, v in params.items()},
                   'BENCH_PARAM_vector_add_warmup': str(warmer).lower()}
            try:
                run = subprocess.run(['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                                      '--output=' + str(prefix), str(binaries[kind])],
                                     env=env, capture_output=True, text=True, timeout=180)
                prefix.with_suffix('.stdout').write_text(run.stdout)
                prefix.with_suffix('.stderr').write_text(run.stderr)
                assert run.returncode == 0, (run.returncode, run.stderr)
                database = prefix.with_suffix('.sqlite')
                export = subprocess.run(['nsys', 'export', '--type=sqlite', '--output=' + str(database),
                                         str(prefix.with_suffix('.nsys-rep'))],
                                        capture_output=True, text=True, timeout=60)
                prefix.with_suffix('.export.log').write_text(export.stdout + export.stderr)
                assert export.returncode == 0, (export.returncode, export.stderr)
                result = check_capture(database, run.stdout, kind, params, warmer)
                result.update(case=label, passed=True)
            except (AssertionError, KeyError, ValueError, IndexError, sqlite3.Error) as error:
                result = dict(case=label, passed=False, error=str(error))
            results.append(result)
            (args.work_dir / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
            print(label + ': ' + ('PASS' if result['passed'] else 'FAIL ' + result['error']), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
