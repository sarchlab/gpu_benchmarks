#!/usr/bin/env python3
"""Check cache_latency's actual CUDA launch/event order with Nsight Systems.

Requires an NVIDIA GPU, nsys, and a compiled benchmark. Captures are retained in
an explicitly new work directory. Profiled timings are not performance results.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def check_capture(database, output, array_bytes, warmer):
    with sqlite3.connect(database) as db:
        kernels = db.execute(
            "SELECT s.value,k.start,k.end,k.correlationId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"
        ).fetchall()
        api = db.execute(
            "SELECT s.value,r.start,r.end,r.correlationId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"
        ).fetchall()
    names = [k[0].split("(")[0] for k in kernels]
    assert names[-2:] == ["pointer_chase_kernel"] * 2, names
    warmers = names[:-2]
    assert all(n == "bench_warmer_kernel" for n in warmers), names
    assert bool(warmers) == warmer, names
    primer, timed = kernels[-2:]
    assert primer[2] <= timed[1], "primer overlaps timed chase"

    # Correlation IDs associate actual GPU kernels with their runtime launches.
    launches = {r[3]: r for r in api}
    primer_launch, timed_launch = launches[primer[3]], launches[timed[3]]
    records = [r for r in api if r[0].startswith("cudaEventRecord_")]
    assert len(records) == 2, records
    assert primer_launch[1] < records[0][1] < timed_launch[1] < records[1][1], (primer_launch, records, timed_launch)
    assert primer[2] <= records[0][1], "start event submitted before primer completed"
    syncs = [r for r in api if r[0].startswith("cudaEventSynchronize_")]
    assert len(syncs) == 1 and syncs[0][2] >= timed[2], "timed completion not synchronized"

    # nsys can add its own status text to stdout after the application's JSON.
    events = [json.loads(line) for line in output.splitlines() if line.startswith('{"type":')]
    assert len(events) == 2 and [e['type'] for e in events] == ['kernel', 'summary'], events
    event, summary = events
    assert event['name'] == 'pointer_chase_kernel' and event['iter'] == 0, event
    params = event['params']
    assert params['cache_entry'] == 'primed' and params['primer_launches'] == 1, params
    assert params['array_bytes'] == array_bytes and params['cacheline_bytes'] == 64, params
    assert params['num_accesses'] == 131072, params
    assert params['warmup_accesses'] == min(array_bytes // 64, (256 << 20) // 64), params
    assert params['measure_laps'] == 4 and params['rng_seed'] == 42, params
    assert math.isfinite(event['time_ms']) and event['time_ms'] >= 0, event
    assert math.isfinite(event['host_time_ms']) and event['host_time_ms'] >= 0, event
    assert summary['total_time_ms'] == event['time_ms'], events
    return dict(order=names, kernels=kernels, event_records=records, params=params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    assert binary.is_file(), binary
    args.work_dir.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    env.update(BENCH_PARAM_cacheline_bytes='64', BENCH_PARAM_measure_laps='4',
               BENCH_PARAM_rng_seed='42', BENCH_PARAM_num_accesses='131072')
    results = []
    for array_bytes in (4096, (256 << 20) + 64):
        for warmer in (False, True):
            label = f'{array_bytes}-warmer-{str(warmer).lower()}'
            prefix = args.work_dir.resolve() / label
            case_env = {**env, 'BENCH_PARAM_array_bytes': str(array_bytes),
                        'BENCH_PARAM_vector_add_warmup': str(warmer).lower()}
            run = subprocess.run([
                'nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                '--output=' + str(prefix), str(binary)
            ], env=case_env, capture_output=True, text=True, timeout=120)
            prefix.with_suffix('.stdout').write_text(run.stdout)
            prefix.with_suffix('.stderr').write_text(run.stderr)
            assert run.returncode == 0, (label, run.stderr)
            database = prefix.with_suffix('.sqlite')
            export = subprocess.run([
                'nsys', 'export', '--type=sqlite', '--output=' + str(database),
                str(prefix.with_suffix('.nsys-rep'))
            ], capture_output=True, text=True, timeout=60)
            prefix.with_suffix('.export.log').write_text(export.stdout + export.stderr)
            assert export.returncode == 0, (label, export.stderr)
            try:
                result = check_capture(database, run.stdout, array_bytes, warmer)
                result.update(case=label, passed=True)
            except (AssertionError, KeyError, sqlite3.Error) as error:
                result = dict(case=label, passed=False, error=str(error))
            results.append(result)
            print(f"{label}: {'PASS' if result['passed'] else 'FAIL ' + result['error']}", flush=True)
            (args.work_dir / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
