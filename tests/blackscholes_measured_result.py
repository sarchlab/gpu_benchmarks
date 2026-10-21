#!/usr/bin/env python3
"""Check every measured Black-Scholes output using actual GPU-buffer faults."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

HOOK = r'''
#include <algorithm>
#include <limits>
static void blackscholes_test_fault(float* calls, float* puts, int n) {
    const char* kind = std::getenv("BLACKSCHOLES_TEST_KIND");
    if (!kind) return;
    const char* channel = std::getenv("BLACKSCHOLES_TEST_CHANNEL");
    const char* location = std::getenv("BLACKSCHOLES_TEST_LOCATION");
    if (!channel || !location || n < 1) std::abort();
    const int component = std::atoi(channel), selection = std::atoi(location);
    if (component < 0 || component > 2 || selection < 0 || selection > 4) std::abort();
    const int positions[] = {0, std::min(9999, n-1), std::min(10000, n-1), n/2, n-1};
    const bool many = !std::strcmp(kind, "many");
    const int count = many ? 12 : 1;
    if (count > n) std::abort();
    for (int i = 0; i < count; ++i) {
        const int at = many ? i : positions[selection];
        float* addresses[] = {calls+at, puts+at};
        for (int c = 0; c < 2; ++c) {
            if (component != 2 && c != component) continue;
            float value;
            if (!std::strcmp(kind, "nan")) value = std::numeric_limits<float>::quiet_NaN();
            else if (!std::strcmp(kind, "inf")) value = std::numeric_limits<float>::infinity();
            else if (!std::strcmp(kind, "negative_inf")) value = -std::numeric_limits<float>::infinity();
            else if (!std::strcmp(kind, "finite") || many) {
                HIP_CHECK(hipMemcpy(&value, addresses[c], sizeof(value), hipMemcpyDeviceToHost));
                value += 1000.0f;
            } else std::abort();
            HIP_CHECK(hipMemcpy(addresses[c], &value, sizeof(value), hipMemcpyHostToDevice));
        }
    }
    std::fprintf(stderr, "BLACKSCHOLES_TEST_INJECT %s channel=%d at=%d count=%d\n",
                 kind, component, positions[selection], count);
}
static void* blackscholes_test_alloc(size_t bytes, const char* name) {
    const char* failed = std::getenv("BLACKSCHOLES_TEST_ALLOCATION");
    if (failed && !std::strcmp(failed, name)) {
        std::fprintf(stderr, "BLACKSCHOLES_TEST_ALLOC_FAIL %s\n", name);
        return nullptr;
    }
    return std::malloc(bytes);
}
'''


def instrument(source):
    include = '#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n'
    download = '    HIP_CHECK(hipMemcpy(h_call, d_call, bytes, hipMemcpyDeviceToHost));'
    for anchor in (include, download):
        if source.count(anchor) != 1:
            raise ValueError(f'expected one production anchor {anchor!r}')
    source = source.replace(include, include+HOOK).replace(
        download, '    blackscholes_test_fault(d_call, d_put, N);\n'+download)
    for name in ('call', 'put'):
        anchor = f'float* ref_{name} = ' if name == 'call' else 'float* ref_put  = '
        anchor += '(float*)malloc((size_t)verifyN * sizeof(float));'
        if source.count(anchor) != 1:
            raise ValueError(f'expected one reference allocation {anchor!r}')
        source = source.replace(anchor, anchor.replace(
            'malloc((size_t)verifyN * sizeof(float))',
            f'blackscholes_test_alloc((size_t)verifyN * sizeof(float), "{name}")'))
    return source


def output_text(value):
    return value.decode(errors='replace') if isinstance(value, bytes) else value or ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir', type=Path, required=True, help='New, nonexistent output directory')
    parser.add_argument('--platform', choices=('cuda', 'rocm'), required=True)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--largest', action='store_true', help='Also corrupt the last price at the largest declared size')
    args = parser.parse_args()
    args.source, args.work_dir = args.source.resolve(), args.work_dir.resolve()
    try:
        args.work_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('--work-dir must name a new, nonexistent directory')
    source_dir = args.source/'tier2/tango_blackscholes/hip'
    target = args.work_dir/'tier2/tango_blackscholes/hip'
    target.mkdir(parents=True)
    shutil.copytree(args.source/'common', args.work_dir/'common')
    for name in ('Makefile', 'params.json'):
        shutil.copy2(source_dir/name, target/name)
    source = (source_dir/'tango_blackscholes.hip').read_text()
    try:
        (target/'tango_blackscholes.hip').write_text(instrument(source))
    except ValueError as exc:
        raise ValueError(f'{source_dir / "tango_blackscholes.hip"}: {exc}') from exc
    with (args.work_dir/'build.log').open('w') as log:
        subprocess.run(['make', 'PLATFORM='+args.platform], cwd=target,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.compile_only:
        print('Black-Scholes measured-result regression compiled; GPU checks not run')
        return 0
    cases = [(n, None, 0, 0, None, 'false') for n in (1, 33, 1024, 9999, 10000, 10001, 32768)]
    cases += [(32768, None, 0, 0, None, 'true')]
    kinds = ('finite', 'nan', 'inf', 'negative_inf')
    cases += [(32768, kind, channel, at, None, 'false')
              for kind in kinds for channel in range(2) for at in range(5)]
    cases += [(32768, 'many', channel, 0, None, 'false') for channel in (0, 2)]
    cases += [(1, kind, channel, 4, None, 'false') for kind in kinds for channel in range(2)]
    cases += [(32768, None, 0, 0, name, 'false') for name in ('call', 'put')]
    if args.largest:
        largest = max(json.loads((source_dir/'params.json').read_text())['scaling_param']['values'])
        cases += [(largest, kind, channel, 4, None, 'true') for kind in kinds for channel in range(2)]
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'BLACKSCHOLES_TEST_'))}
    results = []

    def report(complete=False):
        data = dict(complete=complete, expected=len(cases), total=len(results),
                    passed=sum(row['passed'] for row in results),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(), results=results)
        (args.work_dir/'report.json').write_text(json.dumps(data, indent=2)+'\n')
        return data

    for n, kind, channel, at, allocation, warmer in cases:
        label = f'n{n}-{kind or "control"}-c{channel}-p{at}-alloc{allocation or "none"}-{warmer}'
        env = dict(clean, BENCH_PARAM_size=str(n), BENCH_PARAM_vector_add_warmup=warmer)
        if kind:
            env.update(BLACKSCHOLES_TEST_KIND=kind, BLACKSCHOLES_TEST_CHANNEL=str(channel), BLACKSCHOLES_TEST_LOCATION=str(at))
        if allocation:
            env['BLACKSCHOLES_TEST_ALLOCATION'] = allocation
        start = time.monotonic()
        try:
            p = subprocess.run([str(target/'tango_blackscholes')], env=env,
                               capture_output=True, text=True, timeout=60)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = output_text(exc.stdout), output_text(exc.stderr), 'timeout'
        elapsed = time.monotonic()-start
        (args.work_dir/(label+'.stdout')).write_text(stdout)
        (args.work_dir/(label+'.stderr')).write_text(stderr)
        errors = 12 if kind == 'many' else int(bool(kind))
        passed = code == (1 if kind or allocation else 0) and elapsed < 60
        passed &= ('BLACKSCHOLES_TEST_INJECT' in stderr) == bool(kind)
        passed &= ('BLACKSCHOLES_TEST_ALLOC_FAIL' in stderr) == bool(allocation)
        passed &= stderr.count('Mismatch at ') == (0 if allocation else min(errors, 10))
        if allocation:
            passed &= 'Verification unavailable: CPU reference allocation failed' in stderr
            passed &= 'Black-Scholes verification:' not in stderr
        else:
            passed &= f'Black-Scholes verification: {"FAIL" if kind else "PASS"} ({n} options, {2*n} prices, {errors} mismatching options)' in stderr
            if kind and kind != 'many':
                position = (0, min(9999, n-1), min(10000, n-1), n//2, n-1)[at]
                passed &= f'Mismatch at {position}:' in stderr
        try:
            kernel, summary = [json.loads(line) for line in stdout.splitlines()]
            passed &= kernel['type'] == 'kernel' and kernel['name'] == 'blackscholes_kernel' and kernel['iter'] == 0
            passed &= kernel['params'] == dict(size=n, block_size=256, verification='full_cpu_v1')
            passed &= all(math.isfinite(kernel[k]) and kernel[k] >= 0 for k in ('time_ms', 'host_time_ms'))
            passed &= summary['type'] == 'summary' and abs(summary['total_time_ms']-kernel['time_ms']) < .000003
        except (ValueError, KeyError, TypeError):
            passed = False
        results.append(dict(label=label, n=n, kind=kind, channel=channel, location=at,
                            allocation=allocation, warmer=warmer, exit_code=code,
                            seconds=elapsed, passed=bool(passed)))
        report()
        if not passed:
            print('FAIL:', label, 'see', args.work_dir, flush=True)
        if code == 'timeout':
            return 1
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} Black-Scholes measured-result cases passed")
    return 0 if final['passed'] == len(cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
