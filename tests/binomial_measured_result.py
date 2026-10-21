#!/usr/bin/env python3
"""Verify measured binomial prices, including faults beyond the former ten-price prefix."""
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
static void binomial_test_fault(float* prices, int n) {
    const char* kind = std::getenv("BINOMIAL_TEST_KIND");
    if (!kind) return;
    const char* location = std::getenv("BINOMIAL_TEST_LOCATION");
    if (!location || n < 1) std::abort();
    const int selection = std::atoi(location);
    if (selection < 0 || selection > 4) std::abort();
    const int positions[] = {0, std::min(9,n-1), std::min(10,n-1), n/2, n-1};
    const bool many = !std::strcmp(kind,"many");
    const int count = many ? 12 : 1;
    if (count > n) std::abort();
    for (int i=0;i<count;++i) {
        const int at = many ? i : positions[selection];
        float value;
        if (!std::strcmp(kind,"nan")) value = std::numeric_limits<float>::quiet_NaN();
        else if (!std::strcmp(kind,"inf")) value = std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"negative_inf")) value = -std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"finite") || many) {
            HIP_CHECK(hipMemcpy(&value,prices+at,sizeof(value),hipMemcpyDeviceToHost));
            value += 1000.f;
        } else std::abort();
        HIP_CHECK(hipMemcpy(prices+at,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"BINOMIAL_TEST_INJECT %s at=%d count=%d\n",kind,positions[selection],count);
}
'''


def instrument(source):
    include = '#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n'
    download = '    HIP_CHECK(hipMemcpy(h_prices, d_prices, priceBytes, hipMemcpyDeviceToHost));'
    for anchor in (include, download):
        if source.count(anchor) != 1:
            raise ValueError(f'expected one production anchor {anchor!r}')
    return source.replace(include, include+HOOK).replace(
        download, '    binomial_test_fault(d_prices, numOptions);\n'+download)


def instrument_reference(source):
    allocation = '    std::vector<float> result(count);'
    thread = '        try { threads.emplace_back(work, worker); }'
    workers = 'unsigned workers = std::thread::hardware_concurrency()'
    for anchor in (allocation, thread, workers):
        if source.count(anchor) != 1:
            raise ValueError(f'expected one reference anchor {anchor!r}')
    source = '#include <cstdio>\n#include <cstdlib>\n#include <new>\n'+source
    source = source.replace(workers,
        'unsigned workers = std::getenv("BINOMIAL_TEST_THREADS") ? 16u : std::thread::hardware_concurrency()')
    source = source.replace(allocation, '''    if (std::getenv("BINOMIAL_TEST_ALLOCATION")) {
        std::fprintf(stderr,"BINOMIAL_TEST_ALLOC_FAIL\\n");
        throw std::bad_alloc();
    }
'''+allocation)
    return source.replace(thread, '''        try {
            if (std::getenv("BINOMIAL_TEST_THREADS")) {
                std::fprintf(stderr,"BINOMIAL_TEST_THREAD_FAIL %u\\n",worker);
                throw std::runtime_error("injected thread creation failure");
            }
            threads.emplace_back(work, worker);
        }''')


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
    source_dir = args.source/'tier2/tango_binomial_options/hip'
    target = args.work_dir/'tier2/tango_binomial_options/hip'
    target.mkdir(parents=True)
    shutil.copytree(args.source/'common', args.work_dir/'common')
    for name in ('Makefile', 'params.json'):
        shutil.copy2(source_dir/name, target/name)
    source = (source_dir/'tango_binomial_options.hip').read_text()
    reference = (source_dir/'cpu_reference.h').read_text()
    (target/'tango_binomial_options.hip').write_text(instrument(source))
    (target/'cpu_reference.h').write_text(instrument_reference(reference))
    with (args.work_dir/'build.log').open('w') as log:
        subprocess.run(['make', 'PLATFORM='+args.platform], cwd=target,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.compile_only:
        print('Binomial measured-result regression compiled; GPU checks not run')
        return 0
    cases = [(n, None, 0, None, 'false') for n in (1, 9, 10, 11, 33, 512, 8193)]
    cases += [(33, None, 0, None, 'true')]
    kinds = ('finite', 'nan', 'inf', 'negative_inf')
    cases += [(33, kind, at, None, 'false') for kind in kinds for at in range(5)]
    cases += [(33, 'many', 0, None, 'false')]
    cases += [(1, kind, 4, None, 'false') for kind in kinds]
    cases += [(33, None, 0, failure, 'false') for failure in ('allocation', 'threads')]
    if args.largest:
        largest = max(json.loads((source_dir/'params.json').read_text())['scaling_param']['values'])
        cases += [(largest, kind, 4, None, 'true') for kind in kinds]
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'BINOMIAL_TEST_'))}
    results = []

    def report(complete=False):
        data = dict(complete=complete, expected=len(cases), total=len(results),
                    passed=sum(row['passed'] for row in results),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    reference_sha256=hashlib.sha256(reference.encode()).hexdigest(), results=results)
        (args.work_dir/'report.json').write_text(json.dumps(data, indent=2)+'\n')
        return data

    for n, kind, at, failure, warmer in cases:
        label = f'n{n}-{kind or "control"}-p{at}-{failure or "normal"}-{warmer}'
        env = dict(clean, BENCH_PARAM_options=str(n), BENCH_PARAM_vector_add_warmup=warmer)
        if kind:
            env.update(BINOMIAL_TEST_KIND=kind, BINOMIAL_TEST_LOCATION=str(at))
        if failure:
            env['BINOMIAL_TEST_'+failure.upper()] = '1'
        start = time.monotonic()
        try:
            p = subprocess.run([str(target/'tango_binomial_options')], env=env,
                               capture_output=True, text=True, timeout=60)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = output_text(exc.stdout), output_text(exc.stderr), 'timeout'
        elapsed = time.monotonic()-start
        (args.work_dir/(label+'.stdout')).write_text(stdout)
        (args.work_dir/(label+'.stderr')).write_text(stderr)
        errors = 12 if kind == 'many' else int(bool(kind))
        unavailable = failure == 'allocation'
        passed = code == (1 if kind or unavailable else 0) and elapsed < 60
        passed &= ('BINOMIAL_TEST_INJECT' in stderr) == bool(kind)
        passed &= ('BINOMIAL_TEST_ALLOC_FAIL' in stderr) == unavailable
        passed &= stderr.count('BINOMIAL_TEST_THREAD_FAIL') == (15 if failure == 'threads' else 0)
        passed &= stderr.count('Mismatch at option ') == (0 if unavailable else min(errors, 10))
        if unavailable:
            passed &= 'Verification unavailable: CPU reference failed' in stderr
            passed &= 'Binomial verification:' not in stderr
        else:
            passed &= f'Binomial verification: {"FAIL" if kind else "PASS"} ({n} prices, {errors} mismatches)' in stderr
            if kind and kind != 'many':
                position = (0, min(9,n-1), min(10,n-1), n//2, n-1)[at]
                passed &= f'Mismatch at option {position}:' in stderr
        try:
            kernel, summary = [json.loads(line) for line in stdout.splitlines()]
            passed &= kernel['type'] == 'kernel' and kernel['name'] == 'binomial_kernel' and kernel['iter'] == 0
            passed &= kernel['params'] == dict(options=n, steps=1024, verification='full_cpu_v1')
            passed &= all(math.isfinite(kernel[k]) and kernel[k] >= 0 for k in ('time_ms', 'host_time_ms'))
            passed &= summary['type'] == 'summary' and abs(summary['total_time_ms']-kernel['time_ms']) < .000003
        except (ValueError, KeyError, TypeError):
            passed = False
        results.append(dict(label=label, n=n, kind=kind, location=at, failure=failure,
                            warmer=warmer, exit_code=code, seconds=elapsed, passed=bool(passed)))
        report()
        if not passed:
            print('FAIL:', label, 'see', args.work_dir, flush=True)
        if code == 'timeout':
            return 1
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} binomial measured-result cases passed")
    return 0 if final['passed'] == len(cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
