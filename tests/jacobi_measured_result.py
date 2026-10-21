#!/usr/bin/env python3
"""Qualify complete measured Jacobi grids and reject actual device corruption."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError('Jacobi instrumentation boundary changed: ' + anchor)
    return source.replace(anchor, replacement)


def instrument(source):
    source = replace_once(source, '    jacobi::Grids actual=jacobi::input(n);',
        '    if(getenv("JACOBI_TEST_STORAGE_FAILURE"))throw std::bad_alloc();\n'
        '    jacobi::Grids actual=jacobi::input(n);')
    anchor = '    // Verify both complete actual measured grids, including their distinct borders.'
    source = replace_once(source, anchor, r'''
    if(const char* array=getenv("JACOBI_TEST_ARRAY")) {
        const int k=atoi(array);const size_t i=strtoull(getenv("JACOBI_TEST_CELL"),nullptr,10);
        const size_t count=getenv("JACOBI_TEST_COUNT")?strtoull(getenv("JACOBI_TEST_COUNT"),nullptr,10):1;
        if(k<0 || k>1 || !count || i>=cells || count>cells-i)abort();
        float* pointers[]={d_a,d_b};
        for(size_t j=i;j<i+count;++j) {
            float value;HIP_CHECK(hipMemcpy(&value,pointers[k]+j,sizeof(value),hipMemcpyDeviceToHost));
            const char* mode=getenv("JACOBI_TEST_VALUE");
            value=strcmp(mode,"finite")==0?std::nextafter(value,INFINITY):strtof(mode,nullptr);
            HIP_CHECK(hipMemcpy(pointers[k]+j,&value,sizeof(value),hipMemcpyHostToDevice));
        }
        fprintf(stderr,"JACOBI_TEST_ARRAY %d cell %zu count %zu\n",k,i,count);
    }
''' + anchor)
    return replace_once(source, '    const jacobi::Grids expected=jacobi::reference(problem);',
        '    if(getenv("JACOBI_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();\n'
        '    const jacobi::Grids expected=jacobi::reference(problem);')


def build(repo, work, platform, hooks):
    work.mkdir(parents=True)
    shutil.copytree(repo / 'common', work / 'common')
    folder = work / 'tier2/polybench_jacobi2d/hip'
    folder.mkdir(parents=True)
    for name in ['Makefile', 'jacobi_reference.h', 'params.json']:
        shutil.copy2(repo / 'tier2/polybench_jacobi2d/hip' / name, folder / name)
    source = (repo / 'tier2/polybench_jacobi2d/hip/polybench_jacobi2d.hip').read_text()
    (folder / 'polybench_jacobi2d.hip').write_text(instrument(source) if hooks else source)
    with (work / 'build.log').open('w') as log:
        subprocess.run(['make', 'PLATFORM=' + platform], cwd=folder, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=120)
    return folder / 'polybench_jacobi2d'


def check_output(stdout, stderr, code, n, steps, fault):
    if 'JACOBI_TEST_STORAGE_FAILURE' in fault:
        assert code == 1 and not stdout and 'FAIL: Jacobi setup/reference unavailable:' in stderr
        return
    rows = [json.loads(line) for line in stdout.splitlines()]
    assert len(rows) == 3 and rows[-1]['type'] == 'summary'
    grid = (n - 2 + 15) // 16
    for row, name in zip(rows, ['jacobi2d_kernel', 'jacobi2d_copy_kernel']):
        assert row['type'] == 'kernel' and row['name'] == name and row['iter'] == 0
        want = dict(n=n, tsteps=steps, block_x=16, block_y=16, grid_x=grid, grid_y=grid,
                    threads_per_launch=grid*grid*256, interior_cells=(n-2)**2,
                    kernel_launches=steps, updated_cells=(n-2)**2*steps,
                    stencil_sweeps_per_step=1, copy_sweeps_per_step=1, precision='fp32',
                    boundary='fixed_initial_values', input='polybench_gpu_jacobi_v1',
                    algorithm='polybench_gpu_jacobi_copyback_v1', verification='full_cpu_exact_v1')
        assert row['params'] == want, (row['params'], want)
        assert math.isfinite(row['time_ms']) and row['time_ms'] > 0
        assert math.isfinite(row['host_time_ms']) and row['host_time_ms'] >= 0
    assert abs(sum(row['time_ms'] for row in rows[:-1]) - rows[-1]['total_time_ms']) < 1.6e-6
    if 'JACOBI_TEST_REFERENCE_FAILURE' in fault:
        assert code == 1 and 'FAIL: Jacobi setup/reference unavailable:' in stderr and 'PASS (verified' not in stderr
        return
    count = int(fault.get('JACOBI_TEST_COUNT', '1')) if fault else 0
    assert code == (1 if fault else 0), stderr
    assert f'{"FAIL" if fault else "PASS"} (verified {2*n*n} values across both complete measured grids; {count} mismatches)' in stderr, stderr
    assert stderr.count('Mismatch ') == min(count, 10)
    if fault:
        assert 'JACOBI_TEST_ARRAY ' + fault['JACOBI_TEST_ARRAY'] in stderr
        assert 'GPU=' in stderr and 'CPU=' in stderr
    assert 'HIP error' not in stderr


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['cuda', 'rocm'], default='cuda')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    hooks = build(repo, work / 'instrumented', args.platform, True)
    if args.compile_only:
        print('Jacobi full-grid and storage hooks compiled; GPU tests not executed')
        return
    production = build(repo, work / 'production', args.platform, False)
    manifest = json.loads((repo / 'tier2/polybench_jacobi2d/hip/params.json').read_text())
    sizes = manifest['scaling_param']['values']
    timesteps = next(p['values'] for p in manifest['non_scaling_params'] if p['name'] == 'tsteps')
    cases = [(n, t, {}) for n in sizes for t in timesteps]
    cases += [(n, t, {}) for n, t in [(3, 1), (3, 2), (17, 3), (65, 1), (65, 20)]]
    n = 65
    for k in range(2):
        for i in [0, 1, n-1, n, n+1, n*n//2, n*(n-1), n*n-1]:
            for value in ['finite', 'nan', 'inf', '-inf']:
                cases.append((n, 3, dict(JACOBI_TEST_ARRAY=str(k), JACOBI_TEST_CELL=str(i), JACOBI_TEST_VALUE=value)))
    cases += [(65, 3, dict(JACOBI_TEST_ARRAY='0', JACOBI_TEST_CELL='128', JACOBI_TEST_VALUE='finite', JACOBI_TEST_COUNT='12'))]
    cases += [(max(sizes), max(timesteps), dict(JACOBI_TEST_ARRAY=str(k), JACOBI_TEST_CELL=str(max(sizes)**2-1), JACOBI_TEST_VALUE='finite')) for k in range(2)]
    cases += [(65, 3, dict(JACOBI_TEST_STORAGE_FAILURE='1')), (65, 3, dict(JACOBI_TEST_REFERENCE_FAILURE='1'))]
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'JACOBI_TEST_'))}
    report = []
    for index, (n, steps, fault) in enumerate(cases):
        label = f'{index}-{n}-{steps}'
        start = time.monotonic()
        env = dict(clean, BENCH_PARAM_n=str(n), BENCH_PARAM_tsteps=str(steps),
                   BENCH_PARAM_vector_add_warmup='false' if fault else 'true', **fault)
        try:
            run = subprocess.run([str(hooks if fault else production)], env=env, text=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            (work / (label + '.stdout')).write_bytes(error.stdout or b'')
            (work / (label + '.stderr')).write_bytes(error.stderr or b'')
            raise
        elapsed = time.monotonic() - start
        (work / (label + '.stdout')).write_text(run.stdout)
        (work / (label + '.stderr')).write_text(run.stderr)
        check_output(run.stdout, run.stderr, run.returncode, n, steps, fault)
        report.append(dict(case=label, n=n, tsteps=steps, fault=fault, exit_code=run.returncode, wall_seconds=elapsed))
        (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(label, 'PASS', f'{elapsed:.3f}s', flush=True)
    print(f'{len(cases)} production/control/measured-grid/storage cases pass')


if __name__ == '__main__':
    main()
