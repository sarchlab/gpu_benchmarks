#!/usr/bin/env python3
"""Qualify NAS CG classes and reject faults in measured vectors and history."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

CLASSES = {1400: ('S', 7, 15, 10, 78148), 7000: ('W', 8, 15, 12, 508402),
           14000: ('A', 11, 15, 20, 1853104), 75000: ('B', 13, 75, 60, 13708072),
           150000: ('C', 15, 75, 110, 36121058)}
LAUNCHES = dict(cg_init_kernel=1, spmv_kernel=26, dot_product_kernel=54,
                axpy_kernel=25, direction_kernel=25, normalize_kernel=1)


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError('CG instrumentation boundary changed: '+anchor)
    return source.replace(anchor, replacement)


def instrument(source):
    source = replace_once(source, '    const nas_cg::Matrix matrix=nas_cg::make_matrix(c);',
        '    if (getenv("CG_TEST_STORAGE_FAILURE")) throw std::bad_alloc();\n'
        '    const nas_cg::Matrix matrix=nas_cg::make_matrix(c);')
    source = replace_once(source, '    // Download every actual final vector after the complete measured algorithm.', r'''
    if (const char* array=getenv("CG_TEST_ARRAY")) {
        const int k=atoi(array), i=atoi(getenv("CG_TEST_CELL"));
        const int count=getenv("CG_TEST_COUNT") ? atoi(getenv("CG_TEST_COUNT")) : 1;
        if(k<0 || k>6 || i<0 || count<1 || i+count>(k<5 ? n : c.iterations))abort();
        double* devices[]={d_x,d_z,d_p,d_q,d_r};
        for(int j=i;j<i+count;++j) {
            double value;
            if(k<5) HIP_CHECK(hipMemcpy(&value,devices[k]+j,sizeof(value),hipMemcpyDeviceToHost));
            else value=(k==5 ? actual.zeta : actual.residual)[j];
            const char* mode=getenv("CG_TEST_VALUE");
            value=strcmp(mode,"finite")==0 ? value+0.01 : strtod(mode,nullptr);
            if(k<5) HIP_CHECK(hipMemcpy(devices[k]+j,&value,sizeof(value),hipMemcpyHostToDevice));
            else (k==5 ? actual.zeta : actual.residual)[j]=value;
        }
        fprintf(stderr,"CG_TEST_ARRAY %d cell %d count %d\n",k,i,count);
    }
    // Download every actual final vector after the complete measured algorithm.''')
    source = replace_once(source, '    const nas_cg::Result expected=nas_cg::reference(matrix,c,workers);', r'''
    if(const char* path=getenv("CG_TEST_DUMP")) {
        FILE* file=fopen(path,"wb");if(!file)abort();
        for(const auto* v : {&actual.x,&actual.z,&actual.p,&actual.q,&actual.r,&actual.zeta,&actual.residual})
            if(fwrite(v->data(),sizeof(double),v->size(),file)!=v->size())abort();
        if(fclose(file))abort();
    }
    if(getenv("CG_TEST_REFERENCE_FAILURE")) throw std::bad_alloc();
    const nas_cg::Result expected=nas_cg::reference(matrix,c,workers);''')
    return source


def build(repo, work, platform, hooks):
    work.mkdir(parents=True)
    shutil.copytree(repo/'common', work/'common')
    folder = work/'tier2/npb_cg/hip'
    folder.mkdir(parents=True)
    for name in ['Makefile', 'nas_cg.h', 'params.json']:
        shutil.copy2(repo/'tier2/npb_cg/hip'/name, folder/name)
    source = (repo/'tier2/npb_cg/hip/npb_cg.hip').read_text()
    (folder/'npb_cg.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:
        subprocess.run(['make', 'PLATFORM='+platform], cwd=folder, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=120)
    return folder/'npb_cg'


def check_output(stdout, stderr, code, n, block, fault):
    if 'CG_TEST_STORAGE_FAILURE' in fault:
        assert code == 1 and not stdout and 'FAIL: CG setup/reference/arithmetic unavailable:' in stderr
        return
    rows = [json.loads(line) for line in stdout.splitlines()]
    assert len(rows) == 7 and rows[-1]['type'] == 'summary'
    name, nonzer, iterations, shift, nonzeros = CLASSES[n]
    blocks = (n+block-1)//block
    for row, (kernel, factor) in zip(rows, LAUNCHES.items()):
        assert row['type'] == 'kernel' and row['name'] == kernel and row['iter'] == 0
        p = row['params']
        expected = dict(matrix_size=n, block_size=block, num_blocks=blocks, actual_threads=blocks*block,
            npb_class=name, npb_nonzer=nonzer, outer_iterations=iterations, inner_iterations=25,
            shift=shift, rcond=0.1, matrix_nonzeros=p['matrix_nonzeros'], kernel_launches=factor*iterations,
            spmv_nonzero_products=p['matrix_nonzeros']*factor*iterations if kernel == 'spmv_kernel' else 0,
            input_seed=314159265, input_multiplier=1220703125, reference_workers=p['reference_workers'],
            precision='fp64', algorithm='npb_cg_inverse_power_v1', verification='full_cpu_nas_zeta_v1')
        assert p == expected
        assert p['matrix_nonzeros'] == rows[0]['params']['matrix_nonzeros']
        assert n <= p['matrix_nonzeros'] <= n*(nonzer+1)**2
        assert p['matrix_nonzeros'] == nonzeros
        assert (p['reference_workers'] == 1 if n < 75000 else 1 <= p['reference_workers'] <= 8)
        assert math.isfinite(row['time_ms']) and row['time_ms'] > 0
        assert math.isfinite(row['host_time_ms']) and row['host_time_ms'] >= 0
    assert abs(sum(r['time_ms'] for r in rows[:-1])-rows[-1]['total_time_ms']) < 4e-6
    if 'CG_TEST_REFERENCE_FAILURE' in fault:
        assert code == 1 and 'FAIL: CG setup/reference/arithmetic unavailable:' in stderr and 'PASS (verified' not in stderr
        return
    count = int(fault.get('CG_TEST_COUNT', '1')) if fault else 0
    assert code == (1 if fault else 0)
    assert f'{"FAIL" if fault else "PASS"} (verified {5*n} vector values and {2*iterations} outer scalars, {count} mismatches,' in stderr
    assert stderr.count('Mismatch ') == min(count, 10)
    if fault:
        assert 'CG_TEST_ARRAY '+fault['CG_TEST_ARRAY'] in stderr and 'GPU=' in stderr and 'CPU=' in stderr
    assert 'HIP error' not in stderr


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--platform', choices=['cuda', 'rocm'], default='cuda')
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--compile-only', action='store_true')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    hooks = build(repo, work/'instrumented', args.platform, True)
    if args.compile_only:
        print('CG measured-vector/history and storage hooks compiled; GPU tests not executed')
        return
    production = build(repo, work/'production', args.platform, False)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'CG_TEST_'))}
    cases = [(n, 256, {}) for n in CLASSES]
    cases += [(1400, b, {}) for b in [1, 8, 32, 1024]]
    for k in range(7):
        length = 1400 if k < 5 else 15
        for i in [0, length//2, length-1]:
            for value in ['finite', 'nan', 'inf', '-inf']:
                cases.append((1400, 256, dict(CG_TEST_ARRAY=str(k), CG_TEST_CELL=str(i), CG_TEST_VALUE=value)))
    cases += [(1400, 256, dict(CG_TEST_ARRAY='0', CG_TEST_CELL='700', CG_TEST_VALUE='finite', CG_TEST_COUNT='12')),
              (150000, 256, dict(CG_TEST_ARRAY='4', CG_TEST_CELL='149999', CG_TEST_VALUE='finite')),
              (1400, 256, dict(CG_TEST_STORAGE_FAILURE='1')),
              (1400, 256, dict(CG_TEST_REFERENCE_FAILURE='1'))]
    report = []
    for index, (n, block, fault) in enumerate(cases):
        label = f'{index}-n{n}-b{block}'
        start = time.monotonic()
        env = dict(clean, BENCH_PARAM_matrix_size=str(n), BENCH_PARAM_block_size=str(block),
                   BENCH_PARAM_vector_add_warmup='false' if fault else 'true', **fault)
        try:
            run = subprocess.run([str(hooks if fault else production)], env=env, text=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            (work/(label+'.stdout')).write_bytes(error.stdout or b'')
            (work/(label+'.stderr')).write_bytes(error.stderr or b'')
            raise
        elapsed = time.monotonic()-start
        (work/(label+'.stdout')).write_text(run.stdout)
        (work/(label+'.stderr')).write_text(run.stderr)
        check_output(run.stdout, run.stderr, run.returncode, n, block, fault)
        report.append(dict(case=label, matrix_size=n, block_size=block, fault=fault,
                           exit_code=run.returncode, wall_seconds=elapsed))
        (work/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(label, 'PASS', f'{elapsed:.3f}s', flush=True)
    print(f'{len(cases)} production/control/measured-vector/history/storage cases pass')


if __name__ == '__main__':
    main()
