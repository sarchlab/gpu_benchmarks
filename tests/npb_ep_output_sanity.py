#!/usr/bin/env python3
"""Check EP's measured partials without changing its NAS kernel or sum criterion."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HOOK = r'''
    if (const char* mode = getenv("EP_TEST_MODE")) {
        const bool last = getenv("EP_TEST_LAST") != nullptr;
        const int block = last ? blocks_per_grid - 1 : 0;
        const size_t at = (size_t)block * NQ + (last ? NQ - 1 : 0);
        long long active = nn - (long long)block * threads_per_block;
        if (active > threads_per_block) active = threads_per_block;
        const double budget = (double)active * NK;
        double value = NAN;
        const char* field = getenv("EP_TEST_FIELD");
        if (!field) std::abort();
        double* target = strcmp(field,"q")==0 ? q_device + at :
                         strcmp(field,"sx")==0 ? sx_device + block : sy_device + block;
        if (strcmp(mode,"inf")==0) value = INFINITY;
        else if (strcmp(mode,"negative")==0) value = -1;
        else if (strcmp(mode,"fraction")==0) value = 0.5;
        else if (strcmp(mode,"over")==0) value = budget + 1;
        else if (strcmp(mode,"finite")==0) value = 123456;
        if (strcmp(mode,"total")==0) {
            double counts[NQ] = {};
            counts[0] = budget / 2 + 1; counts[NQ-1] = budget / 2;
            HIP_CHECK(hipMemcpy(q_device + (size_t)block*NQ, counts,
                                sizeof(counts), hipMemcpyHostToDevice));
        } else if (strcmp(mode,"many")==0) {
            value = -1;
            for (size_t i = 0; i < (size_t)blocks_per_grid*NQ; ++i)
                HIP_CHECK(hipMemcpy(q_device+i, &value, sizeof(value), hipMemcpyHostToDevice));
        } else {
            HIP_CHECK(hipMemcpy(target, &value, sizeof(value), hipMemcpyHostToDevice));
        }
        fprintf(stderr,"EP_TEST_INJECT %s %s block=%d\n",field,mode,block);
    }
'''


def instrument(source):
    anchor = '    double* q_host  = (double*)malloc(size_q);'
    if source.count(anchor) != 1:
        raise ValueError('EP post-timing download anchor missing')
    return source.replace(anchor, HOOK + '''
    double* q_host = getenv("EP_TEST_ALLOC_FAIL") ? nullptr : (double*)malloc(size_q);''')


def host_tests(root, work, compiler):
    source = (root / 'tier2/npb_ep/hip/npb_ep.hip').read_text()
    start, end = 'static size_t reduce_and_check_partials(', '// Argument parsing'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('EP production reduction anchors missing')
    constants = re.findall(r'^#define (?:MK|NK|NQ)\s+.*$', source, re.M)
    if len(constants) != 3:
        raise ValueError('EP constants missing')
    text = '#include <cmath>\n#include <cstdio>\n' + '\n'.join(constants) + '\n'
    text += start + source.split(start)[1].split(end)[0]
    text += (root / 'tests/npb_ep_output_sanity_test.cpp').read_text()
    path = work / 'host.cpp'; path.write_text(text)
    binary = work / 'host-test'
    subprocess.run([compiler, '-O3', '-std=c++14', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', str(path), '-o', str(binary)], check=True)
    p = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    (work / 'host.stdout').write_text(p.stdout); (work / 'host.stderr').write_text(p.stderr)
    assert p.returncode == 0, p.stderr
    groups = p.stderr.split('Output sanity: ')[:-1]
    assert groups and all(g.count('Invalid ') <= 10 for g in groups)
    assert any(g.count('Invalid ') == 10 for g in groups)
    print(p.stdout, end='')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work-dir', type=Path, required=True, help='New, nonexistent directory')
    ap.add_argument('--host-only', action='store_true')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--platform', choices=('cuda','rocm'))
    ap.add_argument('--compile-only', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()
    if not args.host_only and not args.platform:
        ap.error('--platform required')
    root, work = args.source.resolve(), args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if args.host_only:
        host_tests(root,work,args.compiler)
        return 0
    relative = Path('tier2/npb_ep/hip'); src = root / relative
    source = (src / 'npb_ep.hip').read_text()
    inputs = [relative / n for n in ('npb_ep.hip','Makefile','params.json')]
    inputs += [Path('common') / n for n in ('bench_timing.h','bench_warmup.h','hip_cuda_compat.h')]
    hashes = {str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
    binaries = {}
    for variant in ('production','faults'):
        dst = work / variant / relative; dst.mkdir(parents=True)
        shutil.copytree(root/'common', work/variant/'common')
        for name in ('Makefile','params.json'):
            shutil.copy2(src/name, dst/name)
        (dst/'npb_ep.hip').write_text(source if variant=='production' else instrument(source))
        with (work/variant/'build.log').open('w') as log:
            subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[variant] = dst/'npb_ep'
    if args.compile_only:
        print('Production and post-timing measured-device fault variants compiled; GPU checks not run')
        return 0
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','EP_TEST_'))}
    # variant, M, block size, warmer, field, mode, last block, allocation failure
    cases = []
    for warm in ('false','true'):
        for m,block in ((24,128),(24,512),(17,3),(17,1)):
            cases.append(('production',m,block,warm,None,None,False,False))
        for m,block in ((24,512),(17,3)):
            for mode in ('over','total'):
                cases.append(('faults',m,block,warm,'q',mode,False,False))
        for m,block in ((24,128),(17,1)):
            for last in (False,True):
                for mode in ('nan','inf','negative','fraction','over','total'):
                    cases.append(('faults',m,block,warm,'q',mode,last,False))
                for field,mode in itertools.product(('sx','sy'),('nan','inf')):
                    cases.append(('faults',m,block,warm,field,mode,last,False))
            cases.append(('faults',m,block,warm,'q','many',False,False))
        cases.append(('faults',24,128,warm,'sx','finite',True,False))
        cases.append(('faults',24,128,warm,None,None,False,True))
    if args.sweep:
        manifest = json.loads((src/'params.json').read_text())
        blocks = next(p['values'] for p in manifest['non_scaling_params'] if p['name']=='block_size')
        cases += [('production',m,b,'true',None,None,False,False)
                  for m,b in itertools.product(reversed(manifest['scaling_param']['values']),blocks)]
    results = []
    def report(complete=False):
        data = dict(complete=complete,source_sha256=hashes,
                    binary_sha256={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in binaries.items()},
                    expected=len(cases),total=len(results),passed=sum(r['passed'] for r in results),cases=results)
        (work/'report.json').write_text(json.dumps(data,indent=2)+'\n')
        return data
    for variant,m,block,warm,field,mode,last,allocation in cases:
        label = f'{len(results):03d}-{variant}-{m}-{block}-{warm}-{field}-{mode}-{last}-{allocation}'
        env = dict(clean,BENCH_PARAM_m=str(m),BENCH_PARAM_block_size=str(block),BENCH_PARAM_vector_add_warmup=warm)
        if mode: env.update(EP_TEST_MODE=mode,EP_TEST_FIELD=field)
        if last: env['EP_TEST_LAST']='1'
        if allocation: env['EP_TEST_ALLOC_FAIL']='1'
        start = time.monotonic()
        try:
            p = subprocess.run([str(binaries[variant])],env=env,capture_output=True,text=True,timeout=60)
            out,err,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as ex:
            out=(ex.stdout or b'').decode(errors='replace'); err=(ex.stderr or b'').decode(errors='replace'); code='timeout'
        seconds=time.monotonic()-start
        (work/(label+'.stdout')).write_text(out); (work/(label+'.stderr')).write_text(err)
        bad=bool(mode or allocation); passed=code==(1 if bad else 0) and seconds<60
        if allocation:
            passed &= 'Verification: FAIL (host output allocation failed)' in err
        else:
            sanity_bad = bool(mode and mode!='finite')
            passed &= ('Output sanity: FAIL:' if sanity_bad else 'Output sanity: PASS: 0 errors') in err
            diagnostics=err.count('Invalid ')
            passed &= diagnostics<=10 and (diagnostics>0)==sanity_bad
            if mode=='many': passed &= diagnostics==10 and '20 errors across 2 blocks' in err
            if mode: passed &= f'EP_TEST_INJECT {field} {mode} block={1 if last else 0}' in err
        if bad:
            passed &= 'Verification: FAIL' in err and 'Verification: PASS' not in err and 'verification skipped' not in err
        elif m==17:
            passed &= 'verification skipped' in err and 'Verification: PASS' not in err
        else:
            passed &= 'Verification: PASS' in err
        try:
            kernel,summary=[json.loads(line) for line in out.splitlines()]
            klass={24:'S',25:'W',28:'A',30:'B',32:'C'}.get(m,'?')
            passed &= kernel['type']=='kernel' and kernel['name']=='gpu_kernel' and kernel['iter']==0
            passed &= kernel['params']==dict(m=m,block_size=block,**{'class':klass},output_sanity='finite_count_bounds_v1')
            passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-kernel['time_ms'])<.000002
        except (ValueError,KeyError,TypeError): passed=False
        results.append(dict(label=label,variant=variant,m=m,block_size=block,warm=warm,field=field,
                            mode=mode,last=last,allocation=allocation,exit=code,seconds=seconds,passed=bool(passed)))
        report(); print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} EP measured-output sanity cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=='__main__':
    raise SystemExit(main())
