#!/usr/bin/env python3
"""Check all measured Q columns under the existing orthogonality criterion."""
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
    if (const char* kind = std::getenv("GRAM_Q_TEST_KIND")) {
        std::vector<float> values((size_t)M*N);
        HIP_CHECK(hipMemcpy(values.data(),d_Q,bytes_Q,hipMemcpyDeviceToHost));
        if (!std::strcmp(kind,"duplicate")) {
            for (int i=0;i<M;++i) values[(size_t)i*N+N-1]=values[(size_t)i*N];
        } else if (!std::strcmp(kind,"scaled")) {
            for (float& value:values) value*=2.0f;
        } else {
            const char* text=std::getenv("GRAM_Q_TEST_INDEX");
            if (!text) std::abort();
            size_t at=std::strtoull(text,nullptr,10);
            if (at>=values.size()) std::abort();
            if (!std::strcmp(kind,"nan")) values[at]=std::numeric_limits<float>::quiet_NaN();
            else if (!std::strcmp(kind,"pinf")) values[at]=std::numeric_limits<float>::infinity();
            else if (!std::strcmp(kind,"ninf")) values[at]=-std::numeric_limits<float>::infinity();
            else if (!std::strcmp(kind,"finite")) values[at]=1234.0f;
            else std::abort();
        }
        HIP_CHECK(hipMemcpy(d_Q,values.data(),bytes_Q,hipMemcpyHostToDevice));
        std::fprintf(stderr,"GRAM_Q_TEST_INJECT %s\n",kind);
    }
'''


def instrument(source):
    anchor = '    // Copy Q back for orthogonality check'
    launch = '                gram_normalize<<<grid_M, block_size>>>'
    if source.count(anchor) != 1 or source.count(launch) != 1:
        raise ValueError('Gram-Schmidt measured-result/launch anchors missing')
    return ('#include <vector>\n#include <limits>\n' + source.replace(anchor, HOOK+anchor)
            .replace(launch, '                if (!std::getenv("GRAM_Q_TEST_SKIP"))\n'+launch))


def host_checks(root, work, compiler):
    source = (root/'tier2/polybench_gramschmidt/hip/polybench_gramschmidt.hip').read_text()
    start = 'static bool check_orthogonality('
    end = '// ---------------------------------------------------------------------------\n// Main'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Gram-Schmidt checker anchors missing')
    text = '#include <cstdio>\n#include <cmath>\n'+start+source.split(start)[1].split(end)[0]
    text += (root/'tests/gramschmidt_q_test.cpp').read_text()
    path = work/'checker.cpp'; path.write_text(text)
    binary = work/'checker'
    subprocess.run([compiler,'-O3','-std=c++14','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(path),'-o',str(binary)],check=True)
    p = subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
    (work/'host.stdout').write_text(p.stdout); (work/'host.stderr').write_text(p.stderr)
    assert p.returncode == 0, p.stderr
    fixtures = re.split(r'GRAM_Q_CASE (\d+) (\d+) (\d+)\n',p.stderr)
    assert not fixtures[0] and len(fixtures)>1
    for i in range(1,len(fixtures),4):
        case,n,errors,diagnostics = fixtures[i:i+4]
        expected = f'Orthogonality: {errors} failing pairs out of {int(n)**2}\n'
        assert diagnostics.endswith(expected) and diagnostics.count('Orthogonality:')==1,(case,diagnostics)
        assert diagnostics.count('ORTH FAIL:')==min(int(errors),10),(case,diagnostics)
    total = int(re.search(r'PASS: (\d+)',p.stdout)[1])
    assert total==(len(fixtures)-1)//4
    print(p.stdout,end='')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work-dir',type=Path,required=True,help='New, nonexistent directory')
    ap.add_argument('--host-only',action='store_true')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--platform',choices=('cuda','rocm'))
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--sweep',action='store_true')
    args = ap.parse_args()
    if not args.host_only and not args.platform: ap.error('--platform required')
    root, work = args.source.resolve(), args.work_dir.resolve()
    work.mkdir(parents=True,exist_ok=False)
    if args.host_only:
        host_checks(root,work,args.compiler); return 0
    relative = Path('tier2/polybench_gramschmidt/hip'); src = root/relative
    source = (src/'polybench_gramschmidt.hip').read_text()
    inputs = [relative/n for n in ('polybench_gramschmidt.hip','Makefile','params.json')]
    inputs += [Path('common_hip')/n for n in ('bench_timing.h','bench_warmup.h','hip_cuda_compat.h')]
    hashes = {str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
    binaries = {}
    for variant in ('production','faults'):
        dst = work/variant/relative; dst.mkdir(parents=True)
        shutil.copytree(root/'common_hip',work/variant/'common_hip')
        for name in ('Makefile','params.json'): shutil.copy2(src/name,dst/name)
        (dst/'polybench_gramschmidt.hip').write_text(source if variant=='production' else instrument(source))
        with (work/variant/'build.log').open('w') as log:
            subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[variant] = dst/'polybench_gramschmidt'
    if args.compile_only:
        print('Production and measured-Q fault variants compiled; GPU checks not run'); return 0
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','GRAM_Q_TEST_'))}
    # variant, rows, columns, warmer, fault kind, element index, skip normalization
    cases = []
    for warm in ('false','true'):
        cases += [('production',m,n,warm,None,None,False)
                  for m,n in ((1,1),(9,7),(17,8),(17,9),(65,33),(257,64))]
        cases += [('faults',257,64,warm,kind,at,False)
                  for kind,at in itertools.product(('finite','nan','pinf','ninf'),(0,7,8,128*64+32,257*64-1))]
        cases += [('faults',257,64,warm,kind,None,False) for kind in ('duplicate','scaled')]
        cases += [('faults',257,64,warm,None,None,skip) for skip in (True,False)]
    if args.sweep:
        manifest = json.loads((src/'params.json').read_text())
        columns = next(x['values'] for x in manifest['non_scaling_params'] if x['name']=='n')
        cases += [('production',m,n,warm,None,None,False)
                  for m,n,warm in itertools.product(manifest['scaling_param']['values'],columns,('false','true'))]
    results = []
    def report(complete=False):
        data = dict(complete=complete,source_sha256=hashes,
                    binary_sha256={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in binaries.items()},
                    expected=len(cases),total=len(results),passed=sum(x['passed'] for x in results),cases=results)
        (work/'report.json').write_text(json.dumps(data,indent=2)+'\n'); return data
    names = ('gram_norm_zero','gram_norm','gram_norm_finish','gram_normalize','gram_project')
    for variant,m,n,warm,kind,index,skip in cases:
        label = f'{len(results):03d}-{variant}-{m}-{n}-{warm}-{kind}-{index}-{skip}'
        env = dict(clean,BENCH_PARAM_m=str(m),BENCH_PARAM_n=str(n),BENCH_PARAM_vector_add_warmup=warm)
        if kind: env['GRAM_Q_TEST_KIND']=kind
        if index is not None: env['GRAM_Q_TEST_INDEX']=str(index)
        if skip: env['GRAM_Q_TEST_SKIP']='1'
        start = time.monotonic()
        try:
            p = subprocess.run([str(binaries[variant])],env=env,capture_output=True,text=True,timeout=60)
            out,err,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as ex:
            out=(ex.stdout or b'').decode(errors='replace'); err=(ex.stderr or b'').decode(errors='replace'); code='timeout'
        seconds = time.monotonic()-start
        (work/(label+'.stdout')).write_text(out); (work/(label+'.stderr')).write_text(err)
        bad = bool(kind or skip)
        passed = code==(1 if bad else 0) and seconds<60
        match = re.search(r'Orthogonality: (\d+) failing pairs out of (\d+)',err)
        errors = int(match[1]) if match else None
        passed &= bool(match) and int(match[2])==n*n and (errors>0 if bad else errors==0)
        passed &= err.count('ORTH FAIL:')==min(errors,10) if errors is not None else False
        passed &= f'Orthogonality check (all {n} columns): '+('FAIL' if bad else 'PASS') in err
        if kind: passed &= f'GRAM_Q_TEST_INJECT {kind}' in err
        if kind=='duplicate': passed &= errors==2
        if kind=='scaled' or skip: passed &= errors==n
        if index is not None:
            pairs = re.findall(r'ORTH FAIL: dot\(Q\[:,(\d+)\], Q\[:,(\d+)\]\)',err)
            passed &= any(index % n in (int(a),int(b)) for a,b in pairs)
        if bad: passed &= ': PASS' not in err
        try:
            records = [json.loads(line) for line in out.splitlines()]
            passed &= len(records)==6
            for record,name in zip(records[:5],names):
                passed &= record['type']=='kernel' and record['name']==name and record['iter']==0
                passed &= record['params']==dict(m=m,n=n,q_orthogonality='all_columns_v1')
                passed &= all(math.isfinite(record[f]) and record[f]>=0 for f in ('time_ms','host_time_ms'))
            summary = records[-1]
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-sum(x['time_ms'] for x in records[:5]))<.000004
        except (ValueError,KeyError,TypeError,IndexError): passed=False
        results.append(dict(label=label,variant=variant,m=m,n=n,warm=warm,kind=kind,index=index,
                            skip=skip,errors=errors,exit=code,seconds=seconds,passed=bool(passed)))
        report(); print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} Gram-Schmidt all-column Q cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=='__main__':
    raise SystemExit(main())
