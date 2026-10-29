#!/usr/bin/env python3
"""Verify every measured element of both CUDA transpose variants."""
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
    if (const char* target=std::getenv("TRANSPOSE_TEST_TARGET")) {
        if (!std::strcmp(target,"@TARGET@")) {
            const size_t at=std::strtoull(std::getenv("TRANSPOSE_TEST_INDEX"),nullptr,10);
            const size_t count=std::getenv("TRANSPOSE_TEST_MANY") ? 12 : 1;
            if (count>num_elems || at>num_elems-count) std::abort();
            const char* kind=std::getenv("TRANSPOSE_TEST_KIND");
            float value=1234.0f;
            if (kind && !std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
            else if (kind && !std::strcmp(kind,"pinf")) value=std::numeric_limits<float>::infinity();
            else if (kind && !std::strcmp(kind,"ninf")) value=-std::numeric_limits<float>::infinity();
            for (size_t i=at;i<at+count;++i)
                HIP_CHECK(hipMemcpy(d_odata+i,&value,sizeof(value),hipMemcpyHostToDevice));
            std::fprintf(stderr,"TRANSPOSE_TEST_INJECT @TARGET@ %zu %zu\n",at,count);
        }
    }
'''


def instrument(source):
    for target in ('naive','optimized'):
        anchor='    // Verify '+target+' transpose'
        launch=f'            transpose_{target}<<<grid, block>>>(d_idata, d_odata, N, N);'
        if source.count(anchor)!=1 or source.count(launch)!=1:
            raise ValueError('Transpose measured-result/launch anchors missing')
        source=source.replace(anchor,HOOK.replace('@TARGET@',target)+anchor)
        guard=f'            if (!std::getenv("TRANSPOSE_TEST_SKIP") || std::strcmp(std::getenv("TRANSPOSE_TEST_SKIP"),"{target}"))\n'
        source=source.replace(launch,guard+launch)
    return '#include <limits>\n'+source


def host_reference(root,work,compiler):
    source=(root/'tier2/cuda_transpose/hip/cuda_transpose.hip').read_text()
    start='static void transpose_cpu('
    end='// ---------------------------------------------------------------------------\n// Main'
    if source.count(start)!=1 or source.count(end)!=1:
        raise ValueError('Transpose CPU reference anchors missing')
    text=start+source.split(start)[1].split(end)[0]
    text+=(root/'tests/transpose_reference_test.cpp').read_text()
    path=work/'reference.cpp'; path.write_text(text); binary=work/'reference'
    subprocess.run([compiler,'-O3','-std=c++14','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(path),'-o',str(binary)],check=True)
    p=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
    (work/'host.stdout').write_text(p.stdout); (work/'host.stderr').write_text(p.stderr)
    assert p.returncode==0,p.stderr
    print(p.stdout,end='')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work-dir',type=Path,required=True,help='New, nonexistent directory')
    ap.add_argument('--host-only',action='store_true')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--platform',choices=('cuda','rocm'))
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--sweep',action='store_true')
    args=ap.parse_args()
    if not args.host_only and not args.platform: ap.error('--platform required')
    root,work=args.source.resolve(),args.work_dir.resolve(); work.mkdir(parents=True,exist_ok=False)
    if args.host_only:
        host_reference(root,work,args.compiler); return 0
    relative=Path('tier2/cuda_transpose/hip'); src=root/relative
    source=(src/'cuda_transpose.hip').read_text()
    inputs=[relative/n for n in ('cuda_transpose.hip','Makefile','params.json')]
    inputs += [Path('common_hip')/n for n in ('bench_timing.h','bench_warmup.h','hip_cuda_compat.h')]
    hashes={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
    binaries={}
    for variant in ('production','faults'):
        dst=work/variant/relative; dst.mkdir(parents=True)
        shutil.copytree(root/'common_hip',work/variant/'common_hip')
        for name in ('Makefile','params.json'): shutil.copy2(src/name,dst/name)
        (dst/'cuda_transpose.hip').write_text(source if variant=='production' else instrument(source))
        with (work/variant/'build.log').open('w') as log:
            subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[variant]=dst/'cuda_transpose'
    if args.compile_only:
        print('Production and measured-device fault variants compiled; GPU checks not run'); return 0
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','TRANSPOSE_TEST_'))}
    # variant, size, warmer, target, fault kind, index, count, skipped variant
    cases=[]
    for warm in ('false','true'):
        cases += [('production',n,warm,None,None,None,0,None) for n in (1,7,31,32,33,65)]
        cases += [('faults',33,warm,target,kind,at,1,None)
                  for target,kind,at in itertools.product(('naive','optimized'),('finite','nan','pinf','ninf'),(0,999,1000,544,1088))]
        cases += [('faults',33,warm,target,'finite',1000,12,None) for target in ('naive','optimized')]
        cases += [('faults',33,warm,None,None,None,0,target) for target in ('naive','optimized')]
        cases += [('faults',33,warm,None,None,None,0,None)]
    if args.sweep:
        manifest=json.loads((src/'params.json').read_text())
        cases += [('production',n,warm,None,None,None,0,None)
                  for n,warm in itertools.product(manifest['scaling_param']['values'],('false','true'))]
    results=[]
    def report(complete=False):
        data=dict(complete=complete,source_sha256=hashes,
                  binary_sha256={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in binaries.items()},
                  expected=len(cases),total=len(results),passed=sum(x['passed'] for x in results),cases=results)
        (work/'report.json').write_text(json.dumps(data,indent=2)+'\n'); return data
    for variant,n,warm,target,kind,index,count,skip in cases:
        label=f'{len(results):03d}-{variant}-{n}-{warm}-{target}-{kind}-{index}-{count}-{skip}'
        env=dict(clean,BENCH_PARAM_size=str(n),BENCH_PARAM_vector_add_warmup=warm)
        if target: env.update(TRANSPOSE_TEST_TARGET=target,TRANSPOSE_TEST_KIND=kind,TRANSPOSE_TEST_INDEX=str(index))
        if count==12: env['TRANSPOSE_TEST_MANY']='1'
        if skip: env['TRANSPOSE_TEST_SKIP']=skip
        start=time.monotonic()
        try:
            p=subprocess.run([str(binaries[variant])],env=env,capture_output=True,text=True,timeout=60)
            out,err,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as ex:
            out=(ex.stdout or b'').decode(errors='replace'); err=(ex.stderr or b'').decode(errors='replace'); code='timeout'
        seconds=time.monotonic()-start
        (work/(label+'.stdout')).write_text(out); (work/(label+'.stderr')).write_text(err)
        bad=bool(target or skip); passed=code==(1 if bad else 0) and seconds<60
        errors={}
        for stage in ('naive','optimized'):
            title=stage.title()
            match=re.search(rf'{title} verification: (PASS|FAIL) \((\d+) mismatches in (\d+) elements\)',err)
            errors[stage]=int(match[2]) if match else None
            wrong=stage in (target,skip)
            passed &= bool(match) and int(match[3])==n*n and match[1]==('FAIL' if wrong else 'PASS')
            if match:
                passed &= errors[stage]>0 if stage==skip else errors[stage]==(count if stage==target else 0)
                passed &= err.count(title+' mismatch at ')==min(errors[stage],5)
        if target:
            passed &= f'TRANSPOSE_TEST_INJECT {target} {index} {count}' in err
            passed &= f'{target.title()} mismatch at {index}:' in err
        try:
            first,second,summary=[json.loads(line) for line in out.splitlines()]
            for record,stage in ((first,'naive'),(second,'optimized')):
                passed &= record['type']=='kernel' and record['name']=='transpose_'+stage and record['iter']==0
                passed &= record['params']==dict(size=n,verification='full_cpu_v1')
                passed &= all(math.isfinite(record[f]) and record[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-first['time_ms']-second['time_ms'])<.000002
        except (ValueError,KeyError,TypeError): passed=False
        results.append(dict(label=label,variant=variant,n=n,warm=warm,target=target,kind=kind,
                            index=index,count=count,skip=skip,errors=errors,exit=code,seconds=seconds,passed=bool(passed)))
        report(); print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} complete measured transpose cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=='__main__':
    raise SystemExit(main())
