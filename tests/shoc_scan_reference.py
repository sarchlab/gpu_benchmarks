#!/usr/bin/env python3
"""Check the SHOC scan reference and actual measured outputs at every size."""
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
    if (const char* kind=std::getenv("SCAN_REFERENCE_TEST_KIND")) {
        if (!std::strcmp(kind,"zero")) {
            HIP_CHECK(hipMemset(d_output,0,bytes));
            std::fprintf(stderr,"SCAN_REFERENCE_TEST_ZERO\n");
        } else {
            int at=std::atoi(std::getenv("SCAN_REFERENCE_TEST_INDEX"));
            if(at<0 || at>=N) std::abort();
            float value;
            HIP_CHECK(hipMemcpy(&value,d_output+at,sizeof(value),hipMemcpyDeviceToHost));
            if(!std::strcmp(kind,"nan")) value=NAN;
            else if(!std::strcmp(kind,"pinf")) value=INFINITY;
            else if(!std::strcmp(kind,"ninf")) value=-INFINITY;
            else value+=fmaxf(1.0f,0.01f*fabsf(value));
            HIP_CHECK(hipMemcpy(d_output+at,&value,sizeof(value),hipMemcpyHostToDevice));
            std::fprintf(stderr,"SCAN_REFERENCE_TEST_INJECT %d\n",at);
        }
    }
'''


def instrument(source):
    anchor='    HIP_CHECK(hipMemcpy(h_output, d_output, bytes_orig, hipMemcpyDeviceToHost));'
    if source.count(anchor)!=1:raise ValueError('SHOC scan measured output anchor missing')
    return source.replace(anchor,HOOK+anchor)


def host_tests(root,work,compiler):
    source=(root/'tier2/shoc_scan/hip/shoc_scan.hip').read_text()
    start='static void scan_cpu('
    end='// ---------------------------------------------------------------------------\n// Main'
    if source.count(start)!=1 or source.count(end)!=1:raise ValueError('SHOC scan CPU reference anchors missing')
    source=start+source.split(start)[1].split(end)[0]
    source+=(root/'tests/shoc_scan_reference_test.cpp').read_text()
    cpp=work/'reference.cpp';cpp.write_text(source);binary=work/'reference'
    subprocess.run([compiler,'-O3','-std=c++14','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    p=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
    (work/'host.stdout').write_text(p.stdout);(work/'host.stderr').write_text(p.stderr)
    assert p.returncode==0,p.stderr
    print(p.stdout,end='')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work-dir',type=Path,required=True,help='New directory for source, binaries and raw logs')
    ap.add_argument('--host-only',action='store_true')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--platform',choices=('cuda','rocm'))
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--sweep',action='store_true')
    args=ap.parse_args()
    if not args.host_only and not args.platform:ap.error('--platform required')
    root,work=args.source.resolve(),args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    if args.host_only:host_tests(root,work,args.compiler);return 0
    relative=Path('tier2/shoc_scan/hip');src=root/relative;source=(src/'shoc_scan.hip').read_text()
    inputs=[relative/n for n in ('shoc_scan.hip','Makefile','params.json')]
    inputs += [Path('common_hip')/n for n in ('bench_timing.h','bench_warmup.h','hip_cuda_compat.h')]
    hashes={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
    binaries={}
    for variant in ('production','faults'):
        dst=work/variant/relative;dst.mkdir(parents=True);shutil.copytree(root/'common_hip',work/variant/'common_hip')
        for name in ('Makefile','params.json'):shutil.copy2(src/name,dst/name)
        (dst/'shoc_scan.hip').write_text(source if variant=='production' else instrument(source))
        with (work/variant/'build.log').open('w') as log:
            subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[variant]=dst/'shoc_scan'
    if args.compile_only:print('Production and measured-device fault variants compiled; GPU checks not run');return 0
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SCAN_REFERENCE_TEST_'))}
    # variant, N, block size, warmer, corruption kind, index
    cases=[]
    for warm in ('false','true'):
        cases += [('production',n,256,warm,None,None) for n in (1,7,511,512,513,1025,262145)]
        cases += [('production',513,b,warm,None,None) for b in (32,128,512)]
        cases += [('faults',513,256,warm,kind,at) for kind,at in itertools.product(('finite','nan','pinf','ninf'),(0,256,512))]
        cases += [('faults',513,256,warm,'zero',None),('faults',513,256,warm,None,None)]
        cases += [('faults',8388608,256,warm,'finite',at) for at in (0,4194304,8388607)]
        cases += [('faults',268435456,256,warm,'finite',268435455)]
    if args.sweep:
        manifest=json.loads((src/'params.json').read_text())
        blocks=next(p['values'] for p in manifest['non_scaling_params'] if p['name']=='block_size')
        cases += [('production',n,b,warm,None,None) for n,b,warm in itertools.product(manifest['scaling_param']['values'],blocks,('false','true'))]
    results=[]
    def report(complete=False):
        data=dict(complete=complete,source_sha256=hashes,binary_sha256={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in binaries.items()},
                  expected=len(cases),total=len(results),passed=sum(x['passed'] for x in results),cases=results)
        (work/'report.json').write_text(json.dumps(data,indent=2)+'\n');return data
    for variant,n,b,warm,kind,at in cases:
        label=f'{len(results):03d}-{variant}-{n}-{b}-{warm}-{kind}-{at}'
        env=dict(clean,BENCH_PARAM_array_size=str(n),BENCH_PARAM_block_size=str(b),BENCH_PARAM_vector_add_warmup=warm)
        if kind:env['SCAN_REFERENCE_TEST_KIND']=kind
        if at is not None:env['SCAN_REFERENCE_TEST_INDEX']=str(at)
        start=time.monotonic()
        try:
            p=subprocess.run([str(binaries[variant])],env=env,capture_output=True,text=True,timeout=60)
            out,err,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as ex:
            out=(ex.stdout or b'').decode(errors='replace');err=(ex.stderr or b'').decode(errors='replace');code='timeout'
        seconds=time.monotonic()-start
        (work/(label+'.stdout')).write_text(out);(work/(label+'.stderr')).write_text(err)
        bad=kind is not None;passed=code==(1 if bad else 0) and seconds<60
        match=re.search(r'FAIL: (\d+) errors out of (\d+)',err)
        if bad:
            expected=n-1 if kind=='zero' else 1
            passed &= bool(match) and int(match[1])==expected and int(match[2])==n and err.count('Mismatch at ')==min(expected,10)
            passed &= ('SCAN_REFERENCE_TEST_ZERO' in err) if kind=='zero' else (f'SCAN_REFERENCE_TEST_INJECT {at}' in err and f'Mismatch at {at}:' in err)
        else:passed &= bool(re.search(r'^PASS$',err,re.M)) and not match
        try:
            first,second,summary=[json.loads(line) for line in out.splitlines()]
            for record,kernel in ((first,'scan_block_kernel'),(second,'add_block_sums_kernel')):
                passed &= record['type']=='kernel' and record['name']==kernel and record['iter']==0
                passed &= record['params']==dict(array_size=n,block_size=b,verification='full_cpu_fp64accum_v1')
                passed &= all(math.isfinite(record[f]) and record[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-first['time_ms']-second['time_ms'])<.000002
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,variant=variant,n=n,block=b,warm=warm,kind=kind,index=at,exit=code,seconds=seconds,passed=bool(passed)))
        report();print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    final=report(complete=True);print(f"{final['passed']}/{final['total']} measured scan cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=='__main__':raise SystemExit(main())
