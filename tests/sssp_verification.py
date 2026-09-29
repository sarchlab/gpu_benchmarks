#!/usr/bin/env python3
"""Verify every measured SSSP distance without changing the current GPU algorithm."""
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
        if (const char* text = std::getenv("SSSP_TEST_INDEX")) {
            const size_t at = std::strtoull(text,nullptr,10);
            const size_t count = std::getenv("SSSP_TEST_MANY") ? 12 : 1;
            if (count > (size_t)N || at > (size_t)N-count) std::abort();
            const int value = -1; // Every valid distance is nonnegative or INT_MAX.
            for (size_t i=at;i<at+count;++i)
                HIP_CHECK(hipMemcpy(d_dist+i,&value,sizeof(value),hipMemcpyHostToDevice));
            std::fprintf(stderr,"SSSP_TEST_INJECT %zu %zu\n",at,count);
        }
'''


def instrument(source):
    anchor = '        std::vector<int> h_gpu_dist(N);'
    launch = '            bellman_ford_kernel<<<dim3(gridSize), dim3(blockSize)>>>'
    if source.count(anchor)!=1 or source.count(launch)!=1:
        raise ValueError('SSSP measured-result/launch anchors missing')
    return source.replace(anchor,HOOK+anchor).replace(launch,
            '            if (!std::getenv("SSSP_TEST_SKIP"))\n'+launch)


def host_reference(root, work, compiler):
    source=(root/'tier2/lonestar_sssp/lonestar_sssp.hip').read_text()
    start,end='static int bellman_ford_cpu(', '// GPU SSSP: run Bellman-Ford iterations'
    if source.count(start)!=1 or source.count(end)!=1:
        raise ValueError('SSSP CPU reference anchors missing')
    text='#include <cstdio>\n#include <climits>\n#include <vector>\n'
    text+=start+source.split(start)[1].split(end)[0]
    text+=(root/'tests/sssp_reference_test.cpp').read_text()
    path=work/'reference.cpp';path.write_text(text);binary=work/'reference'
    subprocess.run([compiler,'-O3','-std=c++14','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(path),'-o',str(binary)],check=True)
    p=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
    (work/'host.stdout').write_text(p.stdout);(work/'host.stderr').write_text(p.stderr)
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
    if not args.host_only and not args.platform:ap.error('--platform required')
    root,work=args.source.resolve(),args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    if args.host_only:
        host_reference(root,work,args.compiler);return 0
    relative=Path('tier2/lonestar_sssp');src=root/relative;source=(src/'lonestar_sssp.hip').read_text()
    inputs=[relative/n for n in ('lonestar_sssp.hip','Makefile','params.json')]
    inputs += [Path('common')/n for n in ('bench_timing.h','bench_warmup.h','hip_cuda_compat.h')]
    hashes={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
    binaries={}
    for variant in ('production','faults'):
        dst=work/variant/relative;dst.mkdir(parents=True);shutil.copytree(root/'common',work/variant/'common')
        for name in ('Makefile','params.json'):shutil.copy2(src/name,dst/name)
        (dst/'lonestar_sssp.hip').write_text(source if variant=='production' else instrument(source))
        with (work/variant/'build.log').open('w') as log:
            subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[variant]=dst/'lonestar_sssp'
    if args.compile_only:
        print('Production and measured-device fault variants compiled; GPU checks not run');return 0
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SSSP_TEST_'))}
    # variant, vertices, degree, block size, warmer, first fault, count, skip launch
    cases=[]
    for warm in ('false','true'):
        cases += [('production',n,d,256,warm,None,0,False)
                  for n,d in itertools.product((1,2,9,1023,1024,1025,2049),(4,16,64))]
        cases += [('faults',2049,4,256,warm,at,1,False) for at in (0,1023,1024,1536,2048)]
        cases += [('faults',2049,4,256,warm,1024,12,False),
                  ('faults',2049,4,256,warm,None,0,True),('faults',2049,4,256,warm,None,0,False)]
    if args.sweep:
        m=json.loads((src/'params.json').read_text());p={x['name']:x['values'] for x in m['non_scaling_params']}
        cases += [('production',n,d,b,'true',None,0,False)
                  for n,d,b in itertools.product(reversed(m['scaling_param']['values']),p['degree'],p['block_size'])]
    results=[]
    def report(complete=False):
        data=dict(complete=complete,source_sha256=hashes,
                  binary_sha256={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in binaries.items()},
                  expected=len(cases),total=len(results),passed=sum(x['passed'] for x in results),cases=results)
        (work/'report.json').write_text(json.dumps(data,indent=2)+'\n');return data
    for variant,n,d,b,warm,index,count,skip in cases:
        label=f'{len(results):03d}-{variant}-{n}-{d}-{b}-{warm}-{index}-{count}-{skip}'
        env=dict(clean,BENCH_PARAM_vertices=str(n),BENCH_PARAM_degree=str(d),
                 BENCH_PARAM_block_size=str(b),BENCH_PARAM_vector_add_warmup=warm)
        if index is not None:env['SSSP_TEST_INDEX']=str(index)
        if count==12:env['SSSP_TEST_MANY']='1'
        if skip:env['SSSP_TEST_SKIP']='1'
        start=time.monotonic()
        try:
            p=subprocess.run([str(binaries[variant])],env=env,capture_output=True,text=True,timeout=60)
            out,err,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as ex:
            out=(ex.stdout or b'').decode(errors='replace');err=(ex.stderr or b'').decode(errors='replace');code='timeout'
        seconds=time.monotonic()-start
        (work/(label+'.stdout')).write_text(out);(work/(label+'.stderr')).write_text(err)
        bad=bool(count or skip);passed=code==(1 if bad else 0) and seconds<60
        match=re.search(r'(PASS|FAIL): (\d+) mismatches in (\d+) vertices',err)
        errors=int(match[2]) if match else None
        passed &= bool(match) and match[1]==('FAIL' if bad else 'PASS') and int(match[3])==n
        if skip:passed &= errors is not None and errors>0
        else:passed &= errors==count
        passed &= err.count('Mismatch at vertex ')==min(errors,10) if errors is not None else False
        if count:passed &= f'SSSP_TEST_INJECT {index} {count}' in err and f'Mismatch at vertex {index}:' in err
        if bad:passed &= 'PASS:' not in err
        try:
            kernel,summary=[json.loads(line) for line in out.splitlines()]
            passed &= kernel['type']=='kernel' and kernel['name']=='bellman_ford_kernel' and kernel['iter']==0
            passed &= kernel['params']==dict(vertices=n,degree=d,block_size=b,verification='full_cpu_distances_v1')
            passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-kernel['time_ms'])<.000002
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,variant=variant,vertices=n,degree=d,block_size=b,warm=warm,
                            index=index,count=count,skip=skip,errors=errors,exit=code,seconds=seconds,passed=bool(passed)))
        report();print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    final=report(complete=True);print(f"{final['passed']}/{final['total']} SSSP complete-distance cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=='__main__':
    raise SystemExit(main())
