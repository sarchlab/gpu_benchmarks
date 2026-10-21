#!/usr/bin/env python3
"""Validate measured buffers, omissions, setup failures, and frozen GPU bodies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from particlefilter_h100_profile import FILES, NAMES, check_records, classify_completed


def once(s,old,new):
    if s.count(old)!=1:raise ValueError('Nonunique instrumentation anchor: '+old)
    return s.replace(old,new)

def instrument(s):
    s=once(s,'struct Verification {',r'''
template<class T> __global__ void pf_fault(T* target,size_t i,int mode) {
    if(threadIdx.x==0&&blockIdx.x==0) {
        if(mode==0) {
            // A unit delta disappears at the scale of unnormalized block sums.
            if(std::is_integral<T>::value)target[i]+=T(1);
            else target[i]+=T(fmax(1.0,fabs(double(target[i]))));
        } else target[i]=mode==1?T(NAN):mode==2?T(INFINITY):T(-INFINITY);
    }
}
struct Verification {''')
    s=once(s,'    const pf::Input in=pf::input(n);','    if(getenv("PF_TEST_SETUP_FAILURE"))throw std::bad_alloc();\n    const pf::Input in=pf::input(n);')
    anchor='    const pf::Reference expected=pf::reference(n,in);'
    hook='''    if(const char* name=getenv("PF_TEST_BUFFER")) {
        size_t index=strtoull(getenv("PF_TEST_INDEX"),nullptr,10);
        int mode=getenv("PF_TEST_MODE")?atoi(getenv("PF_TEST_MODE")):0;
'''
    for name in ['x','y','xj','yj','likelihood','weights','cdf','u','partial','u0','seeds','indices']:
        hook+='        if(!strcmp(name,"'+name+'"))pf_fault<<<1,1>>>('+name+'.data,index,mode);\n'
    hook+='        HIP_CHECK(hipGetLastError());\n    }\n    if(getenv("PF_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();\n'
    s=once(s,anchor,hook+anchor)
    for k in range(6):
        line=next(l for l in s.splitlines() if f'bench_lap_add(&times[{k}]' in l)
        s=once(s,line,'        if(!getenv("PF_TEST_SKIP_'+str(k)+'")||frame!=10) '+line.lstrip())
    return s

def frozen(s):
    s=once(s,'#include "particlefilter_kernels.h"','#include "particlefilter_upstream_gpu.h"\n#include "particlefilter_kernels.h"')
    # Normalize/CDF/u retain exactly the repaired global ordering. The other
    # defining kernels below execute the pinned bodies, with documented RNG fixes.
    s=once(s,'pf_gpu::likelihood_kernel<<<blocks,pf::block>>>(n,frame,video.data,offsets.data,x.data,y.data,xj.data,yj.data,indices.data,likelihood.data,weights.data,seeds.data,partial.data)',
           'pf_upstream_gpu::likelihood_kernel<<<blocks,pf::block>>>(x.data,y.data,xj.data,yj.data,cdf.data,indices.data,offsets.data,likelihood.data,video.data,u.data,weights.data,n,pf::neighbors,pf::video_size,frame,pf::height,pf::frames,seeds.data,partial.data)')
    s=once(s,'pf_gpu::sum_kernel<<<blocks,pf::block>>>(partial.data,blocks)','pf_upstream_gpu::sum_kernel<<<blocks,pf::block>>>(partial.data,n)')
    s=once(s,'pf_gpu::find_index_kernel<<<blocks,pf::block>>>(x.data,y.data,cdf.data,u.data,xj.data,yj.data,n)','pf_upstream_gpu::find_index_kernel<<<blocks,pf::block>>>(x.data,y.data,cdf.data,u.data,xj.data,yj.data,weights.data,n)')
    return s

def build(repo,work,platform,kind):
    for f in FILES:
        dst=work/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/f,dst)
    folder=work/'tier2/altis_particlefilter/hip';source=(folder/'altis_particlefilter.hip').read_text()
    if kind=='instrumented':source=instrument(source)
    if kind=='frozen':
        shutil.copy2(repo/'tests/fixtures/particlefilter_upstream_gpu.h',folder)
        source=frozen(source)
    (folder/'altis_particlefilter.hip').write_text(source)
    with (work/'build.log').open('w') as log:subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'altis_particlefilter'

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda');ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--compile-only',action='store_true');ap.add_argument('--large-size',type=int,default=1000000);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];work=args.work_dir.resolve();binaries={k:build(root,work/k,args.platform,k) for k in ['production','instrumented','frozen']}
    if args.compile_only:print('Particle Filter output/omission hooks and frozen GPU bodies compile; no GPU execution claimed');return
    cases=[(kind,n,{}) for kind in ['production','frozen'] for n in [1,2,31,500,511,512,513,1031,4097,10000,100000]]
    for name in ['x','y','xj','yj','likelihood','weights','cdf','u','partial','u0','seeds','indices']:
        length=1 if name=='u0' else 3 if name=='partial' else 1031*(69 if name=='indices' else 1)
        for index in sorted({0,length-1}):
            for mode in ([0] if name in ['seeds','indices'] else [0,1,2,3]):
                cases.append(('instrumented',1031,dict(PF_TEST_BUFFER=name,PF_TEST_INDEX=str(index),PF_TEST_MODE=str(mode))))
        length=1 if name=='u0' else (args.large_size+511)//512 if name=='partial' else args.large_size*(69 if name=='indices' else 1)
        cases.append(('instrumented',args.large_size,dict(PF_TEST_BUFFER=name,PF_TEST_INDEX=str(length-1))))
    cases += [('instrumented',1031,{'PF_TEST_SKIP_'+str(k):'1'}) for k in range(6)]
    cases += [('instrumented',1031,{k:'1'}) for k in ['PF_TEST_SETUP_FAILURE','PF_TEST_REFERENCE_FAILURE']]
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','PF_TEST_'))};report=[]
    for j,(kind,n,fault) in enumerate(cases):
        start=time.monotonic();p=subprocess.run([str(binaries[kind])],env=dict(clean,BENCH_PARAM_size=str(n),BENCH_PARAM_vector_add_warmup='false' if fault else 'true',**fault),text=True,capture_output=True,timeout=60);elapsed=time.monotonic()-start
        label=f'{j}-{kind}-{n}';(work/(label+'.stdout')).write_text(p.stdout);(work/(label+'.stderr')).write_text(p.stderr)
        if 'PF_TEST_SETUP_FAILURE' in fault:assert p.returncode==1 and not p.stdout and 'setup/reference unavailable' in p.stderr
        elif 'PF_TEST_REFERENCE_FAILURE' in fault:check_records(n,p.stdout);assert p.returncode==1 and 'setup/reference unavailable' in p.stderr and 'verification:' not in p.stderr
        else:
            result=classify_completed(n,p.returncode,p.stdout,p.stderr,elapsed)
            assert result['status']==('unsupported' if fault else 'qualified'),(label,p.stderr)
            if 'PF_TEST_BUFFER' in fault:assert result['mismatches']==1,(label,p.stderr)
        report.append(dict(case=label,fault=fault,exit_code=p.returncode,wall_seconds=elapsed));(work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'PASS: {len(cases)} measured/frozen/negative cases, including twelve large-tail corruptions')
if __name__=='__main__':main()
