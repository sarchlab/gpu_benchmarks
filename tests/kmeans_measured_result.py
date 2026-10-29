#!/usr/bin/env python3
"""Inject faults in actual k-means results without changing timed kernels."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HOOK = r'''
static void kmeans_test_fault(const char* stage, void* data, size_t elements,
                              int clusters, int iteration, bool device) {
    const char* requested=getenv("KMEANS_TEST_STAGE");
    if(!requested || strcmp(stage,requested))return;
    const char* kind=getenv("KMEANS_TEST_KIND");
    const char* selected=getenv("KMEANS_TEST_ITERATION");
    const char* location=getenv("KMEANS_TEST_LOCATION");
    if(!kind || !selected || !location)abort();
    if(iteration!=atoi(selected))return;
    int place=atoi(location);if(place<0 || place>4)abort();
    size_t positions[]={0,std::min(size_t(31),elements-1),std::min(size_t(32),elements-1),elements/2,elements-1};
    bool many=!strcmp(kind,"many");size_t count=many?12:1;if(count>elements)abort();
    bool integer=!strcmp(stage,"membership") || !strcmp(stage,"history");
    for(size_t i=0;i<count;++i) {
        size_t at=many?i:positions[place];
        if(integer) {
            int* address=static_cast<int*>(data)+at;int value;
            if(device)HIP_CHECK(hipMemcpy(&value,address,sizeof(value),hipMemcpyDeviceToHost));
            else value=*address;
            if(!strcmp(kind,"negative"))value=-1;
            else if(!strcmp(kind,"upper"))value=clusters;
            else if(!strcmp(kind,"int_min"))value=INT_MIN;
            else if(!strcmp(kind,"int_max"))value=INT_MAX;
            else if(!strcmp(kind,"wrong") || many) {if(clusters<2)abort();value=(value+1)%clusters;}
            else abort();
            if(device)HIP_CHECK(hipMemcpy(address,&value,sizeof(value),hipMemcpyHostToDevice));
            else *address=value;
        } else {
            float* address=static_cast<float*>(data)+at;float value;
            if(device)HIP_CHECK(hipMemcpy(&value,address,sizeof(value),hipMemcpyDeviceToHost));
            else value=*address;
            if(!strcmp(kind,"nan"))value=NAN;
            else if(!strcmp(kind,"inf"))value=INFINITY;
            else if(!strcmp(kind,"negative_inf"))value=-INFINITY;
            else if(!strcmp(kind,"ulp") || many)value=std::nextafter(value,INFINITY);
            else abort();
            if(device)HIP_CHECK(hipMemcpy(address,&value,sizeof(value),hipMemcpyHostToDevice));
            else *address=value;
        }
    }
    fprintf(stderr,"KMEANS_TEST_INJECT stage=%s iteration=%d count=%zu\n",stage,iteration,count);
}
'''


def instrument(source):
    header='#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n'
    member='            // Retain each measured assignment without adding an in-loop copy.'
    finish='    bool verified = run_pass();'
    for anchor in (header,member,finish):
        if source.count(anchor)!=1:raise ValueError('Expected exactly one anchor: '+anchor)
    return (source.replace(header,header+HOOK)
            .replace(member,'            kmeans_test_fault("membership",membership_d,npoints,nclusters,iter,true);\n'+member)
            .replace(finish,finish+'''
    if (verified) {
        kmeans_test_fault("inversion",feature_d,static_cast<size_t>(npoints)*nfeatures,nclusters,0,true);
        kmeans_test_fault("centers",h_centers,static_cast<size_t>(nclusters)*nfeatures,nclusters,0,false);
        kmeans_test_fault("history",h_membership_all,membership_elements,nclusters,0,false);
    }'''))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--platform',choices=('cuda','rocm'),required=True)
    ap.add_argument('--compile-only',action='store_true')
    args=ap.parse_args();args.source=args.source.resolve();args.work_dir=args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error('--work-dir must be new')
    src=args.source/'tier2/rodinia_kmeans/hip';dst=args.work_dir/'tier2/rodinia_kmeans/hip';dst.mkdir(parents=True)
    shutil.copytree(args.source/'common_hip',args.work_dir/'common_hip')
    for name in ('Makefile','params.json','cpu_reference.h'):shutil.copy2(src/name,dst/name)
    source=(src/'rodinia_kmeans.hip').read_text();(dst/'rodinia_kmeans.hip').write_text(instrument(source))
    with (args.work_dir/'build.log').open('w') as log:
        subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print('K-means measured-result regression compiled; GPU checks not run');return 0
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','KMEANS_TEST_'))}
    cases=[(n,d,k,t,None,None,0,0,'false') for n,d,k,t in
           ((1,1,1,1),(5,34,5,2),(16,1088,1,2),(1088,1,1088,1),
            (17,3,2,2),(31,7,3,4),(32,34,5,10),(33,34,5,10),
            (255,34,5,2),(256,34,5,2),(257,34,5,2),(1024,34,5,10))]
    cases += [(17,34,5,10,None,None,0,0,'true')]
    for stage in ('inversion','centers'):
        cases += [(33,34,5,10,stage,kind,at,0,'false')
                  for kind in ('ulp','nan','inf','negative_inf') for at in range(5)]
        cases += [(33,34,5,10,stage,'many',0,0,'false')]
    cases += [(33,34,5,10,'membership',kind,at,it,'false')
              for kind in ('wrong','negative','upper','int_min','int_max')
              for at in (0,3,4) for it in (0,4,9)]
    cases += [(33,34,5,10,'history','wrong',at,0,'false') for at in range(5)]
    cases += [(33,34,5,10,'history','many',0,0,'false')]
    results=[]
    for n,d,k,t,stage,kind,at,it,warm in cases:
        label=f'{n}-{d}-{k}-{t}-{stage or "control"}-{kind}-{at}-{it}-{warm}'
        env=dict(clean,BENCH_PARAM_points=str(n),BENCH_PARAM_features=str(d),BENCH_PARAM_clusters=str(k),
                 BENCH_PARAM_max_iter=str(t),BENCH_PARAM_vector_add_warmup=warm)
        if stage:env.update(KMEANS_TEST_STAGE=stage,KMEANS_TEST_KIND=kind,KMEANS_TEST_LOCATION=str(at),KMEANS_TEST_ITERATION=str(it))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/'rodinia_kmeans')],env=env,capture_output=True,text=True,timeout=60)
            code,stdout,stderr=p.returncode,p.stdout,p.stderr
        except subprocess.TimeoutExpired as exc:
            def text(value):return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
            code,stdout,stderr='timeout',text(exc.stdout),text(exc.stderr)
        seconds=time.monotonic()-start
        (args.work_dir/(label+'.stdout')).write_text(stdout);(args.work_dir/(label+'.stderr')).write_text(stderr)
        passed=code==(1 if stage else 0) and seconds<60 and ('KMEANS_TEST_INJECT' in stderr)==bool(stage)
        invalid=stage=='membership' and kind not in ('wrong','many')
        if invalid:passed &= 'K-means verification: FAIL invalid membership' in stderr
        else:
            match=re.search(r'K-means verification: (PASS|FAIL) \((\d+) values, (\d+) mismatches\)',stderr)
            if not match:passed=False
            else:
                status,values,mismatches=match.groups();mismatches=int(mismatches)
                passed &= status==('FAIL' if stage else 'PASS') and int(values)==n*d+n*t+k*d
                passed &= mismatches>0 if stage else mismatches==0
                if stage!='membership':passed &= mismatches==(12 if kind=='many' else 1) if stage else mismatches==0
                passed &= stderr.count('MISMATCH ')==min(10,mismatches)
        try:
            first,second,summary=[json.loads(line) for line in stdout.splitlines()]
            for kernel,name in zip((first,second),('invert_mapping','kmeansPoint')):
                passed &= kernel['type']=='kernel' and kernel['name']==name and kernel['iter']==0
                passed &= kernel['params']==dict(points=n,features=d,clusters=k,max_iter=t,threads_per_block=256,verification='full_cpu_fma_trajectory_v1')
                passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and abs(summary['total_time_ms']-first['time_ms']-second['time_ms'])<.000004
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,exit=code,seconds=seconds,passed=bool(passed),fault=bool(stage)))
        report=dict(complete=len(results)==len(cases),expected=len(cases),total=len(results),passed=sum(r['passed'] for r in results),source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not passed:print('FAIL',label,code,stderr[-2000:],flush=True)
    print(f"{report['passed']}/{report['total']} measured-result cases passed")
    return 0 if report['passed']==report['expected'] else 1


if __name__=='__main__':raise SystemExit(main())
