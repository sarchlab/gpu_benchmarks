#!/usr/bin/env python3
"""Verify complete measured N-body state, including odd/even final buffers."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def instrument(source):
    anchor='    // Verify the actual measured final state, including every integration step.'
    if source.count(anchor)!=1:
        raise ValueError('expected one actual measured-state verification boundary')
    hook=r'''
    if (const char* kind=getenv("NBODY_TEST_KIND")) {
        const int component=atoi(getenv("NBODY_TEST_COMPONENT"));
        const int where=atoi(getenv("NBODY_TEST_LOCATION"));
        const int index=where==0 ? 0 : where==1 ? N/2 : N-1;
        if(component<0 || component>7 || where<0 || where>2) abort();
        float value=1e20f;
        if(strcmp(kind,"nan")==0)value=NAN;
        else if(strcmp(kind,"inf")==0)value=INFINITY;
        else if(strcmp(kind,"negative_inf")==0)value=-INFINITY;
        else if(strcmp(kind,"finite")!=0)abort();
        float* target=reinterpret_cast<float*>(component<4 ? final_positions : d_vel);
        HIP_CHECK(hipMemcpy(target+(size_t)index*4+(component%4),&value,sizeof(float),hipMemcpyHostToDevice));
        fprintf(stderr,"NBODY_TEST_INJECT %s component=%d body=%d\n",kind,component,index);
    }
'''
    source=source.replace(anchor,hook+anchor)
    anchor='    std::vector<float3> acc(N, {0.0f, 0.0f, 0.0f});'
    if source.count(anchor)!=1:raise ValueError('expected one reference allocation')
    return source.replace(anchor,'    if (getenv("NBODY_TEST_REFERENCE_FAILURE")) throw std::bad_alloc();\n'+anchor)


def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda')
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--compile-only',action='store_true')
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve();work.mkdir(parents=True)
    shutil.copytree(repo/'common',work/'common');folder=work/'tier2/cuda_nbody/hip';folder.mkdir(parents=True)
    for name in ('Makefile','params.json'):shutil.copy2(repo/'tier2/cuda_nbody/hip'/name,folder/name)
    (folder/'cuda_nbody.hip').write_text(instrument((repo/'tier2/cuda_nbody/hip/cuda_nbody.hip').read_text()))
    with (work/'build.log').open('w') as log:
        subprocess.run(['make','PLATFORM='+args.platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    if args.compile_only:
        print('Measured GPU-state/reference-failure hooks compiled; GPU tests not executed');return
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','NBODY_TEST_'))};results=[]
    cases=[(n,t,{}) for n,t in ((1,1),(2,2),(255,1),(256,2),(257,3),(513,2),(4097,1),(8193,2))]
    for n,t in ((257,2),(4097,3)):
        cases.append((n,t,{}))
        for c in range(8):
            for kind in ('finite','nan','inf','negative_inf'):
                for location in range(3):
                    cases.append((n,t,dict(NBODY_TEST_KIND=kind,NBODY_TEST_COMPONENT=str(c),NBODY_TEST_LOCATION=str(location))))
    cases.append((257,2,dict(NBODY_TEST_REFERENCE_FAILURE='1')))
    for index,(n,t,injection) in enumerate(cases):
        label=f'{index}-n{n}-t{t}';env=dict(clean,BENCH_PARAM_bodies=str(n),BENCH_PARAM_timesteps=str(t),BENCH_PARAM_vector_add_warmup='false',**injection)
        start=time.monotonic();p=subprocess.run([str(folder/'cuda_nbody')],env=env,capture_output=True,text=True,timeout=60);elapsed=time.monotonic()-start
        (work/(label+'.stdout')).write_text(p.stdout);(work/(label+'.stderr')).write_text(p.stderr)
        records=[json.loads(x) for x in p.stdout.splitlines()];assert len(records)==2 and records[0]['type']=='kernel' and records[1]['type']=='summary'
        assert records[0]['params']==dict(bodies=n,timesteps=t,verification='full_cpu_v1')
        assert records[0]['time_ms']==records[1]['total_time_ms']
        if 'NBODY_TEST_KIND' in injection:
            c=int(injection['NBODY_TEST_COMPONENT']);location=int(injection['NBODY_TEST_LOCATION']);body=0 if location==0 else n//2 if location==1 else n-1
            assert f'NBODY_TEST_INJECT {injection["NBODY_TEST_KIND"]} component={c} body={body}' in p.stderr
            assert p.returncode==1 and f'FAIL (verified {n} bodies, {t} timesteps, positions and velocities,' in p.stderr
        elif injection:
            assert p.returncode==1 and 'FAIL: complete CPU reference unavailable:' in p.stderr
        else:
            assert p.returncode==0 and f'PASS (verified {n} bodies, {t} timesteps, positions and velocities, 0 mismatches)' in p.stderr
        assert 'HIP error' not in p.stderr and 'runtime error:' not in p.stderr
        results.append(dict(case=label,bodies=n,timesteps=t,injection=injection,exit_code=p.returncode,wall_seconds=elapsed))
        (work/'report.json').write_text(json.dumps(results,indent=2)+'\n')
        print(label,'PASS',flush=True)
    print(f'{len(cases)} complete measured-state control/fault/reference-resource cases pass')


if __name__=='__main__':main()
