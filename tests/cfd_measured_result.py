#!/usr/bin/env python3
"""Check complete actual CFD results, omitted solver stages and setup failures."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

def replace_once(s,old,new):
    assert s.count(old)==1,old
    return s.replace(old,new)

def instrument(s):
    s=replace_once(s,'static void upload_farfield()',r'''
__global__ void cfd_test_fault(float* target,size_t first,int count,int mode) {
    if(threadIdx.x==0&&blockIdx.x==0)for(int j=0;j<count;++j) {
        float& x=target[first+j];x=mode==1?NAN:mode==2?INFINITY:mode==3?-INFINITY:x+1.0f;
    }
}
static void upload_farfield()''')
    s=replace_once(s,'    cfd::Input in=cfd::input(p);','    if(getenv("CFD_TEST_SETUP_FAILURE"))throw std::bad_alloc();\n    cfd::Input in=cfd::input(p);')
    anchor='    const cfd::Reference expected=cfd::reference(p,in);'
    s=replace_once(s,anchor,r'''
    if(const char* name=getenv("CFD_TEST_BUFFER")) {
        float* target=strcmp(name,"state")==0?state:strcmp(name,"old")==0?old:strcmp(name,"flux")==0?flux:factor;
        size_t first=strtoull(getenv("CFD_TEST_INDEX"),nullptr,10);
        int count=getenv("CFD_TEST_COUNT")?atoi(getenv("CFD_TEST_COUNT")):1;
        int mode=getenv("CFD_TEST_MODE")?atoi(getenv("CFD_TEST_MODE")):0;
        cfd_test_fault<<<1,1>>>(target,first,count,mode);HIP_CHECK(hipGetLastError());
    }
    if(getenv("CFD_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();
'''+anchor)
    targets=[('COPY','bench_lap_add(&times[0]',False),('FACTOR','bench_lap_add(&times[1]',False),('FLUX','bench_lap_add(&times[2]',True),('UPDATE','bench_lap_add(&times[3]',True)]
    for key,anchor,stage in targets:
        lines=[l for l in s.splitlines() if anchor in l];assert len(lines)==1
        condition='iteration==9'+('&&stage==2' if stage else '')
        s=replace_once(s,lines[0],f'        if(!getenv("CFD_TEST_SKIP_{key}")||!({condition})) '+lines[0].lstrip())
    return s

def build(repo,work,platform,hooks):
    work.mkdir(parents=True);shutil.copytree(repo/'common',work/'common');folder=work/'tier2/altis_cfd/hip';folder.mkdir(parents=True)
    for name in ['Makefile','cfd_reference.h','cfd_kernels.h','params.json']:shutil.copy2(repo/'tier2/altis_cfd/hip'/name,folder/name)
    source=(repo/'tier2/altis_cfd/hip/altis_cfd.hip').read_text();(folder/'altis_cfd.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'altis_cfd'

def check(run,n,b,fault):
    if 'CFD_TEST_SETUP_FAILURE' in fault:
        assert run.returncode==1 and not run.stdout and 'FAIL: CFD setup/reference unavailable:' in run.stderr;return
    rows=[json.loads(l) for l in run.stdout.splitlines()];assert len(rows)==5
    padded=(n+b-1)//b*b;side=1
    while 6*side**3<n:side+=1
    names=['copy_variables_kernel','compute_step_factor_kernel','compute_flux_kernel','rk_update_kernel']
    for k,row in enumerate(rows[:-1]):
        params=row['params'];launches=10 if k<2 else 30
        expected=dict(size=n,block_size=b,padded_cells=padded,grid_size=padded//b,mesh_cube_side=side,timesteps=10,rk_stages=3,kernel_launches=launches,cell_updates=padded*launches,
            device_buffer_bytes=132*padded,host_input_reference_bytes=216*padded+min(20*padded,4*1024*1024),algorithm='altis_euler3d_rk3_v1',input='tetrahedral_perturbed_flow_v1',boundary='y0_slip_wall_other_farfield',precision='fp32',verification='full_final_aux_cpu_fp64_v1')
        assert params==expected,(params,expected)
        assert row['type']=='kernel' and row['name']==names[k] and row['iter']==0
        assert math.isfinite(row['time_ms']) and row['time_ms']>0 and math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
    assert rows[-1]['type']=='summary' and abs(sum(r['time_ms'] for r in rows[:-1])-rows[-1]['total_time_ms'])<=2.1e-6
    assert run.returncode==(1 if fault else 0),run.stderr
    if 'CFD_TEST_REFERENCE_FAILURE' in fault:
        assert 'FAIL: CFD setup/reference unavailable:' in run.stderr and 'CFD verification:' not in run.stderr;return
    assert f'CFD verification: {"FAIL" if fault else "PASS"} (verified {16*padded} values:' in run.stderr,run.stderr
    if 'CFD_TEST_BUFFER' in fault:
        count=int(fault.get('CFD_TEST_COUNT','1'));assert f'; {count} mismatches)' in run.stderr and run.stderr.count('Mismatch ')==min(count,10),run.stderr
    elif fault:assert 'Mismatch ' in run.stderr,run.stderr
    else:assert '; 0 mismatches)' in run.stderr

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda');ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--compile-only',action='store_true');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve();instrumented=build(repo,work/'instrumented',args.platform,True)
    if args.compile_only:print('CFD actual-output/omission/failure hooks compiled; no GPU execution claimed');return
    production=build(repo,work/'production',args.platform,False);manifest=json.loads((repo/'tier2/altis_cfd/hip/params.json').read_text());sizes=manifest['scaling_param']['values']
    cases=[(n,256,{}) for n in sizes]
    cases += [(n,b,{}) for n,b in [(1,1),(2,7),(5,32),(6,64),(7,128),(31,256),(503,7),(503,31),(503,512),(503,1024),(97000,256)]]
    n,b=503,256;padded=(n+b-1)//b*b
    for buffer in ['state','old','flux','factor']:
        length=(1 if buffer=='factor' else 5)*padded
        for i in sorted({0,n-1,padded-1,length-1}):
            for mode in ['0','1','2','3']:cases.append((n,b,dict(CFD_TEST_BUFFER=buffer,CFD_TEST_INDEX=str(i),CFD_TEST_MODE=mode)))
        if buffer!='factor':cases.append((n,b,dict(CFD_TEST_BUFFER=buffer,CFD_TEST_INDEX='1040',CFD_TEST_COUNT='12')))
    for buffer in ['state','old','flux','factor']:
        n=max(sizes);padded=(n+255)//256*256;length=(1 if buffer=='factor' else 5)*padded
        cases.append((n,256,dict(CFD_TEST_BUFFER=buffer,CFD_TEST_INDEX=str(length-1))))
    cases += [(503,256,{'CFD_TEST_SKIP_'+s:'1'}) for s in ['COPY','FACTOR','FLUX','UPDATE']]
    cases += [(503,256,{s:'1'}) for s in ['CFD_TEST_SETUP_FAILURE','CFD_TEST_REFERENCE_FAILURE']]
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','CFD_TEST_'))};report=[]
    for index,(n,b,fault) in enumerate(cases):
        env=dict(clean,BENCH_PARAM_size=str(n),BENCH_PARAM_block_size=str(b),BENCH_PARAM_vector_add_warmup='false' if fault else 'true',**fault)
        start=time.monotonic();run=subprocess.run([str(instrumented if fault else production)],env=env,text=True,capture_output=True,timeout=60);elapsed=time.monotonic()-start
        label=f'{index}-{n}-{b}';(work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check(run,n,b,fault);report.append(dict(case=label,size=n,block=b,fault=fault,exit_code=run.returncode,wall_seconds=elapsed));(work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'PASS: all {len(sizes)} declared configurations and {len(cases)} total cases')
if __name__=='__main__':main()
