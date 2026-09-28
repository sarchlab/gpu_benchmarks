#!/usr/bin/env python3
"""Qualify every declared local memory workload and actual measured-result failures."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

NAMES=['readOnly_kernel','writeOnly_kernel','readWrite_kernel']
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Instrumentation anchor changed: '+old)
    return text.replace(old,new)
def instrument(source):
    source=replace_once(source,'    size_t errors=0;\n    for(size_t offset=0;offset<count;) {',r'''    size_t errors=0;
    if(getenv("DM_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();
    if(const char* target=getenv("DM_TEST_TARGET"))if(strcmp(target,name)==0) {
        const char* position=getenv("DM_TEST_POSITION");
        const size_t index=strcmp(position,"last")==0?count-1:strtoull(position,nullptr,10);
        const size_t changes=getenv("DM_TEST_COUNT")?strtoull(getenv("DM_TEST_COUNT"),nullptr,10):1;
        if(!changes || index>=count || changes>count-index)abort();
        for(size_t i=index;i<index+changes;++i) {
            float value;HIP_CHECK(hipMemcpy(&value,device+i,sizeof(value),hipMemcpyDeviceToHost));
            const char* kind=getenv("DM_TEST_VALUE");
            value=strcmp(kind,"finite")==0?std::nextafter(value,INFINITY):strtof(kind,nullptr);
            HIP_CHECK(hipMemcpy(const_cast<float*>(device)+i,&value,sizeof(value),hipMemcpyHostToDevice));
        }
        fprintf(stderr,"DM_TEST_FAULT %s %zu %zu\n",name,index,changes);
    }
    for(size_t offset=0;offset<count;) {''')
    source=replace_once(source,'    std::vector<float> scratch(', '    if(getenv("DM_TEST_STORAGE_FAILURE"))throw std::bad_alloc();\n    std::vector<float> scratch(')
    source=replace_once(source,'        if(!h_init) throw std::bad_alloc();','        if(getenv("DM_TEST_INPUT_FAILURE"))throw std::bad_alloc();\n        if(!h_init) throw std::bad_alloc();')
    source=replace_once(source,'            if(!h_rw) throw std::bad_alloc();','            if(getenv("DM_TEST_RESET_FAILURE"))throw std::bad_alloc();\n            if(!h_rw) throw std::bad_alloc();')
    for name in NAMES:
        anchor='            '+name+'<<<'
        source=replace_once(source,anchor,'            if(!getenv("DM_TEST_SKIP") || strcmp(getenv("DM_TEST_SKIP"),"'+name+'")!=0) '+name+'<<<')
    return source

def build(repo,work,platform,hooks):
    work.mkdir(parents=True);shutil.copytree(repo/'common',work/'common');folder=work/'tier2/shoc_devicememory';folder.mkdir(parents=True)
    for name in ['Makefile','params.json','cpu_reference.h']:shutil.copy2(repo/'tier2/shoc_devicememory'/name,folder/name)
    source=(repo/'tier2/shoc_devicememory/shoc_devicememory.hip').read_text()
    (folder/'shoc_devicememory.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'shoc_devicememory'

def check(run,mib,block,mode,control):
    rows=[json.loads(x) for x in run.stdout.splitlines()];n=mib*262144;blocks=min((n+block-1)//block,65535);stride=blocks*block
    names=[NAMES[0]] if mode=='read' else [NAMES[1]] if mode=='write' else NAMES
    failed_setup=bool(set(control)&{'DM_TEST_STORAGE_FAILURE','DM_TEST_INPUT_FAILURE'})
    reference_failure='DM_TEST_REFERENCE_FAILURE' in control;reset_failure='DM_TEST_RESET_FAILURE' in control
    if failed_setup:
        assert run.returncode==1 and not rows and 'FAIL: device-memory setup/reference unavailable:' in run.stderr;return
    if reference_failure:names=names[:1]
    if reset_failure:names=names[:2]
    kernels=rows if reference_failure or reset_failure else rows[:-1]
    assert len(kernels)==len(names),(rows,run.stderr)
    for row,name in zip(kernels,names):
        assert row['type']=='kernel' and row['name']==name and row['iter']==0
        read=name==NAMES[0];write=name==NAMES[1];count=min(n,stride) if read else n
        expected=dict(array_size_mb=mib,block_size=block,transfer_mode=mode,elements=n,grid_size=blocks,launched_threads=stride,output_elements=count,logical_read_bytes=0 if write else 4*n,logical_write_bytes=4*count,algorithm='local_grid_stride_v1',input='periodic_1000_fp32_v1',precision='fp32',allocation='device_malloc',verification='full_cpu_exact_v1')
        assert row['params']==expected,(row['params'],expected)
        assert math.isfinite(row['time_ms']) and row['time_ms']>=0 and math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
        if not reference_failure and not reset_failure:
            fault=control.get('DM_TEST_TARGET')==name or control.get('DM_TEST_SKIP')==name
            assert f'{name}: {"FAIL" if fault else "PASS"} (verified {count} measured values;' in run.stderr,run.stderr
    if reference_failure or reset_failure:
        assert run.returncode==1 and 'FAIL: device-memory setup/reference unavailable:' in run.stderr;return
    assert rows[-1]['type']=='summary' and abs(rows[-1]['total_time_ms']-sum(r['time_ms'] for r in kernels))<=0.000003
    assert run.returncode==(1 if control else 0),run.stderr
    if 'DM_TEST_TARGET' in control:
        errors=int(control.get('DM_TEST_COUNT','1'))
        assert f'FAIL: complete measured-output verification; {errors} total mismatches' in run.stderr
        assert run.stderr.count('Mismatch ')==min(errors,10)
    elif 'DM_TEST_SKIP' in control:assert 'FAIL: complete measured-output verification;' in run.stderr
    else:assert 'PASS: complete measured-output verification; 0 total mismatches' in run.stderr
    assert 'HIP error' not in run.stderr

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda');ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--compile-only',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];work=args.work_dir.resolve();hooks=build(root,work/'instrumented',args.platform,True)
    if args.compile_only:print('Measured-output hooks compiled; GPU execution not performed');return
    production=build(root,work/'production',args.platform,False);config=json.loads((root/'tier2/shoc_devicememory/params.json').read_text())
    sizes=config['scaling_param']['values'];cases=[(mib,256,mode,{}) for mib in sizes for mode in ['read','write','readwrite']]
    cases += [(mib,block,'readwrite',{}) for mib in [1,96] for block in [1,32,64,255,1024]]
    for name in NAMES:
        for position in ['0','131072','last']:
            for value in ['finite','nan','inf','-inf']:cases.append((1,256,'readwrite',dict(DM_TEST_TARGET=name,DM_TEST_POSITION=position,DM_TEST_VALUE=value)))
        cases += [(4096,256,'readwrite',dict(DM_TEST_TARGET=name,DM_TEST_POSITION='last',DM_TEST_VALUE='finite')),(96,256,'readwrite',dict(DM_TEST_TARGET=name,DM_TEST_POSITION='1048575',DM_TEST_COUNT='2',DM_TEST_VALUE='finite')),(1,256,'readwrite',dict(DM_TEST_TARGET=name,DM_TEST_POSITION='100',DM_TEST_COUNT='12',DM_TEST_VALUE='finite')),(1,256,'readwrite',dict(DM_TEST_SKIP=name))]
    cases += [(1,256,'readwrite',{key:'1'}) for key in ['DM_TEST_STORAGE_FAILURE','DM_TEST_INPUT_FAILURE','DM_TEST_REFERENCE_FAILURE','DM_TEST_RESET_FAILURE']]
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','DM_TEST_'))};report=[]
    for index,(mib,block,mode,control) in enumerate(cases):
        label=f'{index}-{mib}-{block}-{mode}';env=dict(clean,BENCH_PARAM_array_size_mb=str(mib),BENCH_PARAM_block_size=str(block),BENCH_PARAM_transfer_mode=mode,BENCH_PARAM_vector_add_warmup='false' if control else 'true',**control);start=time.monotonic()
        try:run=subprocess.run([str(hooks if control else production)],env=env,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired as e:
            (work/(label+'.stdout')).write_bytes(e.stdout or b'');(work/(label+'.stderr')).write_bytes(e.stderr or b'');raise
        seconds=time.monotonic()-start;(work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check(run,mib,block,mode,control)
        report.append(dict(case=label,mib=mib,block=block,mode=mode,control=control,exit_code=run.returncode,wall_seconds=seconds));(work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(label,'PASS',f'{seconds:.3f}s',flush=True)
    print(f'PASS: {len(cases)} complete production/control/fault/failure cases')
if __name__=='__main__':main()
