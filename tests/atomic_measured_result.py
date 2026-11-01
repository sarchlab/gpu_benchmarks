#!/usr/bin/env python3
"""Qualify fixed atomic attempts and reject corrupted actual measured outputs."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

def replace_once(s,anchor,replacement):
    if s.count(anchor)!=1: raise ValueError('Atomic instrumentation boundary changed: '+anchor)
    return s.replace(anchor,replacement)

def instrument(s):
    s=replace_once(s,'    std::vector<uint32_t> successes(is_cas?num_threads:0);',
        '    if(getenv("ATOMIC_TEST_STORAGE_FAILURE"))throw std::bad_alloc();\n    std::vector<uint32_t> successes(is_cas?num_threads:0);')
    anchor='    // Verify the actual measured counter and every thread\'s CAS success count.'
    s=replace_once(s,anchor,r"""
    if(getenv("ATOMIC_TEST_COUNTER")) {
        unsigned int value;HIP_CHECK(hipMemcpy(&value,d_counter,sizeof(value),hipMemcpyDeviceToHost));
        value^=1u;HIP_CHECK(hipMemcpy(d_counter,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    if(const char* cell=getenv("ATOMIC_TEST_COUNT_CELL")) {
        const size_t index=strtoull(cell,nullptr,10);if(!is_cas || index>=num_threads)abort();
        unsigned int value=UINT_MAX;HIP_CHECK(hipMemcpy(d_successes+index,&value,sizeof(value),hipMemcpyHostToDevice));
    }
"""+anchor)
    return replace_once(s,'    const atomic_bench::Verification result=atomic_bench::verify(p,counter,successes);',
        '    if(getenv("ATOMIC_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();\n    const atomic_bench::Verification result=atomic_bench::verify(p,counter,successes);')

def build(repo,work,platform,hooks):
    work.mkdir(parents=True);shutil.copytree(repo/'common_hip',work/'common_hip')
    folder=work/'tier1/atomic_operations/hip';folder.mkdir(parents=True)
    for name in ['Makefile','atomic_reference.h','params.json']:shutil.copy2(repo/'tier1/atomic_operations/hip'/name,folder/name)
    source=(repo/'tier1/atomic_operations/hip/atomic_operations.hip').read_text()
    (folder/'atomic_operations.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:
        subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'atomic_operations'

def check_output(stdout,stderr,code,ops,blocks,block,operation,fault):
    if 'ATOMIC_TEST_STORAGE_FAILURE' in fault:
        assert code==1 and not stdout and 'FAIL: atomic setup/verification unavailable:' in stderr;return
    rows=[json.loads(x) for x in stdout.splitlines()];assert len(rows)==2
    row,summary=rows;threads=blocks*block;attempts=threads*ops;cas=operation=='atomicCAS'
    want=dict(ops_per_thread=ops,num_blocks=blocks,threads_per_block=block,atomic_op=operation,
              num_threads=threads,attempted_operations=attempts,atomic_read_operations=attempts if cas else 0,
              total_atomic_operations=attempts*(2 if cas else 1),success_count_stores=threads if cas else 0,
              counter_bits=32,initial_counter=0,workload='fixed_attempts_v1',verification='full_counter_and_cas_counts_v1')
    assert row['type']=='kernel' and row['name']==operation+'_kernel' and row['iter']==0 and row['params']==want,row
    assert math.isfinite(row['time_ms']) and row['time_ms']>0 and math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
    assert summary==dict(type='summary',total_time_ms=row['time_ms'])
    if 'ATOMIC_TEST_REFERENCE_FAILURE' in fault:
        assert code==1 and 'FAIL: atomic setup/verification unavailable:' in stderr and 'PASS (' not in stderr;return
    match=re.search(r'(PASS|FAIL) \(counter=(\d+); verified (\d+) CAS counts; successful CAS=(\d+); (\d+) mismatches\)',stderr)
    assert match,stderr
    verdict,counter,count,successes,errors=match.groups();counter,count,successes,errors=map(int,[counter,count,successes,errors])
    assert code==(1 if fault else 0) and verdict==('FAIL' if fault else 'PASS') and bool(errors)==bool(fault),stderr
    assert count==(threads if cas else 0)
    if not fault:
        if cas: assert ops<=successes<=attempts and counter==successes%(1<<32)
        else: assert counter==(attempts%(1<<32) if operation=='atomicAdd' else 0) and successes==0
    assert 'HIP error' not in stderr

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda')
    ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--compile-only',action='store_true');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve()
    hooks=build(repo,work/'instrumented',args.platform,True)
    if args.compile_only:print('Atomic result/storage hooks compiled; GPU tests not executed');return
    production=build(repo,work/'production',args.platform,False)
    manifest=json.loads((repo/'tier1/atomic_operations/hip/params.json').read_text())
    cases=[(ops,2048,256,op,{}) for ops in manifest['scaling_param']['values'] for op in ['atomicAdd','atomicCAS','atomicMin']]
    cases += [(ops,blocks,block,op,{}) for ops,blocks,block in [(1,1,1),(3,1,1),(7,3,17),(31,2,1024)] for op in ['atomicAdd','atomicCAS','atomicMin']]
    cases += [(7,3,17,op,{'ATOMIC_TEST_COUNTER':'1'}) for op in ['atomicAdd','atomicCAS','atomicMin']]
    cases += [(7,3,17,'atomicCAS',{'ATOMIC_TEST_COUNT_CELL':str(i)}) for i in [0,1,25,50]]
    cases += [(7,3,17,'atomicCAS',{key:'1'}) for key in ['ATOMIC_TEST_STORAGE_FAILURE','ATOMIC_TEST_REFERENCE_FAILURE']]
    cases += [(65536,2048,256,'atomicCAS',{'ATOMIC_TEST_COUNT_CELL':'524287'})]
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','ATOMIC_TEST_'))};report=[]
    for index,(ops,blocks,block,op,fault) in enumerate(cases):
        label=f'{index}-{op}-{ops}-{blocks}-{block}';start=time.monotonic()
        env=dict(clean,BENCH_PARAM_ops_per_thread=str(ops),BENCH_PARAM_num_blocks=str(blocks),BENCH_PARAM_threads_per_block=str(block),BENCH_PARAM_atomic_op=op,BENCH_PARAM_vector_add_warmup='false' if fault else 'true',**fault)
        try: run=subprocess.run([str(hooks if fault else production)],env=env,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired as e:
            (work/(label+'.stdout')).write_bytes(e.stdout or b'');(work/(label+'.stderr')).write_bytes(e.stderr or b'');raise
        elapsed=time.monotonic()-start
        (work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check_output(run.stdout,run.stderr,run.returncode,ops,blocks,block,op,fault)
        report.append(dict(case=label,ops=ops,blocks=blocks,block=block,operation=op,fault=fault,exit_code=run.returncode,wall_seconds=elapsed))
        (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'{len(cases)} atomic production/measured-output/storage cases pass')
if __name__=='__main__':main()
