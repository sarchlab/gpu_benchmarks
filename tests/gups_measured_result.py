#!/usr/bin/env python3
"""Qualify GUPS configurations and reject corruption in the actual measured table."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def replace_once(source, anchor, replacement):
    if source.count(anchor)!=1:
        raise ValueError('GUPS instrumentation boundary changed: '+anchor)
    return source.replace(anchor,replacement)


def instrument(source):
    source=replace_once(source,'    std::vector<GupsWord> expected(table_size), measured(table_size);',
        '    if (getenv("GUPS_TEST_STORAGE_FAILURE")) throw std::bad_alloc();\n'
        '    std::vector<GupsWord> expected(table_size), measured(table_size);')
    source=replace_once(source,'    // Download the actual measured table; no reset or second GPU update run.',r'''
    if (const char* index=getenv("GUPS_TEST_CELL")) {
        const size_t cell=static_cast<size_t>(strtoull(index,nullptr,10));
        const int bit=atoi(getenv("GUPS_TEST_BIT"));
        const int count=getenv("GUPS_TEST_COUNT") ? atoi(getenv("GUPS_TEST_COUNT")) : 1;
        if(cell+count>static_cast<size_t>(table_size) || bit<0 || bit>63 || count<1)abort();
        for(int i=0;i<count;++i) {
            GupsWord value;
            HIP_CHECK(hipMemcpy(&value,d_table+cell+i,sizeof(value),hipMemcpyDeviceToHost));
            value.value^=1ULL<<bit;
            HIP_CHECK(hipMemcpy(d_table+cell+i,&value,sizeof(value),hipMemcpyHostToDevice));
        }
        fprintf(stderr,"GUPS_TEST_CELL %zu bit %d count %d\n",cell,bit,count);
    }
    // Download the actual measured table; no reset or second GPU update run.''')
    source=replace_once(source,'    HIP_CHECK(hipFree(d_table));',r'''
    if(const char* path=getenv("GUPS_TEST_DUMP")) {
        FILE* file=fopen(path,"wb");if(!file)abort();
        if(fwrite(measured.data(),sizeof(GupsWord),table_size,file)!=static_cast<size_t>(table_size) || fclose(file))abort();
    }
    HIP_CHECK(hipFree(d_table));''')
    return source


def build(repo,work,platform,hooks):
    work.mkdir(parents=True)
    shutil.copytree(repo/'common',work/'common')
    folder=work/'tier2/altis_gups/hip';folder.mkdir(parents=True)
    for name in ['Makefile','params.json']:
        shutil.copy2(repo/'tier2/altis_gups/hip'/name,folder/name)
    source=(repo/'tier2/altis_gups/hip/altis_gups.hip').read_text()
    (folder/'altis_gups.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:
        subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'altis_gups'


def check_output(stdout,stderr,code,n,b,t,mode,fault):
    if 'GUPS_TEST_STORAGE_FAILURE' in fault:
        assert code==1 and not stdout and 'FAIL: GUPS host storage unavailable:' in stderr
        return
    rows=[json.loads(line) for line in stdout.splitlines()]
    assert len(rows)==2 and rows[0]['type']=='kernel' and rows[1]['type']=='summary'
    row=rows[0];blocks=(t+b-1)//b
    expected=dict(table_size=n,block_size=b,num_threads=t,num_blocks=blocks,actual_threads=blocks*b,
        total_updates=4*n,atomic_mode=mode,atomic_policy='cas64_retry' if mode=='cas' else 'xor32_pair',
        stream_seed=1,lfsr_polynomial=7,algorithm='altis_lfsr4n_v1',verification='full_table_cpu_v1')
    assert row['name']=='gups_'+mode+'_kernel' and row['iter']==0 and row['params']==expected
    assert math.isfinite(row['time_ms']) and row['time_ms']>0
    assert math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
    assert row['time_ms']==rows[1]['total_time_ms']
    count=int(fault.get('GUPS_TEST_COUNT','1')) if fault else 0
    assert code==(1 if fault else 0)
    assert f'{"FAIL" if fault else "PASS"} (verified {n} table entries, {4*n} logical updates, {count} mismatches)' in stderr
    assert stderr.count('Mismatch at entry ')==min(count,10)
    if fault:
        assert 'GUPS_TEST_CELL '+fault['GUPS_TEST_CELL'] in stderr and 'GPU=' in stderr and 'CPU=' in stderr
    assert 'HIP error' not in stderr


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda')
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--sweep',action='store_true')
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve()
    instrumented=build(repo,work/'instrumented',args.platform,True)
    if args.compile_only:
        print('GUPS measured-table/storage hooks compiled; GPU tests not executed');return
    production=build(repo,work/'production',args.platform,False)
    manifest=json.loads((repo/'tier2/altis_gups/hip/params.json').read_text())
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','GUPS_TEST_'))}
    controls=[(n,b,t,mode) for n,b,t in [(1048576,256,65536),(1,1,1),(1,256,1),(16,3,7),
              (1024,32,33),(65536,1024,65536),(131072,64,65537)] for mode in ['cas','xor']]
    if args.sweep:
        values={p['name']:p['values'] for p in manifest['non_scaling_params']}
        controls+=list(itertools.product(manifest['scaling_param']['values'],values['block_size'],values['num_threads'],values['atomic_mode']))
    cases=[(*c,{}) for c in dict.fromkeys(controls)]
    for n in [1,16,65536,16777216]:
        for mode in ['cas','xor']:
            for cell in dict.fromkeys([0,n//2,n-1]):
                for bit in [0,31,32,63]:
                    cases.append((n,256,65536,mode,dict(GUPS_TEST_CELL=str(cell),GUPS_TEST_BIT=str(bit))))
    for mode in ['cas','xor']:
        cases.append((65536,256,65536,mode,dict(GUPS_TEST_CELL='1024',GUPS_TEST_BIT='32',GUPS_TEST_COUNT='12')))
        cases.append((65536,256,65536,mode,dict(GUPS_TEST_STORAGE_FAILURE='1')))
    report=[]
    for index,(n,b,t,mode,fault) in enumerate(cases):
        label=f'{index}-n{n}-b{b}-t{t}-{mode}'
        env=dict(clean,BENCH_PARAM_table_size=str(n),BENCH_PARAM_block_size=str(b),BENCH_PARAM_num_threads=str(t),
                 BENCH_PARAM_atomic_mode=mode,BENCH_PARAM_vector_add_warmup='false' if fault else 'true',**fault)
        start=time.monotonic()
        try:
            run=subprocess.run([str(instrumented if fault else production)],env=env,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired as error:
            (work/(label+'.stdout')).write_bytes(error.stdout or b'')
            (work/(label+'.stderr')).write_bytes(error.stderr or b'')
            raise
        elapsed=time.monotonic()-start
        (work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check_output(run.stdout,run.stderr,run.returncode,n,b,t,mode,fault)
        report.append(dict(case=label,table_size=n,block_size=b,num_threads=t,atomic_mode=mode,fault=fault,
                           exit_code=run.returncode,wall_seconds=elapsed))
        (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'{len(cases)} production/control/measured-table/storage cases pass')


if __name__=='__main__':
    main()
