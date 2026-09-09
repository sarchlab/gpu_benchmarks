#!/usr/bin/env python3
"""Verify actual measured NW matrices, optional full sweep, and corruption rejection."""
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
    if source.count(anchor) != 1:
        raise ValueError('NW instrumentation boundary changed: '+anchor)
    return source.replace(anchor, replacement)


def instrument(source):
    source = replace_once(source, '    // Download the actual measured matrix; no second GPU verification run.', r'''
    if (const char* index = getenv("NW_TEST_CELL")) {
        const size_t cell = static_cast<size_t>(strtoull(index,nullptr,10));
        if(cell >= geometry.cells) abort();
        int value;
        HIP_CHECK(hipMemcpy(&value,d_matrix+cell,sizeof(int),hipMemcpyDeviceToHost));
        value ^= 1;
        HIP_CHECK(hipMemcpy(d_matrix+cell,&value,sizeof(int),hipMemcpyHostToDevice));
        fprintf(stderr,"NW_TEST_CELL %zu\n",cell);
    }
    // Download the actual measured matrix; no second GPU verification run.''')
    source = replace_once(source, '    std::vector<int> previous(columns), current(columns);',
                          '    if (getenv("NW_TEST_REFERENCE_FAILURE")) throw std::bad_alloc();\n'
                          '    std::vector<int> previous(columns), current(columns);')
    source = replace_once(source, '    bench_pass_warm();', r'''
    if (const char* index = getenv("NW_TEST_SCORE_CELL")) {
        const size_t cell = static_cast<size_t>(strtoull(index,nullptr,10));
        if(cell >= geometry.cells) abort();
        const int value = 1000000;
        HIP_CHECK(hipMemcpy(d_reference+cell,&value,sizeof(int),hipMemcpyHostToDevice));
        fprintf(stderr,"NW_TEST_SCORE_CELL %zu\n",cell);
    }
    bench_pass_warm();''')
    source = replace_once(source, '    HIP_CHECK(hipFree(d_reference));', r'''
    if (const char* path = getenv("NW_TEST_DUMP")) {
        FILE* file=fopen(path,"wb");if(!file)abort();
        if(fwrite(h_matrix.data(),sizeof(int),geometry.cells,file)!=geometry.cells || fclose(file))abort();
    }
    HIP_CHECK(hipFree(d_reference));''')
    return source


def build(repo, work, platform, hooks):
    work.mkdir(parents=True)
    shutil.copytree(repo/'common', work/'common')
    folder=work/'tier2/rodinia_nw';folder.mkdir(parents=True)
    for name in ('Makefile','params.json'):
        shutil.copy2(repo/'tier2/rodinia_nw'/name,folder/name)
    source=(repo/'tier2/rodinia_nw/rodinia_nw.hip').read_text()
    (folder/'rodinia_nw.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:
        subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'rodinia_nw'


def check_output(stdout, stderr, code, n, block, penalty, fault):
    length=((n+block-1)//block)*block;cells=(length+1)**2
    rows=[json.loads(line) for line in stdout.splitlines()]
    kernels=rows[:-1]
    names=['nw_kernel1']+(['nw_kernel2'] if length>block else [])
    assert [r['name'] for r in kernels]==names and rows[-1]['type']=='summary'
    expected=dict(sequence_length=n,block_size=block,penalty=penalty,effective_sequence_length=length,
                  matrix_cells=cells,total_cell_updates=length*length,input_seed=7,
                  algorithm='rodinia31_blosum62_v1',verification='full_dp_cpu_v1')
    for row in kernels:
        blocks=length//block
        launches=blocks if row['name']=='nw_kernel1' else blocks-1
        want=dict(expected,kernel_launches=launches,cell_updates=launches*(launches+1)//2*block*block)
        assert row['type']=='kernel' and row['iter']==0 and row['params']==want
        assert math.isfinite(row['time_ms']) and row['time_ms']>0
        assert math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
    assert abs(sum(r['time_ms'] for r in kernels)-rows[-1]['total_time_ms'])<=3e-6
    if not fault:
        assert code==0 and f'PASS (verified {cells} DP cells including borders, 0 mismatches)' in stderr
    elif 'NW_TEST_REFERENCE_FAILURE' in fault:
        assert code==1 and 'FAIL: NW host/reference storage unavailable:' in stderr
    else:
        assert code==1 and f'FAIL (verified {cells} DP cells including borders,' in stderr
        if 'NW_TEST_CELL' in fault:
            assert '1 mismatches)' in stderr and 'NW_TEST_CELL '+fault['NW_TEST_CELL'] in stderr
        else:
            assert 'NW_TEST_SCORE_CELL '+fault['NW_TEST_SCORE_CELL'] in stderr
        assert stderr.count('Mismatch at (')<=10
    assert 'HIP error' not in stderr


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda')
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--sweep',action='store_true',help='Also qualify every declared configuration with the pristine production binary')
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve()
    instrumented=build(repo,work/'instrumented',args.platform,True)
    if args.compile_only:
        print('NW measured-result/score/resource hooks compiled; GPU tests not executed');return
    production=build(repo,work/'production',args.platform,False)
    manifest=json.loads((repo/'tier2/rodinia_nw/params.json').read_text())
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','NW_TEST_'))}
    triples=[(2048,16,2),(1,1,1),(1,16,2),(16,16,10),(17,16,5),(17,3,2),(33,32,1),
             (64,32,5),(65,16,2),(256,16,1),(257,32,10),(4097,16,2)]
    if args.sweep:
        controls={p['name']:p['values'] for p in manifest['non_scaling_params']}
        triples+=list(itertools.product(manifest['scaling_param']['values'],controls['block_size'],controls['penalty']))
    cases=[(n,b,p,{}) for n,b,p in dict.fromkeys(triples)]
    for n,b,p in [(1,16,2),(17,3,2),(65,16,5),(4097,16,2)]:
        length=((n+b-1)//b)*b;cols=length+1
        for k in dict.fromkeys([0,length,length*cols,cols+1,(length//2)*cols+length//2,length*cols+length-1,cols*cols-1]):
            cases.append((n,b,p,dict(NW_TEST_CELL=str(k))))
    for k in [66,32*65+32,65*65-1]:
        cases.append((64,16,2,dict(NW_TEST_SCORE_CELL=str(k))))
    cases.append((64,16,2,dict(NW_TEST_REFERENCE_FAILURE='1')))
    report=[]
    for index,(n,b,p,fault) in enumerate(cases):
        label=f'{index}-n{n}-b{b}-p{p}'
        env=dict(clean,BENCH_PARAM_sequence_length=str(n),BENCH_PARAM_block_size=str(b),
                 BENCH_PARAM_penalty=str(p),BENCH_PARAM_vector_add_warmup='true' if not fault else 'false',**fault)
        start=time.monotonic()
        run=subprocess.run([str(instrumented if fault else production)],env=env,text=True,capture_output=True,timeout=60)
        elapsed=time.monotonic()-start
        (work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check_output(run.stdout,run.stderr,run.returncode,n,b,p,fault)
        report.append(dict(case=label,sequence_length=n,block_size=b,penalty=p,fault=fault,exit_code=run.returncode,wall_seconds=elapsed))
        (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'{len(cases)} production/control/measured-state/score/resource cases pass')


if __name__=='__main__':
    main()
