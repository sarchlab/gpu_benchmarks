#!/usr/bin/env python3
"""Qualify NAS MG classes and reject faults in full measured levels/input/norms."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

CLASSES = {'S':(32,4),'W':(128,4),'A':(256,4),'B':(256,20),'C':(512,20)}
NAMES = ['comm3_gpu_kernel_1','comm3_gpu_kernel_2','comm3_gpu_kernel_3','interp_gpu_kernel',
         'norm2u3_gpu_kernel','psinv_gpu_kernel','resid_gpu_kernel','rprj3_gpu_kernel','coarse_zero']


def layout(n):
    widths=[];offsets=[];cells=0
    k=2
    while k<=n:
        widths.append(k+2);offsets.append(cells);cells+=(k+2)**3;k*=2
    return widths,offsets,cells


def geometry(name,block):
    n,iterations=CLASSES[name];widths,_,_=layout(n)
    stats={k:[] for k in NAMES}
    def add(k,threads,grid):
        stats[NAMES[k]].append((threads,grid*threads))
    def comm(p):
        add(0,block,math.ceil((p-2)/block)*(p-2))
        add(1,block,math.ceil(p/block)*(p-2))
        add(2,block,math.ceil(p/block)*p)
    def residual(p):
        add(6,min(p,block),(p-2)**2);comm(p)
    def smooth(p):
        add(5,min(p,block),(p-2)**2);comm(p)
    def zero(p):
        add(8,block,math.ceil(p**3/block))
    def interp(p):
        add(3,p,(p-1)**2)
    residual(n+2);add(4,block,n*n)
    for _ in range(iterations):
        for p in reversed(widths[:-1]):add(7,p-1,(p-2)**2);comm(p)
        zero(widths[0]);smooth(widths[0])
        for i in range(1,len(widths)-1):
            p=widths[i];zero(p);interp(widths[i-1]);residual(p);smooth(p)
        interp(widths[-2]);residual(n+2);smooth(n+2);residual(n+2)
    add(4,block,n*n)
    return {k:dict(kernel_launches=len(v),launched_threads=sum(x[1] for x in v),
                   min_block_threads=min(x[0] for x in v),max_block_threads=max(x[0] for x in v)) for k,v in stats.items()}


def replace_once(source,anchor,replacement):
    if source.count(anchor)!=1:
        raise ValueError('MG instrumentation boundary changed: '+anchor)
    return source.replace(anchor,replacement)


def instrument(source):
    source=replace_once(source,'    nas_mg::Result actual(layout);actual.v=nas_mg::input(c);',
        '    if(getenv("MG_TEST_STORAGE_FAILURE"))throw std::bad_alloc();\n'
        '    nas_mg::Result actual(layout);actual.v=nas_mg::input(c);')
    anchor='    // Download the entire measured hierarchy, including every periodic ghost.'
    source=replace_once(source,anchor,r'''
    if(const char* array=getenv("MG_TEST_ARRAY")) {
        const int k=atoi(array);const size_t i=strtoull(getenv("MG_TEST_CELL"),nullptr,10);
        const size_t count=getenv("MG_TEST_COUNT")?strtoull(getenv("MG_TEST_COUNT"),nullptr,10):1;
        if(k<0 || k>3 || !count || i+count>(k<2?layout.cells:k==2?actual.v.size():4))abort();
        double* pointers[]={d_u,d_r,d_v};
        for(size_t j=i;j<i+count;++j) {
            double value;
            if(k<3)HIP_CHECK(hipMemcpy(&value,pointers[k]+j,sizeof(value),hipMemcpyDeviceToHost));
            else value=actual.norms[j];
            const char* mode=getenv("MG_TEST_VALUE");
            value=strcmp(mode,"finite")==0?value+0.01:strtod(mode,nullptr);
            if(k<3)HIP_CHECK(hipMemcpy(pointers[k]+j,&value,sizeof(value),hipMemcpyHostToDevice));
            else actual.norms[j]=value;
        }
        fprintf(stderr,"MG_TEST_ARRAY %d cell %zu count %zu\n",k,i,count);
    }
'''+anchor)
    source=replace_once(source,'    const nas_mg::Result expected=nas_mg::reference(c,layout,workers);',
        '    if(getenv("MG_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();\n'
        '    const nas_mg::Result expected=nas_mg::reference(c,layout,workers);')
    return source


def build(repo,work,platform,hooks):
    work.mkdir(parents=True);shutil.copytree(repo/'common',work/'common')
    folder=work/'tier2/npb_mg/hip';folder.mkdir(parents=True)
    for name in ['Makefile','nas_mg.h','mg_reference.h','mg_kernels.h','params.json']:
        shutil.copy2(repo/'tier2/npb_mg/hip'/name,folder/name)
    source=(repo/'tier2/npb_mg/hip/npb_mg.hip').read_text()
    (folder/'npb_mg.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:
        subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'npb_mg'


def check_output(stdout,stderr,code,name,block,fault):
    if 'MG_TEST_STORAGE_FAILURE' in fault:
        assert code==1 and not stdout and 'FAIL: MG setup/reference/arithmetic unavailable:' in stderr
        return
    rows=[json.loads(line) for line in stdout.splitlines()]
    assert len(rows)==10 and rows[-1]['type']=='summary'
    n,iterations=CLASSES[name];widths,_,cells=layout(n);stats=geometry(name,block)
    for row,kernel in zip(rows,NAMES):
        assert row['type']=='kernel' and row['name']==kernel and row['iter']==0
        p=row['params']
        want=dict(npb_class=name,grid_size=n,padded_grid_size=n+2,levels=len(widths),num_vcycles=iterations,
            block_size=block,hierarchy_cells=cells,input_seed=314159265,input_multiplier=1220703125,
            reference_workers=p['reference_workers'],precision='fp64',boundary='periodic',
            algorithm='npb_mg_vcycle_v1',verification='full_cpu_nas_l2_v1',**stats[kernel])
        assert p==want,(p,want)
        assert (p['reference_workers']==1 if n<128 else 1<=p['reference_workers']<=8)
        assert math.isfinite(row['time_ms']) and row['time_ms']>0
        assert math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
    assert abs(sum(r['time_ms'] for r in rows[:-1])-rows[-1]['total_time_ms'])<5.1e-6
    if 'MG_TEST_REFERENCE_FAILURE' in fault:
        assert code==1 and 'FAIL: MG setup/reference/arithmetic unavailable:' in stderr and 'PASS (verified' not in stderr
        return
    count=int(fault.get('MG_TEST_COUNT','1')) if fault else 0
    assert code==(1 if fault else 0),stderr
    assert f'{"FAIL" if fault else "PASS"} (verified {2*cells+(n+2)**3} hierarchy/input values and 4 norms, {count} mismatches,' in stderr,stderr
    assert stderr.count('Mismatch ')==min(count,10)
    if fault:
        assert 'MG_TEST_ARRAY '+fault['MG_TEST_ARRAY'] in stderr and 'GPU=' in stderr and 'CPU=' in stderr
    assert 'HIP error' not in stderr


def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda')
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--compile-only',action='store_true')
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve()
    hooks=build(repo,work/'instrumented',args.platform,True)
    if args.compile_only:
        print('MG full-state and storage hooks compiled; GPU tests not executed');return
    production=build(repo,work/'production',args.platform,False)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','MG_TEST_'))}
    cases=[(name,256,{}) for name in CLASSES]+[('S',b,{}) for b in [1,8,32,1024]]
    widths,offsets,cells=layout(32)
    for k in range(4):
        indices=[]
        if k<2:
            for p,o in zip(widths,offsets):indices += [o,o+p+1,o+p*p+p+1,o+p**3-1]
        elif k==2:indices=[0,1,34**3//2,34**3-1]
        else:indices=[0,1,2,3]
        for i in indices:
            for value in ['finite','nan','inf','-inf']:
                cases.append(('S',256,dict(MG_TEST_ARRAY=str(k),MG_TEST_CELL=str(i),MG_TEST_VALUE=value)))
    cases += [('S',256,dict(MG_TEST_ARRAY='0',MG_TEST_CELL='128',MG_TEST_VALUE='finite',MG_TEST_COUNT='12')),
              ('C',256,dict(MG_TEST_ARRAY='1',MG_TEST_CELL=str(layout(512)[2]-1),MG_TEST_VALUE='finite')),
              ('S',256,dict(MG_TEST_STORAGE_FAILURE='1')),('S',256,dict(MG_TEST_REFERENCE_FAILURE='1'))]
    report=[]
    for index,(name,block,fault) in enumerate(cases):
        label=f'{index}-{name}-b{block}';start=time.monotonic()
        env=dict(clean,BENCH_PARAM_npb_class=name,BENCH_PARAM_block_size=str(block),
                 BENCH_PARAM_vector_add_warmup='false' if fault else 'true',**fault)
        try:run=subprocess.run([str(hooks if fault else production)],env=env,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired as error:
            (work/(label+'.stdout')).write_bytes(error.stdout or b'');(work/(label+'.stderr')).write_bytes(error.stderr or b'');raise
        elapsed=time.monotonic()-start
        (work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check_output(run.stdout,run.stderr,run.returncode,name,block,fault)
        report.append(dict(case=label,npb_class=name,block_size=block,fault=fault,exit_code=run.returncode,wall_seconds=elapsed))
        (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'{len(cases)} production/control/measured-hierarchy/input/norm/storage cases pass')


if __name__=='__main__':main()
