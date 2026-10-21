#!/usr/bin/env python3
"""Qualify complete HotSpot output, padded work counts and real failure paths."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

def replace_once(s,a,b):
    if s.count(a)!=1:raise ValueError('HotSpot instrumentation boundary changed: '+a)
    return s.replace(a,b)

def instrument(source,header):
    source=replace_once(source,'    auto in=hotspot::input(p);','    if(getenv("HS2D_TEST_STORAGE_FAILURE"))throw std::bad_alloc();\n    auto in=hotspot::input(p);')
    source=replace_once(source,'const unsigned workers=hotspot::reference_workers(p.n);','const unsigned workers=getenv("HS2D_TEST_WORKER_FAIL_AFTER")?8:hotspot::reference_workers(p.n);')
    source=replace_once(source,'float time_elapsed){','float time_elapsed,unsigned long long* test_updates){')
    source=replace_once(source,'                  computed = true;','                  computed = true;\n                  if(test_updates)atomicAdd(test_updates,1ULL);')
    source=replace_once(source,'c.rz,c.dt,0.001f);','c.rz,c.dt,0.001f,test_updates);')
    source=replace_once(source,'    float *a=nullptr,*b=nullptr,*power=nullptr;',r"""    float *a=nullptr,*b=nullptr,*power=nullptr;
    unsigned long long* test_updates=nullptr;
    if(getenv("HS2D_TEST_COUNT_WORK")){HIP_CHECK(hipMalloc(&test_updates,sizeof(unsigned long long)));HIP_CHECK(hipMemset(test_updates,0,sizeof(unsigned long long)));}
""")
    source=replace_once(source,'    bench_timer_destroy(&timer);',r"""    if(test_updates){unsigned long long count=0;HIP_CHECK(hipMemcpy(&count,test_updates,sizeof(count),hipMemcpyDeviceToHost));HIP_CHECK(hipFree(test_updates));if(count!=geometry.executed_updates)throw std::runtime_error("executed GPU update count differs from metadata");fprintf(stderr,"HS2D_TEST_WORK %llu\n",count);}
    bench_timer_destroy(&timer);""")
    source=replace_once(source,'    // Download the actual final measured buffer after the last swap.',r"""
    if(const char* cell=getenv("HS2D_TEST_CELL")) {
        const size_t index=strtoull(cell,nullptr,10),count=getenv("HS2D_TEST_COUNT")?strtoull(getenv("HS2D_TEST_COUNT"),nullptr,10):1;
        if(!count || index>=cells || count>cells-index)abort();
        for(size_t i=index;i<index+count;++i){float value;HIP_CHECK(hipMemcpy(&value,src+i,sizeof(value),hipMemcpyDeviceToHost));const char* mode=getenv("HS2D_TEST_VALUE");value=strcmp(mode,"finite")==0?value+static_cast<float>(std::max(1.0,4*bound)):strtof(mode,nullptr);HIP_CHECK(hipMemcpy(src+i,&value,sizeof(value),hipMemcpyHostToDevice));}
        fprintf(stderr,"HS2D_TEST_CELL %zu count %zu\n",index,count);
    }
    if(getenv("HS2D_TEST_STALE_BUFFER"))std::swap(src,dst);
    // Download the actual final measured buffer after the last swap.""")
    source=replace_once(source,'    const auto expected=hotspot::reference(p,c,in,workers);','    if(getenv("HS2D_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();\n    const auto expected=hotspot::reference(p,c,in,workers);')
    anchor='                threads.emplace_back([this,id]() {'
    header=replace_once(header,anchor,'                if(const char* fail=std::getenv("HS2D_TEST_WORKER_FAIL_AFTER"))if(id-1==static_cast<unsigned>(std::strtoul(fail,nullptr,10)))throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));\n'+anchor)
    return source,header

def build(repo,work,platform,hooks):
    work.mkdir(parents=True);shutil.copytree(repo/'common',work/'common');folder=work/'tier2/rodinia_hotspot/hip';folder.mkdir(parents=True)
    for name in ['Makefile','params.json']:shutil.copy2(repo/'tier2/rodinia_hotspot/hip'/name,folder/name)
    source=(repo/'tier2/rodinia_hotspot/hip/rodinia_hotspot.hip').read_text();header=(repo/'tier2/rodinia_hotspot/hip/hotspot_reference.h').read_text()
    if hooks:source,header=instrument(source,header)
    (folder/'rodinia_hotspot.hip').write_text(source);(folder/'hotspot_reference.h').write_text(header)
    with (work/'build.log').open('w') as log:subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'rodinia_hotspot'

def work_counts(n,t,p):
    blocks=(n+15-2*p)//(16-2*p)
    def launch(it):
        total=0
        for k in range(it):
            axis=0
            for b in range(blocks):
                offset=b*(16-2*it)-p
                axis+=max(0,min(n,offset+15-k)-max(0,offset+k+1))
            total+=axis**2
        return total
    full,tail=divmod(t,p);return blocks,full*launch(p)+(launch(tail) if tail else 0)

def timestep(n):
    f=lambda v:struct.unpack('f',struct.pack('f',v))[0]
    chip=f(.0005);dx=f(f(.016)/n);cap=f(.5*1.75e6*chip*dx*dx);rx=f(dx/(2.*100*chip*dx));rz=f(chip/f(f(100*dx)*dx));up=f(.001/f(3e6/(.5*chip*1.75e6)))
    s=f(up/cap);ix=f(1/rx);iz=f(1/rz);wx=s*ix;wa=s*iz
    limited=abs(1-4*wx-wa)+4*wx>1
    return (f(.9*cap/(4./rx+1./rz)) if limited else up),up,limited

def check(stdout,stderr,code,n,t,p,control):
    fault={k:v for k,v in control.items() if k!='HS2D_TEST_COUNT_WORK'}
    if 'HS2D_TEST_STORAGE_FAILURE' in fault:
        assert code==1 and not stdout and 'FAIL: HotSpot setup/reference unavailable:' in stderr;return
    rows=[json.loads(s) for s in stdout.splitlines()];assert len(rows)==2 and rows[-1]['type']=='summary'
    row=rows[0];params=row['params'];blocks,updates=work_counts(n,t,p);launches=(t+p-1)//p
    expected=dict(grid_size=n,total_iterations=t,pyramid_height=p,block_size=16,block_x=16,block_y=16,grid_x=blocks,grid_y=blocks,cells=n*n,kernel_launches=launches,threads_per_launch=blocks**2*256,launched_threads=blocks**2*256*launches,logical_cell_updates=n*n*t,executed_cell_updates=updates,precision='fp32_state_mixed_fp32_fp64_expression',boundary='clamped_four_faces',algorithm='rodinia31_hotspot_pyramid_v1',input='libc_rand42_interleaved_v1',verification='full_cpu_fp64_bound_v1')
    assert row['type']=='kernel' and row['name']=='calculate_temp' and row['iter']==0
    for k,v in expected.items():assert params[k]==v,(k,params[k],v)
    assert set(params)==set(expected)|{'dt','upstream_dt','timestep_limited','simulated_time_seconds','absolute_error_bound','verification_workers'}
    dt,up,limited=timestep(n)
    assert math.isclose(params['dt'],dt,rel_tol=5e-9) and math.isclose(params['upstream_dt'],up,rel_tol=5e-9) and params['timestep_limited']==limited
    assert math.isclose(params['simulated_time_seconds'],dt*t,rel_tol=1e-14)
    assert math.isfinite(params['absolute_error_bound']) and params['absolute_error_bound']>0
    assert 1<=params['verification_workers']<=8
    if n<512:assert params['verification_workers']==(8 if 'HS2D_TEST_WORKER_FAIL_AFTER' in fault else 1)
    assert math.isfinite(row['time_ms']) and row['time_ms']>0 and math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
    assert rows[-1]['total_time_ms']==row['time_ms']
    if 'HS2D_TEST_COUNT_WORK' in control:assert f'HS2D_TEST_WORK {updates}\n' in stderr
    if 'HS2D_TEST_REFERENCE_FAILURE' in fault or 'HS2D_TEST_WORKER_FAIL_AFTER' in fault:
        assert code==1 and 'FAIL: HotSpot setup/reference unavailable:' in stderr and 'PASS (verified' not in stderr;return
    assert code==(1 if fault else 0),stderr
    if 'HS2D_TEST_STALE_BUFFER' in fault:assert 'FAIL (verified' in stderr and 'Mismatch temperature[' in stderr
    else:
        count=int(fault.get('HS2D_TEST_COUNT','1')) if fault else 0
        assert f'{"FAIL" if fault else "PASS"} (verified {n*n} values in the complete measured final grid; {count} mismatches;' in stderr,stderr
        assert stderr.count('Mismatch temperature[')==min(count,10)
    assert 'HIP error' not in stderr

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda');ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--compile-only',action='store_true');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve();hooks=build(repo,work/'instrumented',args.platform,True)
    if args.compile_only:print('HotSpot output/work/worker-failure hooks compiled; GPU tests not executed');return
    production=build(repo,work/'production',args.platform,False);config=json.loads((repo/'tier2/rodinia_hotspot/hip/params.json').read_text());sizes=config['scaling_param']['values'];heights=next(x['values'] for x in config['non_scaling_params'] if x['name']=='pyramid_height')
    cases=[(n,360,p,{}) for n in sizes for p in heights]
    cases += [(n,t,p,{}) for n in [1,2,15,16,17,31,32,33] for p in [1,2,4,7] for t in [1,3,8]]
    cases += [(17,10,p,{}) for p in [3,5,6]]+[(1536,5,3,{}),(8192,3,7,{}),(1385,3,7,{}),(1386,3,7,{})]
    n=17
    for i in [0,16,17,8*17,144,16*17,288]:
        for value in ['finite','nan','inf','-inf']:cases.append((n,3,2,dict(HS2D_TEST_CELL=str(i),HS2D_TEST_VALUE=value)))
    cases += [(n,3,2,dict(HS2D_TEST_CELL='100',HS2D_TEST_VALUE='finite',HS2D_TEST_COUNT='12')),(8192,360,4,dict(HS2D_TEST_CELL=str(8192**2-1),HS2D_TEST_VALUE='finite'))]
    cases += [(n,t,p,dict(HS2D_TEST_STALE_BUFFER='1')) for t,p in [(1,4),(2,2),(3,2),(7,3)]]
    cases += [(n,3,2,{key:'1'}) for key in ['HS2D_TEST_STORAGE_FAILURE','HS2D_TEST_REFERENCE_FAILURE']]
    cases += [(n,3,2,dict(HS2D_TEST_WORKER_FAIL_AFTER=x)) for x in ['0','3']]
    cases += [(n,t,p,dict(HS2D_TEST_COUNT_WORK='1')) for n in [1,17,31] for p in range(1,8) for t in [1,3,8]]
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','HS2D_TEST_'))};report=[]
    for index,(n,t,p,control) in enumerate(cases):
        label=f'{index}-{n}-{t}-{p}';start=time.monotonic();env=dict(clean,BENCH_PARAM_grid_size=str(n),BENCH_PARAM_total_iterations=str(t),BENCH_PARAM_pyramid_height=str(p),BENCH_PARAM_vector_add_warmup='false' if control else 'true',**control)
        try:run=subprocess.run([str(hooks if control else production)],env=env,capture_output=True,text=True,timeout=60)
        except subprocess.TimeoutExpired as e:
            (work/(label+'.stdout')).write_bytes(e.stdout or b'');(work/(label+'.stderr')).write_bytes(e.stderr or b'');raise
        elapsed=time.monotonic()-start;(work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check(run.stdout,run.stderr,run.returncode,n,t,p,control)
        report.append(dict(case=label,n=n,steps=t,pyramid=p,control=control,exit_code=run.returncode,wall_seconds=elapsed));(work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'{len(cases)} production/control/full-grid/work/failure cases pass')
if __name__=='__main__':main()
