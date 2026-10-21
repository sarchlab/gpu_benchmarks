#!/usr/bin/env python3
"""Qualify the full measured HotSpot3D grid and reject real output corruption."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

def replace_once(s,anchor,replacement):
    if s.count(anchor)!=1:raise ValueError('HotSpot3D instrumentation boundary changed: '+anchor)
    return s.replace(anchor,replacement)

def instrument(s):
    s=replace_once(s,'    hotspot3d::Input in=hotspot3d::input(problem);','    if(getenv("HS3D_TEST_STORAGE_FAILURE"))throw std::bad_alloc();\n    hotspot3d::Input in=hotspot3d::input(problem);')
    anchor='    // Verify the complete actual final measured buffer after the last swap.'
    s=replace_once(s,anchor,r"""
    if(const char* cell=getenv("HS3D_TEST_CELL")) {
        const size_t index=strtoull(cell,nullptr,10);
        const size_t count=getenv("HS3D_TEST_COUNT")?strtoull(getenv("HS3D_TEST_COUNT"),nullptr,10):1;
        if(!count || index>=cells || count>cells-index)abort();
        for(size_t i=index;i<index+count;++i) {
            float value;HIP_CHECK(hipMemcpy(&value,src+i,sizeof(value),hipMemcpyDeviceToHost));
            const char* mode=getenv("HS3D_TEST_VALUE");
            value=strcmp(mode,"finite")==0?value+static_cast<float>(std::max(1.0,4*bound)):strtof(mode,nullptr);
            HIP_CHECK(hipMemcpy(src+i,&value,sizeof(value),hipMemcpyHostToDevice));
        }
        fprintf(stderr,"HS3D_TEST_CELL %zu count %zu\n",index,count);
    }
    if(getenv("HS3D_TEST_STALE_BUFFER"))std::swap(src,dst);
"""+anchor)
    return replace_once(s,'    const std::vector<double> expected=hotspot3d::reference(problem,coefficients,in);','    if(getenv("HS3D_TEST_REFERENCE_FAILURE"))throw std::bad_alloc();\n    const std::vector<double> expected=hotspot3d::reference(problem,coefficients,in);')

def build(repo,work,platform,hooks):
    work.mkdir(parents=True);shutil.copytree(repo/'common',work/'common');folder=work/'tier2/rodinia_hotspot3d/hip';folder.mkdir(parents=True)
    for name in ['Makefile','hotspot3d_reference.h','params.json']:shutil.copy2(repo/'tier2/rodinia_hotspot3d/hip'/name,folder/name)
    source=(repo/'tier2/rodinia_hotspot3d/hip/rodinia_hotspot3d.hip').read_text();(folder/'rodinia_hotspot3d.hip').write_text(instrument(source) if hooks else source)
    with (work/'build.log').open('w') as log:subprocess.run(['make','PLATFORM='+platform],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    return folder/'rodinia_hotspot3d'

def check_output(stdout,stderr,code,n,steps,block,ambient,fault):
    if 'HS3D_TEST_STORAGE_FAILURE' in fault:
        assert code==1 and not stdout and 'FAIL: HotSpot3D setup/reference unavailable:' in stderr;return
    rows=[json.loads(x) for x in stdout.splitlines()];assert len(rows)==2 and rows[-1]['type']=='summary'
    row=rows[0];params=row['params'];extent=(n+block-1)//block
    expected=dict(grid_size=n,block_size=block,num_iterations=steps,amb_temp=ambient,
                  block_x=block,block_y=block,block_z=block,grid_x=extent,grid_y=extent,grid_z=extent,
                  cells=n**3,threads_per_launch=(extent*block)**3,kernel_launches=steps,updated_cells=n**3*steps,
                  precision='fp32',boundary='clamped_six_faces',input='libc_rand42_interleaved_v1',
                  algorithm='rodinia31_hotspot3d_v1',verification='full_cpu_fp64_bound_v1')
    assert row['type']=='kernel' and row['name']=='hotspot3d_kernel' and row['iter']==0
    for k,v in expected.items():assert params[k]==v,(k,params[k],v)
    assert set(params)==set(expected)|{'dt','upstream_dt','timestep_limited','simulated_time_seconds','absolute_error_bound','effective_amb_temp'}
    effective=struct.unpack('f',struct.pack('f',ambient))[0];assert abs(params['effective_amb_temp']-effective)<=1e-9*max(1,abs(effective))
    assert math.isfinite(params['absolute_error_bound']) and params['absolute_error_bound']>0
    assert math.isclose(params['dt'],1.4583334e-7,rel_tol=1e-8) and params['dt']==params['upstream_dt'] and params['timestep_limited'] is False
    assert math.isclose(params['simulated_time_seconds'],params['dt']*steps,rel_tol=1e-8)
    assert math.isfinite(row['time_ms']) and row['time_ms']>0 and math.isfinite(row['host_time_ms']) and row['host_time_ms']>=0
    assert rows[-1]['total_time_ms']==row['time_ms']
    if 'HS3D_TEST_REFERENCE_FAILURE' in fault:
        assert code==1 and 'FAIL: HotSpot3D setup/reference unavailable:' in stderr and 'PASS (verified' not in stderr;return
    assert code==(1 if fault else 0),stderr
    if 'HS3D_TEST_STALE_BUFFER' in fault:
        assert 'FAIL (verified' in stderr and 'Mismatch temperature[' in stderr
    else:
        count=int(fault.get('HS3D_TEST_COUNT','1')) if fault else 0
        assert f'{"FAIL" if fault else "PASS"} (verified {n**3} values in the complete measured final grid; {count} mismatches;' in stderr,stderr
        assert stderr.count('Mismatch temperature[')==min(count,10)
    assert 'HIP error' not in stderr

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda');ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--compile-only',action='store_true');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve();hooks=build(repo,work/'instrumented',args.platform,True)
    if args.compile_only:print('HotSpot3D actual-output/storage hooks compiled; GPU tests not executed');return
    production=build(repo,work/'production',args.platform,False);manifest=json.loads((repo/'tier2/rodinia_hotspot3d/hip/params.json').read_text())
    sizes=manifest['scaling_param']['values'];steps=next(p['values'] for p in manifest['non_scaling_params'] if p['name']=='num_iterations')
    cases=[(n,t,8,80.0,{}) for n in sizes for t in steps]
    cases += [(n,t,b,a,{}) for n,t,b,a in [(1,1,1,80.0),(1,2,8,80.0),(2,3,1,80.0),(3,3,10,80.0),(17,2,8,80.0),(17,3,8,80.125),(17,3,8,-20.5),(17,3,8,0.0),(17,3,8,79.9)]]
    n=17
    for i in [0,1,n-1,n,n*n-1,n*n,n**3//2,n*n*(n-1),n**3-1]:
        for value in ['finite','nan','inf','-inf']:cases.append((n,3,8,80.0,dict(HS3D_TEST_CELL=str(i),HS3D_TEST_VALUE=value)))
    cases += [(n,3,8,80.0,dict(HS3D_TEST_CELL='100',HS3D_TEST_VALUE='finite',HS3D_TEST_COUNT='12'))]
    cases += [(max(sizes),max(steps),8,80.0,dict(HS3D_TEST_CELL=str(max(sizes)**3-1),HS3D_TEST_VALUE='finite'))]
    cases += [(n,t,8,80.0,dict(HS3D_TEST_STALE_BUFFER='1')) for t in [1,2,3]]
    cases += [(n,3,8,80.0,{key:'1'}) for key in ['HS3D_TEST_STORAGE_FAILURE','HS3D_TEST_REFERENCE_FAILURE']]
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','HS3D_TEST_'))};report=[]
    for index,(n,t,b,a,fault) in enumerate(cases):
        label=f'{index}-{n}-{t}-{b}';start=time.monotonic()
        env=dict(clean,BENCH_PARAM_grid_size=str(n),BENCH_PARAM_num_iterations=str(t),BENCH_PARAM_block_size=str(b),BENCH_PARAM_amb_temp=str(a),BENCH_PARAM_vector_add_warmup='false' if fault else 'true',**fault)
        try:run=subprocess.run([str(hooks if fault else production)],env=env,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired as e:
            (work/(label+'.stdout')).write_bytes(e.stdout or b'');(work/(label+'.stderr')).write_bytes(e.stderr or b'');raise
        elapsed=time.monotonic()-start
        (work/(label+'.stdout')).write_text(run.stdout);(work/(label+'.stderr')).write_text(run.stderr)
        check_output(run.stdout,run.stderr,run.returncode,n,t,b,a,fault)
        report.append(dict(case=label,n=n,steps=t,block=b,ambient=a,fault=fault,exit_code=run.returncode,wall_seconds=elapsed))
        (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(label,'PASS',f'{elapsed:.3f}s',flush=True)
    print(f'{len(cases)} production/control/full-grid/storage cases pass')
if __name__=='__main__':main()
