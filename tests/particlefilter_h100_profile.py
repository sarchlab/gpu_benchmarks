#!/usr/bin/env python3
"""Qualify every declared Particle Filter size with complete verification in 60s."""
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

NAMES=['likelihood_kernel','sum_kernel','normalize_weights_kernel','cdf_kernel','thresholds_kernel','find_index_kernel']
FILES=['tier2/altis_particlefilter/'+x for x in ['altis_particlefilter.hip','particlefilter_input.h','particlefilter_reference.h','particlefilter_kernels.h','Makefile','params.json','UPSTREAM_LICENSE']]+['common/'+x for x in ['bench_timing.h','bench_warmup.h','hip_cuda_compat.h']]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def expected_params(n):
    return dict(size=n,seed=12345,width=128,height=128,frames=11,timesteps=10,neighbors=69,block_size=512,kernel_launches=10,
                particle_updates=10*n,device_buffer_bytes=344*n+8*((n+511)//512)+8+128*128*11+552,
                algorithm='altis_particlefilter_float_library_v1',input='fixed_video_safe_lcg_v1',precision='fp64',verification='full_final_aux_cpu_fp64_v1')

def check_records(n,stdout):
    records=[json.loads(line) for line in stdout.splitlines()]
    if len(records)!=7 or not all(type(row) is dict for row in records):raise ValueError('Expected six kernels and one summary')
    for row,name in zip(records[:-1],NAMES):
        if row.get('type')!='kernel' or row.get('name')!=name or type(row.get('iter')) is not int or row['iter']!=0:raise ValueError('Invalid kernel identity')
        params=row.get('params');expected=expected_params(n)
        if type(params) is not dict or params!=expected or any(type(params[k])!=type(v) for k,v in expected.items()):raise ValueError('Invalid workload identity')
        for k in ['time_ms','host_time_ms']:
            v=row.get(k)
            if type(v) not in [int,float] or not math.isfinite(v) or v<=0:raise ValueError('Invalid timing')
    total=records[-1].get('total_time_ms')
    if records[-1].get('type')!='summary' or type(total) not in [int,float] or not math.isfinite(total) or total<=0 or abs(total-sum(r['time_ms'] for r in records[:-1]))>3.6e-6:raise ValueError('Invalid total')
    return records

def classify_completed(n,code,stdout,stderr,elapsed):
    check_records(n,stdout)
    if type(elapsed) not in [float,int] or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed time')
    matches=re.findall(r'Particle Filter verification: (PASS|FAIL) \(verified (\d+) particles, 10 timesteps, (\d+) final and auxiliary values; (\d+) mismatches\)',stderr)
    values=78*n+(n+511)//512+1
    if len(matches)!=1:raise ValueError('Missing unique complete verification result')
    status,count,checked,errors=matches[0];errors=int(errors)
    if int(count)!=n or int(checked)!=values:raise ValueError('Incomplete verification')
    if code==0 and status=='PASS' and errors==0:
        return dict(status='qualified' if elapsed<=60 else 'unsupported',reason='complete_measured_state_pass' if elapsed<=60 else 'process_budget_exceeded',mismatches=0)
    if code==1 and status=='FAIL' and 0<errors<=values:return dict(status='unsupported',reason='cpu_gpu_trajectory_mismatch',mismatches=errors)
    raise ValueError('Unexpected process or verification failure')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--device-uuid',required=True);ap.add_argument('--work-dir',type=Path,required=True);args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];work=args.work_dir.resolve();work.mkdir(parents=True)
    fields=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+args.device_uuid,'--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip().split(',')]
    if len(fields)!=3 or 'H100' not in fields[0] or fields[1]!=args.device_uuid or not args.device_uuid.startswith('GPU-'):raise ValueError('Select the physical H100 UUID')
    hashes={f:digest(repo/f) for f in FILES}
    for f in FILES:
        dst=work/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/f,dst)
    folder=work/'source/tier2/altis_particlefilter';binary=folder/'altis_particlefilter'
    build=['make','-B','PLATFORM=cuda','CUDA_ARCH=sm_90']
    with (work/'build.log').open('w') as log:subprocess.run(build,cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    binary_hash=digest(binary);manifest=json.loads((folder/'params.json').read_text())
    profile=dict(complete=False,benchmark='altis_particlefilter',profile='h100_full_cpu_v1',process_timeout_seconds=60,
                 scope='All declared particle counts; full ten-step CPU trajectory and actual final GPU buffers. Single-process qualification, not published performance measurements.',
                 verification='full_final_aux_cpu_fp64_v1',gpu=dict(name=fields[0],uuid=fields[1],driver_version=fields[2]),
                 nvcc_version=subprocess.check_output(['nvcc','--version'],text=True).strip(),host_compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
                 reference_workers_max=16,host_logical_cpus=os.cpu_count(),vector_add_warmup=True,source_sha256=hashes,binary_sha256=binary_hash,build_command=build,configurations=[])
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','PF_TEST_'))};clean.update(CUDA_VISIBLE_DEVICES=args.device_uuid,BENCH_PARAM_vector_add_warmup='true')
    for n in manifest['scaling_param']['values']:
        start=time.monotonic()
        try:
            run=subprocess.run([str(binary)],env=dict(clean,BENCH_PARAM_size=str(n)),text=True,capture_output=True,timeout=60)
            stdout,stderr,code=run.stdout,run.stderr,run.returncode
        except subprocess.TimeoutExpired as e:
            decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x or ''
            stdout,stderr,code=decode(e.stdout),decode(e.stderr),None
        elapsed=time.monotonic()-start
        (work/f'{n}.stdout').write_text(stdout);(work/f'{n}.stderr').write_text(stderr)
        row=dict(size=n,wall_seconds=elapsed,exit_code=code)
        row.update(dict(status='unsupported',reason='process_timeout') if code is None else classify_completed(n,code,stdout,stderr,elapsed))
        profile['configurations'].append(row);(work/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
        print(n,row['status'],row['reason'],f'{elapsed:.3f}s',flush=True)
    if digest(binary)!=binary_hash or {f:digest(repo/f) for f in FILES}!=hashes or {f:digest(work/'source'/f) for f in FILES}!=hashes:raise ValueError('Source or binary changed during qualification')
    sizes=[row['size'] for row in profile['configurations'] if row['status']=='qualified']
    if not sizes:raise ValueError('No complete qualified configuration')
    collection=work/'collection';(collection/'tier1').mkdir(parents=True);target=collection/'tier2/altis_particlefilter';target.mkdir(parents=True)
    selected=json.loads(json.dumps(manifest));selected['scaling_param']['values']=sizes
    for control in selected['non_scaling_params']:
        if control['name']=='vector_add_warmup':control['default']='true';control['values']=['true']
    (target/'params.json').write_text(json.dumps(selected,indent=2)+'\n');shutil.copy2(binary,target/binary.name)
    shutil.copy2(folder/'UPSTREAM_LICENSE',target/'UPSTREAM_LICENSE')
    if digest(target/binary.name)!=binary_hash:raise ValueError('Collection binary changed')
    profile.update(complete=True,qualified_count=len(sizes),unsupported_count=len(profile['configurations'])-len(sizes),collection_workspaces=['collection'])
    (work/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print(f'{len(sizes)} qualified, {profile["unsupported_count"]} explicitly unsupported; complete profile',flush=True)
if __name__=='__main__':main()
