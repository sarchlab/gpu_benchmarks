#!/usr/bin/env python3
"""Qualify a bounded, explicit H100 N-body configuration list with full checking."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_completed(n, t, code, stdout, stderr, elapsed):
    records=[json.loads(line) for line in stdout.splitlines()]
    if len(records)!=2 or not all(isinstance(x,dict) for x in records):
        raise ValueError('Expected one kernel and one summary')
    kernel,summary=records
    params=kernel.get('params',{})
    times=[kernel.get('time_ms'),kernel.get('host_time_ms'),summary.get('total_time_ms')]
    valid_times=all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in times)
    if (kernel.get('type')!='kernel' or kernel.get('name')!='nbody_kernel' or
        type(kernel.get('iter')) is not int or kernel['iter']!=0 or summary.get('type')!='summary' or
        params!=dict(bodies=n,timesteps=t,verification='full_cpu_v1') or
        type(params.get('bodies')) is not int or type(params.get('timesteps')) is not int or
        not valid_times or times[0]<=0 or times[0]!=times[2] or
        not math.isfinite(elapsed) or elapsed<0):
        raise ValueError('Invalid measured-state protocol or process budget')
    match=re.search(r'(PASS|FAIL) \(verified (\d+) bodies, (\d+) timesteps, positions and velocities, (\d+) mismatches\)',stderr)
    if not match or int(match[2])!=n or int(match[3])!=t:
        raise ValueError('Missing complete verification result')
    errors=int(match[4])
    if code==0 and match[1]=='PASS' and errors==0:
        if elapsed>60:
            return dict(status='unsupported',reason='process_budget_exceeded',mismatching_bodies=0)
        return dict(status='qualified',reason='complete_measured_state_pass',mismatching_bodies=0)
    if code==1 and match[1]=='FAIL' and 0<errors<=n:
        return dict(status='unsupported',reason='cpu_gpu_trajectory_mismatch',mismatching_bodies=errors)
    raise ValueError('Unexpected process failure or inconsistent verification result')


def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device-uuid',required=True)
    ap.add_argument('--work-dir',type=Path,required=True)
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    if not args.device_uuid.startswith('GPU-'):raise ValueError('Select a physical GPU UUID explicitly')
    device=subprocess.check_output(['nvidia-smi','--id='+args.device_uuid,'--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    fields=[x.strip() for x in device.split(',')]
    if len(fields)!=3 or 'H100' not in fields[0] or fields[1]!=args.device_uuid:
        raise ValueError('This qualification profile requires the specified H100 GPU')
    work=args.work_dir.resolve();work.mkdir(parents=True)
    manifest=json.loads((repo/'tier2/cuda_nbody/hip/params.json').read_text())
    sizes=manifest['scaling_param']['values'];steps=next(x['values'] for x in manifest['non_scaling_params'] if x['name']=='timesteps')
    files=['tier2/cuda_nbody/hip/cuda_nbody.hip','tier2/cuda_nbody/hip/Makefile','tier2/cuda_nbody/hip/params.json','common_hip/bench_timing.h','common_hip/bench_warmup.h','common_hip/hip_cuda_compat.h']
    hashes={f:digest(repo/f) for f in files}
    for relative in files:
        destination=work/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/relative,destination)
    folder=work/'source/tier2/cuda_nbody/hip';binary=folder/'cuda_nbody'
    with (work/'build.log').open('w') as log:
        subprocess.run(['make','-B','PLATFORM=cuda','CUDA_ARCH=sm_90'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    binary_hash=digest(binary)
    profile=dict(complete=False,benchmark='cuda_nbody',profile='h100_full_cpu_v1',verification='full_cpu_v1',
                 process_timeout_seconds=60,candidate_cpu_pair_interactions_limit=1<<32,
                 scope='Single-process qualification; larger reference-work cases remain unqualified. This is not a hardware time prediction or a benchmark input limit.',
                 gpu=dict(name=fields[0],uuid=fields[1],driver_version=fields[2]),
                 nvcc_version=subprocess.check_output(['nvcc','--version'],text=True).strip(),
                 host_compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
                 reference_workers=1,vector_add_warmup=True,source_sha256=hashes,
                 binary_sha256=binary_hash,build_command=['make','-B','PLATFORM=cuda','CUDA_ARCH=sm_90'],configurations=[])
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','NBODY_TEST_'))}
    clean.update(CUDA_VISIBLE_DEVICES=args.device_uuid,BENCH_PARAM_vector_add_warmup='true')
    for n in sizes:
        for t in steps:
            row=dict(bodies=n,timesteps=t)
            if n*n*t>1<<32:
                row.update(status='unsupported',reason='not_qualified_reference_work_scope')
            else:
                label=f'n{n}-t{t}';start=time.monotonic()
                try:
                    p=subprocess.run([str(binary)],env=dict(clean,BENCH_PARAM_bodies=str(n),BENCH_PARAM_timesteps=str(t)),capture_output=True,text=True,timeout=60)
                    stdout,stderr,code=p.stdout,p.stderr,p.returncode
                except subprocess.TimeoutExpired as exc:
                    decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x or ''
                    stdout,stderr,code=decode(exc.stdout),decode(exc.stderr),None
                elapsed=time.monotonic()-start
                (work/(label+'.stdout')).write_text(stdout);(work/(label+'.stderr')).write_text(stderr)
                row.update(wall_seconds=elapsed,exit_code=code)
                if code is None:
                    row.update(status='unsupported',reason='process_timeout')
                else:
                    row.update(classify_completed(n,t,code,stdout,stderr,elapsed))
                print(label,row['status'],row['reason'],f'{elapsed:.3f}s',flush=True)
            profile['configurations'].append(row)
            (work/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    assert digest(binary)==binary_hash and {f:digest(repo/f) for f in files}==hashes
    assert {f:digest(work/'source'/f) for f in files}==hashes
    profile['qualified_count']=sum(x['status']=='qualified' for x in profile['configurations'])
    profile['unsupported_count']=len(profile['configurations'])-profile['qualified_count']
    assert profile['qualified_count']>0
    # Each generated workspace fixes one timestep value and contains only the
    # qualified body counts. The existing harness can run it without multiplying
    # excluded combinations back into the sweep. Reuse the exact qualified binary.
    profile['collection_workspaces']=[]
    for t in steps:
        admitted=[x['bodies'] for x in profile['configurations'] if x['timesteps']==t and x['status']=='qualified']
        if not admitted:continue
        destination=work/'collection'/f'timesteps-{t}'
        (destination/'tier1').mkdir(parents=True)
        bench=destination/'tier2/cuda_nbody/hip';bench.mkdir(parents=True)
        selected=json.loads(json.dumps(manifest));selected['scaling_param']['values']=admitted
        for control in selected['non_scaling_params']:
            if control['name']=='timesteps':control['default']=t;control['values']=[t]
        (bench/'params.json').write_text(json.dumps(selected,indent=2)+'\n')
        shutil.copy2(binary,bench/'cuda_nbody');assert digest(bench/'cuda_nbody')==binary_hash
        profile['collection_workspaces'].append(str(destination.relative_to(work)))
    profile['complete']=True
    (work/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print(f'{profile["qualified_count"]} qualified, {profile["unsupported_count"]} explicitly unsupported; all declared configurations accounted for')


if __name__=='__main__':main()
