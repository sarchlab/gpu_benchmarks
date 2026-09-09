#!/usr/bin/env python3
"""Check N-body qualification classification against malformed result evidence."""
import copy
import json
import math
from nbody_h100_profile import classify_completed


def main():
    kernel=dict(type='kernel',name='nbody_kernel',iter=0,time_ms=1.5,host_time_ms=2.0,
                params=dict(bodies=256,timesteps=10,verification='full_cpu_v1'))
    summary=dict(type='summary',total_time_ms=1.5)
    passed='PASS (verified 256 bodies, 10 timesteps, positions and velocities, 0 mismatches)'
    failed='FAIL (verified 256 bodies, 10 timesteps, positions and velocities, 17 mismatches)'
    encode=lambda rows:'\n'.join(json.dumps(x) for x in rows)+'\n'
    normal=encode([kernel,summary]);checks=0
    expected=[(0,passed,dict(status='qualified',reason='complete_measured_state_pass',mismatching_bodies=0)),
              (1,failed,dict(status='unsupported',reason='cpu_gpu_trajectory_mismatch',mismatching_bodies=17))]
    for code,stderr,want in expected:
        if classify_completed(256,10,code,normal,stderr,3)!=want:raise AssertionError(want)
        checks+=1
    if classify_completed(256,10,0,normal,passed,60.1)!=dict(status='unsupported',reason='process_budget_exceeded',mismatching_bodies=0):
        raise AssertionError('Slow completed process must be explicitly excluded')
    checks+=1
    invalid=[]
    for field,value in [('time_ms',math.nan),('time_ms',math.inf),('time_ms',0),('time_ms',-1),
                        ('time_ms',True),('host_time_ms',math.inf),('host_time_ms',-1),
                        ('host_time_ms',None),('name','other'),('iter',True),('iter',1)]:
        row=copy.deepcopy(kernel);row[field]=value;invalid.append((0,encode([row,summary]),passed,3))
    for field,value in [('bodies',1),('timesteps',True),('verification','disabled')]:
        row=copy.deepcopy(kernel);row['params'][field]=value;invalid.append((0,encode([row,summary]),passed,3))
    row=copy.deepcopy(summary);row['total_time_ms']=2;invalid.append((0,encode([kernel,row]),passed,3))
    invalid += [(0,encode([kernel]),passed,3),(0,encode([kernel,summary,summary]),passed,3),
                (0,normal,passed,math.nan),(0,normal,failed,3),
                (1,normal,passed,3),(-11,normal,failed,3),(1,normal,'FAIL: complete CPU reference unavailable',3),
                (0,normal,passed.replace('256 bodies','4096 bodies'),3),
                (0,normal,passed.replace('10 timesteps','1 timesteps'),3),
                (1,normal,failed.replace('17 mismatches','257 mismatches'),3)]
    for code,stdout,stderr,wall in invalid:
        try:classify_completed(256,10,code,stdout,stderr,wall)
        except ValueError:checks+=1
        else:raise AssertionError((code,stdout,stderr,wall))
    print(f'{checks} qualification success/mismatch/malformed-evidence cases pass')


if __name__=='__main__':main()
