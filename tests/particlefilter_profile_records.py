#!/usr/bin/env python3
"""Reject malformed or incomplete H100 qualification evidence."""
import copy
import json
import math
from particlefilter_h100_profile import NAMES, expected_params, classify_completed

def main():
    n=503;rows=[dict(type='kernel',name=name,iter=0,time_ms=1.0,host_time_ms=1.5,params=expected_params(n)) for name in NAMES]+[dict(type='summary',total_time_ms=6.0)]
    passed='Particle Filter verification: PASS (verified 503 particles, 10 timesteps, 39236 final and auxiliary values; 0 mismatches)'
    failed=passed.replace('PASS','FAIL').replace('0 mismatches','1 mismatches');encode=lambda x:'\n'.join(json.dumps(v) for v in x)
    normal=encode(rows);checks=0
    for code,stderr,elapsed,status in [(0,passed,1,'qualified'),(1,failed,1,'unsupported'),(0,passed,60.1,'unsupported')]:
        assert classify_completed(n,code,normal,stderr,elapsed)['status']==status;checks+=1
    invalid=[]
    for key,value in [('name','other'),('iter',True),('iter',1),('type','summary'),('time_ms',math.nan),('time_ms',math.inf),('time_ms',-1),('time_ms',0),('time_ms',True),('host_time_ms',math.inf),('host_time_ms',-1)]:
        altered=copy.deepcopy(rows);altered[0][key]=value;invalid.append((0,encode(altered),passed,1))
    for key,value in [('size',1),('seed',False),('frames',10),('timesteps',True),('verification','sampled'),('particle_updates',503)]:
        altered=copy.deepcopy(rows);altered[0]['params'][key]=value;invalid.append((0,encode(altered),passed,1))
    for value in [True,math.nan,math.inf,-1,0,5]:
        altered=copy.deepcopy(rows);altered[-1]['total_time_ms']=value;invalid.append((0,encode(altered),passed,1))
    invalid.extend([(0,normal,failed,1),(1,normal,passed,1),(-11,normal,failed,1),(0,normal,'PASS',1),(0,normal,passed+'\n'+passed,1),
                    (0,normal,passed.replace('39236','503'),1),(0,normal,passed.replace('503 particles','1 particles'),1),
                    (1,normal,failed.replace('1 mismatches','39237 mismatches'),1),(0,normal,passed,math.nan),
                    (0,encode(rows[:-1]),passed,1),(0,encode(rows+rows[-1:]),passed,1)])
    for code,stdout,stderr,elapsed in invalid:
        try:classify_completed(n,code,stdout,stderr,elapsed)
        except ValueError:checks+=1
        else:raise AssertionError((code,stdout,stderr,elapsed))
    print(f'PASS: {checks} qualification success, failure, timeout and malformed-evidence cases')
if __name__=='__main__':main()
