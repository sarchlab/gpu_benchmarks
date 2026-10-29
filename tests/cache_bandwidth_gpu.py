#!/usr/bin/env python3
"""Check CUDA production geometry records and the actual kernel's window addressing."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time


def kernel_test(source):
    anchor = 'int main('
    if source.count(anchor) != 1:
        raise ValueError('expected one main')
    return source.split(anchor)[0] + r'''
int main() {
    unsigned cases=0;
    for (int shared : {0,1}) for (unsigned blocks : {1u,4u,5u})
    for (unsigned threads : {64u,65u,256u,1024u})
    for (unsigned vectors : {1u,63u,65u,1025u}) {
        unsigned windows=shared ? 1 : blocks, iterations=3;
        size_t count=(size_t)windows*vectors, outputs=(size_t)blocks*threads;
        float4* input=new float4[count];
        float* output=new float[outputs];
        for (unsigned w=0;w<windows;++w) for(unsigned i=0;i<vectors;++i)
            input[(size_t)w*vectors+i]=make_float4(float(w+1),float(i%7),2.f,3.f);
        float4* device_input=nullptr;float* device_output=nullptr;
        HIP_CHECK(hipMalloc(&device_input,count*sizeof(float4)));
        HIP_CHECK(hipMalloc(&device_output,outputs*sizeof(float)));
        HIP_CHECK(hipMemcpy(device_input,input,count*sizeof(float4),hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(device_output,0xff,outputs*sizeof(float)));
        bw_read_kernel<<<blocks,threads>>>(device_input,vectors,windows,iterations,shared,device_output);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(output,device_output,outputs*sizeof(float),hipMemcpyDeviceToHost));
        for (unsigned b=0;b<blocks;++b) for(unsigned t=0;t<threads;++t) {
            // Exact integer sums: nonuniform values expose a wrong window or lane.
            unsigned expected=0;
            for(unsigned i=t;i<vectors;i+=threads)
                expected += iterations*((shared ? 1 : b+1)+(i%7)+2+3);
            if(output[(size_t)b*threads+t] != float(expected)) {
                fprintf(stderr,"window/lane mismatch at block %u thread %u\n",b,t);
                return 1;
            }
        }
        HIP_CHECK(hipFree(device_input));HIP_CHECK(hipFree(device_output));
        delete[] input;delete[] output;++cases;
    }
    printf("%u complete kernel topology/output cases pass\n",cases);
}
'''


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization so acceptance assertions execute')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--kernel-only',action='store_true')
    ap.add_argument('--cuda-arch',default=os.environ.get('CUDA_ARCH','sm_70'))
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]; work=args.work_dir.resolve()
    work.mkdir(parents=True)
    for relative in ('common_hip','tier1/cache_bandwidth/hip'):
        shutil.copytree(repo/relative,work/relative,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('cache_bandwidth','__pycache__'))
    directory=work/'tier1/cache_bandwidth/hip'
    subprocess.run(['make','PLATFORM=cuda','CUDA_ARCH='+args.cuda_arch,'-B'],cwd=directory,check=True,timeout=120)
    topology=directory/'topology.hip'
    topology.write_text(kernel_test((directory/'cache_bandwidth.hip').read_text()))
    subprocess.run(['nvcc','-O3','-std=c++14','-arch='+args.cuda_arch,'-x','cu',str(topology),'-o',str(directory/'topology')],check=True,timeout=120)
    if args.compile_only:
        print('Production and actual-kernel topology test compile; GPU execution not performed')
        return
    clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    p=subprocess.run([str(directory/'topology')],env=clean,capture_output=True,text=True,timeout=60)
    (work/'topology.stdout').write_text(p.stdout);(work/'topology.stderr').write_text(p.stderr)
    if p.returncode:raise RuntimeError((p.returncode,p.stdout,p.stderr))
    print(p.stdout.strip(),flush=True)
    if args.kernel_only:return
    manifest=json.loads((directory/'params.json').read_text())
    cases=[dict(working_set_bytes=str(ws),mode=mode) for mode in ('shared','private') for ws in manifest['scaling_param']['values']]
    cases += [{},dict(mode='private',working_set_bytes='1040',threads_per_block='65',blocks_per_cu='3'),
              dict(mode='shared',working_set_bytes='48',threads_per_block='64',blocks_per_cu='1'),
              dict(mode='private',working_set_bytes='1073741824',threads_per_block='256',blocks_per_cu='2'),
              dict(mode='unknown',threads_per_block='63',blocks_per_cu='0'),
              dict(mode='private',working_set_bytes='4294967312'),
              dict(working_set_bytes='17'),dict(blocks_per_cu='2147483647')]
    results=[]
    for index,controls in enumerate(cases):
        env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in controls.items()})
        start=time.monotonic()
        p=subprocess.run([str(directory/'cache_bandwidth')],env=env,capture_output=True,text=True,timeout=60)
        elapsed=time.monotonic()-start
        (work/f'case-{index}.stdout').write_text(p.stdout);(work/f'case-{index}.stderr').write_text(p.stderr)
        rows=[json.loads(line) for line in p.stdout.splitlines()]
        if index >= len(cases)-3:
            reason=('too large for private mode', 'Invalid working_set_bytes',
                    'Effective launch geometry exceeds supported indices or sizes')[index-(len(cases)-3)]
            if p.returncode != 1 or reason not in p.stderr or any(x.get('type')=='kernel' for x in rows):
                raise AssertionError((controls,p.returncode,p.stdout,p.stderr))
        else:
            if p.returncode or len(rows)!=2 or rows[0]['type']!='kernel' or rows[1]['type']!='summary':
                raise AssertionError((controls,p.returncode,p.stdout,p.stderr))
            k=rows[0];r=k['params'];b=r['num_blocks'];ws=r['working_set_bytes'];t=r['threads_per_block']
            assert k['time_ms']>0 and k['time_ms']==rows[1]['total_time_ms']
            assert ws==int(controls.get('working_set_bytes',1048576))
            mode='private' if controls.get('mode')=='private' else 'shared';assert r['mode']==mode
            default_t=1024 if mode=='private' else 256
            override_t=int(controls.get('threads_per_block',default_t))
            assert t==(override_t if 64<=override_t<=1024 else default_t)
            default_bpc=1 if mode=='private' else 8
            override_bpc=int(controls.get('blocks_per_cu',default_bpc))
            resolved_bpc=override_bpc if override_bpc>0 else default_bpc
            units=int(re.search(r'\(id=\d+, (\d+) CUs\)',p.stderr).group(1))
            assert r['blocks_per_cu']==resolved_bpc
            requested=r['requested_num_blocks'];assert requested==resolved_bpc*units
            assert b>0 and b<=requested
            if mode=='private':
                assert b==r['windows'] and r['allocation_bytes']<=4<<30
                assert r['launch_policy']=='distinct_windows_v1'
            else:
                assert b==requested and r['windows']==1 and r['launch_policy']=='shared_window_v1'
            assert r['allocation_bytes']==r['windows']*ws
            assert r['bytes_per_iter']==b*ws
            assert r['iters']==max(3,min(4000000,(8<<30)//(b*ws)))
            assert r['bytes_read']==b*ws*r['iters']
        results.append(dict(index=index,controls=controls,exit_code=p.returncode,wall_seconds=elapsed,records=rows))
        print(f'case {index}: exit {p.returncode}, {elapsed:.3f}s',flush=True)
    (work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'{len(cases)} production GPU success/refusal cases pass; maximum wall {max(x["wall_seconds"] for x in results):.3f}s')


if __name__=='__main__':main()
