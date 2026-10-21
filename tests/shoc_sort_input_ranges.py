#!/usr/bin/env python3
"""Check resolved SHOC sort sizes, rounding and actual buffer-byte arithmetic."""
import argparse
import json
import os
from pathlib import Path
import subprocess

INT_MAX = 2147483647


def section(source, start, end):
    if source.count(start)!=1 or source.count(end)!=1:
        raise ValueError('SHOC sort input anchors missing')
    return start+source.split(start)[1].split(end)[0]


def host_source(source):
    parser=section(source,'static int parseIntParam(', '// Number of scan CTAs')
    prefix=section(source,'    int size = 1 << 22;', '    int device_id = 0;')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
typedef unsigned int uint;
static_assert(sizeof(uint)==4 && sizeof(size_t)>=8, "Tests require a supported 64-bit host");
'''+parser+'\nint main(int argc,char** argv) {\n'+prefix+r'''
    printf("{\"size\":%d,\"bytes\":%llu}\n",size,(unsigned long long)bytes);
}
'''


def expected(size):
    if size<=0: return None,'SHOC sort size must be positive\n'
    rounded=max(1024,size//1024*1024)
    warning=f'Warning: size={size} not a multiple of 1024, using {rounded}\n' if size%1024 else ''
    return dict(size=rounded,bytes=4*rounded),warning


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/shoc_sort/hip/shoc_sort.hip')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--binary',type=Path,help='Also require exact invalid-input errors from the actual binary with GPUs hidden')
    ap.add_argument('--work-dir',type=Path,required=True,help='New directory for source, binary and raw logs')
    args=ap.parse_args();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    sizes=[-INT_MAX-1,-INT_MAX,-1025,-1024,-1023,-1,0,1,2,17,1023,1024,1025,2047,2048,2049,
           2**30-1,2**30,2**30+1,INT_MAX-1,INT_MAX]
    sizes+=json.loads(args.source.with_name('params.json').read_text())['scaling_param']['values']
    cases=[([],dict(size=str(n)),expected(n)) for n in sizes]
    cases += [([],{},expected(4194304)),
              (['--size','3073'],{},expected(3073)),
              (['--size','0'],{},expected(4194304)),
              (['--size','-1024'],{},expected(4194304)),
              (['--size','invalid'],{},expected(4194304)),
              (['--size','0','--size','4096'],{},expected(4194304)),
              (['--size','2048','--size','4096'],{},expected(2048)),
              (['--size','2048'],dict(size='3073'),expected(3073)),
              (['--size','2048'],dict(size='0'),expected(0)),
              (['--size','2048'],dict(size='-1024'),expected(-1024)),
              ([],dict(size=''),expected(0)),
              ([],dict(size='invalid'),expected(0)),
              ([],dict(size=' 2049suffix'),expected(2049))]
    cpp=work/'ranges.cpp';cpp.write_text(host_source(args.source.read_text()));binary=work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SORT_TEST_'))}
    targets=[('host',binary)]
    if args.binary:targets.append(('actual',args.binary.resolve()))
    results=[]
    for kind,target in targets:
        for argv,overrides,(want,diagnostic) in cases:
            invalid=want is None
            if kind=='actual' and not invalid:continue
            env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in overrides.items()})
            if kind=='actual':env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
            p=subprocess.run([str(target),*argv],env=env,capture_output=True,text=True,timeout=30)
            label=f'{len(results):03d}-{kind}'
            (work/(label+'.stdout')).write_text(p.stdout);(work/(label+'.stderr')).write_text(p.stderr)
            passed=p.returncode==(1 if invalid else 0) and p.stderr==diagnostic
            passed &= p.stdout=='' if invalid else json.loads(p.stdout)==want
            results.append(dict(label=label,kind=kind,argv=argv,overrides=overrides,expected=want,
                                diagnostic=diagnostic,exit=p.returncode,passed=passed))
            (work/'report.json').write_text(json.dumps(results,indent=2)+'\n')
            if not passed:raise AssertionError(results[-1],p.stdout,p.stderr)
    print(f'{len(cases)} host size/byte cases and '+f'{sum(r["kind"]=="actual" for r in results)} actual GPU-hidden rejections passed')
    return 0


if __name__=='__main__':raise SystemExit(main())
