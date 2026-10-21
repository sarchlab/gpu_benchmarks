#!/usr/bin/env python3
"""Check resolved transpose dimensions and signed flat-index admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess

INT_MAX = 2147483647


def section(source, start, end):
    if source.count(start)!=1 or source.count(end)!=1:
        raise ValueError('Transpose input anchors missing')
    return start+source.split(start)[1].split(end)[0]


def host_source(source):
    parser=section(source,'static int parseIntParam(', '// Constants — classic CUDA transpose pattern')
    prefix=section(source,'    int N = 4096;', '    // Print device info to stderr')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
static_assert(sizeof(int)==4 && sizeof(float)==4 && sizeof(size_t)>=8,
              "Tests require a supported 64-bit host");
'''+parser+'\nint main(int argc,char** argv) {\n'+prefix+r'''
    printf("{\"size\":%d,\"elements\":%zu,\"bytes\":%zu}\n",N,num_elems,bytes);
}
'''


def expected(n):
    if n<=0: return None,'Transpose size must be positive\n'
    if n*n>INT_MAX: return None,'Transpose size exceeds the signed index range\n'
    return dict(size=n,elements=n*n,bytes=4*n*n),''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/cuda_transpose/hip/cuda_transpose.hip')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--binary',type=Path,help='Also require exact invalid-input errors from the actual binary with GPUs hidden')
    ap.add_argument('--work-dir',type=Path,required=True,help='New directory for source, binary and raw logs')
    args=ap.parse_args();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    sizes=[-INT_MAX-1,-INT_MAX,-46341,-1,0,1,7,31,32,33,46339,46340,46341,46342,65536,INT_MAX]
    sizes+=json.loads(args.source.with_name('params.json').read_text())['scaling_param']['values']
    cases=[([],dict(size=str(n)),expected(n)) for n in sizes]
    cases += [([],{},expected(4096)),
              (['--size','33'],{},expected(33)),
              (['--size','0'],{},expected(4096)),
              (['--size','-1'],{},expected(4096)),
              (['--size','invalid'],{},expected(4096)),
              (['--size','0','--size','7'],{},expected(4096)),
              (['--size','33','--size','7'],{},expected(33)),
              (['--size','46341'],{},expected(46341)),
              (['--size','46341'],dict(size='7'),expected(7)),
              (['--size','33'],dict(size='46341'),expected(46341)),
              (['--size','33'],dict(size='0'),expected(0)),
              ([],dict(size=''),expected(0)),
              ([],dict(size='invalid'),expected(0)),
              ([],dict(size=' 33suffix'),expected(33))]
    cpp=work/'ranges.cpp';cpp.write_text(host_source(args.source.read_text()));binary=work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','TRANSPOSE_TEST_'))}
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
    print(f'{len(cases)} host dimension cases and '+f'{sum(r["kind"]=="actual" for r in results)} actual GPU-hidden rejections passed')
    return 0


if __name__=='__main__':raise SystemExit(main())
