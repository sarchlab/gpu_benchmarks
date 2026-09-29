#!/usr/bin/env python3
"""Check resolved Gram-Schmidt dimensions and actual host size/grid arithmetic."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess

INT_MAX = 2147483647


def section(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Gram-Schmidt input/geometry anchors missing')
    return start + source.split(start)[1].split(end)[0]


def host_source(source):
    parser = section(source, 'static int parseIntParam(',
                     '// ---------------------------------------------------------------------------\n// Kernel: reset')
    prefix = section(source, '    int M      = parseIntParam(', '    // Print device info')
    geometry = section(source, '    const int block_size = 256;', '    char params[160];')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
''' + parser + '\nint main(int argc,char**argv) {\n' + prefix + geometry + r'''
    printf("{\"m\":%d,\"n\":%d,\"a_bytes\":%zu,\"q_bytes\":%zu,\"r_bytes\":%zu,"
           "\"norm_bytes\":%zu,\"block\":%d,\"grid\":%d}\n",
           M,N,bytes_A,bytes_Q,bytes_R,bytes_1,block_size,grid_M);
}
'''


def expected(m, n):
    if m <= 0 or n <= 0:
        return 'Gram-Schmidt dimensions m and n must be positive\n'
    if m*n > INT_MAX:
        return 'Gram-Schmidt M*N exceeds INT_MAX\n'
    if n*n > INT_MAX:
        return 'Gram-Schmidt N*N exceeds INT_MAX\n'
    return dict(m=m,n=n,a_bytes=4*m*n,q_bytes=4*m*n,r_bytes=4*n*n,
                norm_bytes=4,block=256,grid=(m+255)//256)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/
                    'tier2/polybench_gramschmidt/polybench_gramschmidt.hip')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--binary',type=Path,help='Also run invalid cases on the actual binary with GPUs hidden')
    ap.add_argument('--work-dir',type=Path,required=True,help='New directory for source, binary and logs')
    args = ap.parse_args()
    work = args.work_dir.resolve(); work.mkdir(parents=True,exist_ok=False)
    pairs = [(-1,64),(0,64),(512,0),(512,-1),(-INT_MAX-1,1),(1,-INT_MAX-1),(0,0),(1,0),
             (46340,46340),(46341,46341),(1,46340),(1,46341),(INT_MAX,1),(INT_MAX,2),
             (2,INT_MAX),(1,INT_MAX),(33554431,64),(33554432,64),(INT_MAX-1,1),
             (1,1),(17,1),(17,9),(64,64),(65,64),(255,64),(256,64),(257,64)]
    manifest = json.loads(args.source.with_name('params.json').read_text())
    columns = next(x['values'] for x in manifest['non_scaling_params'] if x['name']=='n')
    pairs += list(itertools.product(manifest['scaling_param']['values'],columns))
    cases = [([],dict(m=str(m),n=str(n)),expected(m,n)) for m,n in pairs]
    cases += [([],{},expected(512,64)),
              (['--m','17','--n','9'],{},expected(17,9)),
              (['--m','0','--n','-1'],{},expected(512,64)),
              (['--m','0','--m','17'],{},expected(512,64)),
              (['--m','17','--m','512','--n','9'],{},expected(17,9)),
              (['--m','17','--n','9'],dict(m='65',n='8'),expected(65,8)),
              (['--m','17'],dict(m='0'),expected(0,64)),
              (['--n','9'],dict(n='0'),expected(512,0)),
              ([],dict(m='',n='9'),expected(0,9)),
              ([],dict(m='17',n='invalid'),expected(17,0)),
              ([],dict(m='17suffix',n=' 9'),expected(17,9))]
    cpp = work/'ranges.cpp'; cpp.write_text(host_source(args.source.read_text()))
    binary = work/'ranges'
    subprocess.run([args.compiler,'-std=c++14','-O1','-g','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],
                   check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    results = []
    targets = [('host',binary)]
    if args.binary: targets.append(('actual',args.binary.resolve()))
    for kind,target in targets:
        for argv,overrides,want in cases:
            invalid = isinstance(want,str)
            if kind=='actual' and not invalid: continue
            env = dict(clean,**{'BENCH_PARAM_'+k:v for k,v in overrides.items()})
            if kind=='actual': env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
            run = subprocess.run([str(target),*argv],env=env,capture_output=True,text=True,timeout=30)
            label = f'{len(results):03d}-{kind}'
            (work/(label+'.stdout')).write_text(run.stdout); (work/(label+'.stderr')).write_text(run.stderr)
            if invalid:
                passed = run.returncode==1 and run.stdout=='' and run.stderr==want
            else:
                passed = run.returncode==0 and run.stderr=='' and json.loads(run.stdout)==want
            results.append(dict(label=label,kind=kind,argv=argv,overrides=overrides,expected=want,
                                exit=run.returncode,passed=passed))
            (work/'report.json').write_text(json.dumps(results,indent=2)+'\n')
            if not passed: raise AssertionError(results[-1],run.stdout,run.stderr)
    print(f'{len(cases)} host range/geometry cases and '+
          f'{sum(x["kind"]=="actual" for x in results)} actual GPU-hidden rejections passed')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
