#!/usr/bin/env python3
"""Check large-scan input admission and the actual recursive block-count expression."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess

INT_MAX = 2147483647


def section(source, start, end):
    if source.count(start)!=1 or source.count(end)!=1:
        raise ValueError('Scan input anchors missing')
    return start+source.split(start)[1].split(end)[0]


def host_source(source):
    parser=section(source,'static int parseIntParam(', '// Constants\n')
    prefix=section(source,'    int size = 16777216;', '    // Print device info')
    constants=re.findall(r'^#define (?:BLOCK_SIZE|ELEMENTS_PER_BLOCK) .+$',source,re.M)
    counts=re.findall(r'^    int numBlocks = .+;$',source,re.M)
    if len(constants)!=2 or len(counts)!=1:raise ValueError('Scan geometry anchors missing')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <cstring>',
                      'static_assert(sizeof(int)==4 && sizeof(unsigned int)==4 && sizeof(size_t)>=8, "Tests require a supported 64-bit host");',
                      *constants,parser,'int main(int argc,char** argv) {',prefix,counts[0],
                      r'    printf("{\"size\":%d,\"bytes\":%zu,\"blocks\":%d,\"verification_enabled\":%s}\n",N,bytes,numBlocks,run_verify?"true":"false");','}'])


def expected(n,verify=True):
    if n<=0: return None,'Scan size must be positive\n'
    return dict(size=n,bytes=4*n,blocks=(n+511)//512,verification_enabled=verify),''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/cuda_scan_large/cuda_scan_large.hip')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--binary',type=Path,help='Also require exact invalid-input errors from the actual binary with GPUs hidden')
    ap.add_argument('--work-dir',type=Path,required=True,help='New directory for source, binary and raw logs')
    args=ap.parse_args();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    sizes=[-INT_MAX-1,-INT_MAX,-512,-1,0,1,255,256,511,512,513,1023,1024,1025,
           262143,262144,262145,INT_MAX-512,INT_MAX-511,INT_MAX-510,INT_MAX]
    sizes+=json.loads(args.source.with_name('params.json').read_text())['scaling_param']['values']
    cases=[([],dict(size=str(n)),expected(n)) for n in sizes]
    cases += [([],{},expected(16777216)),
              (['--size','513'],{},expected(513)),
              (['--size','0'],{},expected(16777216)),
              (['--size','-1'],{},expected(16777216)),
              (['--size','invalid'],{},expected(16777216)),
              (['--size','0','--size','7'],{},expected(16777216)),
              (['--size','513','--size','7'],{},expected(513)),
              (['--size',str(INT_MAX)],{},expected(INT_MAX)),
              (['--size','513'],dict(size='7'),expected(7)),
              (['--size','513'],dict(size='0'),expected(0)),
              ([],dict(size=''),expected(0)),
              ([],dict(size='invalid'),expected(0)),
              ([],dict(size=' 513suffix'),expected(513))]
    for value in ('false','0','skip'):
        cases += [([],dict(size='0',verify=value),expected(0)),
                  ([],dict(size='513',verify=value),expected(513,False))]
    cpp=work/'ranges.cpp';cpp.write_text(host_source(args.source.read_text()));binary=work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SCAN_TEST_'))}
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
    print(f'{len(cases)} host size/block cases and '+f'{sum(r["kind"]=="actual" for r in results)} actual GPU-hidden rejections passed')
    return 0


if __name__=='__main__':raise SystemExit(main())
