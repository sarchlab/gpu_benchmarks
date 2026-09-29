#!/usr/bin/env python3
"""Check actual SHOC scan admission, padding and recursive counts without allocation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

INT_MAX = 2147483647


def section(source,start,end):
    if source.count(start)!=1 or source.count(end)!=1:raise ValueError('SHOC scan input anchors missing')
    return start+source.split(start)[1].split(end)[0]


def host_source(source):
    parser=section(source,'static int parseIntParam(', '// Scan kernels\n')
    prefix=section(source,'    int array_size = 1048576;', '    // Print device info to stderr')
    maximum=re.findall(r'^#define MAX_BLOCK_SIZE .+$',source,re.M)
    count=re.findall(r'^    int num_blocks\s+= .+;$',source,re.M)
    if len(maximum)!=1 or len(count)!=1:raise ValueError('SHOC scan geometry anchors missing')
    return '\n'.join(['#include <cstdio>','#include <cstdlib>','#include <cstring>','#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        maximum[0],parser,'static int blocks(int N,int block_size) {',
        '    int elements_per_block = block_size * 2;',count[0],'    return num_blocks;','}',
        'int main(int argc,char** argv) {',prefix,
        r'    printf("{\"array_size\":%d,\"block_size\":%d,\"padded\":%d,\"bytes\":%zu,\"bytes_orig\":%zu,\"blocks\":[",N,block_size,N_padded,bytes,bytes_orig);',
        '    int n=N_padded; bool first=true;',
        '    do { n=blocks(n,block_size); printf("%s%d",first?"":",",n); first=false; } while(n>1);',
        r'    printf("]}\n");','}'])


def expected(n,b):
    if n<=0 or b<=0:return None,'SHOC scan array_size and block_size must be positive\n'
    b=min(b,512)
    if b not in (1,2,4,8,16,32,64,128,256,512):
        return None,'SHOC scan block_size must be a power of two after clamping\n'
    padded=((n+2*b-1)//(2*b))*(2*b)
    if padded>INT_MAX:return None,'SHOC scan padded array size exceeds INT_MAX\n'
    counts=[];level=padded
    while True:
        level=(level+2*b-1)//(2*b);counts.append(level)
        if level==1:break
    return dict(array_size=n,block_size=b,padded=padded,bytes=4*padded,bytes_orig=4*n,blocks=counts),''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/shoc_scan/shoc_scan.hip')
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--binary',type=Path,help='Require exact invalid-input diagnostics from the actual GPU-hidden binary')
    ap.add_argument('--work-dir',type=Path,required=True)
    args=ap.parse_args();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    sizes=json.loads(args.source.with_name('params.json').read_text())['scaling_param']['values']
    cases=[([],dict(array_size=str(n)),expected(n,256)) for n in sizes]
    for b in (1,2,4,8,16,32,64,128,256,512):
        limit=(INT_MAX//(2*b))*(2*b)
        cases += [([],dict(array_size=str(n),block_size=str(b)),expected(n,b))
                  for n in (1,2*b-1,2*b,2*b+1,limit-1,limit,limit+1)]
    cases += [([],dict(array_size=str(n)),expected(n,256)) for n in (-INT_MAX-1,-1,0,INT_MAX)]
    cases += [([],dict(array_size='513',block_size=str(b)),expected(513,b))
              for b in (-INT_MAX-1,-1,0,3,5,6,17,255,511,513,1024,INT_MAX)]
    cases += [([],{},expected(1048576,256)),
        (['--array_size','513','--block_size','32'],{},expected(513,32)),
        (['--array_size','0','--block_size','0'],{},expected(1048576,256)),
        (['--array_size','-1','--block_size','-1'],{},expected(1048576,256)),
        (['--array_size','invalid','--block_size','invalid'],{},expected(1048576,256)),
        (['--array_size','0','--array_size','7'],{},expected(1048576,256)),
        (['--array_size','513','--array_size','7'],{},expected(513,256)),
        (['--array_size','513','--block_size','3'],{},expected(513,3)),
        (['--array_size','513','--block_size','513'],{},expected(513,513)),
        (['--array_size','513','--block_size','32'],dict(array_size='7',block_size='2'),expected(7,2)),
        (['--array_size','513'],dict(array_size='0'),expected(0,256)),
        (['--block_size','32'],dict(block_size='0'),expected(1048576,0)),
        ([],dict(array_size=''),expected(0,256)),
        ([],dict(array_size='invalid'),expected(0,256)),
        ([],dict(array_size=' 513suffix',block_size=' 32suffix'),expected(513,32))]
    cpp=work/'ranges.cpp';cpp.write_text(host_source(args.source.read_text()));binary=work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SCAN_REFERENCE_TEST_'))}
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
            if invalid:passed &= p.stdout==''
            else:
                try:passed &= json.loads(p.stdout)==want
                except json.JSONDecodeError:passed=False
            results.append(dict(label=label,kind=kind,argv=argv,overrides=overrides,expected=want,exit=p.returncode,passed=passed))
            (work/'report.json').write_text(json.dumps(results,indent=2)+'\n')
            if not passed:raise AssertionError(results[-1],p.stdout,p.stderr)
    print(f'{len(cases)} host geometry cases and {sum(x["kind"]=="actual" for x in results)} actual GPU-hidden rejections passed')


if __name__=='__main__':main()
