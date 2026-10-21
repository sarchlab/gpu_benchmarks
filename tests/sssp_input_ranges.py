#!/usr/bin/env python3
"""Check actual SSSP input guards and grid arithmetic without allocating graphs."""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

INT_MAX = 2147483647


def section(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('SSSP input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def host_source(source):
    parser = section(source, 'static int parseIntParam(', '// Simple deterministic PRNG')
    prefix = section(source, '    int vertices   = 65536;', '    // Generate CSR graph')
    grid = re.findall(r'^    int gridSize\s+= .+;$', source, re.M)
    if len(grid) != 1: raise ValueError('SSSP grid anchor missing or duplicated')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <cstring>', '#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        parser, 'static int blocks(int N,int blockSize) {', grid[0], 'return gridSize;}',
        'int main(int argc,char** argv) {', prefix,
        r'printf("{\"vertices\":%d,\"degree\":%d,\"block_size\":%d,\"source\":%d,\"grid\":%d,\"max_edges\":%lld,\"max_candidate\":%lld}\n",',
        'N,avg_deg,block_size,source,blocks(N,block_size),(long long)N*max_degree,(long long)N*100);', '}'])


def expected(n, d, b):
    if n <= 0 or d < 0 or b <= 0:
        return None, 'SSSP vertices and block_size must be positive; degree must be nonnegative\n'
    edges = n * max(1, d//2 + d)
    if edges > INT_MAX:
        return None, 'SSSP maximum generated edge count exceeds INT_MAX\n'
    if 100*n >= INT_MAX:
        return None, 'SSSP maximum distance candidate reaches INT_MAX\n'
    return dict(vertices=n, degree=d, block_size=b, source=0, grid=(n+b-1)//b,
                max_edges=edges, max_candidate=100*n), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/lonestar_sssp/hip/lonestar_sssp.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid cases against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = list(itertools.product(manifest['scaling_param']['values'], controls['degree'], controls['block_size']))
    for n in (1,2,17,1024,8388608,INT_MAX//100):
        lo, hi = 0, INT_MAX
        while lo < hi:
            mid = (lo+hi+1)//2
            if n*max(1,mid//2+mid) <= INT_MAX: lo = mid
            else: hi = mid-1
        values += [(n,d,256) for d in (lo-1,lo,lo+1)]
    values += [(n,0,256) for n in (-INT_MAX-1,-1,0,1,INT_MAX//100-1,INT_MAX//100,INT_MAX//100+1,INT_MAX)]
    values += [(1025,d,256) for d in (-INT_MAX-1,-1,0,1,INT_MAX)]
    values += [(1025,16,b) for b in (-INT_MAX-1,-1,0,1,3,17,33,256,512,1024,INT_MAX)]
    cases = [([],dict(vertices=str(n),degree=str(d),block_size=str(b)),expected(n,d,b)) for n,d,b in values]
    cases += [([],{},expected(65536,16,256)),
        (['--vertices','17','--degree','4','--block_size','3'],{},expected(17,4,3)),
        (['--vertices','0','--degree','0','--block_size','0'],{},expected(65536,16,256)),
        (['--vertices','-1','--degree','-1','--block_size','-1'],{},expected(65536,16,256)),
        (['--vertices','invalid'],{},expected(65536,16,256)),
        (['--vertices','0','--vertices','7'],{},expected(65536,16,256)),
        (['--vertices','17','--vertices','7'],{},expected(17,16,256)),
        (['--vertices','17','--degree','4','--block_size','3'],dict(vertices='33',degree='0',block_size='1'),expected(33,0,1)),
        (['--vertices','17'],dict(vertices='0'),expected(0,16,256)),
        (['--degree','4'],dict(degree='-1'),expected(65536,-1,256)),
        (['--block_size','32'],dict(block_size='0'),expected(65536,16,0)),
        ([],dict(vertices=''),expected(0,16,256)),
        ([],dict(vertices='invalid'),expected(0,16,256)),
        ([],dict(vertices=' 17suffix',degree=' 0suffix',block_size=' 3suffix'),expected(17,0,3))]
    cpp = work/'ranges.cpp'; cpp.write_text(host_source(args.source.read_text())); binary = work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SSSP_TEST_'))}
    targets = [('host',binary)]
    if args.binary: targets.append(('actual',args.binary.resolve()))
    results = []
    for kind,target in targets:
        for argv,overrides,(want,diagnostic) in cases:
            invalid = want is None
            if kind == 'actual' and not invalid: continue
            env = dict(clean,**{'BENCH_PARAM_'+k:v for k,v in overrides.items()})
            if kind == 'actual': env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
            p = subprocess.run([str(target),*argv],env=env,capture_output=True,text=True,timeout=30)
            label = f'{len(results):03d}-{kind}'
            (work/(label+'.stdout')).write_text(p.stdout); (work/(label+'.stderr')).write_text(p.stderr)
            passed = p.returncode == (1 if invalid else 0) and p.stderr == diagnostic
            if invalid: passed &= p.stdout == ''
            else:
                try: passed &= json.loads(p.stdout) == want
                except json.JSONDecodeError: passed = False
            results.append(dict(label=label,kind=kind,argv=argv,overrides=overrides,expected=want,exit=p.returncode,passed=passed))
            (work/'report.json').write_text(json.dumps(results,indent=2)+'\n')
            if not passed: raise AssertionError(results[-1],p.stdout,p.stderr)
    print(f'{len(cases)} host input cases and {sum(x["kind"]=="actual" for x in results)} actual GPU-hidden rejections passed')


if __name__ == '__main__': main()
