#!/usr/bin/env python3
"""Check actual CC guards and grid arithmetic without allocating graphs."""
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
        raise ValueError('CC input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def host_source(source):
    prefix = section(source, '    int num_vertices   = 100000;', '    // Generate CSR graph')
    grid = re.findall(r'^    int gridSize\s*= .+;$', source, re.M)
    rows = re.findall(r'row_offsets.resize\((.+)\);', source)
    if len(grid) != 1 or len(rows) != 1: raise ValueError('CC size anchors missing or duplicated')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        'static int blocks(int N,int block_size) {', grid[0], 'return gridSize;}',
        'int main() {', prefix, 'size_t rows='+rows[0]+';',
        r'printf("{\"num_vertices\":%d,\"avg_degree\":%d,\"max_iterations\":%d,\"block_size\":%d,\"grid\":%d,\"rows\":%zu,\"max_edges\":%llu}\n",',
        'N,avg_degree,max_iterations,block_size,blocks(N,block_size),rows,2ULL*N*max_degree);', '}'])


def expected(n, d, cap, b):
    if n <= 0 or d < 0 or cap <= 0 or b <= 0:
        return None, 'CC num_vertices, max_iterations and block_size must be positive; avg_degree must be nonnegative\n'
    edges = 2*n*max(1,d//2+d)
    if edges > INT_MAX:
        return None, 'CC maximum symmetric edge count exceeds INT_MAX\n'
    return dict(num_vertices=n, avg_degree=d, max_iterations=cap, block_size=b,
                grid=(n+b-1)//b, rows=n+1, max_edges=edges), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/graph_cc/graph_cc.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid cases against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = list(itertools.product(manifest['scaling_param']['values'], controls['avg_degree'],
                                   controls['max_iterations'], controls['block_size']))
    for n in (1,2,17,1024,1000000,INT_MAX//2):
        lo, hi = 0, INT_MAX
        while lo < hi:
            mid = (lo+hi+1)//2
            if 2*n*max(1,mid//2+mid) <= INT_MAX: lo = mid
            else: hi = mid-1
        values += [(n,d,100,256) for d in (lo-1,lo,lo+1)]
    values += [(n,0,100,256) for n in (-INT_MAX-1,-1,0,1,INT_MAX//2-1,INT_MAX//2,INT_MAX//2+1,INT_MAX)]
    values += [(1025,d,100,256) for d in (-INT_MAX-1,-1,0,INT_MAX)]
    values += [(1025,16,t,256) for t in (-INT_MAX-1,-1,0,1,2,100,INT_MAX)]
    values += [(1025,16,100,b) for b in (-INT_MAX-1,-1,0,1,3,33,256,512,1024,INT_MAX)]
    values += [(INT_MAX,INT_MAX,INT_MAX,INT_MAX)] # The guard itself must not overflow signed64.
    cases = [([],dict(num_vertices=str(n),avg_degree=str(d),max_iterations=str(t),block_size=str(b)),expected(n,d,t,b))
             for n,d,t,b in values]
    cases += [([],{},expected(100000,16,100,256)),
        ([],dict(num_vertices='17',avg_degree='0',max_iterations='1',block_size='3'),expected(17,0,1,3)),
        ([],dict(max_iterations=''),expected(100000,16,0,256)),
        ([],dict(max_iterations='invalid'),expected(100000,16,0,256)),
        ([],dict(num_vertices=''),expected(0,16,100,256)),
        ([],dict(num_vertices='invalid'),expected(0,16,100,256)),
        ([],dict(num_vertices=' 17suffix',avg_degree=' 0suffix',max_iterations=' 1suffix',block_size=' 3suffix'),expected(17,0,1,3)),
        (['--num_vertices','0','--max_iterations','0'],{},expected(100000,16,100,256))]
    cpp = work/'ranges.cpp'; cpp.write_text(host_source(args.source.read_text())); binary = work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','CC_TEST_'))}
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
