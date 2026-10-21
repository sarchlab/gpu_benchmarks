#!/usr/bin/env python3
"""Check actual PageRank guards, row sizing and grid arithmetic without allocation."""
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
        raise ValueError('PageRank input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def one(source, pattern):
    matches = re.findall(pattern, source, re.M)
    if len(matches) != 1: raise ValueError('PageRank size anchor missing or duplicated: '+pattern)
    return matches[0]


def host_source(source):
    parser = section(source, 'static int parseIntParam(', '// Simple deterministic PRNG')
    prefix = section(source, '    int vertices      = 131072;', '    // Generate CSC graph')
    rows = one(source, r'col_offsets.resize\((.+)\);')
    allocation = one(source, r'HIP_CHECK\(hipMalloc\(&d_col_offsets, (.+)\)\);')
    copy = one(source, r'hipMemcpy\(d_col_offsets, col_offsets.data\(\), (.+), hipMemcpyHostToDevice\)')
    block = one(source, r'^    int blockSize = .+;$')
    grid = one(source, r'^    int gridSize\s+= .+;$')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <cstring>', '#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        parser, 'int main(int argc,char** argv) {', prefix, '(void)damping;', block, grid,
        'size_t rows='+rows+', allocation='+allocation+', copy='+copy+';',
        r'printf("{\"vertices\":%d,\"degree\":%d,\"pr_iterations\":%d,\"grid\":%d,\"rows\":%zu,\"allocation\":%zu,\"copy\":%zu,\"max_edges\":%lld}\n",',
        'N,avg_deg,pr_iters,gridSize,rows,allocation,copy,(long long)N*max_degree);', '}'])


def expected(n, d, steps):
    if n <= 0 or d < 0 or steps <= 0:
        return None, 'PageRank vertices and pr_iterations must be positive; degree must be nonnegative\n'
    edges = n * max(1, d//2 + d)
    if edges > INT_MAX:
        return None, 'PageRank maximum generated edge count exceeds INT_MAX\n'
    return dict(vertices=n, degree=d, pr_iterations=steps, grid=(n+255)//256,
                rows=n+1, allocation=4*(n+1), copy=4*(n+1), max_edges=edges), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/graph_pr/hip/graph_pr.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid cases against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = list(itertools.product(manifest['scaling_param']['values'], controls['degree'], controls['pr_iterations']))
    for n in (1,2,17,1024,8388608,INT_MAX):
        lo, hi = 0, INT_MAX
        while lo < hi:
            mid = (lo+hi+1)//2
            if n*max(1,mid//2+mid) <= INT_MAX: lo = mid
            else: hi = mid-1
        values += [(n,d,5) for d in (lo-1,lo,lo+1)]
    values += [(n,0,5) for n in (-INT_MAX-1,-1,0,1,INT_MAX-256,INT_MAX-255,INT_MAX-1,INT_MAX)]
    values += [(1025,d,5) for d in (-INT_MAX-1,-1,0,INT_MAX)]
    values += [(1025,16,t) for t in (-INT_MAX-1,-1,0,1,2,5,INT_MAX)]
    cases = [([],dict(vertices=str(n),degree=str(d),pr_iterations=str(t)),expected(n,d,t)) for n,d,t in values]
    cases += [([],{},expected(131072,16,5)),
        (['--vertices','17','--degree','4','--pr-iterations','3'],{},expected(17,4,3)),
        (['--vertices','0','--degree','0','--pr-iterations','0'],{},expected(131072,16,5)),
        (['--vertices','-1','--degree','-1','--pr-iterations','-1'],{},expected(131072,16,5)),
        (['--pr-iterations','invalid'],{},expected(131072,16,5)),
        (['--pr-iterations','0','--pr-iterations','7'],{},expected(131072,16,5)),
        (['--pr-iterations','3','--pr-iterations','7'],{},expected(131072,16,3)),
        (['--vertices','17','--degree','4','--pr-iterations','3'],dict(vertices='33',degree='0',pr_iterations='1'),expected(33,0,1)),
        (['--vertices','17'],dict(vertices='0'),expected(0,16,5)),
        (['--degree','4'],dict(degree='-1'),expected(131072,-1,5)),
        (['--pr-iterations','3'],dict(pr_iterations='0'),expected(131072,16,0)),
        ([],dict(pr_iterations=''),expected(131072,16,0)),
        ([],dict(pr_iterations='invalid'),expected(131072,16,0)),
        ([],dict(vertices=' 17suffix',degree=' 0suffix',pr_iterations=' 3suffix'),expected(17,0,3))]
    cpp = work/'ranges.cpp'; cpp.write_text(host_source(args.source.read_text())); binary = work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','PR_TEST_'))}
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
