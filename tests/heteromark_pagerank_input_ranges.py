#!/usr/bin/env python3
"""Check actual Hetero-Mark PageRank integer guards and size expressions."""
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
        raise ValueError('Hetero-Mark PageRank input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def one(source, pattern):
    matches = re.findall(pattern, source, re.M)
    if len(matches) != 1: raise ValueError('Hetero-Mark PageRank size anchor missing or duplicated: '+pattern)
    return matches[0]


def host_source(source):
    integer = section(source, 'static int parseIntParam(', '// Simple deterministic PRNG')
    floating = section(source, 'static double parseDoubleParam(', 'int main(int argc, char** argv)')
    prefix = section(source, '    int vertices     = 65536;', '    // Print device info')
    rows = one(source, r'rowPtr.resize\((.+)\);')
    allocation = one(source, r'HIP_CHECK\(hipMalloc\(&d_rowPtr, (.+)\)\);')
    copy = one(source, r'hipMemcpy\(d_rowPtr, rowPtr.data\(\), (.+), hipMemcpyHostToDevice\)')
    block = one(source, r'^    int blockSize = .+;$')
    grid = one(source, r'^    int gridSize\s+= .+;$')
    if source.count('generateCSR(N, avgDegree, 42, rowPtr, colIdx, outDegree);') != 1:
        raise ValueError('Hetero-Mark PageRank graph degree anchor changed')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <cstring>', '#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        integer, floating, 'int main(int argc,char** argv) {', prefix, '(void)damping;', block, grid,
        'size_t rows='+rows+', allocation='+allocation+', copy='+copy+';',
        r'printf("{\"vertices\":%d,\"pr_iterations\":%d,\"damping_factor\":%.6f,\"grid\":%d,\"rows\":%zu,\"allocation\":%zu,\"copy\":%zu,\"edges\":%lld}\n",',
        'N,prIters,damping_factor,gridSize,rows,allocation,copy,(long long)N*avgDegree);', '}'])


def expected(n, steps, damping=.85):
    if n <= 0 or steps <= 0:
        return None, 'Hetero-Mark PageRank vertices and pr_iterations must be positive\n'
    if 16*n > INT_MAX:
        return None, 'Hetero-Mark PageRank generated edge count exceeds INT_MAX\n'
    return dict(vertices=n, pr_iterations=steps, damping_factor=damping, grid=(n+255)//256,
                rows=n+1, allocation=4*(n+1), copy=4*(n+1), edges=16*n), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/heteromark_pagerank/heteromark_pagerank.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid cases against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = list(itertools.product(manifest['scaling_param']['values'], controls['pr_iterations'], controls['damping_factor']))
    values += [(n,5,.85) for n in (-INT_MAX-1,-1,0,1,255,256,257,INT_MAX//16-1,INT_MAX//16,INT_MAX//16+1,INT_MAX)]
    values += [(1025,t,.85) for t in (-INT_MAX-1,-1,0,1,2,5,INT_MAX)]
    cases = [([],dict(vertices=str(n),pr_iterations=str(t),damping_factor=str(d)),expected(n,t,d)) for n,t,d in values]
    cases += [([],{},expected(65536,5)),
        (['--vertices','17','--pr-iterations','3','--damping_factor','.7'],{},expected(17,3,.7)),
        (['--vertices','0','--pr-iterations','0'],{},expected(65536,5)),
        (['--vertices','-1','--pr-iterations','-1'],{},expected(65536,5)),
        (['--pr-iterations','invalid'],{},expected(65536,5)),
        (['--pr-iterations','0','--pr-iterations','7'],{},expected(65536,5)),
        (['--pr-iterations','3','--pr-iterations','7'],{},expected(65536,3)),
        (['--vertices','17','--pr-iterations','3','--damping_factor','.7'],dict(vertices='33',pr_iterations='1',damping_factor='.9'),expected(33,1,.9)),
        (['--vertices','17'],dict(vertices='0'),expected(0,5)),
        (['--pr-iterations','3'],dict(pr_iterations='0'),expected(65536,0)),
        ([],dict(pr_iterations=''),expected(65536,0)),
        ([],dict(pr_iterations='invalid'),expected(65536,0)),
        ([],dict(vertices='invalid'),expected(0,5)),
        ([],dict(vertices=' 17suffix',pr_iterations=' 3suffix',damping_factor=' .7suffix'),expected(17,3,.7))]
    cpp = work/'ranges.cpp'; cpp.write_text(host_source(args.source.read_text())); binary = work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
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
