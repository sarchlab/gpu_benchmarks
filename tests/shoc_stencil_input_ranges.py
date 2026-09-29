#!/usr/bin/env python3
"""Check actual SHOC Stencil2D input guards and size/grid arithmetic."""
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
        raise ValueError('SHOC Stencil2D input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def one(source, pattern):
    matches = re.findall(pattern, source, re.M)
    if len(matches) != 1: raise ValueError('SHOC Stencil2D size anchor missing or duplicated: '+pattern)
    return matches[0]


def host_source(source):
    parser = section(source, 'static int parseIntParam(', '// Stencil kernel')
    prefix = section(source, '    int N      = parseIntParam(', '    // Print device info')
    cells = one(source, r'std::vector<float> h_data\((.+), 0.0f\);')
    byte_size = one(source, r'^    size_t bytes = .+;$')
    grid = one(source, r'^    int grid_size = .+;$')
    for anchor in ('dim3 block(block_size, block_size);', 'dim3 grid(grid_size, grid_size);'):
        if source.count(anchor) != 1: raise ValueError('SHOC Stencil2D launch geometry anchor changed')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <cstring>', '#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        parser, 'int main(int argc,char** argv) {', prefix, byte_size, grid,
        'size_t cells='+cells+';',
        r'printf("{\"n\":%d,\"iterations\":%d,\"block_size\":%d,\"grid\":%d,\"cells\":%zu,\"bytes\":%zu}\n",',
        'N,iters,block_size,grid_size,cells,bytes);', '}'])


def expected(n, steps, block=16):
    if n < 3 or steps <= 0 or block <= 0:
        return None, 'SHOC Stencil2D requires n>=3 and positive iterations/block_size\n'
    if n*n > INT_MAX:
        return None, 'SHOC Stencil2D n*n exceeds INT_MAX\n'
    return dict(n=n, iterations=steps, block_size=block, grid=(n-2+block-1)//block,
                cells=n*n, bytes=4*n*n), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/shoc_stencil2d/shoc_stencil2d.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid cases against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = list(itertools.product(manifest['scaling_param']['values'], controls['iterations'], controls['block_size']))
    values += [(n,5,16) for n in (-INT_MAX-1,-1,0,1,2,3,17,46339,46340,46341,INT_MAX)]
    values += [(17,t,16) for t in (-INT_MAX-1,-1,0,1,2,5,INT_MAX)]
    values += [(17,5,b) for b in (-INT_MAX-1,-1,0,1,2,3,7,16,32,1024,INT_MAX)]
    cases = [([],dict(n=str(n),iterations=str(t),block_size=str(b)),expected(n,t,b)) for n,t,b in values]
    cases += [([],{},expected(2048,5)),
        (['--n','17','--iterations','3'],{},expected(17,3)),
        (['--n','2'],{},expected(2,5)),
        (['--n','0','--iterations','0'],{},expected(2048,5)),
        (['--n','-1','--iterations','-1'],{},expected(2048,5)),
        (['--iterations','invalid'],{},expected(2048,5)),
        (['--iterations','0','--iterations','7'],{},expected(2048,5)),
        (['--iterations','3','--iterations','7'],{},expected(2048,3)),
        (['--n','17','--iterations','3'],dict(n='33',iterations='1',block_size='3'),expected(33,1,3)),
        (['--n','17'],dict(n='0'),expected(0,5)),
        (['--iterations','3'],dict(iterations='0'),expected(2048,0)),
        ([],dict(iterations=''),expected(2048,0)),
        ([],dict(iterations='invalid'),expected(2048,0)),
        ([],dict(n=' 17suffix',iterations=' 3suffix',block_size=' 7suffix'),expected(17,3,7))]
    cpp = work/'ranges.cpp'; cpp.write_text(host_source(args.source.read_text())); binary = work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SHOC_STENCIL_TEST_'))}
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
