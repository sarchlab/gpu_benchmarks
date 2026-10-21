#!/usr/bin/env python3
"""Check actual FDTD-2D input guards, initialization bounds and byte arithmetic."""
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
        raise ValueError('FDTD-2D input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def host_source(source):
    parser = section(source, 'static int parseIntParam(', '// FDTD-2D kernels')
    prefix = section(source, '    int N    = parseIntParam(', '    // Print device info')
    bounds = re.findall(r'for \(int i = 0; i < (.+); i\+\+\) h_(?:ex|ey|hz)\[i\] =', source)
    if len(bounds) != 3 or len(set(bounds)) != 1:
        raise ValueError('Expected three equal FDTD-2D initialization bounds')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <cstring>', '#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        parser, 'int main(int argc,char** argv) {', prefix, 'int cells='+bounds[0]+';',
        r'printf("{\"size\":%d,\"tmax\":%d,\"cells\":%d,\"bytes\":%zu}\n",N,TMAX,cells,grid_bytes);', '}'])


def expected(n, steps):
    if n <= 0 or steps <= 0:
        return None, 'FDTD-2D requires positive size and tmax\n'
    if n*n > INT_MAX:
        return None, 'FDTD-2D size*size exceeds INT_MAX\n'
    return dict(size=n, tmax=steps, cells=n*n, bytes=4*n*n), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/polybench_fdtd2d/hip/polybench_fdtd2d.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid cases against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = list(itertools.product(manifest['scaling_param']['values'], controls['tmax']))
    values += [(n,50) for n in (-INT_MAX-1,-1,0,1,2,17,46339,46340,46341,INT_MAX)]
    values += [(17,t) for t in (-INT_MAX-1,-1,0,1,2,200,INT_MAX)]
    cases = [([],dict(size=str(n),tmax=str(t)),expected(n,t)) for n,t in values]
    cases += [([],{},expected(512,50)),
        (['--size','17','--tmax','3'],{},expected(17,3)),
        (['--size','0','--tmax','0'],{},expected(512,50)),
        (['--size','-1','--tmax','-1'],{},expected(512,50)),
        (['--tmax','invalid'],{},expected(512,50)),
        (['--tmax','0','--tmax','7'],{},expected(512,50)),
        (['--tmax','3','--tmax','7'],{},expected(512,3)),
        (['--size','17','--tmax','3'],dict(size='1',tmax='1'),expected(1,1)),
        (['--size','17'],dict(size='0'),expected(0,50)),
        (['--tmax','3'],dict(tmax='0'),expected(512,0)),
        ([],dict(tmax=''),expected(512,0)),
        ([],dict(tmax='invalid'),expected(512,0)),
        ([],dict(size=' 17suffix',tmax=' 3suffix'),expected(17,3))]
    cpp = work/'ranges.cpp'; cpp.write_text(host_source(args.source.read_text())); binary = work/'ranges'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','FDTD_TEST_'))}
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
    print(f'{len(cases)} host cases and {sum(x["kind"]=="actual" for x in results)} actual GPU-hidden rejections passed')


if __name__ == '__main__': main()
