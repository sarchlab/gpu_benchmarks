#!/usr/bin/env python3
"""Check actual LBM integer input guards, array bounds and launch arithmetic."""
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
        raise ValueError('LBM input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def one(source, pattern):
    matches = re.findall(pattern, source, re.M)
    if len(matches) != 1: raise ValueError('LBM input anchor missing or duplicated: '+pattern)
    return matches[0]


def host_source(source):
    prefix = section(source, '    int grid_dim     = 64;', '    int device_id = 0;')
    record = section(source, '    char params[160];', '    // Per-pass accumulator:')
    cells = one(source, r'std::vector<float> h_f\((.+)\);')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <climits>',
        'static_assert(sizeof(int)==4 && sizeof(size_t)>=8,"Supported 64-bit host required");',
        one(source, r'^#define Q .+$'), 'int main(int argc,char** argv) {',
        '(void)argc; (void)argv;', prefix, '(void)omega;',
        one(source, r'^    size_t f_bytes = .+;$'),
        one(source, r'^    int gridSize = .+;$'), 'int entries='+cells+';', record,
        r'printf("{\"params\":{%s},\"nodes\":%d,\"entries\":%d,\"bytes\":%zu,\"blocks\":%d}\n",',
        'params,N,entries,f_bytes,gridSize);', '}'])


def expected(n, steps, block=128, tau=.7):
    if n < 3 or steps <= 0 or block <= 0:
        return None, 'LBM requires grid_dim>=3 and positive num_timesteps/block_size\n'
    if 19*n*n*n > INT_MAX:
        return None, 'LBM 19*grid_dim^3 exceeds INT_MAX\n'
    return dict(params=dict(grid_dim=n, num_timesteps=steps, block_size=block, tau=tau),
                nodes=n**3, entries=19*n**3, bytes=4*19*n**3,
                blocks=(n**3+block-1)//block), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/parboil_lbm/parboil_lbm.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid cases against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = list(itertools.product(manifest['scaling_param']['values'], controls['num_timesteps'], controls['block_size'], controls['tau']))
    values += [(n,1,128,.7) for n in (-INT_MAX-1,-1,0,1,2,3,17,482,483,484,1291,50000,INT_MAX)]
    values += [(17,t,128,.7) for t in (-INT_MAX-1,-1,0,1,2,500,INT_MAX)]
    values += [(17,2,b,.7) for b in (-INT_MAX-1,-1,0,1,2,3,7,128,1024,INT_MAX)]
    values += [(483,1,INT_MAX,.7)]
    cases = [([],dict(grid_dim=str(n),num_timesteps=str(t),block_size=str(b),tau=str(tau)),expected(n,t,b,tau)) for n,t,b,tau in values]
    cases += [([],{},expected(64,100)),
        (['--grid_dim','17','--num_timesteps','2','--block_size','7','--tau','1'],{},expected(64,100)),
        (['--num_timesteps','0'],{},expected(64,100)),
        (['--grid_dim','0'],dict(grid_dim='17',num_timesteps='2',block_size='7',tau='1.0'),expected(17,2,7,1.0)),
        ([],dict(num_timesteps=''),expected(64,0)),
        ([],dict(num_timesteps='invalid'),expected(64,0)),
        ([],dict(grid_dim=''),expected(0,100)),
        ([],dict(block_size='invalid'),expected(64,100,0)),
        ([],dict(grid_dim=' 17suffix',num_timesteps=' 2suffix',block_size=' 7suffix',tau=' 1.0suffix'),expected(17,2,7,1.0))]
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
    print(f'{len(cases)} host cases and {sum(x["kind"]=="actual" for x in results)} actual GPU-hidden rejections passed')


if __name__ == '__main__': main()
