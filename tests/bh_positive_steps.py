#!/usr/bin/env python3
"""Check Barnes-Hut's actual timestep admission and existing input precedence."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess


def section(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Barnes-Hut input anchors missing or duplicated')
    return start + source.split(start)[1].split(end)[0]


def host_source(source):
    parser = section(source, 'static int parseIntParam(', 'static uint32_t xorshift32(')
    prefix = section(source, '    int bodies      = 32768;', '    // Initialize bodies')
    record = section(source, '    char params[192];', '    // Per-kernel accumulators:')
    return '\n'.join(['#include <cstdio>', '#include <cstdlib>', '#include <cstring>',
        parser, 'int main(int argc,char** argv) {', prefix,
        '(void)N; (void)timesteps; (void)theta; (void)dt; (void)eps2;', record,
        r'printf("{%s}\n",params);', '}'])


def expected(steps, bodies=32768, theta=0.5):
    return None if steps <= 0 else dict(bodies=bodies, timesteps=steps, theta=theta)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/lonestar_bh/lonestar_bh.hip')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--binary', type=Path, help='Check invalid requests against the actual GPU-hidden binary')
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.source.with_name('params.json').read_text())
    controls = {p['name']:p['values'] for p in manifest['non_scaling_params']}
    values = itertools.product(manifest['scaling_param']['values'], controls['timesteps'], controls['theta'])
    cases = [([],dict(bodies=str(n),timesteps=str(t),theta=str(theta)),expected(t,n,theta)) for n,t,theta in values]
    cases += [([],dict(timesteps=str(t)),expected(t)) for t in (-2147483648,-1,0,1,2,20,2147483647)]
    cases += [([],{},expected(10)),
        (['--bodies','257','--timesteps','2','--theta','1'],{},expected(2,257,1.0)),
        (['--timesteps','0'],{},expected(10)),
        (['--timesteps','-1'],{},expected(10)),
        (['--timesteps','invalid'],{},expected(10)),
        (['--timesteps','0','--timesteps','7'],{},expected(10)),
        (['--timesteps','3','--timesteps','7'],{},expected(3)),
        (['--bodies','257','--timesteps','3','--theta','1'],dict(bodies='256',timesteps='1',theta='.3'),expected(1,256,.3)),
        (['--timesteps','3'],dict(timesteps='0'),None),
        (['--timesteps','3'],dict(timesteps='-1'),None),
        ([],dict(timesteps=''),None),
        ([],dict(timesteps='invalid'),None),
        ([],dict(timesteps=' 2suffix'),expected(2))]
    cpp = work/'steps.cpp'; cpp.write_text(host_source(args.source.read_text())); binary = work/'steps'
    subprocess.run([args.compiler,'-O1','-g','-std=c++14','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','BH_TEST_'))}
    targets = [('host',binary)]
    if args.binary: targets.append(('actual',args.binary.resolve()))
    results = []
    for kind,target in targets:
        for argv,overrides,want in cases:
            invalid = want is None
            if kind == 'actual' and not invalid: continue
            env = dict(clean,**{'BENCH_PARAM_'+k:v for k,v in overrides.items()})
            if kind == 'actual': env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
            p = subprocess.run([str(target),*argv],env=env,capture_output=True,text=True,timeout=30)
            label = f'{len(results):03d}-{kind}'
            (work/(label+'.stdout')).write_text(p.stdout); (work/(label+'.stderr')).write_text(p.stderr)
            passed = p.returncode == (1 if invalid else 0)
            passed &= p.stderr == ('Barnes-Hut requires positive timesteps\n' if invalid else '')
            if invalid: passed &= p.stdout == ''
            else:
                try: passed &= json.loads(p.stdout) == want
                except json.JSONDecodeError: passed = False
            results.append(dict(label=label,kind=kind,argv=argv,overrides=overrides,expected=want,exit=p.returncode,passed=passed))
            (work/'report.json').write_text(json.dumps(results,indent=2)+'\n')
            if not passed: raise AssertionError(results[-1],p.stdout,p.stderr)
    print(f'{len(cases)} host cases and {sum(x["kind"]=="actual" for x in results)} actual GPU-hidden rejections passed')


if __name__ == '__main__': main()
