#!/usr/bin/env python3
"""Check code_fetch_latency's actual parameter admission without GPU access."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

BODY_OPS = (16, 64, 256, 1024, 4096)


def host_source(source):
    constants = re.findall(r'^static const .*;', source, re.M)
    start = '    const char* env_val;'
    end = '    /* Print device information to stderr. */'
    for anchor in (start, end):
        if source.count(anchor) != 1:
            raise ValueError(f'expected one production anchor {anchor!r}')
    prefix = start + source.split(start, 1)[1].split(end, 1)[0]
    return '''#include <cstdio>
#include <cstdlib>
#include <limits>
''' + '\n'.join(constants) + '''
int main(int argc, char** argv) {
''' + prefix + r'''
    printf("{\"body_ops\":%d,\"total_ops\":%d,\"reps\":%d,\"prime_launches\":%d,"
           "\"num_blocks\":%d,\"threads_per_block\":%d,\"total_threads\":%zu}\n",
           body_ops, total_ops, reps, prime_launches, num_blocks,
           threads_per_block, total_threads);
}
'''


def expected(body_ops=256, total_ops=16384, prime_launches=0, num_blocks=1,
             threads_per_block=64):
    if body_ops not in BODY_OPS:
        return 'Invalid body_ops: need 16, 64, 256, 1024 or 4096\n'
    if total_ops <= 0 or total_ops % body_ops:
        return 'Invalid total_ops: need a positive multiple of body_ops\n'
    if prime_launches not in (0, 1):
        return 'Invalid prime_launches: need 0 or 1\n'
    if num_blocks <= 0 or not 0 < threads_per_block <= 1024:
        return 'Invalid or overflowing launch dimensions\n'
    return dict(body_ops=body_ops, total_ops=total_ops, reps=total_ops // body_ops,
                prime_launches=prime_launches, num_blocks=num_blocks,
                threads_per_block=threads_per_block,
                total_threads=num_blocks * threads_per_block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] /
                        'tier1/code_fetch_latency/hip/code_fetch_latency.hip')
    args = parser.parse_args()
    manifest = json.loads((args.source.parent / 'params.json').read_text())
    declared = manifest['scaling_param']['values']
    if tuple(declared) != BODY_OPS:
        raise AssertionError(f'manifest body_ops {declared} != compiled kernels {BODY_OPS}')
    settings = [dict(body_ops=size, prime_launches=prime)
                for size in declared for prime in (0, 1)]
    settings += [dict(body_ops=size) for size in (0, -16, 8, 17, 32, 8192)]
    settings += [dict(total_ops=count) for count in (0, -16384, 100, 256, 2147483392)]
    settings += [dict(body_ops=4096, total_ops=count) for count in (2048, 4096, 6144, 8192)]
    settings += [dict(prime_launches=count) for count in (-1, 2)]
    settings += [dict(num_blocks=count) for count in (0, -1, 2, 2147483647)]
    settings += [dict(threads_per_block=count) for count in (0, -64, 1, 1024, 1025)]
    settings += [dict(num_blocks=2147483647, threads_per_block=1024)]
    cases = [([], {k: str(v) for k, v in s.items()}, expected(**s)) for s in settings]
    cases += [([], {}, expected()), (['64'], {}, expected(body_ops=64)),
              (['17'], dict(body_ops='64'), expected(body_ops=64)),
              (['64'], dict(body_ops='17'), expected(body_ops=17)),
              ([], dict(body_ops='nonsense'), expected(body_ops=0)),
              ([], dict(total_ops=''), expected(total_ops=0))]
    clean = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='code-fetch-ranges-') as temp:
        root = Path(temp)
        try:
            (root / 'ranges.cpp').write_text(host_source(args.source.read_text()))
        except ValueError as exc:
            raise ValueError(f'{args.source}: {exc}') from exc
        subprocess.run([args.compiler, '-std=c++14', '-O1', '-g', '-fsanitize=address,undefined',
                        '-fno-sanitize-recover=all', str(root / 'ranges.cpp'),
                        '-o', str(root / 'ranges')], check=True, timeout=60)
        for argv, overrides, want in cases:
            env = dict(clean, **{'BENCH_PARAM_' + k: v for k, v in overrides.items()})
            p = subprocess.run([str(root / 'ranges'), *argv], env=env,
                               capture_output=True, text=True, timeout=30)
            if isinstance(want, str):
                if p.returncode != 1 or p.stderr != want or p.stdout:
                    raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
            elif p.returncode or p.stderr or json.loads(p.stdout) != want:
                raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
    print(f'{len(cases)} actual code-fetch-latency admission cases pass (no allocation or GPU access)')


if __name__ == '__main__':
    main()
