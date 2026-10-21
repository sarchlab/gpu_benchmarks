#!/usr/bin/env python3
"""Exercise the actual NPB EP input prefix without allocation or GPU access."""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

INT_MAX = (1 << 31) - 1


def between(source, start, end):
    for anchor in (start, end):
        if source.count(anchor) != 1:
            raise ValueError(f'expected one production anchor {anchor!r}')
    return start + source.split(start, 1)[1].split(end, 1)[0]


def host_source(source):
    constants = re.findall(r'^#define (?:MK|NQ)\s+.*$', source, re.M)
    if len(constants) != 2:
        raise ValueError(f'expected MK and NQ definitions, found {len(constants)}')
    parser = between(source, 'static int parseIntParam(', 'static const char* class_of(')
    prefix = between(source, '    // Default M=28 => Class A.', '    // an = A^(2^MK):')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
''' + '\n'.join(constants) + '\n' + parser + '''
int main(int argc, char** argv) {
''' + prefix + r'''
    printf("{\"m\":%d,\"block_size\":%d,\"nn\":%lld,\"grid\":%d}\n",
           m, block_size, nn, blocks_per_grid);
}
'''


def expected(m, block):
    if m <= 16:
        return 'M must be > 16 (M-MK = number-of-groups exponent must be >= 1)\n'
    if m > 46:
        return 'M produces a thread count that cannot fit int\n'
    if block <= 0:
        return 'block_size must be positive\n'
    nn = 2 ** (m - 16)
    grid = (nn + block - 1) // block
    if grid * 10 - 1 > INT_MAX:
        return 'Per-block histogram index cannot fit int\n'
    assert grid * block - 1 <= INT_MAX
    return dict(m=m, block_size=block, nn=nn, grid=grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] /
                        'tier2/npb_ep/hip/npb_ep.hip')
    args = parser.parse_args()
    pairs = [(m, 256) for m in [-INT_MAX-1, -1, 0, 1, 16, 17, 23, 33, 36, 40, 46,
                               47, 48, 78, 79, 80, INT_MAX]]
    pairs += [(28, b) for b in [-INT_MAX-1, -1, 0, 1, 3, 31, 257, 1024, INT_MAX]]
    pairs += [(43, 1), (44, 1), (44, 2), (45, 2), (45, 3), (46, 5), (46, 6),
              (46, INT_MAX), (17, 1), (17, 3), (17, INT_MAX)]
    manifest = json.loads((args.source.parent / 'params.json').read_text())
    blocks = next(p['values'] for p in manifest['non_scaling_params'] if p['name'] == 'block_size')
    pairs += list(itertools.product(manifest['scaling_param']['values'], blocks))
    cases = [([], dict(m=str(m), block_size=str(b)), expected(m, b)) for m, b in pairs]
    cases += [([], {}, expected(28, 256)), (['--m', '24'], {}, expected(24, 256)),
              (['--m', '0'], {}, expected(28, 256)), (['--m', '-1'], {}, expected(28, 256)),
              (['--m', '24'], dict(m='25'), expected(25, 256)),
              (['--m', '24'], dict(m='0'), expected(0, 256)),
              (['--m', '47'], dict(m='24'), expected(24, 256)),
              (['--m', '24', '--m', '25'], {}, expected(24, 256)),
              (['--block_size', '0'], {}, expected(28, 256))]
    clean = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='npb-ep-ranges-') as temp:
        root = Path(temp)
        try:
            (root / 'ranges.cpp').write_text(host_source(args.source.read_text()))
        except ValueError as exc:
            raise ValueError(f'{args.source}: {exc}') from exc
        subprocess.run([args.compiler, '-std=c++14', '-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                        str(root / 'ranges.cpp'), '-o', str(root / 'ranges')], check=True, timeout=60)
        for argv, overrides, want in cases:
            env = dict(clean, **{'BENCH_PARAM_' + k: v for k, v in overrides.items()})
            p = subprocess.run([str(root / 'ranges'), *argv], env=env,
                               capture_output=True, text=True, timeout=30)
            if isinstance(want, str):
                if p.returncode != 1 or p.stderr != want or p.stdout:
                    raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
            elif p.returncode or p.stderr or json.loads(p.stdout) != want:
                raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
    print(f'{len(cases)} actual NPB EP input/geometry cases pass (no allocation or GPU access)')


if __name__ == '__main__':
    main()
