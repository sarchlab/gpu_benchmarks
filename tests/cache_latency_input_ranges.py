#!/usr/bin/env python3
"""Check the actual cache-latency layout and access guards without allocation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

UINT32_MAX = (1 << 32) - 1


def host_source(source):
    constants = re.findall(r'^static const .*;', source, re.M)
    if len(constants) != 7:
        raise ValueError(f'expected seven production constants, found {len(constants)}')
    start = '    const char* env_val;'
    end = '    /* Print device information to stderr. */'
    for anchor in (start, end):
        if source.count(anchor) != 1:
            raise ValueError(f'expected one production anchor {anchor!r}')
    prefix = start + source.split(start, 1)[1].split(end, 1)[0]
    return '''#include <cstdio>
#include <cstdlib>
#include <cstdint>
''' + '\n'.join(constants) + '''
int main(int argc, char** argv) {
''' + prefix + r'''
    printf("{\"array_bytes\":%zu,\"cacheline_bytes\":%u,\"nodes\":%u,\"stride\":%u,\"timed\":%llu,\"primer\":%llu,\"laps\":%u,\"seed\":%u}\n",
           array_bytes, cacheline_bytes, M, stride,
           (unsigned long long)num_timed, (unsigned long long)num_warmup,
           measure_laps, rng_seed);
}
'''


def expected(size, spacing=64, laps=4, seed=42, accesses=None):
    spacing, laps, seed = (v & UINT32_MAX for v in (spacing, laps, seed))
    if size <= 0:
        return 'Invalid array_bytes'
    if size % 4 or spacing % 4:
        return 'must be word-aligned'
    stride = spacing // 4
    if not stride or size // 4 < stride * 2:
        return 'too small for cacheline_bytes'
    nodes = size // spacing
    if nodes > UINT32_MAX or (nodes - 1) * stride > UINT32_MAX:
        return 'Chain node count or index exceeds uint32_t'
    timed = max(131072, min(16000000, laps * nodes)) if accesses is None else accesses
    if timed == 0:
        return 'num_accesses must be positive'
    return dict(array_bytes=size, cacheline_bytes=spacing, nodes=nodes, stride=stride,
                timed=timed, primer=max(1, min(nodes, (256 << 20) // spacing)),
                laps=laps, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] /
                        'tier1/cache_latency/cache_latency.hip')
    args = parser.parse_args()
    pairs = [(0, 64), (-1, 64), (1, 64), (127, 64), (129, 64), (130, 64), (131, 64),
             (128, 64), (132, 64), (8, 4), (256, 12), (4096, 0), (4096, 3),
             (4096, 5), (4096, 6), (4096, -4), (128, 128),
             (UINT32_MAX * 4, 4), ((UINT32_MAX + 1) * 4, 4),
             (2147483648 * 8, 8), (2147483649 * 8, 8),
             (1431655766 * 12, 12), (1431655767 * 12, 12),
             ((1 << 63) - 4, 64), (2 * 4294967292, -4)]
    manifest = json.loads((args.source.parent / 'params.json').read_text())
    pairs += [(size, 64) for size in manifest['scaling_param']['values']]
    cases = [([], dict(array_bytes=str(size), cacheline_bytes=str(spacing)),
              expected(size, spacing)) for size, spacing in pairs]
    cases += [([], {}, expected(2 << 20)), (['132'], {}, expected(132)),
              (['129'], dict(array_bytes='132'), expected(132)),
              (['132'], dict(array_bytes='129'), expected(129)),
              ([], dict(measure_laps='0'), expected(2 << 20, laps=0)),
              ([], dict(measure_laps='-1', rng_seed='-1'), expected(2 << 20, laps=-1, seed=-1))]
    for text, number in [('0', 0), ('', 0), ('nonsense', 0), ('1', 1), ('7', 7),
                         ('16000001', 16000001), ('-1', (1 << 64) - 1)]:
        cases.append(([], dict(num_accesses=text), expected(2 << 20, accesses=number)))
    clean = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='cache-latency-ranges-') as temp:
        root = Path(temp)
        try:
            (root / 'ranges.cpp').write_text(host_source(args.source.read_text()))
        except ValueError as exc:
            raise ValueError(f'{args.source}: {exc}') from exc
        subprocess.run([args.compiler, '-std=c++14', '-O1', '-g', '-fsanitize=address,undefined',
                        str(root / 'ranges.cpp'), '-o', str(root / 'ranges')], check=True, timeout=60)
        for argv, overrides, want in cases:
            env = dict(clean, **{'BENCH_PARAM_' + k: v for k, v in overrides.items()})
            p = subprocess.run([str(root / 'ranges'), *argv], env=env,
                               capture_output=True, text=True, timeout=30)
            if isinstance(want, str):
                if p.returncode != 1 or want not in p.stderr or p.stdout:
                    raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
            elif p.returncode or json.loads(p.stdout) != want:
                raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
    print(f'{len(cases)} actual cache-latency layout/access cases pass (no allocation or GPU access)')


if __name__ == '__main__':
    main()
