#!/usr/bin/env python3
"""Check code_fetch_latency's actual parameter admission and host reference chain without GPU access."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

BODY_OPS = (16, 64, 256, 1024, 4096)
MAX_TOTAL_OPS = 1 << 20
INT_MAX = (1 << 31) - 1
INCLUDES = '''#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
'''
# Exact stored values; a missing or extra 16-FMA loop iteration; non-finite
# values; the total_ops bound; and the exactly representable seed.
CHAIN_CHECKS = r'''
int main() {
    int checks = 0;
    for (int t = 0; t < DEFAULT_THREADS_PER_BLOCK; ++t, checks += 3) {
        const float want = reference_chain(t, DEFAULT_TOTAL_OPS);
        if (!chain_value_matches(want, want) ||
            chain_value_matches(reference_chain(t, DEFAULT_TOTAL_OPS - 16), want) ||
            chain_value_matches(reference_chain(t, DEFAULT_TOTAL_OPS + 16), want)) {
            fprintf(stderr, "thread %d: wrong chain length accepted or right value rejected\n", t);
            return 1;
        }
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    if (chain_value_matches(nan, nan) || chain_value_matches(inf, inf)) {
        fprintf(stderr, "non-finite value accepted\n");
        return 1;
    }
    checks += 2;
    for (int t = 0; t < MAX_THREADS_PER_BLOCK; t += MAX_THREADS_PER_BLOCK - 1, ++checks) {
        if (!std::isfinite(reference_chain(t, MAX_TOTAL_OPS))) {
            fprintf(stderr, "thread %d: chain not finite at MAX_TOTAL_OPS\n", t);
            return 1;
        }
    }
    for (int t = 0; t < MAX_THREADS_PER_BLOCK; ++t, ++checks) {
        if (static_cast<double>(FETCH_SEED(t)) != 1.0 + t / 1024.0) {
            fprintf(stderr, "thread %d: seed is not exact\n", t);
            return 1;
        }
    }
    printf("%d\n", checks);
}
'''


def production_parts(source):
    """(shared declarations, parameter prefix) taken from the benchmark source."""
    def between(start, end):
        for anchor in (start, end):
            if source.count(anchor) != 1:
                raise ValueError(f'expected one production anchor {anchor!r}')
        return start + source.split(start, 1)[1].split(end, 1)[0]
    constants = re.findall(r'^static const .*;', source, re.M)
    seed = re.findall(r'^#define FETCH_SEED\(thread\) .*$', source, re.M)
    if len(seed) != 1:
        raise ValueError('expected one FETCH_SEED definition')
    helpers = between('static bool parse_int(', '/**\n * Compare one launch')
    prefix = between('    const char* env_val;', '    /* Print device information to stderr. */')
    return INCLUDES + '\n'.join(constants + seed) + '\n' + helpers, prefix


def admission_source(shared, prefix):
    return shared + '''
int main(int argc, char** argv) {
''' + prefix + r'''
    printf("{\"body_ops\":%d,\"total_ops\":%d,\"reps\":%d,\"prime_launches\":%d,"
           "\"num_blocks\":%d,\"threads_per_block\":%d,\"total_threads\":%zu}\n",
           body_ops, total_ops, reps, prime_launches, num_blocks,
           threads_per_block, total_threads);
}
'''


def parse(text, low, high):
    """The benchmark's checked parse: a whole decimal integer in [low, high], else None."""
    if not re.fullmatch(r'\s*[+-]?[0-9]+', text):
        return None
    value = int(text)
    return value if low <= value <= high else None


def expected(body_ops=None, total_ops=None, prime_launches=None, num_blocks=None,
             threads_per_block=None):
    """Expected JSON for the given parameter texts (None = not provided), or the error."""
    body = 256 if body_ops is None else parse(body_ops, 16, 4096)
    if body not in BODY_OPS:
        return 'Invalid body_ops: need 16, 64, 256, 1024 or 4096\n'
    total = 16384 if total_ops is None else parse(total_ops, 1, MAX_TOTAL_OPS)
    if total is None or total % body:
        return f'Invalid total_ops: need a positive multiple of body_ops, at most {MAX_TOTAL_OPS}\n'
    prime = 0 if prime_launches is None else parse(prime_launches, 0, 1)
    if prime is None:
        return 'Invalid prime_launches: need 0 or 1\n'
    blocks = 1 if num_blocks is None else parse(num_blocks, 1, INT_MAX)
    if blocks is None:
        return 'Invalid num_blocks: need a positive integer\n'
    threads = 64 if threads_per_block is None else parse(threads_per_block, 1, 1024)
    if threads is None:
        return 'Invalid threads_per_block: need 1 to 1024\n'
    return dict(body_ops=body, total_ops=total, reps=total // body, prime_launches=prime,
                num_blocks=blocks, threads_per_block=threads, total_threads=blocks * threads)


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
    malformed = ['', 'true', 'nonsense', '1junk', '1 ', '1.0', '0x10', '99999999999999999999']
    settings = [dict(body_ops=str(size), prime_launches=prime)
                for size in declared for prime in ('0', '1')]
    settings += [dict(body_ops=text) for text in ['0', '-16', '8', '17', '32', '8192'] + malformed]
    settings += [dict(total_ops=text) for text in
                 ['0', '-16384', '100', '256', '1048576', '1048832', '2147483392'] + malformed]
    settings += [dict(body_ops='4096', total_ops=text) for text in ('2048', '4096', '6144', '8192')]
    settings += [dict(prime_launches=text) for text in ['-1', '2'] + malformed]
    settings += [dict(num_blocks=text) for text in
                 ['0', '-1', '2', '2147483647', '2147483648'] + malformed]
    settings += [dict(threads_per_block=text) for text in
                 ['0', '-64', '1', '1024', '1025'] + malformed]
    settings += [dict(num_blocks='2147483647', threads_per_block='1024')]
    cases = [([], s, expected(**s)) for s in settings]
    cases += [([], {}, expected()), (['64'], {}, expected(body_ops='64')),
              (['64junk'], {}, expected(body_ops='64junk')),
              (['17'], dict(body_ops='64'), expected(body_ops='64')),
              (['64'], dict(body_ops='17'), expected(body_ops='17'))]
    clean = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='code-fetch-host-') as temp:
        root = Path(temp)
        try:
            shared, prefix = production_parts(args.source.read_text())
        except ValueError as exc:
            raise ValueError(f'{args.source}: {exc}') from exc
        programs = dict(admission=admission_source(shared, prefix), chain=shared + CHAIN_CHECKS)
        for name, text in programs.items():
            (root / f'{name}.cpp').write_text(text)
            subprocess.run([args.compiler, '-std=c++14', '-O1', '-g', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all', str(root / f'{name}.cpp'),
                            '-o', str(root / name)], check=True, timeout=60)
        for argv, overrides, want in cases:
            env = dict(clean, **{'BENCH_PARAM_' + k: v for k, v in overrides.items()})
            p = subprocess.run([str(root / 'admission'), *argv], env=env,
                               capture_output=True, text=True, timeout=30)
            if isinstance(want, str):
                if p.returncode != 1 or p.stderr != want or p.stdout:
                    raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
            elif p.returncode or p.stderr or json.loads(p.stdout) != want:
                raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
        chain = subprocess.run([str(root / 'chain')], env=clean, capture_output=True,
                               text=True, timeout=120)
        if chain.returncode or chain.stderr:
            raise AssertionError((chain.returncode, chain.stdout, chain.stderr))
    print(f'{len(cases)} actual code-fetch-latency admission cases and {int(chain.stdout)} '
          'reference-chain checks pass (no allocation or GPU access)')


if __name__ == '__main__':
    main()
