#!/usr/bin/env python3
"""Check code_fetch_latency's actual parameter admission and host reference chains without GPU access."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

BODY_OPS = tuple(16 << i for i in range(12))  # powers of two, 16 to 32768
OPS = ('fma', 'xor_add')
MAX_CHAIN_OPS = 1 << 20
INT_MAX = (1 << 31) - 1
CHAIN = 16384  # chain length used for the reference-chain checks
INCLUDES = '''#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
'''
# Exact stored values; a missing or extra 16-operation loop pass; non-finite
# values; the chain bound; the exactly representable FMA seed; and integer
# chain values printed for an independent recomputation.
CHAIN_CHECKS = r'''
int main() {
    const int chain = %d;
    int checks = 0;
    for (int t = 0; t < DEFAULT_THREADS_PER_BLOCK; ++t, checks += 5) {
        const float want = reference_chain(t, chain);
        if (!chain_value_matches(want, want) ||
            chain_value_matches(reference_chain(t, chain - 16), want) ||
            chain_value_matches(reference_chain(t, chain + 16), want)) {
            fprintf(stderr, "thread %%d: wrong FMA chain length accepted or right value rejected\n", t);
            return 1;
        }
        const unsigned want_int = reference_int_chain(t, chain);
        if (reference_int_chain(t, chain - 16) == want_int ||
            reference_int_chain(t, chain + 16) == want_int) {
            fprintf(stderr, "thread %%d: integer chain does not tell a missing or extra pass\n", t);
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
        if (!std::isfinite(reference_chain(t, MAX_CHAIN_OPS))) {
            fprintf(stderr, "thread %%d: chain not finite at MAX_CHAIN_OPS\n", t);
            return 1;
        }
    }
    for (int t = 0; t < MAX_THREADS_PER_BLOCK; ++t, ++checks) {
        if (static_cast<double>(FETCH_SEED(t)) != 1.0 + t / 1024.0) {
            fprintf(stderr, "thread %%d: seed is not exact\n", t);
            return 1;
        }
    }
    printf("%%d\n", checks);
    for (int t = 0; t < MAX_THREADS_PER_BLOCK; t += 341) {
        printf("%%d %%u %%u\n", t, reference_int_chain(t, 0), reference_int_chain(t, chain));
    }
}
''' % CHAIN


def production_parts(source):
    """(shared declarations, parameter prefix) taken from the benchmark source."""
    def between(start, end):
        for anchor in (start, end):
            if source.count(anchor) != 1:
                raise ValueError(f'expected one production anchor {anchor!r}')
        return start + source.split(start, 1)[1].split(end, 1)[0]
    constants = re.findall(r'^static const .*;', source, re.M)
    seeds = re.findall(r'^#define FETCH_(?:INT_)?SEED\(thread\) .*$', source, re.M)
    if len(seeds) != 2:
        raise ValueError('expected one FETCH_SEED and one FETCH_INT_SEED definition')
    helpers = between('static bool parse_int(', '/**\n * Compare the launch')
    prefix = between('    const char* env_val;', '    /* Print device information to stderr. */')
    return INCLUDES + '\n'.join(constants + seeds) + '\n' + helpers, prefix


def admission_source(shared, prefix):
    return shared + '''
int main(int argc, char** argv) {
''' + prefix + r'''
    printf("{\"body_ops\":%d,\"loops\":%d,\"chain_ops\":%d,\"op\":\"%s\",\"xor_add\":%s,"
           "\"num_blocks\":%d,\"threads_per_block\":%d,\"total_threads\":%zu}\n",
           body_ops, loops, chain_ops, op, xor_add ? "true" : "false", num_blocks,
           threads_per_block, total_threads);
}
'''


def parse(text, low, high):
    """The benchmark's checked parse: a whole decimal integer in [low, high], else None."""
    if not re.fullmatch(r'\s*[+-]?[0-9]+', text):
        return None
    value = int(text)
    return value if low <= value <= high else None


def expected(body_ops=None, loops=None, op=None, num_blocks=None, threads_per_block=None):
    """Expected JSON for the given parameter texts (None = not provided), or the error."""
    body = 256 if body_ops is None else parse(body_ops, BODY_OPS[0], BODY_OPS[-1])
    if body not in BODY_OPS:
        return f'Invalid body_ops: need a power of two from {BODY_OPS[0]} to {BODY_OPS[-1]}\n'
    passes = 1 if loops is None else parse(loops, 1, MAX_CHAIN_OPS // body)
    if passes is None:
        return f'Invalid loops: need a positive integer with body_ops * loops at most {MAX_CHAIN_OPS}\n'
    kind = 'fma' if op is None else op
    if kind not in OPS:
        return 'Invalid op: need fma or xor_add\n'
    blocks = 1 if num_blocks is None else parse(num_blocks, 1, INT_MAX)
    if blocks is None:
        return 'Invalid num_blocks: need a positive integer\n'
    threads = 64 if threads_per_block is None else parse(threads_per_block, 1, 1024)
    if threads is None:
        return 'Invalid threads_per_block: need 1 to 1024\n'
    return dict(body_ops=body, loops=passes, chain_ops=body * passes, op=kind,
                xor_add=kind == 'xor_add', num_blocks=blocks, threads_per_block=threads,
                total_threads=blocks * threads)


def int_chain(thread, ops):
    """The integer chain recomputed here, independently of the benchmark's helper."""
    x = (thread * 2654435761 + 1) & 0xffffffff
    for _ in range(ops):
        x = ((x ^ 0x9E3779B9) + 0x7F4A7C15) & 0xffffffff
    return x


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
    swept = {p['name']: p['values'] for p in manifest['non_scaling_params']}
    if tuple(swept['op']) != OPS:
        raise AssertionError(f"manifest op {swept['op']} != compiled kinds {OPS}")
    kernels = [f'{prefix}_{size}' for prefix in ('fetch_kernel', 'fetch_xor_add_kernel')
               for size in BODY_OPS]
    if manifest['kernels'] != kernels:
        raise AssertionError('manifest kernels differ from the compiled kernel names')
    malformed = ['', 'true', 'nonsense', '1junk', '1 ', '1.0', '0x10', '99999999999999999999']
    settings = [dict(body_ops=str(size), loops=str(passes), op=op)
                for size in declared for passes in swept['loops'] for op in swept['op']]
    settings += [dict(body_ops=text) for text in
                 ['0', '-16', '8', '15', '17', '24', '48', '6144', '32767', '32769', '65536'] + malformed]
    # Default body_ops is 256: the chain bound admits at most 4096 loops.
    settings += [dict(loops=text) for text in
                 ['0', '-1', '2', '3', '4096', '4097', '1048576', '2147483647'] + malformed]
    settings += [dict(body_ops='16', loops=text) for text in ('65536', '65537')]
    settings += [dict(body_ops='32768', loops=text) for text in ('32', '33')]
    settings += [dict(op=text) for text in ['', 'FMA', 'fma ', 'xor', 'xor_add_', 'add', '1']]
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
    checks, *values = chain.stdout.splitlines()
    if len(values) != 4:
        raise AssertionError(f'expected 4 integer-chain lines, got {values}')
    for line in values:
        thread, seed, value = (int(field) for field in line.split())
        if (seed, value) != (int_chain(thread, 0), int_chain(thread, CHAIN)):
            raise AssertionError(f'integer chain of thread {thread} differs from its recomputation')
    print(f'{len(cases)} actual code-fetch-latency admission cases and {int(checks) + 2 * len(values)} '
          'reference-chain checks pass (no allocation or GPU access)')


if __name__ == '__main__':
    main()
