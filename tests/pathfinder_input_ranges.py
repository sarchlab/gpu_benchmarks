#!/usr/bin/env python3
"""Exercise Pathfinder's actual input and index arithmetic without allocations or GPU access."""
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
    constants = re.findall(r'^#define (?:BLOCK_SIZE|HALO)\s+.*$', source, re.M)
    if len(constants) != 2:
        raise ValueError('expected BLOCK_SIZE and HALO definitions')
    parser = between(source, 'static int parseIntParam(',
                     '// ---------------------------------------------------------------------------\n// Constants and kernel')
    prefix = between(source, '    int cols           = parseIntParam(', '    int device_id = 0;')
    cells = between(source, '    long long total_cells =', '    // Preflight BOTH allocations')
    buffers = between(source, '    long long row_bytes_ll  =', '    if (dev_needed + warmer_reserve >')
    coordinates = between(source, '        // calculate the small block size', '        int W = tx-1;')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
''' + '\n'.join(constants) + '\n' + parser + '''
static int last_column(int bx, int iteration, int cols, int border) {
    int tx = BLOCK_SIZE - 1;
''' + coordinates + '''
    return xidx;
}
int main(int argc, char** argv) {
''' + prefix + cells + buffers + r'''
    int maximum_column = -1;
    if (rows > 1) {
        const int first = std::min(rows - 1, pyramid_height);
        const int last = (rows - 2) % pyramid_height + 1;
        maximum_column = std::max(last_column(blockCols - 1, first, cols, borderCols),
                                  last_column(blockCols - 1, last, cols, borderCols));
    }
    printf("{\"cols\":%d,\"rows\":%d,\"pyramid_height\":%d,\"blocks\":%d,"
           "\"cells\":%lld,\"grid_bytes\":%lld,\"device_bytes\":%lld,\"maximum_column\":%d}\n",
           cols, rows, pyramid_height, blockCols, total_cells, grid_bytes, dev_needed, maximum_column);
}
'''


def expected(cols, rows, height):
    if cols <= 0 or rows <= 0 or height <= 0:
        return 'cols, rows and pyramid_height must be positive\n'
    if height >= 128:
        return 'pyramid_height too large for BLOCK_SIZE=256 (need pyramid_height < 128)\n'
    if 9 * rows > INT_MAX:
        return 'Accumulated path costs may exceed INT_MAX\n'
    blocks = (cols + 255 - 2 * height) // (256 - 2 * height)
    maximum_column = -1
    if rows > 1:
        full_launches, tail = divmod(rows - 1, height)
        iterations = ([height] if full_launches else []) + ([tail] if tail else [])
        # Python integers independently describe the furthest lane of each launch kind.
        maximum_column = max((blocks - 1) * (256 - 2 * count) - height + 255
                             for count in iterations)
        if maximum_column > INT_MAX:
            return 'Pathfinder padded column indices exceed INT_MAX\n'
    return dict(cols=cols, rows=rows, pyramid_height=height, blocks=blocks,
                cells=rows * cols, grid_bytes=4 * rows * cols,
                device_bytes=4 * cols * (rows + 1), maximum_column=maximum_column)


def cases_for(manifest):
    axes = {p['name']: p['values'] for p in manifest['non_scaling_params']}
    triples = list(itertools.product(manifest['scaling_param']['values'], axes['rows'], axes['pyramid_height']))
    triples += [(c, 100, 20) for c in (-INT_MAX-1, -1, 0, 1, 2, 127, 128, 255, 256, 257, INT_MAX)]
    triples += [(1000, r, 20) for r in (-INT_MAX-1, -1, 0, 1, 2, 3, 20, 21, INT_MAX//9, INT_MAX//9+1, INT_MAX)]
    triples += [(1000, 100, p) for p in (-INT_MAX-1, -1, 0, 1, 2, 4, 20, 40, 127, 128, 129, INT_MAX//2, INT_MAX//2+1, INT_MAX)]
    triples += [(INT_MAX, 1, p) for p in (1, 20, 127)]
    for p in (1, 2, 4, 20, 40, 127):
        for r in (2, 3, 100, 500, 1000, 10000, INT_MAX//9):
            last = (r - 2) % p + 1
            last_safe_block = (INT_MAX + p - 255) // (256 - 2 * last)
            boundary = min(INT_MAX, (last_safe_block + 1) * (256 - 2 * p))
            triples += [(c, r, p) for c in (boundary - 1, boundary, boundary + 1) if 0 < c <= INT_MAX]
    cases = [([], dict(cols=str(c), rows=str(r), pyramid_height=str(p)), expected(c, r, p))
             for c, r, p in dict.fromkeys(triples)]
    cases += [([], {}, expected(100000, 100, 20)),
              (['--cols', '33', '--rows', '3', '--pyramid_height', '2'], {}, expected(33, 3, 2)),
              (['--cols', '0'], {}, expected(100000, 100, 20)),
              (['--rows', '-1'], {}, expected(100000, 100, 20)),
              (['--pyramid_height', '0'], {}, expected(100000, 100, 20)),
              (['--pyramid_height', '128'], {}, expected(100000, 100, 128)),
              (['--rows'], {}, expected(100000, 100, 20)),
              (['--cols', '33', '--cols', '65'], {}, expected(33, 100, 20)),
              (['--cols', '0', '--cols', '65'], {}, expected(100000, 100, 20)),
              (['--cols', '33'], dict(cols='65'), expected(65, 100, 20)),
              (['--rows', '3'], dict(rows='0'), expected(100000, 0, 20)),
              (['--pyramid_height', '128'], dict(pyramid_height='2'), expected(100000, 100, 2)),
              ([], dict(cols=''), expected(0, 100, 20)),
              ([], dict(cols='33x', rows='3', pyramid_height='0002'), expected(33, 3, 2))]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] /
                        'tier2/rodinia_pathfinder/rodinia_pathfinder.hip')
    args = parser.parse_args()
    cases = cases_for(json.loads((args.source.parent / 'params.json').read_text()))
    clean = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='pathfinder-ranges-') as temp:
        root = Path(temp)
        try:
            (root / 'ranges.cpp').write_text(host_source(args.source.read_text()))
        except ValueError as exc:
            raise ValueError(f'{args.source}: {exc}') from exc
        subprocess.run([args.compiler, '-std=c++14', '-O1', '-g', '-fsanitize=address,undefined',
                        '-fno-sanitize-recover=all', str(root / 'ranges.cpp'), '-o', str(root / 'ranges')],
                       check=True, timeout=60)
        for argv, overrides, want in cases:
            env = dict(clean, **{'BENCH_PARAM_' + k: v for k, v in overrides.items()})
            p = subprocess.run([str(root / 'ranges'), *argv], env=env,
                               capture_output=True, text=True, timeout=30)
            if isinstance(want, str):
                if p.returncode != 1 or p.stderr != want or p.stdout:
                    raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
            elif p.returncode or p.stderr or json.loads(p.stdout) != want:
                raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
    print(f'{len(cases)} actual Pathfinder range/index cases pass (no allocation or GPU access)')


if __name__ == '__main__':
    main()
