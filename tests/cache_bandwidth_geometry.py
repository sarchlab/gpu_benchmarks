#!/usr/bin/env python3
"""Exercise production cache-bandwidth geometry/records with simulated limits."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def section(source, start, end):
    for anchor in (start, end):
        if source.count(anchor) != 1:
            raise ValueError(f'expected one source anchor {anchor!r}')
    return start + source.split(start, 1)[1].split(end, 1)[0]


def host_source(source):
    constants = re.findall(r'^static const .*;', source, re.M)
    if len(constants) != 9:
        raise ValueError('expected nine production constants')
    return r'''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
struct float4 {float x,y,z,w;};
struct hipDeviceProp_t {
    int multiProcessorCount, maxGridSize[3], maxThreadsPerBlock;
};
static int hipGetDevice(int* d) {*d=0;return 0;}
static int hipGetDeviceProperties(hipDeviceProp_t* p, int) {
    return sscanf(getenv("BW_TEST_LIMITS"),"%d,%d,%d",
      &p->multiProcessorCount,&p->maxGridSize[0],&p->maxThreadsPerBlock)==3 ? 0 : 2;
}
static int hipMemGetInfo(size_t* f, size_t* t) {
    *f=strtoull(getenv("BW_TEST_FREE"),nullptr,10);*t=*f;return 0;
}
#define HIP_CHECK(call) do {if(call) return 2;} while(0)
''' + '\n'.join(constants) + '\n' + section(source, 'int main(', '    fprintf(stderr, "=== cache_bandwidth') + section(source, '    char params[', '    double total_gpu_ms =') + '\n    printf("{%s}\\n",params);\n}\n'


def expected(ws, mode, bpc, tpb, units, free, grid, threads):
    # Python arbitrary-precision arithmetic is independent of C++ casts/overflow.
    if ws < 16 or ws % 16:
        return 'Invalid working_set_bytes'
    if units <= 0 or grid <= 0 or tpb > threads:
        return 'Unsupported device launch limits'
    if ws // 16 + tpb - 1 > (1 << 32) - 1:
        return 'Working-set traversal exceeds uint32_t'
    cap = free // 2 if mode == 'shared' else min(free // 2, 4 << 30)
    if ws > cap:
        return 'exceeds memory budget' if mode == 'shared' else 'too large for private mode'
    requested = bpc * units
    blocks = requested if mode == 'shared' else min(requested, cap // ws)
    if blocks > grid or blocks * tpb > 1 << 32 or blocks * ws > (1 << 64) - 1:
        return 'Effective launch geometry exceeds supported indices or sizes'
    windows = 1 if mode == 'shared' else blocks
    per_iter = blocks * ws
    iters = max(3, min(4000000, (8 << 30) // per_iter))
    if per_iter * iters >= 1 << 64:
        return 'Launch byte count exceeds uint64_t'
    return dict(working_set_bytes=ws, mode=mode, blocks_per_cu=bpc,
                requested_num_blocks=requested, num_blocks=blocks,
                threads_per_block=tpb, windows=windows, allocation_bytes=windows * ws,
                iters=iters, bytes_per_iter=per_iter, bytes_read=per_iter * iters,
                launch_policy='shared_window_v1' if mode == 'shared' else 'distinct_windows_v1')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'tier1/cache_bandwidth/hip/cache_bandwidth.hip')
    args = ap.parse_args()
    cases = []

    def add(ws=1 << 20, mode='shared', bpc=None, tpb=None, units=114,
            free=16 << 30, grid=(1 << 31)-1, threads=1024, argv=None, overrides=None):
        bpc = bpc if bpc is not None else 8 if mode == 'shared' else 1
        tpb = tpb if tpb is not None else 256 if mode == 'shared' else 1024
        env = dict(working_set_bytes=str(ws), mode=mode, blocks_per_cu=str(bpc), threads_per_block=str(tpb))
        if overrides is not None:
            env = overrides
        cases.append((argv or [], env, units, free, grid, threads,
                      expected(ws, mode, bpc, tpb, units, free, grid, threads)))

    manifest = json.loads((args.source.parent / 'params.json').read_text())
    for mode in ('shared', 'private'):
        for ws in manifest['scaling_param']['values']:
            for units in (1, 114, 120):
                add(ws, mode, units=units)
        for ws in (0, -16, 15, 17, 16, 48, 1008, 1040, 4 << 30, (4 << 30)+16,
                   ((1 << 32)-1024)*16, ((1 << 32)-1024)*16+16, (1 << 63)-16):
            add(ws, mode)
        # Synthetic capacity boundaries, including one/no window and odd free bytes.
        for windows in (0, 1, 2, 113, 114, 115):
            for delta in (-1, 0, 1):
                add(1024, mode, free=max(0, windows*2048+delta))
        for tpb in (64, 65, 255, 1024):
            for bpc in (1, 3, 2147483647):
                add(1040, mode, tpb=tpb, bpc=bpc)
        add(mode=mode, units=0)
        add(mode=mode, grid=0)
        add(mode=mode, threads=32)
        add(mode=mode, grid=1)
    add(overrides={})
    add(48, argv=['48'], overrides={})
    add(1040, argv=['17'], overrides={'working_set_bytes':'1040'})
    add(overrides={'mode':'unknown', 'blocks_per_cu':'-1', 'threads_per_block':'63'})
    add(mode='private', overrides={'mode':'private', 'blocks_per_cu':'0', 'threads_per_block':'1025'})
    # Exact sink-index and traversal-increment boundaries, without allocation.
    for blocks in ((1 << 26), (1 << 26)+1):
        add(16, bpc=blocks, tpb=64, units=1)
    for ws in (((1 << 32)-64)*16, ((1 << 32)-64)*16+16):
        add(ws, bpc=1, tpb=64, units=1, free=(1 << 64)-1)
    add(((1 << 32)-64)*16, bpc=1 << 26, tpb=64, units=1, free=(1 << 64)-1)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'BW_TEST_'))}
    with tempfile.TemporaryDirectory(prefix='cache-bandwidth-geometry-') as tmp:
        cpp = Path(tmp)/'geometry.cpp'; binary = Path(tmp)/'geometry'
        cpp.write_text(host_source(args.source.read_text()))
        subprocess.run([args.compiler, '-std=c++14', '-O1', '-g', '-fsanitize=address,undefined',
                        '-fno-sanitize-recover=all', str(cpp), '-o', str(binary)], check=True, timeout=60)
        for argv, overrides, units, free, grid, threads, want in cases:
            env = dict(clean, BW_TEST_LIMITS=f'{units},{grid},{threads}', BW_TEST_FREE=str(free))
            env.update({'BENCH_PARAM_'+k:v for k,v in overrides.items()})
            p = subprocess.run([str(binary), *argv], env=env, capture_output=True, text=True, timeout=30)
            if isinstance(want, str):
                if p.returncode != 1 or want not in p.stderr or 'runtime error:' in p.stderr or 'Sanitizer' in p.stderr:
                    raise AssertionError((argv, overrides, want, p.returncode, p.stdout, p.stderr))
                if p.stdout and json.loads(p.stdout).get('type') != 'error':
                    raise AssertionError(p.stdout)
            elif p.returncode or p.stderr or json.loads(p.stdout) != want:
                raise AssertionError((argv, overrides, units, free, want, p.returncode, p.stdout, p.stderr))
    print(f'{len(cases)} production geometry/record cases pass under fatal sanitizers (no GPU or allocation)')


if __name__ == '__main__':
    main()
