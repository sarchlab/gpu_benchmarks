#!/usr/bin/env python3
"""Measure and check code_fetch_latency's compiled AMD kernels (no GPU required).

Reads the code objects embedded in a ROCm build of the benchmark, reports each
fetch kernel's code bytes, and checks the layout the benchmark depends on: one
loop whose body holds exactly body_ops FMA instructions and nothing else.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

BODY_OPS = (16, 64, 256, 1024, 4096)
LINE_BYTES = 64
FUNCTION = re.compile(r'^([0-9a-f]+) <(\w+)>:$')
# "\tmnemonic operands   // 000000003628: D1CB0001 040A0204 <symbol+0x28>"
INSTRUCTION = re.compile(r'^\s+(\S+)[^/]*// ([0-9A-F]+):((?: [0-9A-F]{8})+)(?: <\w+\+0x([0-9a-f]+)>)?')


def run(*command):
    return subprocess.run([str(c) for c in command], check=True,
                          capture_output=True, text=True).stdout


def kernel_layout(disassembly, sizes):
    """{body_ops: row} for the fetch kernels found in one code object."""
    rows, row = {}, None
    for line in disassembly.splitlines():
        match = FUNCTION.match(line)
        if match:
            kernel = re.search(r'fetch_kernel_(\d+)', match[2])
            row = None
            if kernel:
                row = rows[int(kernel[1])] = dict(start=int(match[1], 16), bytes=sizes[match[2]],
                                                  fma=[], loops=[], branches=0)
            continue
        match = INSTRUCTION.match(line) if row else None
        if not match:
            continue
        address = int(match[2], 16)
        if address >= row['start'] + row['bytes']:
            continue  # alignment padding after the kernel
        if match[1].startswith('v_fma_f32'):
            row['fma'].append(address)
        elif 'branch' in match[1] and match[4]:
            row['branches'] += 1
            target = row['start'] + int(match[4], 16)
            if target <= address:  # backward branch closes a loop
                row['loops'].append((target, address + 4 * len(match[3].split())))
    return rows


def check(body_ops, row):
    """Problems that make this kernel's timing unusable for code-fetch analysis."""
    problems = []
    if len(row['loops']) != 1:
        return ['%d loops (want 1)' % len(row['loops'])]
    begin, end = row['loops'][0]
    if len(row['fma']) != body_ops:
        problems.append('%d FMA instructions (want %d)' % (len(row['fma']), body_ops))
    if any(not begin <= address < end for address in row['fma']):
        problems.append('FMA instructions outside the loop body')
    if row['branches'] != 2:  # zero-trip guard plus the loop's backward branch
        problems.append('%d branches (want 2)' % row['branches'])
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1] /
                        'tier1/code_fetch_latency/hip/code_fetch_latency')
    parser.add_argument('--llvm-bin', type=Path, default=Path('/opt/rocm/llvm/bin'))
    args = parser.parse_args()
    tool = lambda name: args.llvm_bin / name
    failed = False
    with tempfile.TemporaryDirectory(prefix='code-fetch-size-') as temp:
        fatbin, code = Path(temp) / 'fatbin', Path(temp) / 'code.co'
        run(tool('llvm-objcopy'), '--dump-section', '.hip_fatbin=%s' % fatbin, args.binary)
        targets = [t for t in run(tool('clang-offload-bundler'), '--list', '--type=o',
                                  '--input=%s' % fatbin).split() if 'amdgcn' in t]
        if not targets:
            raise SystemExit('no AMD code object embedded in %s' % args.binary)
        for target in targets:
            run(tool('clang-offload-bundler'), '--unbundle', '--type=o', '--targets=' + target,
                '--input=%s' % fatbin, '--output=%s' % code)
            sizes = {f[7]: int(f[2]) for f in (line.split() for line in run(
                tool('llvm-readelf'), '--symbols', '--wide', code).splitlines())
                if len(f) == 8 and f[3] == 'FUNC'}
            rows = kernel_layout(run(tool('llvm-objdump'), '-d',
                                     '--mcpu=' + target.rsplit('-', 1)[1], code), sizes)
            print('%s\n| kernel | code bytes | %d-byte lines | FMA instructions | loop body bytes |'
                  '\n|---|---:|---:|---:|---:|' % (target, LINE_BYTES))
            for body_ops in BODY_OPS:
                row = rows.get(body_ops)
                problems = check(body_ops, row) if row else ['kernel missing']
                if row and not problems:
                    begin, end = row['loops'][0]
                    first, last = row['start'] // LINE_BYTES, (row['start'] + row['bytes'] - 1) // LINE_BYTES
                    print('| `fetch_kernel_%d` | %d | %d | %d | %d |' % (
                        body_ops, row['bytes'], last - first + 1, len(row['fma']), end - begin))
                else:
                    failed = True
                    print('FAIL: fetch_kernel_%d: %s' % (body_ops, '; '.join(problems)))
    if failed:
        raise SystemExit('compiled kernels do not have the layout this benchmark measures')
    print('PASS: one loop with exactly body_ops FMA instructions in every fetch kernel')


if __name__ == '__main__':
    main()
