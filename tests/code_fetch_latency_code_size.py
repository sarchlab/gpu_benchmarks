#!/usr/bin/env python3
"""Measure and check code_fetch_latency's compiled AMD kernels (no GPU required).

Reads the code objects embedded in a ROCm build of the benchmark, reports each
fetch kernel's code bytes, and checks the layout the benchmark depends on: one
loop whose body holds exactly body_ops FMA instructions, its loop control and at
most one loop-invariant register move.
--self-test checks the layout checker itself and needs no ROCm tools.
"""
import argparse
import collections
from pathlib import Path
import re
import subprocess
import tempfile

BODY_OPS = tuple(16 << i for i in range(10))  # powers of two, 16 to 8192
LINE_BYTES = 64
FMA = 'v_fma_f32'
# The loop control the measured loop must hold, once each: counter decrement,
# compare and backward branch, as the gfx942 build emits them. A compiler that
# emits another loop-control sequence fails the check until its disassembly has
# been inspected and this list updated.
LOOP_CONTROL = ('s_add_i32', 's_cmp_eq_u32', 's_cbranch_scc0')
# AMD clang 24 (ROCm 10.2) leaves the move of the addend into a vector register
# at the head of the loop instead of hoisting it (clang 22 hoists it). At most
# one such move is accepted; it adds 4 bytes to the loop and no FMA.
LOOP_MOVE = 'v_mov_b32_e32'
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
                                                  instructions=[], loops=[], branches=0)
            continue
        match = INSTRUCTION.match(line) if row else None
        if not match:
            continue
        address, width = int(match[2], 16), 4 * len(match[3].split())
        if address >= row['start'] + row['bytes']:
            continue  # alignment padding after the kernel
        row['instructions'].append((address, match[1], width))
        if 'branch' in match[1] and match[4]:
            row['branches'] += 1
            target = row['start'] + int(match[4], 16)
            if target <= address:  # backward branch closes a loop
                row['loops'].append((target, address + width))
    return rows


def check(body_ops, row):
    """Problems that make this kernel's timing unusable for code-fetch analysis."""
    if len(row['loops']) != 1:
        return ['%d loops (want 1)' % len(row['loops'])]
    begin, end = row['loops'][0]
    loop = sorted(i for i in row['instructions'] if begin <= i[0] < end)
    inside = collections.Counter(name for address, name, width in loop)
    fma = sum(name == FMA for address, name, width in row['instructions'])
    problems = []
    # Every byte of the loop must belong to exactly one decoded instruction, so
    # nothing the parser skipped or the disassembly elided can hide in it.
    expected = begin
    for address, name, width in loop:
        if address != expected:
            problems.append('decoded instructions %s at 0x%x in the loop (0x%x..0x%x)' % (
                'leave a gap' if address > expected else 'overlap', expected, begin, end))
            break
        expected = address + width
    else:
        if expected != end:
            problems.append('decoded instructions leave a gap at 0x%x in the loop (0x%x..0x%x)'
                            % (expected, begin, end))
    if fma != body_ops:
        problems.append('%d FMA instructions (want %d)' % (fma, body_ops))
    if inside[FMA] != fma:
        problems.append('FMA instructions outside the loop body')
    other = inside - collections.Counter({FMA: inside[FMA]})
    control = other - collections.Counter({LOOP_MOVE: other[LOOP_MOVE]})
    if control != collections.Counter(LOOP_CONTROL) or other[LOOP_MOVE] > 1:
        problems.append('loop body holds %s besides FMAs (want one each of %s, and at most one %s)' % (
            ', '.join('%d %s' % (n, name) for name, n in sorted(other.items())) or 'nothing',
            ', '.join(LOOP_CONTROL), LOOP_MOVE))
    if row['branches'] != 2:  # zero-trip guard plus the loop's backward branch
        problems.append('%d branches (want 2)' % row['branches'])
    return problems


def synthetic_kernel(body_ops, in_loop=(), hidden=(), loop_head=()):
    """(disassembly, sizes) shaped like the gfx942 build, with extra in-loop instructions.

    `loop_head` instructions are the first of the loop (the backward branch
    targets them); `in_loop` ones sit before the loop control. Instructions
    whose text is in `hidden` keep their bytes but get no disassembly line, as
    when a line is elided or cannot be parsed.
    """
    symbol, start = '_Z15fetch_kernel_%dPfffi' % body_ops, 0x3600
    fma = (FMA + ' v1, s4, v1, v2', 2)
    loop = list(loop_head) + [fma] * (body_ops - 1) + list(in_loop) + [
        ('s_add_i32 s6, s6, -1', 1), ('s_cmp_eq_u32 s6, 0', 1), fma, ('s_cbranch_scc0 65501', 1)]
    program = [('s_load_dwordx4 s[4:7], s[0:1], 0x8', 2), ('s_cmp_lt_i32 s6, 1', 1),
               ('s_cbranch_scc1 36', 1)] + loop + [
               ('global_store_dword v0, v1, s[0:1]', 2), ('s_endpgm', 1)]
    offsets = [0]
    for _, words in program:
        offsets.append(offsets[-1] + 4 * words)
    loop_begin, loop_end = offsets[3], offsets[3 + len(loop)]
    lines = ['%016x <%s>:' % (start, symbol)]
    for (text, words), offset in zip(program, offsets):
        if text in hidden:
            continue
        target = {'s_cbranch_scc1 36': loop_end, 's_cbranch_scc0 65501': loop_begin}.get(text)
        lines.append('\t%-40s // %012X:%s%s' % (
            text, start + offset, ' BF800000' * words,
            '' if target is None else ' <%s+0x%x>' % (symbol, target)))
    return '\n'.join(lines), {symbol: offsets[-1]}


def self_test():
    """The checker accepts the supported loop and rejects extra or undecoded work inside it."""
    def problems(body_ops, **changes):
        return check(body_ops, kernel_layout(*synthetic_kernel(body_ops, **changes))[body_ops])
    load = 'global_load_dword v3, v0, s[0:1]'
    move = (LOOP_MOVE + ' v2, s5', 1)
    cases = [
        ('supported layout', problems(16), False),
        # AMD clang 24: the addend's register move stays at the head of the loop.
        ('one register move at the loop head', problems(16, loop_head=[move]), False),
        ('two register moves in the loop', problems(16, loop_head=[move], in_loop=[move]), True),
        ('in-loop move of another kind', problems(16, loop_head=[('s_mov_b32 s7, s5', 1)]), True),
        ('in-loop memory load', problems(16, in_loop=[(load, 2)]), True),
        ('in-loop arithmetic', problems(16, in_loop=[('v_add_f32_e32 v1, v1, v2', 1)]), True),
        ('repeated loop control', problems(16, in_loop=[('s_add_i32 s6, s6, -1', 1)]), True),
        ('extra FMA', problems(16, in_loop=[(FMA + ' v1, s4, v1, v2', 2)]), True),
        # Lines removed while later addresses stay: only byte coverage sees the
        # hidden load, whose removal leaves 16 FMAs and the right loop control.
        ('undecoded in-loop instruction', problems(16, in_loop=[(load, 2)], hidden=[load]), True),
        ('missing loop-control line', problems(16, hidden=['s_cmp_eq_u32 s6, 0']), True),
    ]
    gap = problems(16, in_loop=[(load, 2)], hidden=[load])
    if len(gap) != 1 or 'leave a gap' not in gap[0]:
        raise SystemExit('self-test failed: undecoded instruction not reported as a gap: %s' % gap)
    wrong = [name for name, found, want in cases if bool(found) != want]
    if wrong:
        raise SystemExit('self-test failed: ' + ', '.join(wrong))
    print('PASS: layout checker self-test, %d cases' % len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1] /
                        'tier1/code_fetch_latency/hip/code_fetch_latency')
    parser.add_argument('--llvm-bin', type=Path, default=Path('/opt/rocm/llvm/bin'))
    parser.add_argument('--self-test', action='store_true',
                        help='check the layout checker on synthetic disassembly; needs no ROCm tools')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    tool = lambda name: args.llvm_bin / name
    failed = False
    with tempfile.TemporaryDirectory(prefix='code-fetch-size-') as temp:
        fatbin, code = Path(temp) / 'fatbin', Path(temp) / 'code.co'
        # With no output operand objcopy rewrites its input: send the copy to
        # the temporary directory so the inspected binary is only read.
        run(tool('llvm-objcopy'), '--dump-section', '.hip_fatbin=%s' % fatbin, args.binary,
            Path(temp) / 'discarded-copy')
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
            # llvm-objdump takes the processor from the code object itself.
            rows = kernel_layout(run(tool('llvm-objdump'), '-d', code), sizes)
            print('%s\n| kernel | code bytes | %d-byte lines | FMA instructions | loop body bytes |'
                  '\n|---|---:|---:|---:|---:|' % (target, LINE_BYTES))
            for body_ops in BODY_OPS:
                row = rows.get(body_ops)
                problems = check(body_ops, row) if row else ['kernel missing']
                if row and not problems:
                    begin, end = row['loops'][0]
                    first, last = row['start'] // LINE_BYTES, (row['start'] + row['bytes'] - 1) // LINE_BYTES
                    print('| `fetch_kernel_%d` | %d | %d | %d | %d |' % (
                        body_ops, row['bytes'], last - first + 1, body_ops, end - begin))
                else:
                    failed = True
                    print('FAIL: fetch_kernel_%d: %s' % (body_ops, '; '.join(problems)))
    if failed:
        raise SystemExit('compiled kernels do not have the layout this benchmark measures')
    print('PASS: one loop holding exactly body_ops FMA instructions, its loop control and at most one '
          '%s in every fetch kernel' % LOOP_MOVE)


if __name__ == '__main__':
    main()
