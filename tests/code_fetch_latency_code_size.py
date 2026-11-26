#!/usr/bin/env python3
"""Measure and check code_fetch_latency's compiled AMD kernels (no GPU required).

Reads the code objects embedded in a ROCm build of the benchmark, reports each
fetch kernel's code bytes, and checks the layout the benchmark depends on: one
loop, closed by the kernel's only backward jump, whose body holds exactly
body_ops operations of the kernel's kind, its loop control and at most one
loop-invariant register move.
--self-test checks the layout checker itself and needs no ROCm tools.
"""
import argparse
import collections
from pathlib import Path
import re
import subprocess
import tempfile

BODY_OPS = tuple(16 << i for i in range(12))  # powers of two, 16 to 32768
LINE_BYTES = 64
# Operation kind -> (kernel name prefix, instructions of one body operation).
OPS = collections.OrderedDict([
    ('fma', ('fetch_kernel', ('v_fma_f32',))),
    ('xor_add', ('fetch_xor_add_kernel', ('v_xor_b32', 'v_add_u32'))),
])
# The loop control the measured loop must hold, once each, as the gfx942 build
# emits it. Short form: counter decrement, compare, backward conditional branch.
# Long form, when the body exceeds the conditional branch's range (16384
# operations and up): the conditional branch leaves the loop and a PC-relative
# jump (s_getpc_b64, 64-bit add of the offset, s_setpc_b64) goes back.
# A compiler that emits another sequence fails the check until its disassembly
# has been inspected and these lists updated.
SHORT_CONTROL = ('s_add_i32', 's_cmp_eq_u32', 's_cbranch_scc0')
LONG_JUMP = ('s_getpc_b64', 's_add_u32', 's_addc_u32', 's_setpc_b64')
LONG_CONTROL = ('s_add_i32', 's_cmp_eq_u32', 's_cbranch_scc1') + LONG_JUMP
# AMD clang 24 (ROCm 10.2) leaves the move of the FMA addend into a vector
# register at the head of the loop instead of hoisting it (clang 22 hoists it).
# At most one such move is accepted; it adds 4 bytes to the loop and no operation.
LOOP_MOVE = 'v_mov_b32_e32'
FUNCTION = re.compile(r'^([0-9a-f]+) <(\w+)>:$')
KERNEL = re.compile(r'fetch_(xor_add_)?kernel_(\d+)')
# "\tmnemonic operands   // 000000003628: D1CB0001 040A0204 <symbol+0x28>"
INSTRUCTION = re.compile(r'^\s+(\S+)[^/]*// ([0-9A-F]+):((?: [0-9A-F]{8})+)(?: <\w+\+0x([0-9a-f]+)>)?')


def run(*command):
    return subprocess.run([str(c) for c in command], check=True,
                          capture_output=True, text=True).stdout


def base(name):
    """Mnemonic without its encoding suffix: v_add_u32_e32 and v_add_u32_e64 are one opcode."""
    return re.sub(r'_e(32|64)$', '', name)


def long_jumps(instructions):
    """(address, end, target) of each s_getpc_b64 / s_add_u32 / s_addc_u32 / s_setpc_b64 jump."""
    jumps = []
    for i in range(len(instructions) - 3):
        group = instructions[i:i + 4]
        if tuple(name for _, name, _, _ in group) != LONG_JUMP:
            continue
        (address, _, _, _), (base_pc, _, _, low), (_, _, _, high), (last, _, width, _) = group
        if len(low) != 2 or len(high) != 2:
            continue  # the offset must be a 32-bit literal in each add
        offset = (int(high[1], 16) << 32) | int(low[1], 16)
        if offset >= 1 << 63:
            offset -= 1 << 64
        # s_getpc_b64 yields the address of the instruction after it.
        jumps.append((address, last + width, base_pc + offset))
    return jumps


def kernel_layout(disassembly, sizes):
    """{(op, body_ops): row} for the fetch kernels found in one code object."""
    rows, row = {}, None
    for line in disassembly.splitlines():
        match = FUNCTION.match(line)
        if match:
            kernel = KERNEL.search(match[2])
            row = None
            if kernel:
                key = ('xor_add' if kernel[1] else 'fma', int(kernel[2]))
                row = rows[key] = dict(start=int(match[1], 16), bytes=sizes[match[2]],
                                       instructions=[], branches=[])
            continue
        match = INSTRUCTION.match(line) if row else None
        if not match:
            continue
        words = match[3].split()
        address, width = int(match[2], 16), 4 * len(words)
        if address >= row['start'] + row['bytes']:
            continue  # alignment padding after the kernel
        row['instructions'].append((address, match[1], width, words))
        if 'branch' in match[1] and match[4]:
            row['branches'].append((address, address + width, row['start'] + int(match[4], 16)))
    for row in rows.values():
        row['long_jumps'] = long_jumps(row['instructions'])
        # A jump whose target is not after it closes a loop.
        row['loops'] = [(target, end, form)
                        for form, jumps in (('short', row['branches']), ('long', row['long_jumps']))
                        for address, end, target in jumps if target <= address]
    return rows


def check(op, body_ops, row):
    """Problems that make this kernel's timing unusable for code-fetch analysis."""
    if len(row['loops']) != 1:
        return ['%d loops (want 1)' % len(row['loops'])]
    begin, end, form = row['loops'][0]
    body = OPS[op][1]
    loop = sorted(i[:3] for i in row['instructions'] if begin <= i[0] < end)
    inside = collections.Counter(base(name) if base(name) in body else name
                                 for address, name, width in loop)
    total = collections.Counter(base(name) for address, name, width, words in row['instructions'])
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
    for name in body:
        if total[name] != body_ops:
            problems.append('%d %s instructions (want %d)' % (total[name], name, body_ops))
        if inside[name] != total[name]:
            problems.append('%s instructions outside the loop body' % name)
    other = inside - collections.Counter({name: inside[name] for name in body})
    control = other - collections.Counter({LOOP_MOVE: other[LOOP_MOVE]})
    wanted = SHORT_CONTROL if form == 'short' else LONG_CONTROL
    if control != collections.Counter(wanted) or other[LOOP_MOVE] > 1:
        problems.append('loop body holds %s besides its operations (want one each of %s, and at most one %s)' % (
            ', '.join('%d %s' % (n, name) for name, n in sorted(other.items())) or 'nothing',
            ', '.join(wanted), LOOP_MOVE))
    # Zero-trip guard plus the loop's own conditional branch; the long form
    # adds one PC-relative jump to each.
    jumps = (len(row['branches']), len(row['long_jumps']))
    if jumps != ((2, 0) if form == 'short' else (2, 2)):
        problems.append('%d conditional branches and %d PC-relative jumps (want %s)' % (
            jumps + ('2 and 0' if form == 'short' else '2 and 2',)))
    return problems


def synthetic_kernel(body_ops, op='fma', long_form=False, in_loop=(), hidden=(), loop_head=(),
                     straight_line=False):
    """(disassembly, sizes) shaped like the gfx942 build, with extra in-loop instructions.

    `loop_head` instructions are the first of the loop (the backward jump
    targets them); `in_loop` ones sit before the loop control. Instructions
    whose text is in `hidden` keep their bytes but get no disassembly line, as
    when a line is elided or cannot be parsed. `straight_line` drops the jump
    that closes the loop.
    """
    prefix = OPS[op][0]
    symbol, start = '_Z%d%s_%dP' % (len(prefix) + 1 + len(str(body_ops)), prefix, body_ops), 0x3600
    one = [('v_fma_f32 v1, s4, v1, v2', 2)] if op == 'fma' else [
        ('v_xor_b32_e32 v1, s4, v1', 1), ('v_add_u32_e32 v1, s5, v1', 1)]
    # (text, words, jump): jump is None, ('branch', label) or ('pc', label) on
    # the s_add_u32 / s_addc_u32 pair that carries a PC-relative offset.
    def long_jump(label):
        return [('s_getpc_b64 s[8:9]', 1, None), ('s_add_u32 s8, s8, 0', 2, ('pc_low', label)),
                ('s_addc_u32 s9, s9, 0', 2, ('pc_high', label)), ('s_setpc_b64 s[8:9]', 1, None)]
    plain = lambda items: [(text, words, None) for text, words in items]
    body = one * body_ops
    loop = plain(list(loop_head) + body[:-1] + list(in_loop) + [
        ('s_add_i32 s6, s6, -1', 1), ('s_cmp_eq_u32 s6, 0', 1)] + body[-1:])
    head = plain([('s_load_dwordx4 s[4:7], s[0:1], 0x8', 2), ('s_cmp_lt_i32 s6, 1', 1)])
    if long_form:
        head += [('s_cbranch_scc0 6', 1, ('branch', 'loop_begin'))] + long_jump('loop_end')
        close = [('s_cbranch_scc1 6', 1, ('branch', 'loop_end'))] + long_jump('loop_begin')
    else:
        head += [('s_cbranch_scc1 36', 1, ('branch', 'loop_end'))]
        close = [('s_cbranch_scc0 65501', 1, ('branch', 'loop_begin'))]
    if straight_line:
        close = []
    tail = plain([('global_store_dword v0, v1, s[0:1]', 2), ('s_endpgm', 1)])
    program = head + loop + close + tail
    offsets = [0]
    for _, words, _ in program:
        offsets.append(offsets[-1] + 4 * words)
    labels = dict(loop_begin=offsets[len(head)], loop_end=offsets[len(head) + len(loop) + len(close)])
    lines = ['%016x <%s>:' % (start, symbol)]
    for index, ((text, words, jump), offset) in enumerate(zip(program, offsets)):
        if text in hidden:
            continue
        encoded, suffix = ' BF800000' * words, ''
        if jump and jump[0] == 'branch':
            suffix = ' <%s+0x%x>' % (symbol, labels[jump[1]])
        elif jump:
            # The offset is relative to the s_add_u32 (the instruction after s_getpc_b64).
            add = offset if jump[0] == 'pc_low' else offsets[index - 1]
            delta = (labels[jump[1]] - add) & ((1 << 64) - 1)
            encoded = ' 8008FF08 %08X' % (delta & 0xffffffff if jump[0] == 'pc_low' else delta >> 32)
        lines.append('\t%-40s // %012X:%s%s' % (text, start + offset, encoded, suffix))
    return '\n'.join(lines), {symbol: offsets[-1]}


def self_test():
    """The checker accepts the supported loops and rejects extra or undecoded work inside them."""
    def problems(body_ops, op='fma', **changes):
        rows = kernel_layout(*synthetic_kernel(body_ops, op, **changes))
        return check(op, body_ops, rows[(op, body_ops)])
    load = 'global_load_dword v3, v0, s[0:1]'
    move = (LOOP_MOVE + ' v2, s5', 1)
    fma, xor, add = ('v_fma_f32 v1, s4, v1, v2', 2), ('v_xor_b32_e32 v1, s4, v1', 1), ('v_add_u32_e32 v1, s5, v1', 1)
    cases = [
        ('fma, short loop', problems(16), False),
        ('xor_add, short loop', problems(16, 'xor_add'), False),
        ('fma, long-jump loop', problems(16, long_form=True), False),
        ('xor_add, long-jump loop', problems(16, 'xor_add', long_form=True), False),
        # AMD clang 24: the addend's register move stays at the head of the loop.
        ('one register move at the loop head', problems(16, loop_head=[move]), False),
        ('one register move, long-jump loop', problems(16, long_form=True, loop_head=[move]), False),
        ('two register moves in the loop', problems(16, loop_head=[move], in_loop=[move]), True),
        ('in-loop move of another kind', problems(16, loop_head=[('s_mov_b32 s7, s5', 1)]), True),
        ('in-loop memory load', problems(16, in_loop=[(load, 2)]), True),
        ('in-loop memory load, long-jump loop', problems(16, long_form=True, in_loop=[(load, 2)]), True),
        ('in-loop arithmetic', problems(16, in_loop=[('v_add_f32_e32 v1, v1, v2', 1)]), True),
        ('repeated loop control', problems(16, in_loop=[('s_add_i32 s6, s6, -1', 1)]), True),
        ('extra FMA', problems(16, in_loop=[fma]), True),
        ('extra xor without its add', problems(16, 'xor_add', in_loop=[xor]), True),
        ('extra add without its xor', problems(16, 'xor_add', in_loop=[add]), True),
        ('FMA in an xor_add body', problems(16, 'xor_add', in_loop=[fma]), True),
        ('no loop', problems(16, straight_line=True), True),
        # Lines removed while later addresses stay: only byte coverage sees the
        # hidden load, whose removal leaves 16 FMAs and the right loop control.
        ('undecoded in-loop instruction', problems(16, in_loop=[(load, 2)], hidden=[load]), True),
        ('missing loop-control line', problems(16, hidden=['s_cmp_eq_u32 s6, 0']), True),
        ('incomplete PC-relative jump', problems(16, long_form=True, hidden=['s_addc_u32 s9, s9, 0']), True),
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
            print('%s\n| kernel | op | code bytes | %d-byte lines | body instructions | loop bytes | loop jump |'
                  '\n|---|---|---:|---:|---:|---:|---|' % (target, LINE_BYTES))
            for op, (prefix, body) in OPS.items():
                for body_ops in BODY_OPS:
                    row = rows.get((op, body_ops))
                    problems = check(op, body_ops, row) if row else ['kernel missing']
                    if row and not problems:
                        begin, end, form = row['loops'][0]
                        first, last = row['start'] // LINE_BYTES, (row['start'] + row['bytes'] - 1) // LINE_BYTES
                        print('| `%s_%d` | %s | %d | %d | %d | %d | %s |' % (
                            prefix, body_ops, op, row['bytes'], last - first + 1,
                            body_ops * len(body), end - begin, form))
                    else:
                        failed = True
                        print('FAIL: %s_%d: %s' % (prefix, body_ops, '; '.join(problems)))
    if failed:
        raise SystemExit('compiled kernels do not have the layout this benchmark measures')
    print('PASS: one loop holding exactly body_ops operations, its loop control and at most one '
          '%s in every fetch kernel' % LOOP_MOVE)


if __name__ == '__main__':
    main()
