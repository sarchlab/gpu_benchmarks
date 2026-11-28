# loop_wait — Tier 1 Microbenchmark

## Overview

`loop_wait` times loops that differ from the loop of
[`instruction_cost`](../../instruction_cost/hip/README.md) in one respect, or
for the `compiled` layout in two. Like that
benchmark's, its kernels are hand-written gfx942 assembly
([`loop_wait_kernels.S`](loop_wait_kernels.S)), so every loop is what the
source says.

The reference loop is the chained `instruction_cost` kernel
`ic_<instruction>_chained_k<ops_per_iter>`: `ops_per_iter` copies of one VALU
instruction, then the loop control, with the loop head on a 64-byte boundary.

```
loop:
    <ops_per_iter copies of the instruction>     V x K
    s_add_i32      s3, s3, -1                    S
    s_cmp_lg_u32   s3, 0                         S
    s_cbranch_scc1 loop                          B
```

A compiled loop differs from it in several ways at once. On gfx942 with ROCm
10.2, the loops of `fp16_throughput`, `fp64_throughput`, `fp32_throughput` and
`int32_throughput` start 8, 20, 28 and 52 bytes into a 64-byte line, place the
counter decrement before or among the VALU instructions, and place a VALU
instruction between the compare and the branch. A compiler may also give a
VALU instruction a scalar-register operand. Each `layout` here makes one such
change alone and keeps everything else: the same `ops_per_iter` VALU
instructions and the same three loop-control instructions per iteration,
computing the same result. The exception is `compiled`, which makes both
changes of instruction order, the decrement after the first VALU instruction
and one VALU instruction before the branch, to measure the cost of the
combined layout relative to the reference. `add_at_head` and
`valu_before_branch` make one each; `add_at_head` puts the decrement at the
loop head, so `compiled` is not those two layouts combined.

The time per iteration of a layout, less that of the reference kernel with the
same instruction and `ops_per_iter`, is what the change costs, or for
`compiled` what its two changes cost together. The other exceptions are in the
two `sgpr` layouts, which change two things at once for four instructions:
`v_mov_b64` also drops a dependency, and `v_mul_f32`, `v_add_f32` and
`v_add_u32` also move X to the second source (see
[The kernels](#the-kernels)). The binary emits raw times only.

## Platform Support

| Platform | Source files | Tools |
|----------|--------------|-------|
| AMD ROCm, gfx942 (MI300) | `loop_wait.hip`, `loop_wait_kernels.S`, `loop_wait_kernels.h` | `hipcc`, ROCm `clang` and `ld.lld` |

There is no CUDA version. The manifest declares `"platforms": ["rocm"]`, so
`gpubench` leaves the benchmark out on CUDA, and `make PLATFORM=cuda` builds
nothing here.

The kernels are assembled for gfx942 only. On any other AMD device the
benchmark stops before loading them, with a fatal error that names gfx942 and
the architecture the device reports.

## The kernels

One kernel exists per layout, instruction and `ops_per_iter`, named
`lw_<layout>_<instruction>_k<ops_per_iter>`, with the signature
`(uint64_t *out, int iterations)`. The instructions are the eight VALU
instructions of `instruction_cost`, with the same operands, and every copy is
the chained one, X = op(X, ...), unless noted. In the table K is
`ops_per_iter`, V a copy of the instruction, S a loop-control scalar
instruction and B the loop branch.

| `layout` | Loop | Loop head | What differs from the reference |
|----------|------|-----------|---------------------------------|
| `head_offset_8` | `V x K, S, S, B` | 8 bytes into a 64-byte line | The alignment |
| `head_offset_20` | `V x K, S, S, B` | 20 bytes | The alignment |
| `head_offset_28` | `V x K, S, S, B` | 28 bytes | The alignment |
| `head_offset_52` | `V x K, S, S, B` | 52 bytes | The alignment |
| `add_at_head` | `S, V x K, S, B` | Aligned | `s_add_i32` is the first instruction of the loop |
| `valu_before_branch` | `V x (K-1), S, S, V, B` | Aligned | One VALU instruction sits between `s_cmp_lg_u32` and the branch |
| `compiled` | `V, S, V x (K-2), S, V, B` | Aligned | Both: `s_add_i32` follows the first VALU instruction, and one VALU instruction precedes the branch |
| `sgpr_at_head` | `V', V x (K-1), S, S, B` | Aligned | The first VALU instruction, the branch target, reads a scalar register |
| `sgpr_in_middle` | `V x (K/2), V', V x (K-K/2-1), S, S, B` | Aligned | VALU instruction K/2, counting from 0, reads a scalar register |

Notes on the layouts:

- **`head_offset_N`**: the padding is `s_nop` before the loop, outside it, so
  it executes once. At 20, 28 and 52 bytes, an 8-byte instruction of the body
  can straddle two 64-byte lines.
- **`valu_before_branch`, `compiled`**: a VALU instruction leaves SCC alone, so
  the branch still tests the compare.
- **`compiled`** fixes the position of `s_add_i32` after the first VALU
  instruction, as in the `fp32_throughput` and `fp64_throughput` loops. At
  `ops_per_iter=2` it is the same loop as `valu_before_branch`.
- **`sgpr_at_head`, `sgpr_in_middle`**: `V'` is the instruction with one
  operand read from `s6` (or `s[6:7]`), which is loaded before the loop with
  the bits of the vector register it replaces. The other copies are unchanged.

| `instruction` | Reference copy | `V'` | Scalar register holds |
|---------------|----------------|------|-----------------------|
| `v_fma_f32` | `v_fma_f32 v2, v2, v4, v6` | `v_fma_f32 v2, v2, s6, v6` | M |
| `v_mul_f32` | `v_mul_f32 v2, v2, v4` | `v_mul_f32 v2, s6, v2` | M |
| `v_add_f32` | `v_add_f32 v2, v2, v6` | `v_add_f32 v2, s6, v2` | A |
| `v_mul_lo_u32` | `v_mul_lo_u32 v2, v2, v4` | `v_mul_lo_u32 v2, v2, s6` | M |
| `v_add_u32` | `v_add_u32 v2, v2, v6` | `v_add_u32 v2, s6, v2` | A |
| `v_fma_f64` | `v_fma_f64 v[2:3], v[2:3], v[4:5], v[6:7]` | `v_fma_f64 v[2:3], v[2:3], s[6:7], v[6:7]` | M |
| `v_pk_fma_f32` | `v_pk_fma_f32 v[2:3], v[2:3], v[4:5], v[6:7]` | `v_pk_fma_f32 v[2:3], v[2:3], s[6:7], v[6:7]` | M, in both lanes |
| `v_mov_b64` | `v_mov_b64 v[2:3], v[2:3]` | `v_mov_b64 v[2:3], s[6:7]` | The initial X |

The scalar register stands for a loop-invariant operand: M for a multiply or
an FMA, A for an add. A two-operand instruction (`v_mul_f32`, `v_add_f32`,
`v_add_u32`) accepts a scalar register as its first source only, so there X
moves to the second source.

`v_mov_b64` is the exception among the eight. Its one source is X, which a
move never changes, so its `V'` reads `s[6:7]` in place of X and no longer
reads the register the previous copy wrote: one read-after-write dependency
per iteration is gone, and in `sgpr_at_head` it is the one that crosses the
loop branch. For the eight `lw_sgpr_*_v_mov_b64_k*` kernels the difference
from the reference is therefore the combined effect of the scalar operand and
of the missing dependency, not the cost of the scalar operand alone. The seven
arithmetic instructions keep X as an operand of `V'`, so their accumulator
dependency is intact. For `v_fma_f32`, `v_mul_lo_u32`, `v_fma_f64` and
`v_pk_fma_f32` X also stays in the same source, so the scalar operand is the
only change. `v_mul_f32`, `v_add_f32` and `v_add_u32` also move X to the second
source, so their difference from the reference includes that move.

To see a loop as assembled, with the address of its head:

```bash
/opt/rocm/lib/llvm/bin/llvm-objdump -d --mcpu=gfx942 \
    --disassemble-symbols=lw_head_offset_52_v_fma_f32_k4 loop_wait_kernels.hsaco
```

## Build

```bash
make PLATFORM=rocm
```

The Makefile assembles the kernels into a code object,
`loop_wait_kernels.hsaco`, and links that file's bytes into the host program,
which loads them with `hipModuleLoadData` and launches the selected kernel
with `hipModuleLaunchKernel`. The assembler runs the C preprocessor first:
[`loop_wait_kernels.h`](loop_wait_kernels.h) lists the layouts and the
`ops_per_iter` values once for the assembly, the code-object metadata and the
host program. The eight VALU instructions, their operand constants, the kernel
entry and exit and the host-side launch and verification are in
[`gfx942_loop_kernels.h`](../../../common_hip/gfx942_loop_kernels.h) and
[`gfx942_loop_host.h`](../../../common_hip/gfx942_loop_host.h), which
`instruction_cost` shares.

## Run

```bash
BENCH_PARAM_layout=head_offset_28 BENCH_PARAM_instruction=v_fma_f32 ./loop_wait
```

Each execution launches and times one kernel **exactly once**. Unless
`BENCH_PARAM_vector_add_warmup` is `false`, the vector-add warmer runs before
the timed launch. Repeat measurements come from fresh processes:
`gpubench run --reps N`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_iterations` | 65536 | Loop iterations per thread, at least 1 |
| `BENCH_PARAM_layout` | `head_offset_8` | One of the nine layouts above |
| `BENCH_PARAM_instruction` | `v_fma_f32` | VALU instruction in the loop body |
| `BENCH_PARAM_ops_per_iter` | 4 | Copies per iteration: 2, 4, 8 or 16 |
| `BENCH_PARAM_threads_per_block` | 64 | Threads per block; 64 is one wavefront (declared sweep: 64) |
| `BENCH_PARAM_num_blocks` | 1 | Blocks in the grid (declared sweep: one block) |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer before the timed launch |

Every layout has a kernel for every instruction and every `ops_per_iter`, so
each declared combination runs.

## Output

One record named after the launched kernel, then a summary. `time_ms` is the
GPU event time of the single launch; the times are left out here.

```
{"type":"kernel","name":"lw_head_offset_28_v_fma_f32_k4","iter":0,"time_ms":...,"host_time_ms":...,"params":{"iterations":65536,"layout":"head_offset_28","instruction":"v_fma_f32","ops_per_iter":4,"num_blocks":1,"threads_per_block":64,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":...}
```

## Verification

Every thread stores X after the loop, and after the timed bracket the host
compares every thread's value with a CPU reference; a mismatch is a fatal
error. The store is part of the measured kernel, outside the loop.

No layout changes what the loop computes: the reference is
`ops_per_iter × iterations` applications of the instruction, the value the
chained `instruction_cost` kernel stores. A scalar register holds the bits of
the operand it replaces. GPU and CPU both round to nearest and fuse the FMAs,
so the values match bit for bit. As in `instruction_cost`, a match does not
prove the count for an instruction whose result saturates or never changes.

## Interpreting the result

- The time per iteration is a slope over `iterations`, which removes the
  launch, the operand setup, the padding before the loop and the final store.
- The reference kernel is in `instruction_cost`, not here. Collect both
  benchmarks in one session, at the same clock, with one block of 64 threads.
- Layouts execute the same instructions per iteration as the reference, so a
  difference in time per iteration is not a difference in instruction count.
- Event timer resolution and run-to-run variation must be qualified on the
  target; see [measurement interpretation](../../../docs/measurement_interpretation.md).
