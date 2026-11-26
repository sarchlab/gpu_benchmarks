# instruction_cost — Tier 1 Microbenchmark

## Overview

`instruction_cost` times a loop whose body is a known, fixed instruction
sequence: `ops_per_iter` copies of one instruction followed by three
loop-control instructions. The kernels are hand-written gfx942 assembly
([`instruction_cost_kernels.S`](instruction_cost_kernels.S)), so the loop is
what the source says rather than what a compiler chose to emit. The throughput
benchmarks cannot offer that: `fp32_throughput`'s four `fmaf` calls, for
example, compile to two `v_pk_fma_f32` on gfx942.

The time of a launch is then a count of known instructions:

```
time ≈ launch + iterations × (ops_per_iter × instruction_cost + loop_control_cost)
```

At a fixed `ops_per_iter`, the slope of time over `iterations` is the time per
iteration. The slope of that over `ops_per_iter` is the cost of one instruction;
its intercept is the cost of the loop control. The binary emits raw times only.

## Platform Support

| Platform | Source files | Tools |
|----------|--------------|-------|
| AMD ROCm, gfx942 (MI300) | `instruction_cost.hip`, `instruction_cost_kernels.S`, `instruction_cost_kernels.h` | `hipcc`, ROCm `clang` and `ld.lld` |

There is no CUDA version. The manifest declares `"platforms": ["rocm"]`, so
`gpubench` leaves the benchmark out on CUDA, and `make PLATFORM=cuda` builds
nothing here. The code object targets gfx942 only.

## The kernels

One kernel exists per instruction, dependency and `ops_per_iter`, named
`ic_<instruction>_<dependency>_k<ops_per_iter>`, plus `ic_empty`, the loop
control alone. `ops_per_iter=0` selects `ic_empty` whatever the instruction and
dependency. Every kernel has the signature `(uint64_t *out, int iterations)`
and this loop, which starts on a 64-byte boundary:

```
loop:
    <ops_per_iter copies of the instruction>
    s_add_i32      s3, s3, -1
    s_cmp_lg_u32   s3, 0
    s_cbranch_scc1 loop
```

Per lane, `v[2:3]` is the accumulator X, `v[4:5]` a multiplier M and `v[6:7]`
an addend A. M and A never change. A 32-bit instruction uses the low register
of each pair.

| `instruction` | One copy (chained) | Bytes |
|---------------|--------------------|-------|
| `v_fma_f32` | `v_fma_f32 v2, v2, v4, v6` | 8 |
| `v_mul_f32` | `v_mul_f32 v2, v2, v4` | 4 |
| `v_add_f32` | `v_add_f32 v2, v2, v6` | 4 |
| `v_mul_lo_u32` | `v_mul_lo_u32 v2, v2, v4` | 8 |
| `v_add_u32` | `v_add_u32 v2, v2, v6` | 4 |
| `v_fma_f64` | `v_fma_f64 v[2:3], v[2:3], v[4:5], v[6:7]` | 8 |
| `v_pk_fma_f32` | `v_pk_fma_f32 v[2:3], v[2:3], v[4:5], v[6:7]` | 8 |
| `v_mov_b64` | `v_mov_b64 v[2:3], v[2:3]` | 4 |

The loop control is 12 bytes.

`dependency` chooses the destination:

- **`chained`**: every copy writes X, as in the table. Each instruction reads
  the register the previous one wrote, within an iteration and across
  iterations. `v_mov_b64` has a single source, so chained it moves X onto
  itself: the value never changes, but the read-after-write remains.
- **`independent`**: copy *j* writes `v[8+2j : 9+2j]` instead, and X is never
  written. No instruction reads a register written inside the loop, and no two
  copies share a destination.

`v_pk_fma_f32` performs two FMAs per lane, one on the low registers and one on
the high registers of its operands.

To see a loop as assembled:

```bash
/opt/rocm/lib/llvm/bin/llvm-objdump -d --mcpu=gfx942 \
    --disassemble-symbols=ic_v_pk_fma_f32_chained_k4 instruction_cost_kernels.hsaco
```

## Build

```bash
make PLATFORM=rocm
```

The Makefile assembles the kernels into a code object,
`instruction_cost_kernels.hsaco`, and links that file's bytes into the host
program, which loads them with `hipModuleLoadData` and launches the selected
kernel with `hipModuleLaunchKernel`. The assembler runs the C preprocessor
first: [`instruction_cost_kernels.h`](instruction_cost_kernels.h) lists the
instructions, the `ops_per_iter` values and the operand constants once for the
assembly, the code-object metadata and the host program.

## Run

```bash
BENCH_PARAM_instruction=v_fma_f32 BENCH_PARAM_ops_per_iter=8 ./instruction_cost
```

Each execution launches and times one kernel **exactly once**. Unless
`BENCH_PARAM_vector_add_warmup` is `false`, the vector-add warmer runs before
the timed launch. Repeat measurements come from fresh processes:
`gpubench run --reps N`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_iterations` | 65536 | Loop iterations per thread, at least 1 |
| `BENCH_PARAM_instruction` | `v_fma_f32` | Instruction in the loop body |
| `BENCH_PARAM_dependency` | `chained` | `chained` or `independent` |
| `BENCH_PARAM_ops_per_iter` | 4 | Copies per iteration: 0, 1, 2, 4, 8 or 16 |
| `BENCH_PARAM_threads_per_block` | 64 | Threads per block; 64 is one wavefront |
| `BENCH_PARAM_num_blocks` | 1 | Blocks in the grid (declared sweep: one block) |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer before the timed launch |

## Example Output

```
{"type":"kernel","name":"ic_v_fma_f32_chained_k4","iter":0,"time_ms":1.589660,"host_time_ms":1.599150,"params":{"iterations":65536,"instruction":"v_fma_f32","dependency":"chained","ops_per_iter":4,"num_blocks":1,"threads_per_block":64,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":1.589660}
```

The record is named after the launched kernel. `time_ms` is the GPU event time
of the single launch.

## Verification

Every thread stores one 64-bit result after the loop, and after the timed
bracket the host compares every thread's value with a CPU reference; a mismatch
is a fatal error. The store is part of the measured kernel, outside the loop.

- **Chained** kernels store X. The host compares it with a CPU reference
  evaluated for `ops_per_iter × iterations` applications of the instruction.
  GPU and CPU both round to nearest and fuse the FMAs, so the values match bit
  for bit. A match does not prove the count: rounding or integer wraparound
  can make different counts indistinguishable. Chained `v_add_f32`, for
  example, reaches 2.0 after 2^23 applications and stays there, and
  `v_mov_b64` never changes X.
- **Independent** kernels store `v[8:9]`, one application of the instruction
  to the initial operands. This checks the instruction and the kernel, not the
  iteration count; the loop control is the same code as in the chained kernels.

## Interpreting the result

- Compare kernels at the same `threads_per_block`. One block of 64 threads is a
  single wavefront; 256 threads are four wavefronts on one compute unit.
- The time per iteration is a slope over `iterations`, which removes the
  launch, the operand setup and the final store.
- A fit over `ops_per_iter` assumes every copy costs the same. Check the
  residuals: the first copies can cost more or less than later ones.
- Loop bodies differ in size (see the byte counts above), so kernels of one
  `ops_per_iter` do not span the same number of 64-byte lines.
- Event timer resolution and run-to-run variation must be qualified on the
  target; see [measurement interpretation](../../../docs/measurement_interpretation.md).

## Observed on MI300X (ROCm 10.2)

Qualification notes from one collection on 2026-11-26, on an AMD Developer
Cloud "MI300X VF" instance with ROCm 10.2.0: the declared sweep, five fresh
processes per configuration, reduced to the minimum GPU event time. They say
how to read this target's output; they are not performance claims.

- The time per iteration was linear in `ops_per_iter` for every instruction,
  dependency and block size: the largest residual of a fit was 0.4 ns.
- The fitted cost of one copy was 1.88–1.92 ns for all eight instructions,
  chained or independent, with 64 or 256 threads. The fitted loop-control cost
  was 15.6–15.8 ns per iteration.
- Repetitions of one configuration differed by about 2% at 262144 iterations
  (5% at most) and by tens of microseconds at 4096, where a launch lasts under
  0.3 ms. The intercept of time over `iterations` ranged from 20 to 130 µs
  across configurations. Use the slope, not one launch divided by its
  iteration count.
- `rocprofv3 --kernel-trace` dispatch durations were about 30 µs shorter than
  the event times and gave the same slopes (15.35 against 15.32 ns per
  iteration for `ic_empty`, 46.08 against 46.07 for
  `ic_v_pk_fma_f32_chained_k16`).
