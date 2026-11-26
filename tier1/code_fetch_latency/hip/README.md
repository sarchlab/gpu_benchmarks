# code_fetch_latency — what a kernel's code costs versus its size

Each process launches one kernel once. Every thread runs a straight-line body
of `body_ops` dependent FMAs, `x = fmaf(x, a, b)`, on register values, `loops`
times, in a loop that is not unrolled, and stores the result once. One kernel,
`fetch_kernel_<body_ops>`, exists per body size, so a launch fetches only the
code of the size under test. `a` and `b` reach the kernels as runtime
arguments, so no compiler folds the chain.

Sweeping `body_ops` and `loops` separates three costs: the fixed cost of a
dispatch, the cost of an FMA whose code has already been fetched once, and the
extra cost of the first pass over code that has not been executed yet.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `BENCH_PARAM_body_ops` | 256 | FMAs in the loop body: a power of two from 16 to 65536. Selects the kernel and its code footprint |
| `BENCH_PARAM_loops` | 1 | Times the body is executed; at least 1, with `body_ops * loops` at most 1048576. The declared sweep uses 1, 4, 8 and 16 |
| `BENCH_PARAM_num_blocks` | 1 | Thread blocks |
| `BENCH_PARAM_threads_per_block` | 64 | Threads per block, at most 1024 |
| `BENCH_PARAM_vector_add_warmup` | true | Clock/cache warmer before the timed launch |

## Entry sequence

1. Query the selected kernel's attributes (`hipFuncGetAttributes`;
   `cudaFuncGetAttributes` under nvcc). This requests that the runtime load the
   kernel without running it.
2. Run the optional vector-add clock/cache warmer.
3. Time one launch with GPU events. It is the first dispatch of that kernel in
   the process.

A runtime may defer loading a kernel's code to its first launch (CUDA lazy
loading does); without step 1 that load would sit inside the timed bracket and
grow with code size. Whether the query fully pre-loads a kernel is
runtime-specific. `CUDA_MODULE_LOADING=EAGER` and
`HIP_ENABLE_DEFERRED_LOADING=0` are environment-level alternatives; the HIP
variable was read from the ROCm/clr source, not measured. Each repetition is a
fresh process.

## Deriving the costs

With `T(N, R)` the time for `body_ops = N` and `loops = R`:

```
T(N, R) ≈ fixed dispatch cost + R · (loop control)
          + N · (first-pass cost per FMA)
          + (R − 1) · N · (repeat-pass cost per FMA)
```

- **Repeat-pass cost per FMA** =
  `(T(N, R2) − T(N, R1)) / ((R2 − R1) · N)`: the cost of an FMA whose code was
  already fetched once. The fixed cost and the first pass cancel. It rises
  when the body no longer fits an instruction-cache level.
- **Fixed dispatch cost**: what `T(N, 1)` tends to as `N` gets small.
- **First-pass extra** =
  `T(N, 1) − fixed cost − N · (repeat-pass cost)`: what bringing the code in
  costs the first time. Fit it against the measured code bytes of each kernel
  (below), not against `body_ops`.

The loop-control cost depends only on `loops`, so within one `loops` series it
does not distort the shape versus `N`. Differences of small bodies are small
against timer noise; judge each cell's signal before using it.

## Limits

- **The body's own execution hides fast refetches.** A refetch that delivers
  code faster than the body consumes it does not show in the repeat-pass cost.
  A flat repeat-pass cost therefore bounds the refetch cost; it does not prove
  an instruction cache holds the body. On the two targets measured (MI300X and
  H100), a lone wavefront/warp issued one instruction every 2–3 ns regardless
  of the instruction, so a different instruction does not expose more; a body
  with more code bytes per instruction would.
- **Fetch and translation are not separated.** Code is contiguous, so the
  number of lines and the number of pages grow together. The first-pass extra
  holds whatever the first touch of the code costs.
- **What the first pass contains.** Any loading the attribute query leaves
  undone falls inside the timed launch, as does other first-use work of the
  kernel inside the event bracket.
- **One wavefront.** The declared sweep launches one block of 64 threads. With
  more blocks, several SMs/CUs fetch the same code and can share lower cache
  levels. That is a different experiment and is not declared.

## Observed on MI300X (ROCm 10.2)

Notes on reading this target's output, from collections on 2026-11-26 on an AMD
Developer Cloud "MI300X VF" instance. They are not performance claims.

- **GPU event times were bimodal.** In an earlier collection about 8 % of runs
  reported about 9 µs while the profiler's dispatch duration was about 43 µs.
  Use `rocprofv3 --kernel-trace` dispatch durations, or the median of event
  times; never the minimum across repetitions.
- **Longer kernels ran faster per FMA.** Kernels running longer than
  roughly 0.7 ms ran faster per FMA (2.5 → 2.1 ns), so compare cells of similar
  duration.

## Compiled code: size and layout

Code size must be measured from the compiled code object, not estimated from
`body_ops`, and the layout must be checked: one loop, not unrolled, holding
exactly `body_ops` FMA instructions. `unroll 1` requests that, but only the
disassembly shows what a compiler produced. For a ROCm build:

```sh
make PLATFORM=rocm
python3 ../../../tests/code_fetch_latency_code_size.py   # needs /opt/rocm/llvm/bin
```

The script extracts the code objects embedded in the binary, prints one table
per target and fails when a kernel does not have the expected layout: the
kernel's only backward jump closes one loop that holds exactly `body_ops`
`v_fma_f32`, the loop control, and at most one register move
(`v_mov_b32_e32`). Measured for `gfx942` at `-O3 -std=c++14` with ROCm 7.2.4
(HIP 7.2.53211, AMD clang 22.0.0git):

| kernel | code bytes | 64-byte lines | FMA instructions | loop bytes | loop jump |
|---|---:|---:|---:|---:|---|
| `fetch_kernel_16` | 244 | 4 | 16 | 140 | short |
| `fetch_kernel_32` | 372 | 6 | 32 | 268 | short |
| `fetch_kernel_64` | 628 | 10 | 64 | 524 | short |
| `fetch_kernel_128` | 1140 | 18 | 128 | 1036 | short |
| `fetch_kernel_256` | 2164 | 34 | 256 | 2060 | short |
| `fetch_kernel_512` | 4212 | 66 | 512 | 4108 | short |
| `fetch_kernel_1024` | 8308 | 130 | 1024 | 8204 | short |
| `fetch_kernel_2048` | 16500 | 258 | 2048 | 16396 | short |
| `fetch_kernel_4096` | 32884 | 514 | 4096 | 32780 | short |
| `fetch_kernel_8192` | 65652 | 1026 | 8192 | 65548 | short |
| `fetch_kernel_16384` | 131236 | 2051 | 16384 | 131108 | long |
| `fetch_kernel_32768` | 262308 | 4099 | 32768 | 262180 | long |
| `fetch_kernel_65536` | 524452 | 8195 | 65536 | 524324 | long |

- **Bytes per FMA.** Each FMA is one 8-byte `v_fma_f32`.
- **Loop jump.** Up to 8192 FMAs the loop closes with a backward conditional
  branch ("short": `s_add_i32`, `s_cmp_eq_u32`, `s_cbranch_scc0`).
  From 16384 the body exceeds that branch's range, so the conditional branch
  leaves the loop and a PC-relative jump goes back ("long": `s_cbranch_scc1`
  over `s_getpc_b64`, `s_add_u32`, `s_addc_u32`, `s_setpc_b64`); the zero-trip
  guard before the loop takes the same form. This adds 24 bytes and four
  scalar instructions per loop pass, part of the loop control.
- **Other toolchains.** With ROCm 10.2.0 (AMD clang 24.0.0git) the kernels up
  to 32768 were checked on a build of the previous revision: same code bytes,
  one 4-byte register move kept inside the loop (loop bytes 4 larger), and the
  same short and long loop jumps. `fetch_kernel_65536` has not been
  layout-checked with clang 24. Rerun the script for the toolchain that built
  the measured binary.
- The "64-byte lines" column counts the lines the kernel's address range spans
  in the code object. The line size is an assumption to qualify on the target.

No equivalent check exists yet for NVIDIA builds; inspect the SASS (for example
with `cuobjdump -sass`) before using CUDA results. With nvcc 12.9 for `sm_90`
each FMA is one 16-byte `FFMA`.

Compilers recurse once per chained FMA. AMD clang 24 (ROCm 10.2) segfaults on
`fetch_kernel_65536` under the default 8 MB stack, so the Makefile raises the
stack limit for the ROCm compile. With the raise this file compiles in about
1.5 minutes with ROCm 7.2.4; the same thirteen kernels took about 2 minutes
with ROCm 10.2 in an earlier revision. nvcc 12.9 takes about 12 s under the
default stack.

## Verification

After timing, every thread's stored value is compared with the same chain
computed on the host. The comparison is exact: the per-thread seed
`1 + thread/1024` is exactly representable, each step is one correctly rounded
`fmaf` on both sides and every FMA changes the value, so a chain with a missing
or extra loop pass does not match. Non-finite values are rejected, and the
buffer starts as NaN, so a thread that never stores fails. A mismatch exits
nonzero.

## Timing, traces and simulation

Stdout contains one record named after the launched kernel and one summary
equal to it. `host_time_ms` spans event submission, launch, checking and
synchronization.

A process launches the benchmark kernel once, so a trace (collected with the
warmer off) holds that one launch and the suite's whole-trace simulation
report compares like with like. A simulator must start it without that
kernel's code in its instruction cache.

`scripts/simulation_run.py select` keeps one non-scaling combination, that is
one `loops` value, and six body sizes at evenly spaced positions of the
declared list: with thirteen sizes, every second one (starting at 16, 32 or
64) when those are usable, otherwise six consecutive ones. Among equally wide
windows it prefers the default `loops=1`. One combination is a single
`T(N, R)` series, so a selection cannot give the repeat-pass cost or the
first-pass extra, which need at least two of the four `loops` values. Simulate
the full
sweep with
`python3 scripts/simulation_run.py run --benchmarks code_fetch_latency --all-params`.
No `params_selected.json` is committed for this benchmark.

## Build and run

```sh
make PLATFORM=cuda       # nvcc; use PLATFORM=rocm for hipcc
./code_fetch_latency     # default: body_ops=256, loops=1
./code_fetch_latency 4096   # positional body_ops
BENCH_PARAM_body_ops=1024 BENCH_PARAM_loops=16 ./code_fetch_latency
```

Environment values take priority over the positional body size. Integer
parameters must be whole decimal numbers: empty, non-numeric and trailing-junk
values are rejected, not read as zero. Unsupported body sizes, a `loops` below
1 or with `body_ops * loops` above 1048576, and nonpositive or oversized launch
dimensions fail before any device access. The chain limit keeps the chain
finite for every thread seed and the host reference cheap; the largest
declared cell, 65536 × 16, is exactly at it.
`python3 ../../../tests/code_fetch_latency_host.py` checks those rules and the
reference chain on the host.

Illustrative JSON (times are not performance claims):

```json
{"type":"kernel","name":"fetch_kernel_1024","iter":0,"time_ms":0.020000,"host_time_ms":0.060000,"params":{"body_ops":1024,"loops":4,"num_blocks":1,"threads_per_block":64}}
{"type":"summary","total_time_ms":0.020000}
```
