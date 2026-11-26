# code_fetch_latency — first-dispatch cost versus kernel code size

Every thread runs one dependent chain of `total_ops` fused multiply-adds,
`x = fmaf(x, a, b)`, on register values and stores the result once. The chain is
laid out as a straight-line body of `body_ops` FMAs inside a loop of
`reps = total_ops / body_ops` iterations. The number of dependent FMAs executed
is the same for every `body_ops` while the static code footprint changes.
Loop-control work is not the same: smaller bodies run more iterations. One
kernel, `fetch_kernel_<body_ops>`, exists per supported body size (every power of two
from 16 to 8192), so a launch fetches only the code of the size under test.

Each execution times one launch. Cold and primed times are paired within one
body size. The raw result is first-dispatch time minus primed-dispatch time at
that body size. The benchmark is built so that this difference exposes the cost
of fetching code that has not been executed yet; reading it that way requires
target qualification (see [Deriving the difference](#deriving-the-difference)).

## Entry policy

0. Query the selected kernel's attributes (`hipFuncGetAttributes`;
   `cudaFuncGetAttributes` under nvcc) and check the result. This requests that
   the runtime load the kernel without running it.
1. Run the optional vector-add clock/cache warmer.
2. Run `prime_launches` untimed launches of the same kernel with the same
   `reps` and coefficients, and wait for them to finish. A primer stores to its
   own output buffer, never to the timed launch's.
3. Time one launch with GPU events. No warmer runs between the primer and this
   launch.

Step 0 runs in both entries. A runtime may defer loading a kernel's code to its
first launch (CUDA lazy loading does); without step 0 the cold run would pay
that load inside the timed bracket and the primed run in its untimed primer,
and the load can grow with code size. Querying the kernel's attributes is the
documented way to load it under CUDA lazy loading. Whether the call fully
pre-loads a kernel is runtime-specific and part of target qualification; one
ROCm 10.2 observation is under
[Observed on MI300X](#observed-on-mi300x-rocm-102).

`prime_launches=0` is the suite's default cold entry: the timed launch is the
first dispatch of that kernel in the process. `prime_launches=1` is an explicit
primed exception to cold entry, declared as a swept parameter so both states
are collected. Setting `BENCH_PARAM_vector_add_warmup=false` disables only
step 1; the attribute query and a requested primer still run. Each repetition
is a fresh process. The attribute query, warmer and primer are outside the GPU
and host timing brackets.

`code_entry` in the output is `"cold"` or `"primed"` and repeats what
`prime_launches` requested. Neither label certifies a hardware state:

- **Cold** means the kernel has not been dispatched in this process. Where its
  loaded code resides after the warmer (which cache level, if any) is not
  established for any target.
- **Primed** means the kernel was dispatched once. A separate launch is not
  guaranteed to run on the same SM/CU or to find its instruction cache intact.

## Deriving the difference

For one `body_ops`, with all other parameters equal:

```
first_dispatch_cost(body_ops) = time_ms(prime_launches=0) − time_ms(prime_launches=1)
```

This is first-dispatch time minus primed-dispatch time at a fixed body size.
The separate primer buffer removes direct reuse of the timed launch's output.
The two entries can still differ in more than instruction fetch: the primer may
have run on a different SM/CU, and address translation and other first-use
state of the kernel differ too. Interpreting the difference, or its slope, as
instruction-fetch cost therefore requires target qualification.

Fit the difference against the measured code size of each kernel. The slope per
64-byte line is the size-dependent part, the candidate per-line fetch cost; the
intercept is the part that does not depend on code size. A slope that is not
constant is itself a result (for example prefetching or overlapped fetches),
not a defect of the fit.

Reduce the repetitions of each entry to one value before subtracting. The
suite's usual choice is the minimum GPU time; it is valid only where the event
times of one configuration form a single mode. Where they do not (see
[Observed on MI300X](#observed-on-mi300x-rocm-102)), use the median, or the
profiler's dispatch durations. Qualify the event timer's resolution and the
run-to-run variation on the target against the cold-minus-primed difference
being estimated, which can be much smaller than the launch time (see
[measurement interpretation](../../../docs/measurement_interpretation.md#empty-dispatch-and-timing-resolution)).

## Observed on MI300X (ROCm 10.2)

Qualification notes from one collection on 2026-11-26, on an AMD Developer
Cloud "MI300X VF" instance with ROCm 10.2.0 (HIP 7.17.26434). They say how to
read this target's output; they are not performance claims.

- **GPU event times were bimodal.** Roughly 8 % of primed runs and about 1 % of
  cold runs reported about 9 µs regardless of body size; the rest reported
  about 45 µs. `rocprofv3 --kernel-trace` dispatch durations for launches of
  the same configurations were about 43 µs. On this target do not take the
  minimum across repetitions: it selects the short mode. Use the median, or
  better the profiler's dispatch durations.
- **One profiled process gives both entries.** A `prime_launches=1` process
  under `rocprofv3 --kernel-trace --output-format csv` records both dispatches
  of the kernel: the first is the cold one, the second the primed one. Their
  paired difference had a much tighter spread than differences of event times
  across processes. Order the two rows by start timestamp; the CSV is not
  always chronological.
- **Attribute query.** With `hipFuncGetAttributes` (entry step 0) the paired
  cold-minus-primed dispatch difference was smaller than without it: about
  12 µs versus about 14 µs at the 8192-FMA body. Runtime first-use work is
  therefore small on this target compared with CUDA lazy loading.

## Compiled code: size and layout

Code size must be measured from the compiled code object, not estimated from
`body_ops`, and the layout must be checked: one loop, not unrolled, whose body
holds exactly `body_ops` FMA instructions. `#pragma unroll 1` requests that, but
only the disassembly shows what a compiler produced. For a ROCm build:

```sh
make PLATFORM=rocm
python3 ../../../tests/code_fetch_latency_code_size.py   # needs /opt/rocm/llvm/bin
```

The script extracts the code objects embedded in the binary, prints one table
per target and fails when a kernel does not have the expected layout: besides
the FMAs, the loop may hold only its counter decrement, compare and backward
branch, and at most one register move (`v_mov_b32_e32`). Measured for `gfx942`
at `-O3 -std=c++14` with two toolchains, ROCm 7.2.4 (HIP 7.2.53211, AMD clang
22.0.0git) and ROCm 10.2.0 (HIP 7.17.26434, AMD clang 24.0.0git). Code bytes
and lines are identical for both; the loop-body column is the clang 22 one:

| kernel | code bytes | 64-byte lines | FMA instructions | loop body bytes |
|---|---:|---:|---:|---:|
| `fetch_kernel_16` | 244 | 4 | 16 | 140 |
| `fetch_kernel_32` | 372 | 6 | 32 | 268 |
| `fetch_kernel_64` | 628 | 10 | 64 | 524 |
| `fetch_kernel_128` | 1140 | 18 | 128 | 1036 |
| `fetch_kernel_256` | 2164 | 34 | 256 | 2060 |
| `fetch_kernel_512` | 4212 | 66 | 512 | 4108 |
| `fetch_kernel_1024` | 8308 | 130 | 1024 | 8204 |
| `fetch_kernel_2048` | 16500 | 258 | 2048 | 16396 |
| `fetch_kernel_4096` | 32884 | 514 | 4096 | 32780 |
| `fetch_kernel_8192` | 65652 | 1026 | 8192 | 65548 |

Each FMA is one 8-byte `v_fma_f32`, so code bytes are `8 * body_ops + 116` with
both compilers. Clang 22 hoists the move of the addend into a vector register
out of the loop (loop body `8 * body_ops + 12` bytes). Clang 24 keeps that one
4-byte loop-invariant move as the first instruction of the loop (loop body
`8 * body_ops + 16` bytes), so it executes once per iteration and belongs to
the loop-control work that cancels within one body size. Rerun the script for
the toolchain that built the measured binary; another compiler version or
target can change sizes and layout. The "64-byte lines" column counts the lines
the kernel's address range spans in the code object. The line size is an
assumption to qualify on the target.

The sweep stops at 8192 because a 16384-FMA body exceeds the range of the short
conditional branch on `gfx942`: with clang 22 it is built with a
long-branch sequence (`s_getpc_b64` ... `s_setpc_b64`) around the loop, a
different loop shape from the other sizes.

No equivalent check exists yet for NVIDIA builds. Inspect the SASS (for example
with `cuobjdump -sass`) before using CUDA results.

## Limits

- **Instruction-cache capacity.** When a kernel's code exceeds the instruction
  cache and `reps > 1`, later loop iterations refetch code even in the primed
  run, so the difference no longer contains one fetch of the code. Every
  declared size runs at least two iterations (`reps` is 2 at `body_ops=8192`),
  so this applies to every size whose code exceeds that cache. Exclude such
  sizes from the slope. Where the primed time rises with `body_ops` is evidence
  about that capacity, to be qualified separately.
- **One wavefront.** The declared sweep launches one block of 64 threads. With
  more blocks, several SMs/CUs fetch the same code and can share lower cache
  levels. That is a different experiment and is not declared.
- **What "cold" contains.** Both entries query the kernel's attributes before
  timing (entry step 0) to request that the runtime load it. Whether that fully
  pre-loads the kernel is established by target qualification; any loading it
  leaves undone falls inside the cold timed launch. Otherwise the cold time
  holds first-dispatch hardware and driver effects inside the GPU event
  bracket: the fetch itself, address translation of the code, and any other
  first-use state of the kernel. Costs that do not grow with code size land in
  the intercept; they are part of the measurement and are not separated here.
  Both output buffers are set up before the warmer in either entry and a primer
  stores to its own buffer, which removes direct reuse of the timed launch's
  destination. It does not make the output store's cost equal in the two
  entries: only the primed entry executes a store after the warmer, which can
  affect shared translation or memory-system state.
- **FMA count is constant, loop-control work is not.** Smaller bodies run more
  loop iterations. Loop overhead is present in both entries of one body size
  and cancels in their difference; it does not cancel when comparing absolute
  times across `body_ops`.

## Verification

After timing, every thread's value stored by the timed launch is compared with
the same chain computed on the host. The comparison is exact: the per-thread
seed `1 + thread/1024` is exactly representable, each step is one correctly
rounded `fmaf` on both sides, and every FMA changes the value, so a chain with a
missing or extra loop iteration does not match. Non-finite values are rejected.
The output buffer starts as NaN, so a thread that never stores fails. A mismatch
exits nonzero. With `prime_launches=1` the primer's own buffer is checked the
same way, separately from the timed launch's.

## Timing and trace replay

Stdout contains one record named after the launched kernel
(`fetch_kernel_<body_ops>`) for the timed launch and one summary; the primer has
no timing record. The summary equals the timed record's GPU time.
`host_time_ms` spans event submission, launch, checking and synchronization.

Trace collection forces the clock warmer off and must retain every launch in
order. With `prime_launches=1` the raw capture has two launches of the same
kernel even though timing output has one record. They have the same `reps` and
coefficients and differ only in the output pointer. A simulator must replay the
primer to establish instruction-cache state and exclude its time from the
result; it must not reset caches between the two launches. With
`prime_launches=0` the simulator must start the timed launch without that
kernel's code in its instruction cache.

The suite's whole-trace simulation report aggregates every launch of a trace.
With `prime_launches=1` it would compare primer plus timed simulated work with
the timed-only hardware record, so a primed comparison needs per-launch
simulator output. General trace-region identification and artifact/replay
qualification remain R-23/R-24; this benchmark's ordering alone does not qualify
a simulator.

No `params_selected.json` is committed for this benchmark, so the
selected-configuration simulation workflow skips it until one is written.
`scripts/simulation_run.py select` keeps one non-scaling combination and six
scaling values: the values at positions `s, s+k, ..., s+5k` of the declared
`body_ops` list must all be usable (hardware timing, an ok trace and an ok
simulation). With ten declared sizes only stride `k=1` fits, so a selection is
six consecutive body sizes of a single `prime_launches` value. The widest
stride wins, then the smallest start, then the combination closest to the
defaults, so `prime_launches=0` is preferred when both entries qualify. A
selection therefore never holds both entries of one body size and cannot give
the cold-minus-primed difference. Simulate the full sweep explicitly with
`python3 scripts/simulation_run.py run --benchmarks code_fetch_latency --all-params`.
Primed comparisons still need per-launch simulator output.

## Build and run

```sh
make PLATFORM=cuda       # nvcc; use PLATFORM=rocm for hipcc
./code_fetch_latency     # default: body_ops=256, cold entry
./code_fetch_latency 4096   # positional body_ops
BENCH_PARAM_body_ops=1024 BENCH_PARAM_prime_launches=1 ./code_fetch_latency
```

Environment values take priority over the positional body size:

| Parameter | Default | Meaning |
|---|---:|---|
| `BENCH_PARAM_body_ops` | 256 | FMAs in the loop body: a power of two from 16 to 8192 |
| `BENCH_PARAM_total_ops` | 16384 | FMAs executed per thread; positive multiple of `body_ops`, at most 1048576 |
| `BENCH_PARAM_prime_launches` | 0 | Untimed launches of the same kernel before timing: 0 or 1 |
| `BENCH_PARAM_num_blocks` | 1 | Thread blocks |
| `BENCH_PARAM_threads_per_block` | 64 | Threads per block, at most 1024 |
| `BENCH_PARAM_vector_add_warmup` | true | Clock/cache warmer before any primer |

Integer parameters must be whole decimal numbers: empty, non-numeric and
trailing-junk values are rejected, not read as zero. Unsupported body sizes, a
`total_ops` that is not a positive multiple of `body_ops` or exceeds 1048576, a
`prime_launches` other than 0 or 1, and nonpositive or oversized launch
dimensions fail before any device access. The `total_ops` limit keeps the chain
finite for every thread seed and the host reference cheap.
`python3 ../../../tests/code_fetch_latency_host.py` checks those rules and the
reference chain on the host.

Illustrative JSON for the primed 1024-FMA body (times are not performance claims):

```json
{"type":"kernel","name":"fetch_kernel_1024","iter":0,"time_ms":0.020000,"host_time_ms":0.060000,"params":{"body_ops":1024,"total_ops":16384,"reps":16,"prime_launches":1,"num_blocks":1,"threads_per_block":64,"code_entry":"primed"}}
{"type":"summary","total_time_ms":0.020000}
```
