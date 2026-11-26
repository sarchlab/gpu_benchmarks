# code_fetch_latency — cost of fetching kernel code (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

Every thread runs one dependent chain of `total_ops` fused multiply-adds,
`x = fmaf(x, a, b)`, on register values and stores the result once. The chain is
laid out as a straight-line body of `body_ops` FMAs inside a loop of
`reps = total_ops / body_ops` iterations. The number of dependent FMAs executed
is the same for every `body_ops` while the static code footprint changes.
Loop-control work is not the same: smaller bodies run more iterations. One
kernel, `fetch_kernel_<body_ops>`, exists per supported body size (16, 64, 256,
1024, 4096), so a launch fetches only the code of the size under test.

Each execution times one launch. Cold and primed times are paired within one
body size: the two entries differ in whether that kernel's code was fetched
before the timed launch, and their time difference is the cost of fetching the
code under this protocol.

## Entry policy

1. Run the optional vector-add clock/cache warmer.
2. Run `prime_launches` untimed launches of the same kernel with the same
   `reps` and coefficients, and wait for them to finish. A primer stores to its
   own output buffer, never to the timed launch's.
3. Time one launch with GPU events. No warmer runs between the primer and this
   launch.

`prime_launches=0` is the suite's default cold entry: the timed launch is the
first launch of that kernel in the process. `prime_launches=1` is an explicit
resident-code exception to cold entry, declared as a swept parameter so both
states are collected. Setting `BENCH_PARAM_vector_add_warmup=false` disables
only step 1; a requested primer still runs. Each repetition is a fresh process.
The warmer and primer are outside the GPU and host timing brackets.

`code_entry` in the output is `"cold"` or `"primed"` and repeats what
`prime_launches` requested. Neither label certifies a hardware state:

- **Cold** means the kernel has not run in this process. Where the uploaded
  code resides after the warmer (which cache level, if any) is not established
  for any target.
- **Primed** means one launch fetched the code. A separate launch is not
  guaranteed to run on the same SM/CU or to find its instruction cache intact.

## Deriving fetch cost

For one `body_ops`, with all other parameters equal:

```
fetch_cost(body_ops) = time_ms(prime_launches=0) − time_ms(prime_launches=1)
```

Fit `fetch_cost` against the measured code size of each kernel. The slope per
64-byte line is the per-line fetch cost; the intercept is the cost that does not
depend on code size. A slope that is not constant is itself a result (for
example prefetching or overlapped fetches), not a defect of the fit.

Use the minimum GPU time across repetitions for each of the two entries before
subtracting, as for any other benchmark in the suite. Small bodies produce
intervals of a few microseconds; qualify the event timer's resolution on the
target first (see [measurement interpretation](../../../docs/measurement_interpretation.md#empty-dispatch-and-timing-resolution)).

## Compiled code: size and layout

Code size must be measured from the compiled code, not estimated from
`body_ops`, and the layout must be checked: one loop, not unrolled, whose body
holds exactly `body_ops` FMA instructions. `#pragma unroll 1` requests that, but
only the disassembly shows what a compiler produced.

No size table or automated check exists yet for NVIDIA builds. Inspect the SASS
of each `fetch_kernel_<body_ops>` (for example with `cuobjdump -sass
code_fetch_latency`) and record its code bytes before using CUDA results. The
[HIP README](../hip/README.md#compiled-code-size-and-layout) reports the AMD
`gfx942` sizes and the script that measures them; those numbers do not transfer
to NVIDIA targets.

## Limits

- **Instruction-cache capacity.** When a kernel's code exceeds the instruction
  cache and `reps > 1`, later loop iterations refetch code even in the primed
  run, so the difference no longer isolates one fetch of the code. Exclude such
  sizes from the slope. Where the primed time rises with `body_ops` is evidence
  about that capacity, to be qualified separately.
- **One block.** The declared sweep launches one block of 64 threads. With
  more blocks, several SMs fetch the same code and can share lower cache
  levels. That is a different experiment and is not declared.
- **What "cold" contains.** Any first-use cost of the kernel that falls inside
  the GPU event bracket is in the cold time: runtime-side first-launch work for
  that kernel and address translation of its code, as well as the fetch itself.
  Costs that do not grow with code size land in the intercept; they are part of
  the measurement and are not separated here. The output store is not one of
  them: both output buffers are set up before the warmer in either entry and a
  primer stores to its own buffer, so the timed launch's output is equally
  untouched in both entries and its cost cancels in the difference.
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

## Build and run

```sh
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./code_fetch_latency     # default: body_ops=256, cold entry
./code_fetch_latency 4096   # positional body_ops
BENCH_PARAM_body_ops=1024 BENCH_PARAM_prime_launches=1 ./code_fetch_latency
```

Environment values take priority over the positional body size:

| Parameter | Default | Meaning |
|---|---:|---|
| `BENCH_PARAM_body_ops` | 256 | FMAs in the loop body: 16, 64, 256, 1024 or 4096 |
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
`python3 ../../../tests/code_fetch_latency_host.py --source "$PWD/code_fetch_latency.cu"`
checks those rules and the reference chain on the host.

Illustrative JSON for the primed 1024-FMA body (times are not performance claims):

```json
{"type":"kernel","name":"fetch_kernel_1024","iter":0,"time_ms":0.020000,"host_time_ms":0.060000,"params":{"body_ops":1024,"total_ops":16384,"reps":16,"prime_launches":1,"num_blocks":1,"threads_per_block":64,"code_entry":"primed"}}
{"type":"summary","total_time_ms":0.020000}
```
