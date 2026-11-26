# code_fetch_latency — what a kernel's code costs versus its size (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

Each process launches one kernel once. Every thread runs a straight-line body
of `body_ops` dependent operations on register values, `loops` times, in a loop
that is not unrolled, and stores the result once. One kernel exists per body
size and operation kind, so a launch fetches only the code of the size under
test.

Sweeping `body_ops` and `loops` separates three costs: the fixed cost of a
dispatch, the cost of an operation whose code has already been fetched once,
and the extra cost of the first pass over code that has not been executed yet.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `BENCH_PARAM_body_ops` | 256 | Operations in the loop body: a power of two from 16 to 32768. Selects the kernel and its code footprint |
| `BENCH_PARAM_loops` | 1 | Times the body is executed; at least 1, with `body_ops * loops` at most 1048576. The declared sweep uses 1, 4, 8 and 16 |
| `BENCH_PARAM_op` | fma | Body operation: `fma` or `xor_add` |
| `BENCH_PARAM_num_blocks` | 1 | Thread blocks |
| `BENCH_PARAM_threads_per_block` | 64 | Threads per block, at most 1024 |
| `BENCH_PARAM_vector_add_warmup` | true | Clock/cache warmer before the timed launch |

The two operation kinds:

- **`fma`**: `x = fmaf(x, a, b)` on `float`, one instruction per operation
  (kernels `fetch_kernel_<body_ops>`).
- **`xor_add`**: `x = (x ^ a) + b` on `unsigned`, two cheap instructions per
  operation (kernels `fetch_xor_add_kernel_<body_ops>`). It consumes code
  faster than the FMA body; see [Limits](#limits).

`a` and `b` reach the kernels as runtime arguments, so no compiler folds a
chain.

## Entry sequence

1. Query the selected kernel's attributes (`cudaFuncGetAttributes`). This
   requests that the runtime load the kernel without running it.
2. Run the optional vector-add clock/cache warmer.
3. Time one launch with GPU events. It is the first dispatch of that kernel in
   the process.

CUDA lazy loading defers loading a kernel's code to its first launch; without
step 1 that load would sit inside the timed bracket and grow with code size.
Whether the query fully pre-loads a kernel is runtime-specific.
`CUDA_MODULE_LOADING=EAGER` is an environment-level alternative. Each
repetition is a fresh process.

## Deriving the costs

For one `op`, with `T(N, R)` the time for `body_ops = N` and `loops = R`:

```
T(N, R) ≈ fixed dispatch cost + R · (loop control)
          + N · (first-pass cost per operation)
          + (R − 1) · N · (repeat-pass cost per operation)
```

- **Repeat-pass cost per operation** =
  `(T(N, R2) − T(N, R1)) / ((R2 − R1) · N)`: the cost of an operation whose
  code was already fetched once. The fixed cost and the first pass cancel. It
  rises when the body no longer fits an instruction-cache level.
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
  The dependent FMA is slow per code byte; the `xor_add` body exists to consume
  code faster. A flat repeat-pass cost therefore bounds the refetch cost; it
  does not prove an instruction cache holds the body.
- **Fetch and translation are not separated.** Code is contiguous, so the
  number of lines and the number of pages grow together. The first-pass extra
  holds whatever the first touch of the code costs.
- **What the first pass contains.** Any loading the attribute query leaves
  undone falls inside the timed launch, as does other first-use work of the
  kernel inside the event bracket.
- **One block.** The declared sweep launches one block of 64 threads. With
  more blocks, several SMs fetch the same code and can share lower cache
  levels. That is a different experiment and is not declared.

## Compiled code: size and layout

Code size must be measured from the compiled code, not estimated from
`body_ops`, and the layout must be checked: one loop, not unrolled, holding
exactly `body_ops` operations. `unroll 1` requests that, but only the
disassembly shows what a compiler produced.

No size table or automated check exists yet for NVIDIA builds. Inspect the SASS
of each kernel (for example with `cuobjdump -sass code_fetch_latency`) and
record its code bytes before using CUDA results. With nvcc 12.9 for `sm_90` one
`fma` operation is one 16-byte `FFMA` and one `xor_add` operation is two
16-byte instructions, so the two kinds differ in code bytes per operation. The
[HIP README](../hip/README.md#compiled-code-size-and-layout) reports the AMD
`gfx942` sizes and the script that measures them; those numbers do not transfer
to NVIDIA targets. It also explains why the sweep stops at 32768.

## Verification

After timing, every thread's stored value is compared with the same chain
computed on the host. The comparison is exact. For `fma` the per-thread seed
`1 + thread/1024` is exactly representable, each step is one correctly rounded
`fmaf` on both sides and every FMA changes the value; non-finite values are
rejected, and the buffer starts as NaN, so a thread that never stores fails.
For `xor_add` the chain is plain `uint32` arithmetic. A chain with a missing or
extra loop pass does not match. A mismatch exits nonzero.

## Timing, traces and simulation

Stdout contains one record named after the launched kernel and one summary
equal to it. `host_time_ms` spans event submission, launch, checking and
synchronization.

A process launches the benchmark kernel once, so a trace (collected with the
warmer off) holds that one launch and the suite's whole-trace simulation
report compares like with like. A simulator must start it without that
kernel's code in its instruction cache.

`scripts/simulation_run.py select` keeps one non-scaling combination, that is
one `loops` and one `op`, and six body sizes at evenly spaced positions of the
declared list: with twelve sizes, every second one when those are usable. It
prefers the defaults (`loops=1`, `op=fma`). One combination is a single
`T(N, R)` series, so a selection cannot give the repeat-pass cost or the
first-pass extra, which need at least two `loops` values. Simulate the full
sweep with
`python3 scripts/simulation_run.py run --benchmarks code_fetch_latency --all-params`.
No `params_selected.json` is committed for this benchmark.

## Build and run

```sh
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./code_fetch_latency     # default: body_ops=256, loops=1, op=fma
./code_fetch_latency 4096   # positional body_ops
BENCH_PARAM_body_ops=1024 BENCH_PARAM_loops=16 BENCH_PARAM_op=xor_add ./code_fetch_latency
```

Environment values take priority over the positional body size. Integer
parameters must be whole decimal numbers: empty, non-numeric and trailing-junk
values are rejected, not read as zero. Unsupported body sizes, a `loops` below
1 or with `body_ops * loops` above 1048576, an unknown `op`, and nonpositive or
oversized launch dimensions fail before any device access. The chain limit
keeps the FMA chain finite for every thread seed and the host reference cheap.
`python3 ../../../tests/code_fetch_latency_host.py --source "$PWD/code_fetch_latency.cu"`
checks those rules and the reference chains on the host.

Illustrative JSON (times are not performance claims):

```json
{"type":"kernel","name":"fetch_kernel_1024","iter":0,"time_ms":0.020000,"host_time_ms":0.060000,"params":{"body_ops":1024,"loops":4,"op":"fma","num_blocks":1,"threads_per_block":64}}
{"type":"summary","total_time_ms":0.020000}
```
