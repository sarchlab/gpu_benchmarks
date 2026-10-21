# polybench_3dconv — fixed PolyBench/GPU 3D stencil (native CUDA)

The benchmark applies the fixed PolyBench/GPU stencil to every interior point
of an N×N×N input volume. It runs one timed GPU kernel, then copies and verifies
the actual measured output on the CPU. Repetition is performed by launching a
fresh process (`gpubench run --reps N`).

## Provenance and operator

The operator is pinned to [PolyBench-ACC commit 70ea4ca](https://github.com/cavazos-lab/PolyBench-ACC/blob/70ea4ca9047dc4ec9c3ca79b4c91b3b2cc1510e7/CUDA/stencils/convolution-3d/3DConvolution.cu).
Both its `conv3D` CPU reference and `convolution3D_kernel` GPU implementation
contain 15 ordered terms with the same coefficients and coordinates. The
repeated coordinates in the k−1 plane are intentional here because they occur
in both pinned implementations; they are not replaced by a symmetric filter.
Their combined spatial weights are:

| Offset (di, dj, dk) | Combined coefficient |
|---|---:|
| (−1, −1, −1) | −1 (= 2 + 5 − 8) |
| (+1, −1, −1) | 21 (= 4 + 7 + 10) |
| (0, −1, 0) | −3 |
| (0, 0, 0) | 6 |
| (0, +1, 0) | −9 |
| (−1, −1, +1) | 2 |
| (+1, −1, +1) | 4 |
| (−1, 0, +1) | 5 |
| (+1, 0, +1) | 7 |
| (−1, +1, +1) | −8 |
| (+1, +1, +1) | 10 |

All other spatial weights are zero. The GPU expression retains all 15 terms
and their order; the CPU verifier independently uses the combined weights.
The source expression has 15 multiplies and 14 additions/subtractions per
interior output: nominal work is `29 * (N-2)^3` FLOPs. This is a source model;
compiled instructions and physical memory traffic require target qualification.

The implementation uses one cubic 3D thread grid instead of the upstream's
sequence of independent 2D slice launches. There are no dependencies between
slices. Input generation retains the existing `srand(42)` and
`(rand()%100)/100.0f` sequence, rather than the upstream coordinate-based
initializer; this is a synthetic dataset, and its cross-libc identity remains
subject to input provenance. Output borders are explicitly initialized to zero
before timing and left untouched by the kernel. The upstream only defines and
compares interior outputs; its uninitialized output borders are not reproduced.

Earlier revisions computed a different dense, normalized random-filter
convolution at filter sizes 3, 5 and 7. Those measurements are not comparable
with this fixed stencil and must be recollected. `filter_size` is removed from
the manifest; its old environment variable and command-line option now fail
explicitly, including value 3. Fixed `operator=polybench_gpu_70ea4ca` metadata
identifies the restored operator without declaring an inert execution control.

## Parameters and measurement

| Parameter | Default | Declared sweep / behavior |
|---|---:|---|
| `BENCH_PARAM_size` / `--size` | 128 | 25 volume dimensions from 16 to 512; N≥3 and N³ must fit signed integer indexing |
| `BENCH_PARAM_block_size` / `--block_size` | 8 | Cubic block side; manifest fixes 8 (512 threads), custom positive values must satisfy the selected device's launch limits |
| `BENCH_PARAM_vector_add_warmup` | true | Fixed per harness run; optional untimed warmer before the measured stencil |

The declared sweep now has 25 configurations, rather than 75 combinations
with the old filter-size parameter. Environment variables override command-line
values. Invalid values fail before a timing record is emitted.

Host allocation, input initialization, initial upload and output zeroing occur
before the optional warmer. The warmer is intended to raise clocks and disturb
cache state; qualify its achieved state on the target. One event-timed
`conv3d_kernel` launch follows. The output copy and CPU verification occur after
the timing bracket. No verification kernel or recomputation replaces the measured
output. GPU event time is the measured operation's time; host launch-bracket time
also excludes the subsequent verifier.

## Verification and cost

Every interior value is compared with an independent double-precision CPU
reference. Every output border must remain exactly zero. Non-finite output,
reference or error scale fails verification, and the process exits nonzero.
The fixed `verification=full_cpu_v1` policy is recorded in kernel parameters.
It is always enabled.

The interior tolerance is `2e-6 * max(1, sum(abs(source_term)))`, with the
absolute weights of all 15 terms retained even where coordinates repeat. This
allows float accumulation rounding near cancellation while bounding the error;
it does not divide by a near-zero reference. Qualify the actual error/tolerance
ratios on each target.

Verification costs O(N³) CPU work and a host output buffer of `4*N³` bytes
(512 MiB at N=512). The program has two host and two device volume buffers in
total, plus the optional warmer's device scratch. These allocations and the
complete process time must fit the target's capacity and collection timeout.
The post-timing CPU cost does not change an already measured GPU interval.

## Build, run and test

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./polybench_3dconv
BENCH_PARAM_size=256 ./polybench_3dconv
make test   # actual GPU required
```

The GPU regression uses all 27 spatial basis inputs, signed volumes, partial
blocks, untouched borders and allocation guards. Exactly representable test
inputs permit exact comparisons. The finite-injection suite separately exercises
the production verifier, including the last measured interior output.

Stdout contains one kernel record and its matching summary (illustrative times):

```json
{"type":"kernel","name":"conv3d_kernel","iter":0,"time_ms":0.123456,"host_time_ms":0.145678,"params":{"size":128,"block_size":8,"operator":"polybench_gpu_70ea4ca","verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":0.123456}
```

Device information and `PASS` or `FAIL` diagnostics go to stderr. A verification
failure may follow valid timing records; the nonzero process exit makes the
complete run an error. Derive rates downstream from successful complete runs.
