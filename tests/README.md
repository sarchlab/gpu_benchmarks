# GPU regression checks

`finite_verification.py` exercises the actual verification paths of 27 Tier 2
benchmarks. It makes isolated source copies, injects values immediately before
the existing comparisons, and builds those copies with the benchmark Makefiles.
The production sources and executables receive no injection interface.

```sh
python3 tests/finite_verification.py --platform cuda --work-dir /tmp/finite-cuda
python3 tests/finite_verification.py --platform rocm --work-dir /tmp/finite-rocm
```

Use a new output directory for each run. Select a subset by repeating
`--benchmark NAME`. `--source PATH` selects a checkout for before/after checks.
`--compile-only` checks the injection anchors and builds the test executables
without claiming GPU verification; CI uses this mode because its runner has no
GPU. Runtime checks require the selected compiler and a compatible GPU.

Each compared output/reference pair receives a single NaN, positive infinity,
and negative infinity, independently. An incorrect finite output control affects
all compared samples so it exceeds Barnes–Hut's existing approximation allowance.
A normal finite run must still pass. FFT checks both float and double precision;
Parboil SpMV checks its host self-test and actual GPU result. Complex/vector
comparisons cover each compared component independently. CG/MG injection targets
the reduced residual and reference norms that their verifiers actually compare.

Failure requires the hook to have executed, a verification failure diagnostic,
and exit status 1, with no HIP runtime error. A crash, timeout, missing hook,
or later cleanup error cannot satisfy the test. A timeout aborts the run; the
last completed benchmark report may therefore be partial.
Logs and `report.json` retain each result and the original source hash. The
minimum declared scaling value and existing non-scaling defaults are used, with
verification enabled and the clock/cache warmer disabled to keep these functional
checks short. These are verification tests, not performance measurements.

The R-04 audit fixes finite comparisons and error propagation, preserving existing
tolerances, sampled regions, and measured work. It also checks for non-finite
intermediate error metrics where normalization or max accumulation could hide
invalid values. One non-finite Barnes–Hut sample fails regardless of the quota
for finite approximation errors. SHOC reduction uses the same pass condition for
its diagnostic and exit status.

The remaining existing floating-point predicates in Rodinia LUD/Gaussian,
NPB EP's reference comparison, and Altis
particle filter use positive acceptance comparisons; those comparisons reject
NaN. Parboil LBM checks NaN and density bounds. Integer/byte verifiers have no
non-finite values. Missing verification, unchecked output components, unverified
NPB EP sizes, and verification on a separate small workload remain coverage
issues under R-13; this test does not establish their correctness.

CUDA execution and ROCm execution must be reported separately. A CUDA pass or
ROCm compile alone does not establish AMD runtime qualification (R-19).

`cache_latency_order.py` checks the actual warmer → untimed primer → timed chase
sequence and CUDA event boundaries using Nsight Systems. It requires an NVIDIA
GPU, `nsys`, a built `cache_latency` binary, and a new output directory:

```sh
python3 tests/cache_latency_order.py \
  --binary tier1/cache_latency/cache_latency --work-dir /tmp/cache-order
```

The four cases cover warmer on/off, a small chain, and the primer-cap boundary.
Captures, profiler logs, and `report.json` are retained. A nonzero profiler/export
exit is recorded as a failed case; a timeout aborts the run and may leave a
partial report. CI does not run this GPU check. Profiled times establish ordering,
not performance or a physical cache hit rate. See
[the benchmark documentation](../tier1/cache_latency/README.md) for interpretation
and trace-replay requirements.

`shuffle_output_test.hip` checks every output lane of the production Tier 1
shuffle kernels against closed-form lane origins, including odd/even operation
counts, both wave boundaries, multiple blocks and guarded output allocations.
Run `make -C tier1/warp_shuffle PLATFORM=cuda test` (or `PLATFORM=rocm`) on a
compatible GPU. It uses the selected device's native width. CI compiles the test;
CUDA and AMD native wave32/wave64 runtime qualification must be recorded
separately. Correct outputs do not replace target instruction-count checks.

`two_mm_output_test.hip` calls PolyBench 2MM's complete production pass and checks
both matrix results against independent double products, pinned upstream square
inputs, and exact rank-one references through N=4096. Guarded output and a
NaN-poisoned initial temporary detect missing writes or an invalid initial
contribution. Run `make -C tier2/polybench_2mm PLATFORM=cuda test` (or `rocm`), or
run `bench_output_test --rank-one N` for an individual declared size. CI compiles
the test; GPU runtime qualification is separate. This checks the restored R-09
expression and does not claim that ordinary timing runs perform verification
(R-13).

`bh_entry_order.py` checks actual Barnes–Hut CUDA timelines: initial body/tree
transfers before the warmer, all force/integration timestep launches with their
event brackets, and the existing separate verification launch. It requires an
NVIDIA GPU, `nsys`, a built binary and a new directory. Run
`python3 tests/bh_entry_order.py --binary tier2/lonestar_bh/lonestar_bh --work-dir /tmp/bh-entry`.
Assertions must be enabled; optimized Python is rejected. Captures and reports
are retained; timeouts abort and may leave a partial report. CI does not run this
functional GPU check, and its profiled timings are not performance results.

`algorithmic_zero_output.hip` tests the actual Gram–Schmidt and MG reset kernels
with exact positive-zero and guard checks. Run `make test PLATFORM=cuda` (or
`rocm`) from each benchmark directory. CI builds both test variants.

`algorithmic_zero_order.py` requires an NVIDIA GPU, `nsys`, both built benchmarks
and a new output directory:

```sh
python3 tests/algorithmic_zero_order.py \
  --gram-binary tier2/polybench_gramschmidt/polybench_gramschmidt \
  --mg-binary tier2/npb_mg/npb_mg --work-dir /tmp/reset-order
```

It checks the full timed kernel sequence (including resets), individual event
brackets, absence of algorithmic runtime memsets/copies, verification extras
and all records/summary totals. Eight captures cover small/default cases and
warmer on/off. Run with assertions enabled; optimized Python is rejected.
Ordinary capture/export failures are recorded; timeouts or missing tools abort
and may leave partial reports. Captures are retained. CI does not execute this
GPU check, and its profiled times are not performance results.

`backprop_output_test.hip` calls the real Rodinia backprop hybrid epoch with
non-saturating signed inputs, nonzero momentum and pinned initialization. It
compares every input/output weight and momentum value, activation and delta
against an independent dense double reference. Run `make test PLATFORM=cuda`
(or `rocm`) from `tier2/rodinia_backprop`; `bench_output_test --upstream N`
checks an individual declared size. CI compiles the test; GPU execution and
ordinary timing-run verification are separate qualification steps.

`two_d_convolution_output.hip` checks the pinned signed PolyBench/GPU operator
with nine exact unit-impulse outputs and full signed-matrix references, including
rectangular strides, partial tiles, NaN-poisoned interiors and untouched borders
and guards. Run `make test PLATFORM=cuda` (or `rocm`) in `tier2/polybench_2dconv`;
`bench_output_test --size N` checks a declared size, and `--impulse 0` through
`--impulse 8` isolate the nine coefficients. CI compiles the test; runtime and
ordinary timing-run verification remain separate qualification steps.
