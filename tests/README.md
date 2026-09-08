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

`backprop_output_test.hip` also exercises the production measured-epoch verifier
on every test state while retaining its independent dense reference. Ordinary
runs verify all retained forward partials and the complete updated state after
timing; finite-injection tests cover the comparator and tail partial, weight,
momentum and output/reference values.

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
`--impulse 8` isolate the nine coefficients. Ordinary timing runs also check
every measured output after timing under fixed policy `full_cpu_v1`.
`finite_verification.py --benchmark polybench_2dconv` injects non-finite and
incorrect finite values into that comparison, including the last interior
output and original reference input. CI compiles these tests; GPU execution
requires target qualification.

`stencil_output_test.hip` exercises the actual Parboil production timestep
sequence against an independent dense reference: exact center/neighbor
impulses, signed rectangular grids, partial tiles, odd/even/100-step sequences,
NaN-poisoned first destinations, both buffers' boundaries and allocation guards.
Run `make test PLATFORM=cuda` (or `rocm`) in `tier2/parboil_stencil`, or
`BENCH_PARAM_vector_add_warmup=false ./bench_output_test --case N T` for a
particular declared configuration. CI compiles the test; runtime execution and
ordinary timing-run verification remain separate qualification steps.

`common_header_dependencies.py` checks the real Make rebuild decisions for
all Tier 1/2 benchmark targets and their `bench_output_test` targets under
both `PLATFORM=cuda` and `PLATFORM=rocm`. In temporary source copies, timestamp
fixtures must stay current with unchanged inputs and become stale when each
shared timing, warmer, or compatibility header changes. It executes `make -q`
without running compiler recipes. Run `python3 tests/common_header_dependencies.py`;
`--report PATH` preserves every decision and `--repo PATH` selects a checkout.
CI runs this host-only regression. Target compilation and GPU execution require
separate toolchains and hardware qualification.

### Fixed 3D convolution output

`make -C tier2/polybench_3dconv PLATFORM=cuda test` (or `PLATFORM=rocm`)
requires an actual GPU. `three_d_convolution_output.hip` includes the production
kernel and compares all 27 spatial basis inputs and 40 signed-volume/block
combinations with an independent spatial-weight reference. Exactly representable
inputs permit exact comparison; NaN destination interiors, untouched border
sentinels and allocation guards detect missing or out-of-region writes.
`--case N B` selects a cubic volume/block side; `--impulse 0..26` isolates one
basis input. The production executable also checks every actual measured output
after timing. The existing finite-injection harness covers that comparator and
the last interior output/reference input, while CI only compiles these GPU tests.

`cedd_output_test.hip` checks every output byte of all four actual Chai CEDD
kernels with independent integer/double stage references, allocation guards,
unchanged borders, partial tiles, constant images, ramps and alternating pixels.
It also executes the real timed production pass and its full-output verifier.
An exhaustive test covers all 2,082,841 same-parity integer Sobel gradient pairs in
[-1020,1020]² against the production float quantization helper. Opposite-parity
pairs cannot come from a byte-valued Sobel patch: both sums have the same four
corner values modulo two. Some impossible pairs cross a float angular bin
boundary differently from the ideal double reference; these are excluded by
the proved stencil invariant, not by an output mismatch allowance. Run
`make -C tier2/chai_cedd PLATFORM=cuda test` (or `PLATFORM=rocm`).
`bench_output_test --case W H` checks a chosen shape. Byte outputs have no
NaN/Inf representation; corruption tests should alter actual output bytes,
including the last interior pixel and preserved border, after timing.

`matvec_reference_test.cpp` checks the production CPU reference helpers for
ATAX, BiCG and MVT without a GPU or GPU compiler. Rectangular and square cases,
including dimensions below/equal/above eight, compare every output value
bit-for-bit with complete float references and protect the output boundaries.
Run `c++ -O2 -std=c++14 -fsanitize=address,undefined tests/matvec_reference_test.cpp
-o /tmp/matvec_reference_test && /tmp/matvec_reference_test` from the repository
root. CI runs this test under address/undefined-behavior sanitizers. The GPU
regression below separately checks the actual measured device outputs.

`lud_measured_result.py` checks that ordinary LUD runs factor exactly once and
verify the matrix produced by that measured factorization. It injects faults
into the actual device result after the timed pass, and requires failed exit
status while preserving the timing records. Valid cases and manual bypasses
check the emitted verification policy. Run `python3 tests/lud_measured_result.py
--platform cuda --work-dir /tmp/lud-check` (or `rocm`). Choose a new, nonexistent
`--work-dir` for each invocation; previous results are retained for inspection.
CI compiles the instrumented driver; runtime qualification requires a GPU.
The checker retains the full O(N³) residual; target process budgets still
require qualification even with its tiled CPU reconstruction.

`triangle_total_verification.py` checks exact raw triangle-count acceptance in
the production driver. It alters actual device edge counts after timing and
requires nonzero exit for every signed nonzero raw-total error, including the
+1/+2 cases formerly hidden by integer division and negative errors for a
zero-triangle graph. Run `python3
tests/triangle_total_verification.py --platform cuda --work-dir /tmp/tc-check`
(or `rocm`) with a new, nonexistent output directory. CI compiles the test;
execution requires a GPU. A timeout aborts with failure and retains completed
case logs. Target process budgets and AMD qualification remain separate.

`lud_residual_test.cpp` checks the full CPU LU residual against the original
unblocked formula. It compares every reconstructed double and the final
residual bit-for-bit for finite values, preserves non-finite classification
and pass/fail behavior, and checks partial tile boundaries, a known LU product,
near-threshold residuals and output guards. Run `c++ -O3 -std=c++14
-fsanitize=address,undefined tests/lud_residual_test.cpp -o /tmp/lud_residual_test
&& /tmp/lud_residual_test`. CI runs this host-only test under sanitizers. The
GPU regression above separately checks the actual measured device output.

`triangle_host_test.cpp` checks complete generated CSR arrays and subsequent
RNG values against the original `std::set` generator, and compares the full raw
triangle total with both the original reference and an independent cubic oracle.
It exhausts all simple undirected graphs through six vertices, checks sparse
large vertex IDs, and exercises the compact edge table's collisions, wraparound,
duplicate handling, packing and capacity bounds. Run `c++ -O3 -std=c++14
-fsanitize=address,undefined tests/triangle_host_test.cpp -o /tmp/triangle_host_test
&& /tmp/triangle_host_test`. CI runs this host-only check under sanitizers; the
actual-device regression above separately checks measured GPU results.

`matvec_measured_result.py` injects faults into actual ATAX, BiCG and MVT
device outputs after timing and before the existing result download. It checks
finite corruption and NaN/+Inf/-Inf at the first element, indices 7 and 8, the
middle and the last element of every output vector, plus small/rectangular and
warmer controls. Unequal row/column shapes also receive tail faults to catch
swapped output bounds. Run `python3 tests/matvec_measured_result.py --platform cuda
--work-dir /tmp/matvec-check` (or `rocm`) with a new, nonexistent output
directory. CI compiles the test; execution requires a GPU. A timeout fails the
test and retains completed-case logs and a partial report. The test requires
`full_cpu_v1` in both kernel records and checks exact workload identity, normal
verification diagnostics and failure exit codes.


`bfs_reference_test.cpp` compares every CPU distance with an independent
Floyd-Warshall oracle on 66,066 exhaustive directed graphs and 25 structured
cases, including self-loops, duplicates, deep chains and unreachable nodes.
Run `c++ -O2 -std=c++14 -fsanitize=address,undefined tests/bfs_reference_test.cpp
-o /tmp/bfs_reference_test && /tmp/bfs_reference_test`. CI runs this host test.

`bfs_measured_result.py` checks measured BFS distances on random and test-only
structured graphs, then injects device-output faults at the source, middle and
tail. It requires exact result diagnostics, failure exits and `full_cpu_v1` in
both kernel rows. Run `python3 tests/bfs_measured_result.py --platform cuda
--work-dir /tmp/bfs-check` (or `rocm`) with a new, nonexistent output directory.
CI compiles this test; execution requires a GPU. A timeout fails and retains
completed-case logs plus a partial report. Result correctness does not qualify
the existing GPU concurrent accesses or upstream fidelity.

## Parboil stencil measured-output verification

```sh
c++ -O2 -std=c++14 -fsanitize=address,undefined tests/stencil_reference_test.cpp -o /tmp/stencil-reference
/tmp/stencil-reference
python3 tests/stencil_measured_result.py --platform cuda --work-dir /tmp/stencil-measured-result
```

The host test checks 107 full reference states against a dense matrix operator,
input preservation, exact boundaries and finite/tolerance predicates. The GPU
regression runs 212 ordinary-path cases (16 controls, 196 faults) using fresh
source copies and a fault hook on the final measured device buffer. It checks
full-grid rejection, one-ULP boundary faults, parity, non-finite outputs, and
all mismatch counts with at most ten individual diagnostics. JSON records must
retain finite times and `verification=full_cpu_v1`; faulty runs must exit 1.
`--work-dir` must be new; reports and process output are retained there.
Use `--platform rocm` for AMD or `--compile-only` when no GPU is available.
The independent production-pass oracle remains `tests/stencil_output_test.hip`.

## 2MM measured intermediate and final products

```sh
c++ -O3 -std=c++14 -fsanitize=address,undefined tests/two_mm_reference_test.cpp -o /tmp/two-mm-reference
/tmp/two-mm-reference
python3 tests/two_mm_measured_result.py --platform cuda --work-dir /tmp/two-mm-measured-result
```

The host oracle compares 168 complete matrix pairs bit for bit against the
qualified unblocked double products, with input preservation and finite/tolerance
checks. The GPU regression runs 158 ordinary-path cases (53 controls and 105
faults), injecting corruption in either or both actual measured device matrices
after timing. It checks first/prefix/middle/tail elements, partial tiles,
finite/NaN/±Inf values, zero/negative coefficients and complete mismatch counts
with capped diagnostics. Both kernel JSON records must retain finite times and
`verification=full_cpu_v1`; faults must exit 1. The output directory must be new;
reports and process output remain there. CI runs the host oracle and compiles
the GPU regression. Use `--compile-only` without a GPU or `--platform rocm` on
AMD. The existing `two_mm_output_test.hip` keeps its independent oracle.

## SHOC Stencil2D measured final state

```sh
c++ -O3 -std=c++14 -fsanitize=address,undefined tests/shoc_stencil_reference_test.cpp -o /tmp/shoc-stencil-reference
/tmp/shoc-stencil-reference
python3 tests/shoc_stencil_measured_result.py --platform cuda --work-dir /tmp/shoc-stencil-measured-result
python3 tests/finite_verification.py --platform cuda --benchmark shoc_stencil2d --work-dir /tmp/shoc-stencil-finite
```

The host oracle compares 240 complete float-reference states bit for bit,
checks input/boundary preservation and isolates all nine coefficients with
exact one-step impulses. The GPU regression checks 210 ordinary-path cases
(16 controls, 194 faults), injecting finite/NaN/±Inf errors into the actual
final measured buffer across interior/boundary positions and timestep parity.
It checks exact boundary rejection, complete mismatch counts with ten-line
diagnostic caps, normal failure exits, finite JSON times and `full_cpu_v1`.
The existing finite-value regression also checks the new actual-result
comparator's CPU-reference rejection. Output directories must be new; logs and
reports remain there. CI executes the host test and compiles the GPU tests.
Use `--platform rocm` for AMD or `--compile-only` without GPU execution.
