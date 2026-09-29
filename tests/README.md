# GPU regression checks

The CI CPU-reference builds and Python host sanitizer checks use
`-fno-sanitize-recover=all`, so detected sanitizer errors fail the check. Include
this flag when compiling a sanitized host test manually.

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
comparisons cover each compared component independently. CG injection covers all
five actual/reference vectors and both outer histories; MG covers the full
hierarchy solution/residual, input and initial/final norm arrays.

`python3 tests/nbody_reference.py --compiler clang++` checks the production
N-body comparator and CPU recurrence with 59 analytic/boundary checks, plus
54 production parsing/index cases under fatal host sanitizers.
`python3 tests/nbody_measured_result.py --platform cuda --work-dir /tmp/nbody-measured`
uses a new directory and runs 203 actual measured-state cases: ten controls,
192 GPU-buffer faults spanning all position/velocity components and three body
locations, and one unavailable-reference case. It covers odd/even timestep
buffers and bodies beyond the former subset. CI compiles the hooks only.
`nbody_h100_profile.py` builds an isolated H100 binary, qualifies a bounded
configuration list with complete checks and the 60-second process limit, records
all exclusions, and generates workspaces containing only admitted combinations
for the existing harness. `python3 tests/nbody_profile_records.py` checks 28
qualification classifications, including malformed timing/verification evidence,
explicit completed-process overruns and unexpected failures. See the [N-body README](../tier2/cuda_nbody/README.md).

`python3 tests/h100_profile_identity.py` checks that the committed N-body and
Particle Filter H100 profiles still match the current benchmark sources,
headers, Makefiles, manifests and preserved licenses. Both existing profile
record tests also run this check in CI. Each current candidate must have exactly
one classification, with consistent qualified/unsupported counts.
`python3 tests/h100_profile_identity_test.py` exercises isolated source/hash
corruption, missing or new helpers, shared-header changes and changed candidate
coverage, including attempts to hide a changed manifest by refreshing its hash.
Changes to any of the three common headers invalidate both profiles: run each
benchmark's existing H100 qualification procedure and commit its fresh evidence.
Never update only a hash to make this check pass. Documentation is not a runtime
input. This check does not rerun the GPU, authenticate a claimed qualification,
or establish that a locally rebuilt binary or different toolchain/GPU is covered;
the documented requirement to use the exact qualified collection binary remains.

`python3 tests/shoc_spmv_verification.py --host-only --work-dir /tmp/spmv-host`
checks the actual CPU SpMV reference against analytical CSR examples with fatal
host sanitizers. Use `--platform cuda --work-dir /tmp/spmv-gpu --sweep` for
production controls, actual post-timing device faults (including the former
32-row cutoff and final row), skipped-work rejection and all 81 production
manifest combinations. Every GPU process must finish within 60 seconds. CI
compiles these GPU tests; their execution requires a GPU. Raw output, source and
binary hashes and per-case outcomes remain in the new work directory.

`python3 tests/transpose_input_ranges.py --compiler clang++ --work-dir /tmp/transpose-input`
checks the actual production parser, positivity and signed flat-index bounds,
and buffer arithmetic with fatal host sanitizers. It covers all declared sizes
and both sides of the 46340 boundary without allocating large matrices. Add
`--binary /path/to/cuda_transpose` to require exact invalid-input errors with
GPUs hidden; CI runs this path too. Large manual sizes still need memory/time
qualification.

`python3 tests/shoc_sort_input_ranges.py --compiler clang++ --work-dir /tmp/sort-input`
checks the actual production parser, nonpositive-input rejection, existing size
rounding and wide buffer-byte arithmetic with fatal host sanitizers. Cases cross
the former 4 GiB wrap boundary without allocating large arrays. Add `--binary
/path/to/shoc_sort` to require exact invalid-input errors with GPUs hidden; CI
runs this path too. Memory/time qualification of large manual sizes is separate.

`python3 tests/shoc_sort_verification.py --host-only --compiler clang++ --work-dir /tmp/sort-host`
checks the production sort verifier against an independent complete pair-multiset
and key-order oracle, including all small permutations with ties/extreme keys,
invalid indices, wrong associations, bounded diagnostics, unavailable allocation
and immutable input/output guards under fatal host sanitizers. Use `--platform
cuda --sweep --work-dir /tmp/sort-gpu` for actual measured-device faults, skipped
sorting and both new allocation failures, plus all 13 declared sizes and both
warmer modes. Each GPU process has a 60-second budget. Use a new directory for
raw logs, source/binary hashes and outcomes. CI executes host fixtures and
compiles GPU variants; execution requires hardware.

`python3 tests/transpose_verification.py --host-only --compiler clang++ --work-dir /tmp/transpose-host`
checks the production CPU transpose against a known answer and independent
rectangular mappings, bit preservation, inverse transposes and memory guards
under fatal host sanitizers. Use `--platform cuda --sweep --work-dir
/tmp/transpose-gpu` for faults in each actual measured variant beyond the former
1000-element cutoff, nonfinite results, skipped launches, partial tiles and all
22 declared sizes with both warmer modes. Every process has a 60-second budget.
Use a new work directory to retain logs, hashes and outcomes. CI executes host
fixtures and compiles GPU variants; execution requires hardware.

`python3 tests/gramschmidt_input_ranges.py --compiler clang++ --work-dir /tmp/gram-input-host`
checks the actual parser, resolved dimension guards, storage sizes and widened
launch geometry with fatal host sanitizers, including signed-index boundaries
without allocating large matrices. Add `--binary /path/to/polybench_gramschmidt`
to require exact invalid-input diagnostics and no output from the actual binary
with GPUs hidden. CI runs both paths. These are arithmetic-safety checks, not
memory-capacity or arbitrary-override performance qualification.

`python3 tests/gramschmidt_q_verification.py --host-only --compiler clang++ --work-dir /tmp/gram-q-host`
exercises the production Q orthogonality checker with independent known
orthonormal fixtures, element corruption, duplicate columns, nonfinite values
and immutable inputs/guards under fatal sanitizers. Host fixtures pin the existing
inclusive `0.01f` criterion with boundary and adjacent-float off-diagonal cases,
check positive/negative norm errors, and require every fixture's complete error
count, N² pair total and ten-message cap. Use
`--platform cuda --sweep --work-dir /tmp/gram-q-gpu` for actual measured-Q
corruptions beyond the former eight-column cutoff, skipped normalization,
bounded diagnostics and all 20 declared sizes with both warmer modes. Each GPU
process has a 60-second limit. Use new directories to retain raw logs, hashes
and outcomes. CI runs host checks and compiles GPU variants; their execution
requires hardware. This checks Q orthogonality only; full QR reconstruction
and final A workspace verification remain separate.

`python3 tests/sssp_verification.py --host-only --compiler clang++ --work-dir /tmp/sssp-host`
checks the actual CPU Bellman-Ford reference against 32,768 independent
Floyd-Warshall cases and a weighted parallel/self-edge/unreachable fixture with
fatal host sanitizers. Use `--platform cuda --sweep --work-dir /tmp/sssp-gpu`
for actual measured-distance faults beyond the former 1024-vertex cutoff,
skipped-work rejection, bounded diagnostics, both warmer modes and all 81
declared vertex/degree configurations. CI runs host fixtures and compiles GPU
variants; execution requires hardware. Use a new work directory to retain raw
logs, source/binary hashes and outcomes. Each GPU process has a 60-second limit.
These checks cover final distances; SSSP concurrency and deterministic work
remain separate repairs.

`python3 tests/convolution_verification.py --host-only --work-dir /tmp/conv-host`
checks the production separable CPU reference against asymmetric known answers,
direct 2D fixtures, zero-padding boundaries, immutable inputs and output guards
with fatal host sanitizers. Use `--platform cuda --sweep --work-dir /tmp/conv-gpu`
to check both measured GPU stages, actual post-timing finite/NaN faults beyond
the former 512×512 region, partial tiles, both warmer modes and unavailable
references. The sweep adds all 352 declared width/height/radius combinations
with the warmer enabled. CI runs host checks and compiles GPU variants; their
execution requires hardware. Use a new nonexistent work directory; raw logs,
source/binary hashes and per-case outcomes are retained. Each GPU process must
finish within 60 seconds.

`python3 tests/parboil_sgemm_verification.py --host-only --work-dir /tmp/parboil-sgemm-host`
checks the production FP32 reference against the original ordered recipe,
independent rectangular/zero/identity results, nonzero initial C and beta, and
automatic worker selection at the 512² output threshold. Fatal host sanitizers
also exercise failed thread creation and memory guards. Use `--platform cuda
--work-dir /tmp/parboil-sgemm-gpu --sweep` for actual measured finite/NaN faults,
partial tiles, both warmers, unavailable references and all 22 declared sizes.
CI executes host checks and compiles the GPU variants. Runtime GPU checks require
hardware, enforce a 60-second process budget and retain raw logs and source/binary
hashes in a new nonexistent work directory.

`python3 tests/shoc_gemm_verification.py --host-only --work-dir /tmp/shoc-gemm-host`
checks complete bitwise agreement with the original FP32 reference, independent
rectangular known answers, zero/identity cases, guards and input preservation.
Default worker selection is exercised below and at the 512² output threshold.
Fatal host sanitizers also exercise zero and partial worker-creation failure.
Use `--platform cuda --work-dir /tmp/shoc-gemm-gpu --sweep` for actual measured
device-tail finite/NaN faults, partial tiles, bounded diagnostics, unavailable
references, both warmers and all 22 declared sizes. CI runs host checks and
compiles GPU variants; execution requires hardware. Use a new nonexistent work
directory. Logs and source/binary hashes are retained; each GPU process has a
60-second budget.

`python3 tests/fir_verification.py --host-only --work-dir /tmp/fir-host`
checks the production CPU FIR against exact startup and signed-coefficient cases,
complete impulse responses beyond the former 1024-sample cutoff, and output
guards with fatal host sanitizers. Use `--platform cuda --work-dir /tmp/fir-gpu
--sweep` for actual device-output finite/nonfinite faults, partial blocks, all tap
counts, bounded diagnostics and all 108 production configurations. CI runs the
host check and compiles both GPU variants; execution requires hardware. Every
GPU process has a 60-second budget, with raw output and source/binary hashes
retained in the new work directory.

`python3 tests/cutcp_concurrency.py --platform cuda --work-dir /tmp/cutcp-race
--sweep` runs Compute Sanitizer against actual CUTCP kernels, evaluates the
production CPU potential at every lattice point in small cases, and runs
all 120 declared production configurations. An optional `--baseline-source`
pre-repair `.hip` file also reproduces its two racecheck failures and compares
seven complete baseline/current lattices. The current production verifier
checks every point; the historical baseline checks 64. CI compiles the variants;
runtime checks require a CUDA GPU, retain hashes/logs and enforce 60 seconds per
process. Temporary full-array dumps are removed after comparison.

`python3 tests/aes_verification.py --host-only --work-dir /tmp/aes-host` checks
the unchanged CPU cipher and key expansion against all four NIST SP 800-38A
F.1.5 AES-256 vectors, with fatal sanitizers and buffer/key guards. Use
`--platform cuda --work-dir /tmp/aes-gpu --sweep` for actual GPU known answers,
partial blocks, measured ciphertext faults beyond the former 256-block cutoff,
all 16 byte positions, bounded diagnostics, both warmers and all 27 declared
configurations. CI runs the host checks and compiles both GPU variants. Runtime
checks retain raw output and source/binary hashes with a 60-second process limit.

`python3 tests/graph_pr_verification.py --host-only --work-dir /tmp/pr-host`
checks the production PageRank reference against exact trajectories, duplicated
edges, a single vertex and total mass with fatal host sanitizers. Use `--platform
cuda --work-dir /tmp/pr-gpu --sweep` for balanced measured-rank faults (which
preserve total mass), nonfinite ranks, odd/even final-buffer selection and all 81
production configurations. The existing numerical criterion is unchanged by
this coverage check. CI runs the host check and compiles the GPU variants; actual
GPU runs retain raw output and source/binary hashes with a 60-second process limit.

`python3 tests/graph_cc_verification.py --host-only --work-dir /tmp/cc-host`
checks the production CPU label propagation against exact capped and converged
trajectories, including disconnected and isolated vertices. Use `--platform cuda
--work-dir /tmp/cc-gpu --sweep` for actual device-label faults, odd-step final-buffer
faults and all 69 production configurations. CPU and GPU use the same existing
Jacobi termination rule; the regression does not silently replace capped runs
with converged answers. CI runs the fatal-sanitizer host check and compiles the
GPU tests. GPU execution requires hardware; each test process has a 60-second
budget and retained raw output and source/binary hashes.

`python3 tests/parboil_spmv_controls.py --compiler clang++` checks SpMV’s
resolved seed, requested/effective launch geometry and JSON identity without a
GPU. Its 72 cases cover CLI/environment precedence and simulated device
geometry branches under host sanitizers. CI runs the test with g++; simulated
properties do not qualify hardware support for arbitrary block sizes.
Missing or duplicated extraction anchors report the source file and anchor.

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

`python3 tests/cache_bandwidth_geometry.py --compiler clang++` checks 230
production geometry and JSON cases under fatal host sanitizers, including the
private allocation cap and integer boundaries without allocating large buffers.
`python3 tests/cache_bandwidth_gpu.py --work-dir /tmp/cache-bandwidth-gpu`
requires a new output directory and CUDA GPU. It checks 96 complete kernel
outputs with distinct, nonuniform window inputs and 50 production success/refusal
cases, including all 42 declared configurations with the default warmer.
`--compile-only` builds both executables without claiming hardware qualification;
CI uses this mode. The production kernel is preserved, and ordinary timing runs
do not add output verification. CUDA results do not qualify AMD hardware or
establish physical cache hit rates.

`python3 tests/cache_latency_input_ranges.py --compiler clang++` runs the actual
parameter prefix under host sanitizers without allocation or GPU access. Its 64
cases cover word alignment, chain-count/index boundaries, positive timed work,
all declared sizes, and existing precedence/clamp/override behavior. Large valid
host cases establish representable layout, not available memory or a supported
process budget. CI runs this test with g++.

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
brackets, absence of algorithmic runtime memsets/copies (except the pinned MG
host-finished initial norm transfer), and all records/summary totals. Eight
captures cover small/reference cases and
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


`python3 tests/pathfinder_input_ranges.py --compiler clang++` runs the actual
Pathfinder parameter prefix, allocation-size arithmetic and kernel column-index
expressions under fatal host sanitizers, without allocation or GPU access. Its
414 cases cover all 240 declared combinations, partial-pyramid index boundaries,
positive dimensions/heights, maximum accumulated costs, argv fallback and
environment precedence. Rejections require the exact intended diagnostic; valid
host cases require empty stderr. The one-row arithmetic control preserves the
existing absence of a traversal launch and is outside the performance sweep.
Passing these checks establishes arithmetic validity, not memory availability
or a runtime budget. CI runs the test with g++.

`python3 tests/npb_ep_input_ranges.py --compiler clang++` executes the actual EP
parameter and geometry prefix with host sanitizers. It checks exponent/thread
and per-block histogram index limits, positive block sizes, all declared
configurations, and existing argv/environment precedence without allocating
large inputs. Host acceptance establishes representable indices, not device
support, verification coverage or runtime feasibility. CI runs it with g++.

`python3 tests/npb_ep_output_sanity.py --host-only --compiler clang++ --work-dir
/tmp/ep-host` tests the actual EP reduction with fatal host sanitizers, including
partial blocks, integral/count bounds, nonfinite partials, reduced-sum overflow
and bounded diagnostics. `--platform cuda --sweep --work-dir /tmp/ep-device`
builds production and post-timing device-fault variants, checks all measured
partial buffers and all 15 declared configurations. Both warmer modes and
healthy non-reference-M skip behavior are covered. Each work directory must be
new. CI runs host fixtures and compiles the GPU variants. These sanity bounds
supplement the unchanged official NAS Sx/Sy criterion; they do not establish
an independent exact histogram reference.

`python3 tests/bfs_input_ranges.py --compiler clang++` executes the actual
BFS parameter prefix and row-size expression under host sanitizers. It checks
positive vertex/block counts, nonnegative out-degree, signed edge/index limits,
wide row sizing and CLI/environment precedence without allocating the graph or
accessing a GPU. Representable host-test cases do not establish memory capacity
or legal hardware launch dimensions. CI runs this test with g++.

`bfs_reference_test.cpp` compares CPU distances and logical work with an independent
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
completed-case logs plus a partial report.

`python3 tests/bfs_concurrent_kernels.py --compiler clang++ --sanitizer thread`
executes the actual kernel bodies concurrently through a host adapter. Competing
discoveries and completion-flag updates exercise the access-ownership and atomic
exchange policy. CI uses the default AddressSanitizer/UndefinedBehaviorSanitizer
mode; this host check supplements the source argument and GPU tests, and does
not claim that GPU Racecheck proves freedom from global-memory races.

`python3 tests/bfs_work_counts.py --platform cuda --work-dir /tmp/bfs-work-check`
(or `rocm`) instruments test copies and checks every level's actual edge,
discovery-atomic and promotion-atomic counts against independent CPU frontier
snapshots. It covers 165 cases across graph shapes, block sizes and repetitions.
CI compiles the GPU test. Counters and observation copies are absent from the
production benchmark. `python3 tests/bfs_work_counts_test.py` runs in CI without
a GPU and rejects missing or duplicate instrumentation anchors explicitly.

The [BFS benchmark notes](../tier2/rodinia_bfs/README.md) describe the immutable
level-entry visited mask, race-free atomic stores, pinned upstream stages and
intentional instruction/traffic changes. Logical operation counts are not
physical memory-traffic measurements; AMD execution remains deferred.

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

## Verification-control records

`python3 tests/verification_control_records.py --compiler clang++` checks
NPB IS, Rodinia Gaussian and large scan's actual flag parsing and JSON
formatting without a GPU. Its 117 cases cover default/empty values, the three
case-sensitive bypass values, other spellings and integer formatting bounds.
CI runs the test with g++ and host sanitizers. Every kernel row records the
resolved JSON boolean `verification_enabled`; enabled is not a success verdict,
and a successful process with checks disabled is not verified evidence.

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

`blackscholes_measured_result.py` checks all actual call/put prices after timing.
It corrupts the first, middle and last outputs and positions 9,999 and 10,000,
checks finite and non-finite failures, exercises both reference-allocation
failures and bounds mismatch logging. Run on a CUDA GPU with a new output
directory:

```bash
python3 tests/blackscholes_measured_result.py --platform cuda --work-dir /tmp/blackscholes-check --largest
```

Use `rocm` for an AMD run. The default 60 cases gain eight largest-size tail
faults with `--largest`. Each process retains the 60-second budget; a timeout
stops the test with logs and a partial report. CI uses `--compile-only` and does
not claim GPU qualification.


`tests/binomial_cpu_reference.cpp` compares the optimized binomial CPU reference
bit for bit against the frozen original per-node-power recurrence, using 665
input/depth combinations. It checks 42 complete parallel runs, input preservation,
49 finite/non-finite maximum selections and six invalid-reference arguments.
Compile it with C++14, `-O3 -pthread -fsanitize=address,undefined
-fno-sanitize-recover=all`; CI runs the g++ build. No GPU is needed.

`python3 tests/binomial_measured_result.py --platform cuda --work-dir /tmp/binomial-measured --largest`
checks actual measured GPU prices through the ordinary full verifier. It injects
finite/NaN/±Inf corruption at the first price, indices 9 and 10, the middle and
last price, plus a 12-error diagnostic-cap case. It checks full-reference failure
and all 15 thread-creation failures with successful sequential fallback. The
optional largest-size cases corrupt the final price with all four fault kinds.
The test requires exact counts, verification identity, finite timing records and
appropriate process status within 60 seconds; a timeout preserves partial logs
and stops the run. Use `--platform rocm` on AMD or `--compile-only` for CI without
a GPU. The existing finite-value test also covers the production comparator.

## Rodinia Needleman–Wunsch

`python3 tests/nw_reference.py --compiler clang++` checks the pinned BLOSUM62
table, ordered seed-7 input recipe, independent dense integer oracles, every-cell
verification failures and actual production input/index/score guards under fatal
ASan/UBSan. `python3 tests/nw_measured_result.py --platform cuda --sweep
--work-dir /tmp/nw-check` runs complete production configurations and controls,
then corrupts actual measured cells and uploaded scores and injects reference
allocation failure. It checks per-kernel and whole-problem descriptors, timing
sums and complete verification counts. Use a fresh directory. `--compile-only`
compiles hooks without claiming GPU execution; CI uses that mode.

## Altis GUPS

`python3 tests/gups_reference.py --compiler clang++` checks the actual production
LFSR, large-offset jump, partitioned kernel bodies with host atomic stubs, full
CPU reference, every-word failures and input/geometry admission under fatal
ASan/UBSan. Independent polynomial and matrix oracles check stream identity.
The host stubs check logical work and atomic counts; they do not model concurrent
GPU atomics.

`python3 tests/gups_measured_result.py --platform cuda --sweep --work-dir /tmp/gups-check`
executes both real atomic variants across all declared configurations and extra
launch controls. It corrupts first/middle/last measured words in both 32-bit halves,
checks the ten-diagnostic cap and complete counts, and injects host storage failure.
Each process has a 60-second budget. Use a fresh output directory; logs and the
incremental report remain there. `--platform rocm` selects AMD;
`--compile-only` checks the hooks without claiming GPU execution. CI runs the host
oracles and compiles the GPU regression.

## NAS CG

`python3 tests/cg_host.py --compiler clang++` compares the production RNG with
pinned floating randlc, the complete Class S matrix and trajectory with pinned
NAS code, and serial/2-worker/8-worker references under fatal ASan/UBSan. It also
checks every result array's finite/nonfinite failure paths, complete diagnostic
counts, actual class/input admission and CLI precedence, and cleanup after
partial worker creation failures. `--all-matrices` extends the exact matrix
comparison to all five complete classes (allow substantial memory and time).
`--thread-sanitizer` selects TSan for the reference and worker cleanup checks.

`python3 tests/cg_measured_result.py --platform cuda --work-dir /tmp/cg-check`
executes all five standard classes, extra launch sizes, 86 faults in actual
measured vectors/history (including the last Class C vector entry and a
12-error diagnostic-cap case), and two allocation-failure cases. It checks
problem/work descriptors, timing sums, complete verification counts and exit
status. Each process has a 60-second budget. Use a fresh output directory.
`--platform rocm` selects AMD; `--compile-only` compiles the instrumentation
without claiming GPU execution. CI runs the host tests and compiles these hooks.

## NAS MG

`python3 tests/mg_host.py --compiler clang++` checks the exact RNG, pinned Class S
full hierarchy/input/norm trajectory with serial/2/8 workers, 192 corruption
cases across every level and periodic ghost region, 54 actual admission cases,
and partial worker creation cleanup under fatal host sanitizers. `--all-inputs`
compares all five complete S--C inputs with original zran3; `--thread-sanitizer`
checks the parallel reference on a supported host.

`python3 tests/mg_measured_result.py --platform cuda --work-dir /tmp/mg-check`
runs 205 cases: five standard classes, four extra launch controls, 194 faults in
actual measured hierarchy/input/norm values (including the largest-class tail
and a 12-error diagnostic cap), and two allocation failures. Each process has
a 60-second limit. All records, actual launch geometry/work, full verification
counts and exit status are checked. `--compile-only` builds the hooks without
claiming GPU execution; CI uses it alongside the host tests. `--platform rocm`
selects AMD, whose hardware qualification remains separate.

## PolyBench/GPU Jacobi-2D

`python3 tests/jacobi_host.py --compiler clang++` compares complete initialization
and CPU trajectories with frozen PolyBench/GPU 1.0 bodies, checks all boundary
values and CLI/environment precedence, and exercises exact finite comparison
under fatal ASan/UBSan. `--all-inputs` covers initialization at all 20 declared
sizes without running their full CPU trajectories under sanitizers.

`python3 tests/jacobi_measured_result.py --platform cuda --work-dir /tmp/jacobi-check`
builds clean production and isolated test binaries, runs all 60 declared size/step
pairs and partial-block controls, and checks actual device A/B corruption,
nonfinite output, diagnostic caps and setup/reference failure. Each process has
a 60-second limit and must preserve both kernel records, geometry/work metadata,
timing sums and failure status. `--compile-only` does not claim GPU execution.

`make -C tier2/polybench_jacobi2d PLATFORM=cuda test` compares the actual
production kernels against frozen upstream GPU bodies under the original 32×8
mapping, including both entire output grids and inputs. `bench_output_test`
accepts the same `--n` and `--tsteps` controls for full-domain qualification.
The shared `finite_verification.py --benchmark polybench_jacobi2d` additionally
poisons the actual/reference tails of both grids. All injection is confined to
isolated test copies. AMD and trace qualification remain separate work.

## Fixed-attempt atomic operations

`python3 tests/atomic_host.py --compiler clang++` exercises actual parameter
admission and environment precedence, exact add/min counters, every CAS count's
bounds, complete success sums and unsigned 32-bit wrap under fatal ASan/UBSan.
The success-count oracle does not predict a concurrent GPU schedule.

`python3 tests/atomic_measured_result.py --platform cuda --work-dir /tmp/atomic-check`
builds production and isolated instrumented binaries, executes all 87 declared
configurations and additional launch controls, and corrupts the actual counter
and first/middle/last CAS count slots. It also injects setup/reference failures.
All 109 cases check the effective source-level work descriptors, timing records,
result diagnostics and failure status within a 60-second per-process limit.
Use a fresh output directory; raw logs and an incremental report remain there.
`--compile-only` builds the hooks without claiming GPU execution, as used by CI
alongside the host tests. `--platform rocm` selects AMD; AMD hardware and trace
qualification remain separate.

## Rodinia HotSpot3D

`python3 tests/hotspot3d_host.py --compiler clang++` checks all 1290 integer-safe
sizes against pinned physical parameters, including the 56 sizes requiring a
smaller timestep, plus 24 small-grid trajectories using limited coefficients
from the first and last limited sizes. These exercise the numerical branch
without claiming hardware qualification of the corresponding large grids.
It compares complete small CPU trajectories with both a frozen
upstream CPU body and a separately indexed long-double oracle, and checks all
91 parameter/precedence cases under fatal ASan/UBSan. `--all-inputs` checks the
complete legacy input recipe at all 22 declared sizes.

`python3 tests/hotspot3d_measured_result.py --platform cuda --work-dir /tmp/hs3d-check`
builds clean production and isolated instrumented binaries, then runs 118 cases:
all 66 declared size/step pairs, nine extra controls, 36 finite/nonfinite faults
at faces/interior/tail, a diagnostic-cap case, a largest-grid tail fault, three
stale-buffer checks and two storage/reference failures. Each process must finish
within 60 seconds. All cases check metadata, timing sums, complete verification
counts and exit status. Use a fresh work directory. `--compile-only` compiles the
fault hooks without claiming GPU execution; CI uses it plus host tests.

`make -C tier2/rodinia_hotspot3d PLATFORM=cuda test` compares the entire actual
production result with the frozen z-marching GPU body on a partial grid.
`bench_output_test --grid_size N --num_iterations T` supports full-profile
comparisons. Both complete GPU grids must satisfy the derived bound against the
independent FP64 trajectory, and their mutual difference must not exceed twice
that bound. A test-only build with FMA contraction disabled can additionally
require exact equality via `HS3D_TEST_REQUIRE_EXACT=1`. Normal production
compilation retains FMA. This fixture requires N>=2, ambient=80 and unchanged upstream dt;
its only kernel adaptation is an x/y guard. CI compiles it. `--platform rocm`
and `PLATFORM=rocm` select AMD, whose hardware qualification remains deferred.

## Rodinia HotSpot

`python3 tests/hotspot_host.py --compiler clang++` checks all 46340 physical
parameter sets against the frozen CUDA source, 224 independently enumerated
padded/partial-pyramid work counts, 100 complete long-double/mixed/serial/2/8-worker
trajectories, steady and impulse cases, finite comparisons and legacy input
identity. It also checks 96 actual control/precedence/overflow cases and three
partial worker-creation failures under fatal ASan/UBSan. `--all-inputs` compares
complete inputs at all 20 declared sizes. `--thread-sanitizer` selects TSan on
hosts that support it.

`python3 tests/hotspot_measured_result.py --platform cuda --work-dir /tmp/hotspot-check`
builds production and isolated instrumented binaries, then runs 264 cases:
60 declared profiles, 103 small/partial/limited-timestep controls, 28 finite or
nonfinite output faults, a diagnostic-cap and largest-tail case, four stale-buffer
checks, four storage/reference/worker failures, and 63 actual GPU work-count
checks. Every process has a 60-second limit; all records, work, timesteps,
verification counts and exit statuses are checked. Use a fresh output directory.
`--compile-only` builds the hooks without claiming GPU execution; CI uses it.

`make -C tier2/rodinia_hotspot PLATFORM=cuda test` compares the entire production
result with the frozen original CUDA kernel. `bench_output_test --grid_size N
--total_iterations T --pyramid_height P` supports full-profile and partial-tail
comparison. Both use the approved dt for N; the fixture kernel body is unchanged.
AMD hardware and trace/replay qualification remain separate.

## Altis CFD solver and complete results

`python3 tests/cfd_host.py --compiler clang++` (or `g++`) runs independent
geometric reconstruction for 342 tetrahedral meshes, 36 complete frozen-FP32
versus independent-FP64 trajectories, all final auxiliary arrays, and 68 actual
control/padding/index/precedence cases under fatal ASan/UBSan.
Host far-field constants are compared bitwise against frozen upstream initialization
statements; the GPU qualification below checks the actual uploaded constants too. Two mutations reject the former double-literal
gamma rounding in boundary and initial-state energy. Separate isolated
header copies exercise both system-error and allocation-failure paths during
zero-worker and partial-worker creation, retaining complete reference output.

`python3 tests/cfd_measured_result.py --platform cuda --work-dir /tmp/cfd-check`
builds clean production and isolated instrumented binaries. It checks every
declared size, nonstandard and partial blocks, actual finite/NaN/Inf faults in
all final and auxiliary buffers, padding and largest tails, capped diagnostics,
omitted solver stages, and setup/reference failure. Each process has a 60-second
limit and preserves its output. Use a fresh directory. CI uses `--compile-only`.

`make -C tier2/altis_cfd PLATFORM=cuda test` compares every stage with frozen
Altis revision `042e2926d1fe9a5ea5b3e2d659b75bf925b152cc` kernels. The
`bench_output_test --size N --block_size B` binary also supports full-size
comparisons; all 10 copies, 10 step-factor buffers, 30 flux buffers and 30 state
updates are compared exactly, including padded cells. The owner-approved
synthetic tetrahedral input is shared with the original solver for this test;
no upstream random-dataset equivalence is claimed. Qualification downloads do
not occur in the measured production run. AMD and trace work remain separate.

### CUTCP complete measured potentials

`cutcp_verification.py --host-only --compiler c++ --work-dir /tmp/cutcp-host`
runs analytical and complete pointwise CPU comparisons under fatal ASan/UBSan,
including no-thread and partial-thread creation fallback. Run
`cutcp_verification.py --platform cuda --work-dir /tmp/cutcp-full --sweep`
on hardware to check actual device-tail/NaN/multiple-output faults, partial
region boundaries, reference allocation failure and all 120 manifest
atom/grid/cutoff configurations within a 60-second process budget.
`--compile-only` builds the isolated production/fault variants without claiming
hardware execution. Output directories must not already exist. Production
contains no test fault controls. `cutcp_concurrency.py` now uses the complete
production reference; its optional historical pre-race baseline remains a
64-point checker.
