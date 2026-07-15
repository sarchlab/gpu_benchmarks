# Current Methodology Audit — 2026-07-13

This report supersedes `review_findings.md`. It contains only problems that remain in the current
tree, including unresolved or deferred items from the earlier review and additional findings from a
fresh repository-wide scan.

## Audit basis

- Repository state was pulled from `origin/main` before the audit.
- Audited code snapshot: `47d29db`; Rule 1 was clarified in the working tree during audit
  consolidation.
- Normative specification: the Tier 1/Tier 2 methodology in `README.md`, especially benchmark
  rules 1–10, sweep rules 1–5, and harness rules 1–5.
- Scope: all 14 Tier 1 and 63 Tier 2 benchmark directories, their `.hip` sources, `params.json`
  manifests, READMEs and Makefiles; the three headers in `common/`; the Go harness; SQLite schema;
  and protocol/CLI documentation.
- The manifest sweep contains 7,020 Tier 1/Tier 2 configurations at one repetition, or 49,140
  process executions at the default seven repetitions.
- This was a static and CPU-side audit. Neither `hipcc` nor `nvcc` is available in the audit
  environment, and no GPU runtime verification was possible.

Validation performed:

- `go test ./...` — pass.
- `go vet ./...` — pass.
- `go test -race ./cmd/gpubench ./tracer/...` — pass.
- `go run ./cmd/gpubench validate` — 126/126 manifests accepted, zero errors, four density
  warnings. The Tier 1/Tier 2 warnings are `npb_ep` (5 scaling values), `npb_mg` (7), and
  `parboil_histogram` (8); the fourth is Tier 4 `triton_flash_attention` (9).

## Executive conclusion

The repository is structurally consistent but is **not compliant with the methodology currently
written in `README.md`**.

Strong areas:

- Every Tier 1/Tier 2 benchmark has exactly one `.hip` source, one README, one Makefile, and one
  `params.json`.
- There are exactly three shared headers in `common/` and no vendor-specific `.cu`/`.cpp` copies.
- All 77 sources use the shared timing path and emit raw timing summaries.
- All manifests have one scaling object, ordered unique values, valid defaults, and a fixed
  `vector_add_warmup` measurement control.
- Repetition is implemented as fresh benchmark processes.
- The shared warmer now performs approximately 100 ms of sustained load over a 768 MiB buffer.
- Database transaction failures, process-group timeout handling, and several earlier kernel bugs
  identified in the old review have been repaired.

The main remaining risks are:

1. Several Tier 2 benchmarks change the upstream algorithm or measured workload, beyond the
   independent-source implementations that Rule 1 permits.
2. Several Tier 1 kernels do not reliably measure the operation named by the benchmark.
3. There are still concrete Tier 2 correctness and measurement-path defects.
4. Many non-scaling parameters are actually secondary work axes.
5. The runner accepts materially incomplete or inconsistent benchmark output as a successful run.
6. Trace mode includes primer and verification kernels that were not part of the measured records.

## Methodology compliance matrix

| Methodology requirement | Status | Current result |
|---|---|---|
| Benchmark rule 1: algorithmically faithful Tier 2 workload | **Partial** | Independent source is allowed, but several documented variants change the problem, recurrence, stages, or operation class. |
| Rule 2: one HIP/CUDA source | Pass | All 77 benchmarks use one `.hip` source. |
| Rule 3: three header-only common files | Pass | `common/` contains exactly the three declared headers. |
| Rule 4: operation executes exactly once | **Fail** | Primer and verification GPU launches execute benchmark kernels outside the recorded pass. |
| Rule 5: cold cache at boosted clock immediately before timing | **Partial** | Shared warmer is correct, but some benchmarks prime or upload working data on the wrong side of it. |
| Rule 6: timed bracket contains only the operation | **Fail** | Host-assisted algorithm stages are omitted; a runtime memset is recorded as a kernel in `npb_mg`. |
| Rule 7: GPU-event timing and raw output | Pass for benchmark output | No Tier 1/Tier 2 benchmark emits derived GFLOPS/GB/s metrics. Documentation is stale. |
| Rule 8: one accumulated record per kernel; sum equals summary | **Partial** | Sources usually follow it; the harness does not enforce it. |
| Rule 9: deterministic equivalent work | **Fail** | `lonestar_sssp` can perform a scheduling-dependent number of iterations. |
| Rule 10: declared parameters drive execution | Mostly pass | Declared parameters are read, but one is not emitted and four undeclared controls alter execution. |
| Sweep rule 1: exactly one scaling axis | Pass structurally | Every manifest has one scaling object. |
| Sweep rule 2: dense range through relevant capacity regimes | **Partial** | Three Tier 1/Tier 2 manifests are short; several ranges deliberately do not cross the stated LLC target. |
| Sweep rule 3: non-scaling knobs change how, not how much | **Fail** | At least 22 knobs in 19 benchmarks change total work. |
| Sweep rule 4: every value/combination matters and works | **Fail** | Unsupported TF32, saturated CUTCP work, oversized Pathfinder configurations, and duplicate variants remain. |
| Sweep rule 5: controls are not swept | Pass | `vector_add_warmup` is fixed by the harness. |
| Harness rule 1: orchestration only | Pass | No timing mathematics is performed in the runner. |
| Harness rule 2: repetitions are fresh processes | Pass | One process is launched for each repetition. |
| Harness rule 3: measurement controls fixed and recorded | Pass | The control is removed from the cartesian sweep and stored per run. |
| Harness rule 4: store raw results faithfully | **Partial** | Valid records are stored, but incomplete/malformed protocols can still be marked successful. |
| Harness rule 5: trace only measured kernels | **Fail** | Trace mode disables only the shared warmer, not local primer/verification kernels. |

## A. Methodology and provenance contract

### A-01 — Critical — Several variants change the algorithm, not merely the source code

`README.md:35` now defines fidelity at the algorithm/workload level. Independent source code is
allowed: a tiled GEMM, a different scan decomposition, a CSR SpMV reimplementation, or a different
thread mapping can be faithful when it preserves the same problem, mathematics, stages,
dependencies, termination behavior, and work scaling. A README statement that the file is “not
upstream source” is therefore not evidence of a violation by itself, and the prior audit's count of
30 source-level discrepancies is retired.

At minimum, the following documented differences cross the clarified semantic boundary:

- `npb_is`: generic one-pass bucket sort instead of NPB IS's ten ranking iterations and key model.
- `tango_binomial_options`: American puts with early exercise and per-node `powf` instead of the
  upstream European FMA induction.
- `altis_gups`: atomic XOR updates instead of HPCC RandomAccess's non-atomic update semantics,
  changing the operation class being measured.
- `altis_particlefilter`: closed-form Gaussian likelihood instead of the upstream video-frame
  gather, with the inter-block scan moved to the CPU.
- `parboil_lbm`: different domain/obstacle problem and boundary representation, not merely a
  different D3Q19 memory layout.
- `parboil_stencil`: a different stencil formula.
- `polybench_2mm`: documented different matrix-chain semantics.
- `polybench_3dconv`: generic dense runtime convolution with different boundary behavior instead
  of the fixed sparse PolyBench stencil.
- `polybench_jacobi2d`: one sweep per timestep instead of the canonical two sweeps.
- `npb_mg`: a simple 7-point geometric multigrid operator instead of NPB MG's four-coefficient
  stencil problem.
- `lonestar_sssp`: topology-driven Bellman-Ford instead of Lonestar's worklist algorithm.
- `lonestar_bh`: the defining GPU tree-build stage is moved to untimed CPU work, leaving only
  force traversal and integration in the measured GPU workload.
- `shoc_devicememory`: three generic grid-stride kernels replace SHOC's defining
  unrolled/local-memory/texture access set.

These benchmarks should either be changed to preserve the upstream algorithm or renamed/described
as distinct workloads without presenting their result population as the upstream benchmark.
This is a minimum set based on explicit current documentation, not a final semantic disposition of
every independent port. Other independent implementations are not Rule 1 failures merely because
their source, tiling, layout, launch mapping, or kernel breakdown differs.

### A-02 — Major — The protocol and schema describe a different measurement model

`docs/output_protocol.md` conflicts with the root methodology in several places:

- It recommends one event per invocation or an average across internal iterations
  (`docs/output_protocol.md:21-44`), while the root requires one accumulated record per kernel and
  no in-program measurement repetitions.
- It permits wall-clock timing and derived metrics in benchmark output
  (`docs/output_protocol.md:39`, `:48-72`), while the root requires raw GPU-event time and
  downstream derivation.
- It states that `info` events are recorded (`:97-115`), but the parser discards them.
- It says the CLI can verify actual parameters (`:159-167`), but the runner only stores the
  emitted value and does not compare it with the requested configuration.
- Its example still shows ten internal iterations and GFLOPS computed in the benchmark
  (`:117-134`).

`docs/schema.sql:94-125` likewise describes one row per in-program Tier 1 iteration and a metrics
table for benchmark-derived GFLOPS/bandwidth. These documents should be made subordinate to, or
rewritten to match, the root methodology.

## B. Tier 1 benchmark findings

### B-01 — Critical — `fp16_throughput` does not reliably force FP16 FMA on AMD — RESOLVED (PR #66)

`tier1/fp16_throughput/fp16_throughput.hip:74-90` converts `1.0000001f` to FP16. That value rounds
to exactly `1.0` in FP16. NVIDIA's explicit `__hfma2` intrinsic is normally protected by inline
PTX lowering, but the AMD/LLVM path can legally reduce `fma(x, 1, z)` to addition. The benchmark
can therefore count packed FMAs while executing packed adds.

Use a representable FP16 multiplier different from one, such as `1.001953125`, and inspect the
generated ISA on both toolchains.

> **Resolved (PR #66).** The multiplier is now `mul = 1 - 2^-10 (0.9990234375)` with
> `add = 2^-10`, both exactly representable, chosen so the recurrence contracts toward the fixed
> point 1.0 and the accumulators stay bounded across the whole sweep (no overflow to `inf`, no
> subnormal flush). Every launched thread now stores its own checksum, so no thread's chain can be
> eliminated as dead. Verified on nvcc 12.9 / `sm_90` / H100: the loop body emits `HFMA2` against a
> `0.9990234375` multiplier (previously foldable `1.0`). AMD ISA not re-checked (no ROCm host
> available); the fix removes the `mul == 1.0` degeneracy that the finding hinges on for any
> toolchain.

### B-02 — Critical — `int32_throughput` relies on signed-overflow undefined behavior — RESOLVED (PR #66)

`tier1/int32_throughput/int32_throughput.hip:67-86` repeatedly evaluates `a = a * 3 + 1` on signed
`int`. Overflow is guaranteed at the declared iteration counts and is undefined in C++. Constant
multiplication by three also invites strength reduction to shift/add instructions, so the claimed
integer MAD operation mix is compiler-dependent.

Use unsigned arithmetic and a runtime-loaded or otherwise instruction-forced multiplier, then
verify the emitted ISA.

> **Resolved (PR #66).** Accumulators are now `unsigned` (defined wraparound) and the
> multiplier/addend are passed as runtime kernel arguments, so the multiply cannot be
> strength-reduced or the recurrence algebraically collapsed. Every launched thread stores its own
> checksum. Verified on nvcc 12.9 / `sm_90` / H100: the loop body is now `IMAD Rd, Rd, UR4, R4`
> against the runtime operand — the previous constant-folded `a*81+40` collapse (four `a*3+1` steps
> fused into one IMAD, running ~1/4 of the reported MADs) is gone. At `mads_per_thread=262144`
> kernel time rises from ~0.24 ms to ~0.70 ms, confirming the full work now executes.

### B-03 — Critical — `cache_latency` evicts its own primer before timing

`tier1/cache_latency/cache_latency.hip:236-257` launches an untimed pointer-chase primer and then
calls `bench_run_once`. `bench_run_once` invokes the 768 MiB warmer after the primer, evicting the
chain that the primer was intended to make resident. The L2/MALL points consequently mix cold and
hit accesses while the benchmark describes a steady-state hit measurement.

Prime after the shared warmer and immediately before an explicit event bracket, or define a
documented cache-latency exception to the generic cold-cache rule.

### B-04 — Major — `cache_bandwidth` private mode aliases large working sets

`tier1/cache_bandwidth/cache_bandwidth.hip:147-196` caps the total allocation at 4 GiB and computes
fewer windows than blocks once `num_blocks * working_set_bytes` exceeds the cap. Blocks are then
mapped modulo the window count. Large configurations therefore do not provide the documented
private window per block; many blocks access the same data.

The emitted params omit the actual `windows` value and `threads_per_block`, so downstream analysis
cannot identify or correct the aliasing. Either constrain configurations, allocate enough private
storage, or report the effective sharing topology.

### B-05 — Critical — `tensor_core_throughput` has unsupported and misleading configurations

`tier1/tensor_core_throughput/tensor_core_throughput.hip:116-165` assumes 32 lanes for both CUDA
WMMA and rocWMMA. On the intended CDNA/MI300 wave64 population, a 32-thread block does not represent
the intended full-wave rocWMMA execution model.

The `precision=tf32` value is more directly broken: on ROCm, and on CUDA devices without TF32, the
benchmark emits an unsupported message, a zero summary, and exits successfully without a kernel.
Because the harness accepts summary-only output, these configurations can be stored as valid
zero-time measurements. In addition, one dependent accumulator tile and one warp/block measure a
latency-limited issue stream rather than peak device tensor throughput.

Unsupported combinations must fail before measurement or be removed conditionally from the
sweep. Platform-correct wave size, accumulator type, and throughput semantics should be explicit.

> **Resolved (2026-07-14):** the `tensor_core_throughput` benchmark was removed from the suite
> entirely (issue #63, B-05 / D-02(a)). The unsupported-config and wave-size problems no longer
> apply. A future tensor/matrix-throughput benchmark, if added, must handle wave32/wave64
> populations and fail unsupported precisions before measurement.

### B-06 — Major — Atomic cases primarily measure failed or compiler-transformed paths

`tier1/atomic_operations/atomic_operations.hip:67-86` resets the counter to zero before
`atomicMin` with nonnegative operands, so the minimum never changes. The CAS case performs one
attempt without retry. These measure failed-update paths, not successful atomic update behavior.
Uniform `atomicAdd` may also be warp-aggregated by the compiler, making counted source operations
different from hardware atomic instructions.

If these paths are intentional, rename and document the records as failed-min/one-shot-CAS cases.
Otherwise construct inputs that exercise successful operations and verify the ISA.

### B-07 — Major — `warp_shuffle` has a latent AMD wave-size assumption

`tier1/warp_shuffle/warp_shuffle.hip:72-77` assigns `WARP_SIZE=64` for every AMD build. That is not
valid for AMD targets configured for wave32. The serial dependent chain and single-block design are
also latency measurements, not peak shuffle throughput; this limitation is documented but remains
important for interpretation.

The declared `shuffle_variant` parameter drives which kernels execute but is missing from emitted
kernel params. The `all` value also repeats the same kernels already covered by individual variant
configurations.

### B-08 — Major — Manifest coverage and undeclared controls weaken reproducibility

- `tier1/memory_bandwidth/params.json` declares H2D, D2H, and D2D kernel names, but its
  `transfer_type` sweep contains only `d2d`; H2D and D2H are unreachable through the shipped
  manifest.
- `cache_bandwidth` reads undeclared `BENCH_PARAM_threads_per_block` and
  `BENCH_PARAM_blocks_per_cu`; only part of their effect is emitted.
- `cache_latency` reads undeclared `BENCH_PARAM_num_accesses`.
- `parboil_spmv` reads undeclared `BENCH_PARAM_block_size`.

These environment variables can alter execution without being represented in the manifest-derived
configuration identity. They should be declared, removed, or clearly treated as unsupported debug
controls and fully recorded.

### B-09 — Minor — Remaining Tier 1 interpretation limitations

- `mixbench` samples operational intensity at only `0.5, 2, 4, 8, 32`; this is legal under the
  3–5-value guidance but sparse around many contemporary roofline knees.
- The smallest `fmas_per_thread` points in the FP32/FP64 throughput sweeps approach the fixed
  launch/event floor, so they should not be interpreted as steady-state arithmetic throughput.
- `empty_kernel.time_ms` measures device-side grid dispatch/retirement, not host launch latency;
  the smallest points approach GPU-event resolution noise.
- Pageable `pcie_bandwidth` bidirectional mode uses async calls on pageable memory, which generally
  requires runtime staging and does not establish the same concurrent PCIe transfer semantics as
  pinned mode.
- `shared_mem_bandwidth` constructs 32-lane logical groups even on wave64 AMD hardware. The
  advertised conflict multiplicity is therefore not the same hardware access pattern on NVIDIA
  warp32 and AMD wave64 targets.

These are interpretation/coverage limitations rather than direct structural violations.

## C. Tier 2 correctness and measurement findings

### C-01 — Critical — `parboil_spmv` can corrupt the final JDS row

`tier2/parboil_spmv/parboil_spmv.hip:187-207` combines the “row changed” and “last nonzero” cases
when generating row counts. If the final row contains exactly one nonzero, the previous row count
can be incremented while the final row is never recorded. The resulting JDS structure is corrupt
and verification can fail for a deterministic subset of declared configurations.

Separate final-element flushing from row-transition accounting and add a regression case whose
last row has exactly one nonzero.

### C-02 — Critical — `parboil_cutcp` silently drops work as the sweep grows

`tier2/parboil_cutcp/parboil_cutcp.hip:365-390` drops atoms after a fixed per-bin `BIN_DEPTH` is
reached. The CPU reference excludes the same dropped atoms, so verification can pass even though
the requested problem was not executed. At small `lattice_dim`, upper `num_atoms` values saturate
near the bin capacity and perform near-identical effective work.

This remains explicitly open in the prior review. Constrain `num_atoms` by `lattice_dim`, make bin
storage dynamic, or report and scale on the accepted atom count rather than the requested count.

### C-03 — Major — `npb_mg` verification is too weak

`tier2/npb_mg/npb_mg.hip:458-477` accepts a relative residual below `10.0`. A solution with a
residual several times the RHS norm therefore passes. This threshold previously hid a broken
residual implementation and remains too loose to establish solver correctness.

Use a meaningful convergence tolerance and, where possible, a known reference norm or a monotonic
residual-reduction requirement.

### C-04 — Major — `lonestar_sssp` does scheduling-dependent work

`tier2/lonestar_sssp/lonestar_sssp.hip:102-126` reads and updates `dist` using `atomicMin` in the
same launch. Later vertices can observe same-launch updates depending on block scheduling. The host
then repeats launches until no values change (`:187-199`). The fixed point may be correct, but the
number of launches and total measured work can differ between identical runs, violating benchmark
rule 9.

Use a double-buffered/Jacobi relaxation or otherwise define a deterministic frontier/iteration
schedule.

### C-05 — Major — `lonestar_bh` defeats the cold-cache/boosted-clock entry state

`tier2/lonestar_bh/lonestar_bh.hip:444-464` runs the shared warmer and then, for each timestep,
downloads positions, builds the tree on the CPU, allocates/uploads the tree, and finally times the
force kernel. The force kernel reads freshly uploaded data rather than a cold working set, and the
CPU tree-build interval can let the GPU clock fall.

The prior review deferred the GPU tree-build redesign under issue #57, but the current sequence is
still a methodology violation. At minimum, separate host-tree construction from the measurement
entry sequence and perform the final warmer only after all uploads.

### C-06 — Major — `altis_particlefilter` omits host-assisted scan work

`tier2/altis_particlefilter/altis_particlefilter.hip:462-516` performs a device copy, D2H transfer,
CPU prefix-sum/inter-block adjustment, H2D transfer, and scalar D2H transfers between recorded GPU
kernels. These stages are part of the particle-filter algorithm but are absent from per-kernel
records and `summary.total_time_ms`.

Either implement the inter-block scan on the GPU and time its kernels, or define this benchmark as
a kernel-only subset instead of claiming the recorded sum is the full algorithm time.

### C-07 — Major — Device-runtime operations are handled inconsistently

- `npb_is` performs an unrecorded mid-algorithm D2D copy.
- `polybench_gramschmidt` performs an unrecorded device memset between measured kernels.
- `npb_mg` records `hipMemsetAsync` as the `coarse_zero` kernel record.

The root methodology says Tier 2 brackets contain kernel launches and nothing else, with transfer
exceptions limited to the Tier 1 transfer benchmarks. The repository needs one policy for
algorithmically required device copies/memsets: include them as explicitly typed raw operations,
replace them with measured kernels, or explicitly redefine summary time as kernel-only.

### C-08 — Major quality gap — Eighteen Tier 2 benchmarks lack a full reference verifier

The root methodology does not require a verifier, so this is not itself a compliance failure, but
it leaves algorithm and porting defects undetected:

- PolyBench: `polybench_2dconv`, `polybench_2mm`, `polybench_3dconv`, `polybench_3mm`,
  `polybench_correlation`, `polybench_fdtd2d`, `polybench_gemm`, `polybench_syr2k`.
- Rodinia: `rodinia_backprop`, `rodinia_bfs`, `rodinia_hotspot`, `rodinia_hotspot3d`,
  `rodinia_kmeans`, `rodinia_lavamd`, `rodinia_nw`, `rodinia_srad`.
- Other: `parboil_stencil`, `shoc_devicememory`.

At minimum, add sampled CPU/reference checks or invariant checks that fail the process on mismatch.

### C-09 — Minor/deferred — Input generation is not portable across C libraries

Several benchmarks still use libc `srand`/`rand`, including BFS, Gaussian, Hotspot, Hotspot3D,
Backprop, graph triangle counting, CEDD, Parboil SpMV, and Parboil stencil. Fixed seeds are
deterministic on a given libc but do not define the same dataset across libc implementations.

The prior review deferred replacement because supported CUDA/ROCm hosts are normally glibc and a
generator change would reset result comparability. Keep it documented as a future dataset-version
change rather than silently changing the sequence.

## D. Sweep design findings

### D-01 — Major — At least 22 non-scaling knobs change the amount of work

Sweep rule 3 says non-scaling parameters change how work is performed, not how much is performed.
The following current knobs are secondary work axes:

| Benchmark | Non-scaling work parameter(s) |
|---|---|
| `altis_gups` | `num_updates`, `num_threads` |
| `altis_raytracing` | `height`, `spheres` |
| `chai_cedd` | `height` |
| `cuda_convolution_separable` | `height` |
| `cuda_nbody` | `timesteps` |
| `lonestar_bh` | `timesteps` |
| `npb_cg` | `max_cg_iterations` |
| `npb_mg` | `num_vcycles`, `num_smooth_steps` |
| `parboil_lbm` | `num_timesteps` |
| `parboil_stencil` | `num_timesteps` |
| `polybench_atax` | `ny` |
| `polybench_bicg` | `n` |
| `polybench_fdtd2d` | `tmax` |
| `polybench_jacobi2d` | `tsteps` |
| `polybench_syr2k` | `inner_size` |
| `rodinia_hotspot3d` | `num_iterations` |
| `rodinia_pathfinder` | `rows` |
| `rodinia_srad` | `num_iterations` |
| `shoc_stencil2d` | `iterations` |

The earlier review accepted several of these by owner decision, but `README.md:74` still forbids
them. Either introduce a schema for multiple work dimensions or pin them and leave one amount axis.

### D-02 — Major — Some declared combinations do not work or do not matter

- `tensor_core_throughput precision=tf32` performs no kernel on unsupported platforms and exits
  successfully.
- `parboil_cutcp` upper `num_atoms` values saturate bin capacity at small lattice sizes.
- `rodinia_pathfinder` declares grids up to 5,000,000 × 10,000 integers, approximately 200 GB for
  one grid. The source now rejects oversized allocations cleanly, but sweep rule 4 requires every
  declared combination to work; clean rejection does not make the combination valid.
- `warp_shuffle shuffle_variant=all` duplicates operations separately represented by the
  individual variant values.

The manifest format currently has no platform predicates or cross-parameter constraints. Either
add them or restrict manifests to their valid intersection.

### D-03 — Major policy mismatch — Several ranges do not cross the stated LLC regime

Sweep rule 2 asks ranges to extend far beyond the last-level cache. The previous review deliberately
left several ranges below or at the 256 MiB MI300X MALL boundary for simulator tractability:
`shoc_sort`, `cuda_scan_large`, `heteromark_fir`, `chai_hsto`, `polybench_gramschmidt`,
`altis_gups`, and `shoc_fft`.

That owner decision may be reasonable, but it is not reflected in the root methodology. Document a
simulator-specific exception or split simulator and hardware sweep profiles.

### D-04 — Minor — Sweep density and operational cost

- Tier 1/Tier 2 scaling counts below the ten-value guidance are `npb_ep` (5), `npb_mg` (7), and
  `parboil_histogram` (8).
- There are 58 singleton non-scaling parameters. This is legal because the value-count language is
  guidance, but those entries increase schema surface without exploring a choice.
- Default Tier 1/Tier 2 execution is 49,140 fresh processes. The 100 ms warmer alone contributes
  roughly 82 minutes before benchmark time, verification, process startup, or trace overhead.

### D-05 — Major operational risk — Full CPU references can dominate or time out runs

Several benchmarks verify only a small output subset but first compute a full CPU reference:

- `polybench_atax` calls the full `O(nx * ny)` CPU reference before checking eight outputs.
- `polybench_bicg` computes both full matrix-vector products before checking eight entries each.
- `polybench_mvt` computes both full `O(n^2)` references before checking eight entries.
- `graph_tc` runs full CPU sorted-adjacency intersections at up to 1,048,576 vertices and average
  degree 128 unless its undeclared debug control disables verification.
- `rodinia_gaussian` computes a full `O(n^2)` residual at sizes up to 8192 after GPU elimination.

This work is outside the timing brackets, so it does not bias `time_ms`, but it can make an otherwise
valid GPU configuration exceed the harness timeout and be recorded as failed. Use sampled references,
size-dependent verification policies, or a separately budgeted verification phase.

## E. Harness, output, and trace findings

### E-01 — Critical — Incomplete output can be stored with `status=ok`

`cmd/gpubench/commands.go:1278-1345` rejects a successful process only when both kernels and summary
are absent, or when a parsed kernel lacks `time_ms`. It does not require the complete protocol
defined by methodology rules 7–8.

An exit-zero run can currently be accepted with:

- a summary and zero kernel events;
- kernel events and no summary;
- a summary missing `total_time_ms`;
- a summary total different from the sum of kernel times;
- missing, non-object, or mismatched params;
- a fatal `error` event but exit code zero;
- negative times;
- unknown or missing kernel names; or
- multiple summaries, in which case the last one wins.

`cmd/gpubench/protocol.go:65-75` warns and drops malformed JSON lines, while an otherwise parseable
summary can still make the run successful. Unknown event types are also ignored. This allows
truncated or corrupt data to enter the successful result population.

The runner should enforce exactly one summary, at least the declared kernel set as appropriate,
finite nonnegative times, parameter equality, no fatal events, and summary equality within a small
floating-point tolerance.

### E-02 — Critical — Trace mode includes kernels outside the measurement records

Trace mode only adds `BENCH_PARAM_vector_add_warmup=false`
(`cmd/gpubench/commands.go:1102-1112`). At least ten benchmarks execute extra GPU kernels for
priming or verification:

`cache_latency`, `altis_cfd`, `altis_gups`, `altis_particlefilter`, `cuda_nbody`, `lonestar_bh`,
`npb_cg`, `npb_mg`, `polybench_jacobi2d`, and `shoc_stencil2d`.

Their traces therefore contain launches that do not correspond to emitted timing records. Add a
standard trace/verification control, or filter captured launches using an explicit measured-region
marker rather than relying only on warmer suppression.

### E-03 — Major — Trace success is based only on process exit status

`cmd/gpubench/commands.go:1118-1141` marks a trace captured when the process exits zero. It does not
validate benchmark stdout, confirm that the expected trace files were created and are nonempty, or
compare observed kernel names/counts with the manifest. A zero-exit empty trace can be reported as
successful.

### E-04 — Major — Trace selection ignores non-scaling configuration identity

`loadSuccessfulConfigs` in `cmd/gpubench/commands.go:1585-1612` selects successful data using only
`(benchmark, scaling_param_value)`. If one non-scaling combination succeeded, trace mode treats all
non-scaling combinations at that scaling value as eligible. The query and lookup must include the
canonical non-scaling parameter JSON.

### E-05 — Major — Manifest validation still accepts wrong structural types

`validateParams` in `cmd/gpubench/commands.go:290-447` validates many leaf values but does not reject
all wrong container shapes:

- a non-object `scaling_param` becomes `nil` and skips all scaling checks;
- a non-array `non_scaling_params` becomes an empty loop;
- a present `values` field of the wrong type skips value checks;
- non-object entries in `non_scaling_params` are silently skipped;
- top-level `benchmark`, `tier`, `suite`, and `description` types are not comprehensively checked;
- scaling-value ordering is not checked.

Current manifests are well formed, but malformed future manifests can pass. Add explicit type errors
for every required container and field.

### E-06 — Minor — `info` events remain unstored

The protocol advertises `info` metadata, but `parseOutput` has no `info` case and the schema has no
table for it. Either store these events or remove them from the supported protocol.

## F. Shared infrastructure and reproducibility

### F-01 — Minor — The warmer creates a hidden 768 MiB availability requirement

`common/bench_warmup.h:77-88` lazily allocates the warmer buffer after benchmark allocations. A
configuration may run with `vector_add_warmup=false` but OOM with it enabled, especially on smaller
GPUs or at top-of-sweep buffer sizes. The resulting message identifies only a warmup HIP error and
does not report combined memory demand.

Preflight the warmer allocation, reserve it before benchmark buffers, or include its memory cost in
configuration feasibility checks.

### F-02 — Minor — Effective runtime configuration is not fully self-describing

Although all declared workload parameters are used, several effective choices are missing from
event records: cache-bandwidth window count and threads-per-block, `warp_shuffle` variant, and the
presence/effect of undeclared debug controls. Rule 7 depends on raw records containing enough work
descriptors to derive and compare results downstream.

Record all effective dimensions, including rounded/capped values such as accepted atom count,
actual windows, actual grid dimensions, and platform-specific feature availability.

## Recommended remediation order

### P0 — Prevent invalid measurements from entering the database

1. Enforce the complete output protocol and summary/kernel/parameter invariants in `gpubench`.
2. Make unsupported tensor-core configurations fail and correct the ROCm launch model.
3. ~~Fix FP16 and integer throughput instruction semantics and validate generated ISA.~~ **Done (PR #66, B-01/B-02).**
4. Repair the Parboil SpMV final-row conversion defect.
5. Remove CUTCP's silent atom dropping or constrain the sweep to accepted work.

### P1 — Restore methodology fidelity

6. Bring the algorithmically divergent Tier 2 variants in A-01 into alignment or identify them as
   distinct workloads.
7. Fix `cache_latency` primer placement and `cache_bandwidth` window aliasing/reporting.
8. Make `lonestar_sssp` iteration work deterministic.
9. Reorder `lonestar_bh` warming/uploads and include or redefine Particle Filter host-assisted work.
10. Establish a consistent policy for algorithmic copies and memsets.
11. Suppress or explicitly mark primer/verification kernels during tracing and validate trace
    artifacts.

### P2 — Make sweeps and documentation enforceable

12. Model secondary work axes and cross-parameter/platform constraints in the manifest schema.
13. Align LLC-range policy with simulator tractability and split profiles if needed.
14. Rewrite the output-protocol and schema documents to match the root methodology.
15. Harden manifest container/type validation and add regression tests for invalid shapes.
16. Add reference or invariant verification to the 18 uncovered Tier 2 benchmarks.

## Historical findings intentionally not carried forward

The following prominent findings from the deleted review are omitted because current code shows
them resolved or removed:

- Metal/macOS ports and their protocol/build defects were removed from the project.
- The shared warmer now runs to an approximately 100 ms sustained-load target.
- Earlier races or math defects in CUDA N-body, Parboil LBM, SHOC reduction, Particle Filter's
  block scan, Binomial Options, NPB MG residual sign, and graph connected components were repaired.
- Verification failures now return nonzero in benchmarks that have verifiers.
- Echo-only declared parameters identified in the prior audit were removed or wired up.
- Database write failures propagate; timeout handling kills process groups; trace repetitions are
  forced to one; run timestamps and child working directories were repaired.
- The prior `npb_mg` duplicate scaling values, graph-CC iteration duplicates, PCIe transfer-type
  duplicate, SRAD pre-warmer upload, and several untimed verification relaunches were repaired.

These fixes should remain covered by regression tests, but they are not open findings in this
current-state audit.
