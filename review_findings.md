# Tier 1 & Tier 2 Benchmark Review — 2026-07-10

Scope: all 14 tier-1 and 63 tier-2 benchmarks (`.hip` canonical implementation, Metal port,
`params.json`, per-benchmark README, Makefile), the three shared headers in `common/`, and the
`gpubench` Go harness. Reviewed by 18 parallel agents against the methodology rules in the root
README (benchmark rules 1–10, sweeping rules 1–5, harness rules 1–5), with every finding verified
against actual code lines. Static review only (no GPU on this machine); the harness was built and
its tests run.

**Totals: 20 critical / 119 major / 166 minor findings.**

> **Update (same day):** Metal/macOS support was dropped entirely — all `*.metal`, `*_metal.mm`,
> and `bench_mps.py` files deleted; Makefiles, gpubench, and docs now target CUDA and ROCm only.
> This resolves S2 wholesale, the Metal halves of S7 (including the `PLATFORM=auto` forwarding
> bug and the cache_bandwidth/empty_kernel `$(error)` aborts, both fixed), and every
> per-benchmark Metal finding below. Metal-related items are kept for the record but need no
> further action.

Headline: the HIP/CUDA implementations are mostly methodologically sound — timing brackets,
per-kernel records, re-initialization, and determinism are right far more often than not — but
(1) the methodology's central "boosted clock" premise is likely unmet suite-wide, (2) the Metal
ports are a systematically broken population, (3) seven benchmarks have genuine GPU kernel bugs
(races/UB/wrong math), (4) the tier-1 compute family cannot measure device peak because
`num_blocks` is pinned to 1, and (5) the harness can silently lose runs while reporting success.

---

## Part A — Suite-wide systemic findings

### S1. [major, suite-wide] One warmer pass almost certainly does not boost the clock
> **RESOLVED:** `bench_warmer_run` now repeats the pass to a ~100 ms sustained-load target
> (`BENCH_WARMER_TARGET_MS`), measuring the first pass and folding the remaining budget into a
> single launch via the kernel's `reps` argument; the final pass still evicts the LLC.
> README rule 5 and the header comments updated to match.
`bench_run_once` / `bench_pass_warm` invoke `bench_warmer_run()` exactly once
(common/bench_timing.h:145,166). One pass moves 1.5 GB (768 MB RMW): ~0.43 ms on MI300X,
~0.75 ms on H100, ~5 ms on a 300 GB/s consumer card. DVFS ramp from idle needs tens of ms of
sustained load, so the first (and only) timed kernel starts on a still-ramping clock — a
systematic slow bias, worst for short kernels on fast GPUs. This contradicts README rule 5's
"sustained ALU + memory traffic boosts the clock." The infrastructure to fix it already exists
and is unused: `bench_warmer_kernel` has a `reps` parameter hardcoded to 1
(common/bench_warmup.h:59,85-86), and bench_warmup.h:81 says "call it repeatedly at startup" —
no caller does. Mitigation today is only the nondeterministic warmth carried across `--reps`
processes. **Fix direction:** run the warmer for a time-based target (e.g. loop until ~50–100 ms
of load) before the timed launch.

### S2. [critical/major, all Metal ports] Metal is a separate, broken measurement population
Every one of the ~70 `_metal.mm` hosts diverges from the methodology; on macOS (the auto-detected
platform on this machine) results are not comparable to HIP/CUDA and often not even ingestible:

- **Malformed JSON kernel records** (missing comma between the first two `params` keys, or a
  trailing comma) in ~30 benchmarks — every kernel line unparseable by `gpubench`
  (protocol.go skips them), so Metal runs store summaries only. Affected (agent-verified):
  all 7 PolyBench-A, 4 of 6 PolyBench-B (fdtd2d, gemm, mvt, syr2k), 8 of 9 SHOC (all but
  stencil2d), 3 Rodinia-A (bfs, hotspot, gaussian), 2 Rodinia-B (lud, pathfinder), 3 Parboil
  (cutcp, histogram, sgemm), 5 CUDA/HeteroMark (transpose, scan_large, aes, fir, pagerank),
  4 Tango/NPB (blackscholes, binomial, ep, is), 3 Altis (cfd, particlefilter, raytracing),
  2 Graph/Chai (pr trailing comma, hsto).
- **Wrong methodology everywhere:** host wall-clock (`mach_absolute_time` around
  commit+`waitUntilCompleted`; `GPUStartTime/GPUEndTime` unused), no LLC-eviction warmer,
  `BENCH_PARAM_vector_add_warmup` never read (yet the harness records `true` for those rows —
  the DB metadata describes a warmer that never ran), derived GFLOPS/GB-s metrics computed
  in-binary (rule 7), several ports average multiple in-process repetitions (rule 4), and
  multi-kernel ports stamp the same whole-pass time into every kernel record so
  Σrecords ≠ summary (rule 8).
- **Different algorithms / dead scaling params on Metal:** shoc_fft (fixed-size radix-2 FFT,
  ignores `n_ffts` — the whole sweep runs identical work), npb_ep (ignores `m`; fp32 Box-Muller,
  different binning), rodinia_lavamd (reads `num_boxes`, not the declared `boxes1d` — sweep
  collapses; and computes 12-6 LJ instead of the Gaussian potential), rodinia_backprop (6-kernel
  sigmoid net, hidden=1024, wrong kernel names), rodinia_hotspot (drops pyramid blocking, reads
  undeclared params, 10 iterations vs HIP's 360), rodinia_pathfinder (no pyramid), rodinia_srad
  (fixed q0sqr, ignores lambda, different seeds/names), rodinia_kmeans (early-exit convergence vs
  fixed max_iter), shoc_sort (bitonic keys-only, silently rounds sizes up), shoc_devicememory
  (ignores `transfer_mode` — 3 declared values give byte-identical runs), parboil cutcp/histogram/
  spmv (different algorithms, disclosed), fp64_throughput (FP32 proxy not marked in the JSON),
  graph_cc / lonestar_sssp / npb_cg (run the algorithm 5× and average, reading an undeclared
  `iterations` param).
- **Broken builds/dispatch:** `atomic_operations_metal.mm:143` and `warp_shuffle_metal.mm:144`
  reference an undeclared `avg_ms` — **compile error**; `shared_mem_bandwidth_metal.mm:110,122`
  requests 48 KB threadgroup memory vs Apple's 32 KB limit — **cannot dispatch on any Apple GPU**;
  `cuda_nbody_metal` (bodies=384) and `cuda_convolution_separable_metal` (width=64) read
  **uninitialized threadgroup memory** at those declared sizes (non-uniform `dispatchThreads`
  with cooperative smem loads); memory_bandwidth_metal silently emits zero kernels and exits 0
  if the `.metal` file is missing.
- Assorted printf-vararg UB in Metal stderr banners (more `%d` than arguments) in ~6 files.

**Fix direction:** decide the Metal policy first — either bring ports up to the methodology
(GPU timestamps, warmer equivalent, raw output, fix the JSON commas — mostly mechanical) or
explicitly mark `platform=metal` rows non-comparable and stop recording `vector_add_warmup=true`
for them. The comma bug alone is a one-line fix per file and unblocks ingestion.

### S3. [minor→major in aggregate] Stale pre-refactor per-benchmark READMEs (~70 of 77)
> **RESOLVED:** all 77 tier-1/tier-2 READMEs regenerated against the current code and
> params.json — phantom BENCH_PARAM_warmup/iteration protocol, multi-iteration examples, CSV
> claims, and nonexistent flags removed; parameter tables and example JSON now match actual
> emission; verification exit behavior documented; overstated lineage claims corrected
> (2dconv "PolyBench standard" coefficients, fdtd2d equations, hotspot's pyramid actually
> described, hotspot3d/bfs/lbm rewrites disclosed, lavamd fs=2·vij comment, syr2k symmetry).
> Missing READMEs created for empty_kernel and cache_bandwidth. Stale .hip header comments
> fixed suite-wide. In-code standalone defaults aligned to params.json where they disagreed
> (atomic num_blocks 2048, memory_bandwidth d2d, shared_mem no_conflict, spmv avg_nnz 80,
> pagerank+graph_pr pr_iterations 5, gramschmidt n 64); npb_cg's missing scale_kernel added
> to its kernels list; altis_raytracing's false "also sets height" param description fixed.
Nearly every benchmark README documents `BENCH_PARAM_warmup` (default 3) / `BENCH_PARAM_iteration`
(default 7), "N warmup + M timed passes," CSV output, or multi-iteration examples. No binary reads
these variables; all run exactly once per rule 4. Several `.hip` headers carry the same stale
language ("one JSON line per timed iteration"), as does common/bench_warmup.h:23-26 ("taking the
min over the timed iterations discards it" — there are no in-process iterations; every rep pays
the first-launch I-cache cost) and bench_warmup.h:91-92 (claims disabling the warmer "falls back
to warming with the real kernel" — no such path exists; bench_timing.h:14 "timed as-is" is
correct). Also stale defaults contradicting code: atomic_operations (2048 vs 1024), hotspot,
spmv (50000 vs 131072; 80 vs 128), pagerank (5 vs 20), particlefilter ratio (1.0 vs 0.5),
backprop (65536 vs 1000000), srad iterations, fp32 "20 TFLOPS on RTX 4090" (~82 real).

### S4. [major] Echo-only parameters violate rule 10 across the suite
> **RESOLVED:** all verified echo-only params removed from params.json + code + READMEs across
> 27 benchmarks (`block_size`/`tile_dim` where launches are compile-time, `precision` where
> kernels are float-only — incl. tango_binomial_options, found beyond the original list —
> `trials`, `key_length`, `num_bins`, `seed`, `num_particles_ratio`, phantom `iterations`).
> `verify` is now a real undeclared debug knob: `BENCH_PARAM_verify=false|skip` gates the CPU
> verification block (default on) in cuda_scan_large, graph_tc, rodinia_gaussian, rodinia_lud,
> npb_is — no longer swept, no longer echoed. Params verified as genuinely driving execution
> were kept (npb_is/chai_hsto/correlation/reduction/scan/stencil2d/devicememory `block_size`,
> shoc_fft/tensor_core `precision`, alpha/beta everywhere). Emitted-JSON format strings of all
> 32 touched .hip files machine-checked to parse as valid JSON. Not yet compiled on a GPU
> machine.
Declared params that are read, validated, echoed into records — but change nothing:
- `block_size`: hardcoded launches in cuda_transpose (also `tile_dim`), cuda_nbody,
  cuda_scan_large, polybench 2dconv/2mm/3mm/fdtd2d/gemm/mvt/syr2k/gramschmidt/jacobi2d,
  parboil_sgemm/stencil, rodinia_gaussian/lud, shoc_gemm, npb_is (Metal), chai_hsto (< 256
  is a latent device OOB in the merge loop).
- `precision`: float-only kernels in shoc gemm/reduction/scan/spmv/sort/triad, polybench
  2dconv/2mm/3mm/correlation/gemm/mvt, parboil_sgemm, altis_cfd, tango_blackscholes — several
  accept `double`/`half` and would emit records labeled with a precision that never ran.
- `verify`: verification runs unconditionally for all of true/false/skip in cuda_scan_large,
  rodinia_gaussian, rodinia_lud, npb_is, graph_tc — sweeping it produces duplicate runs
  (rule 4) and, for graph_tc, an O(E·deg) single-threaded CPU check (~10^10 ops at the top of
  the sweep) that dwarfs the measured kernel and risks harness timeouts.
- Others: `key_length` (heteromark_aes always runs AES-256), `num_bins` (chai_hsto hardwired 256),
  `trials` (polybench_jacobi2d: no-op on HIP, but repeats the algorithm on Metal),
  `num_particles_ratio` + `seed` (altis_particlefilter: never used; description fabricated),
  `iterations` (cuda_nbody README documents it; binary neither parses nor echoes it).

### S5. [major] Spec–validator–practice disagreement on params.json
> **RESOLVED (by reclassification):** value counts are now officially empirical guidance, not
> hard rules — README sweeping rules 2–3 and docs/params_format.md reworded ("typically 20–30 /
> 3–5"), and the validator's ≥10-scaling-values check demoted to a warning. `gpubench validate`
> now exits 0 on the shipped repo (126/126 valid, 3 warnings: npb_ep, parboil_histogram,
> triton_flash_attention). The other validator gaps noted here (type checking, kernel-name
> cross-checks) remain open as harness finding #12.
docs/params_format.md rule 5 requires 3–5 values per non-scaling param; the validator never checks
it; single-value lists are the de-facto suite convention (present in ~50 benchmarks). Also
unchecked: declared types vs values, duplicate scaling values, kernel-name uniqueness, and emitted
kernel names vs the `kernels` list (npb_cg emits `scale_kernel`, absent from params.json; a run
with empty stdout stores `status='ok'` with zero kernel rows). Meanwhile
**`gpubench validate` fails on the shipped repo** (exit 1): npb_ep (5 scaling values),
parboil_histogram (8), tier4 triton_flash_attention (9) — `make validate` is red on a clean
checkout. Note `vector_add_warmup` in every params.json is fine: discover.go:149-177 special-cases
it out of the cartesian sweep per harness rule 3.

### S6. [critical for tier-1 results] `num_blocks` pinned to `[1]`
> **RESOLVED (by clarification):** owner confirmed single-SM/CU measurement is the intended
> design — device peak is derived downstream as per-SM value × SM/CU count (an upper bound,
> since all-SM loads throttle below single-SM boost clock). In-code defaults aligned to 1
> (were 1024/128), stale "saturate the device" comments rewritten, params.json descriptions
> and all six READMEs now document the per-SM semantics + ×SM-count derivation + clock caveat;
> root README tier-1 table marks these rows "(single SM/CU)". Exception documented honestly:
> warp_shuffle's serial dependent chain is latency-bound and does NOT extrapolate to device
> peak (its tier-1 finding #2 about ILP remains open if peak shuffle throughput is ever wanted).
fp32/fp64/fp16/int32_throughput, shared_mem_bandwidth, and warp_shuffle all declare
`num_blocks: [1]` (default 1), so every harness run occupies one SM/CU. The kernels' own comments
say "1024 blocks … saturate even high-end GPUs," and each in-code default is ≥128. A single-SM
measurement is a legitimate metric but it is not "peak device throughput," is documented nowhere,
and for warp_shuffle (serial dependent chains) makes the result latency-bound rather than
throughput. `fmas_per_thread` (the scaling axis) scales duration, not occupancy, so no sweep
point ever saturates the device (sweeping rule 2).

### S7. [major] macOS build flow is broken
> **RESOLVED:** the Metal/macOS half went away with the platform drop; the `PLATFORM=auto`
> forwarding bug is fixed (leaf Makefiles resolve `auto` themselves); and `CUDA_ARCH` now
> defaults to `native` in all 95 Makefiles (compiles for the GPU present; override with
> `make CUDA_ARCH=sm_XX` for cross-builds). README states the CUDA >= 11.6 floor that
> `-arch=native` requires.
- tier1/cache_bandwidth/Makefile:47 and tier1/empty_kernel/Makefile:46 hard-`$(error)` under
  `PLATFORM=metal`: root `make` (auto-detects metal on Darwin) aborts at cache_bandwidth,
  leaving 7 later tier-1 benchmarks unbuilt; `gpubench compile` always exits 1 on macOS.
- tier1/Makefile:7,14 forwards `PLATFORM=auto` literally to leaf Makefiles, which reject it —
  `make -C tier1` fails standalone.
- 94 Makefiles default `CUDA_ARCH ?= sm_90` (Hopper-only): on A100/RTX 30/40 the build succeeds
  and every run fails with no-kernel-image; README promises CUDA ≥ 11.0, which cannot even
  compile sm_90.

### S8. [minor] Doc rot in the root README
> **RESOLVED:** counts corrected (126 total, tier1 14 incl. empty_kernel/cache_bandwidth in
> table + tree, tier2 63); four dead reference links removed (parameter_selection.md,
> schema.sql, spec.md, roadmap.md); docs/ tree listing matches reality. Also fixed the
> harness-audit doc rot: cli_design.md's unimplemented --clean and 10,000-run prompt removed,
> --jobs/--timeout defaults corrected (60 s), and gpubench's rocm detection aligned to the
> Makefiles' /opt/rocm/bin/hipcc check (platform.go + doc snippet).
Tier-1 table and directory tree omit `empty_kernel` and `cache_bandwidth` ("Microbenchmarks (12)",
14 exist); tier-2 says 64, 63 exist; "130 benchmarks" total vs 126 discovered.

### S9. [minor] Input-generation portability
> **DEFERRED (owner decision):** with macOS dropped, all supported platforms are effectively
> glibc, so cross-OS input divergence is moot and run-to-run determinism (rule 9) holds.
> Standardizing the remaining libc-`rand()` users on the portable LCG would change generated
> inputs and invalidate comparability with previously collected results — to be bundled into a
> future intentional reset, not done casually.
bfs/gaussian/hotspot/hotspot3d/backprop/graph_tc/chai_cedd/parboil spmv+stencil use libc
`srand`/`rand` (sequences differ between glibc and macOS BSD libc → different inputs per OS from
the same seed), while kmeans/cutcp/lbm/histogram/graph_cc/graph_pr/chai_hsto use portable LCG/
xorshift. Deterministic per platform (rule 9 holds run-to-run) but cross-OS inputs differ;
rodinia_pathfinder's Metal port additionally uses different seeds (42/123) than HIP (9).

### S10. [minor] Verification failures never affect exit status
> **RESOLVED:** verification failure now exits non-zero in all 45 verifying tier-2 benchmarks
> (38 fixed, 7 were already correct), always after records + summary are emitted so failed-run
> data is still stored. 18 benchmarks have no verifier (8 polybench, 8 rodinia, parboil_stencil,
> shoc_devicememory) — adding references there is future work, noted per benchmark in Part C.
> Two unsound verifiers repaired in the same pass: chai_cedd's CPU reference now uses the GPU's
> swept thresholds (was hardwired 50/100), with the excluded border raised to 6 px and tolerance
> tightened to 1%; graph_pr's vacuous 1e-4 absolute check replaced by a scale-aware per-vertex
> tolerance plus a mass-conservation check. Watch-item: shoc_scan's verification was previously
> non-fatal by design (fp32 summation-order rounding at large N) — if large-N runs now fail on
> real hardware, fix by making its tolerance scale-aware, not by reverting the exit code.
> Consequence, intended: benchmarks with open kernel bugs (cuda_nbody, parboil_lbm,
> tango_binomial_options races) may now legitimately record failed runs when verification
> catches the corruption.
Across tier-2, a FAIL prints to stderr and the process still exits 0, so a bad run is stored as a
valid measurement (e.g. the binomial-options race below would be recorded as `ok`).

---

## Part B — Tier 1: does each benchmark measure what its name claims?

| Benchmark | Verdict | Why |
|---|---|---|
| memory_bandwidth | Partially | Sound D2D blit measurement (>LLC top sizes, warmer correct), but only `d2d` is swept (h2d/d2h kernels declared yet unreachable), ×2 read+write accounting is ambiguous downstream, smallest sizes sit on the copy-launch floor. |
| pcie_bandwidth | Yes, with caveats | Pinned by default, correct two-stream bidir bracketing. D2H/bidir pay a first-touch DMA-registration cost inside the bracket on lazily-registering platforms (fresh pinned dst is never touched before timing); `pageable × bidir` measures two sequential staged copies, not overlap. |
| fp32_throughput | Partially | Kernel is right (4 independent FMA accumulators, exact FLOP accounting, DCE-safe). But `num_blocks=[1]` → one SM (S6); bottom sweep points below the launch-overhead floor. |
| fp64_throughput | Partially | Same structure/pinning. Metal runs an FP32 proxy that is honest in comments/stderr but unmarked in the machine-readable output — fp64 queries silently ingest Apple fp32 rows. |
| fp16_throughput | **No on AMD/Metal; partially on NVIDIA** | `__float2half2_rn(1.0000001f)` rounds to exactly 1.0h; LLVM folds `fma(x,1,z)→add` (no fast-math needed), so AMD likely executes packed ADDs — wrong op type, 2× FLOP overcount. NVIDIA is protected only because `__hfma2` is inline PTX. Fix: multiplier representable in fp16 and ≠1 (e.g. 1.001953125). |
| int32_throughput | Partially | `int a = a*3+1` overflows with certainty — formal UB (works today). Constant ×3 invites strength reduction to shift-add on both backends, so the "MAD" op mix is compiler-dependent (up to ~2× GIOPS spread). Fix: `unsigned` + runtime-loaded multiplier. |
| cache_latency | Partially | Chain construction is excellent (per-line shuffled Hamiltonian cycle, single thread, DCE-safe). But the priming lap runs **before** `bench_run_once`, whose warmer then evicts it: L2/MALL sweep points (2 MB–256 MB) blend a flat ~25% cold-miss fraction while the code claims pure hits (cache_latency.hip:240 vs bench_timing.h:145). L1 and DRAM points are clean. Fix: prime after the warmer (bench_pass_warm + hand bracket). |
| cache_bandwidth | Partially | Shared mode is well-designed (re-read amortization ≥99.95%). Private mode silently aliases: `windows = min(num_blocks, mem/ws)` means every `working_set ≥ ~16 MB` shares "private" windows across blocks, and `windows` isn't emitted so downstream can't correct (cache_bandwidth.hip:148-157). Undeclared functional knobs `threads_per_block`/`blocks_per_cu` make runs unreproducible from records. |
| tensor_core_throughput | **No on AMD; partially on NVIDIA** | `dim3 block(32)` with rocWMMA on wave64 CDNA runs half-populated waves — invalid results on the MI300X target (tensor_core_throughput.hip:110,281). Single dependent accumulator tile + one warp/block is latency-bound below peak; accumulate is fp16 (not the conventional fp32, undocumented); `precision=tf32` on AMD emits zero kernels and "succeeds" with total_time_ms=0. |
| mixbench | Partially | Structure is right (buffer size scales, ratio is non-scaling; FLOPs/bytes derivable; DCE-safe both directions), but rule 3's 5-value cap samples the roofline at 5 points and drops the knee ratios (1, 16) the README's own table lists. |
| shared_mem_bandwidth | Partially | Conflict/no-conflict kernels and byte accounting are exact; but `num_blocks=[1]` makes it an undocumented single-CU measurement contradicting its README (128), and the Metal port cannot dispatch (48 KB > 32 KB). Conflict-kernel has a benign intra-warp race; conflict multiplicity differs NVIDIA (32-way) vs AMD wave64. |
| warp_shuffle | Partially | Masks/widths/convergence verified correct (incl. the compat macros). But serial dependent chains + 1 block = latency-bound issue rate, not throughput; `WARP_SIZE=64` whenever `__HIP_PLATFORM_AMD__` is latent UB on RDNA wave32; GB/s vs Gops/s naming inconsistent. |
| atomic_operations | Partially | Bracketing/determinism correct; but it measures only one regime (single-address max contention), compilers may warp-aggregate the uniform `atomicAdd` (counted ops ≠ hardware ops, unevenly across Add/CAS/Min), `atomicMin(counter, tid+i)` after reset-to-0 never succeeds (measures the fail path), and the Metal port doesn't compile. |
| empty_kernel | Partially | What's in `total_time_ms` is device-side grid dispatch/retire cost — launch overhead lives only in `host_time_ms` (inflated by the bracket's own API calls). Lower half of the sweep is at event-resolution noise. 11 coarse scaling values; absent from root README; no README of its own. |

---

## Part C — Tier 2 findings by suite

### Disposition & work checklist

Direction (approved 2026-07-11): we do **not** vendor upstream source. Instead we stop *claiming* to
be the upstream suite — attribution without a faithfulness claim. This lowers the bar from
"reproduces suite X detail-for-detail" to "is a correct, representative instance of workload Y."

**Resolution (2026-07-12): directory/identifier renames are SKIPPED.** De-prefixing produced too many
cryptic names (`cc`, `mg`, `is`, `ep`, `bh`, `nw`, `lud`) and `graph_`/`heteromark_`/`chai_` are
categories, not suite claims. The honesty goal is instead delivered by a **README provenance sweep**
(done — 57 READMEs): each benchmark carries a "Provenance & fidelity" section stating what it is
derived from and exactly how it differs from upstream (faithful ones get a lighter fidelity note). So
the **`rename` tag below is satisfied by that README disclosure**, not by moving directories. This
does **not** excuse correctness bugs or fix a workload whose identity is itself broken.

Tags:
- **rename** — honest provenance disclosure in the README (✅ done via the sweep). Directory names
  are kept as-is (see Resolution above); this tag no longer implies a directory move.
- **fix-bug** — correctness/race defect; must fix regardless of naming.
- **fix-identity** — deviation breaks the workload's identity; either fix the algorithm to match
  the name, or rename to what it honestly is.
- **methodology** — timed-path/measurement violation (rules 4/6/8); orthogonal to naming.
- **sweep** — also has a sweep-quality issue (details in the Sweep-quality section below).
- **decision** — needs an owner call before code (fix-vs-honest-rename).

Naming caveat (moot now that dir renames are skipped, kept for the record): de-prefixing collides on
`gemm` (shoc+polybench) and `spmv` (shoc+parboil); everything else is unique but often cryptic.

Status: ✅ = fixed (2026-07-11, code committed to worktree); unmarked = open.

> **Update 2026-07-13:** the `methodology` and `sweep` tags below are now RESOLVED
> for shoc_scan, shoc_stencil2d, parboil_sgemm, polybench_gramschmidt, rodinia_srad,
> npb_mg, npb_cg, graph_cc, and tango_binomial_options — see the dated banners in
> "Methodology violations in the timed path" and "Sweep-quality problems". The `rename`
> tags and non-Part-C items (e.g. particlefilter's host-side scan, parboil_spmv's
> coo_to_jds bug) are unchanged.

| Dir | → workload | Tags | What the work is |
|---|---|---|---|
| cuda_nbody | nbody | ✅ fix-bug | Restored position double buffer (read `posIn` / write `posOut`, ping-pong per step); verification launch + frees updated. |
| parboil_lbm | lbm | ✅ fix-bug; rename open | Replaced the `is_boundary` branch with per-link bounce-back (bounce iff destination leaves the domain) → every f_dst slot has exactly one writer; no face race, no stale slot. |
| shoc_reduction | reduction | ✅ fix-bug, rename | Driver now ping-pongs `d_partial`↔`d_temp` so no level reads and writes the same buffer. |
| altis_particlefilter | particlefilter | ✅ fix-bug (barrier); methodology, rename open | Added missing `__syncthreads` before the up-sweep in `prefix_sum_kernel`. Host-side inter-block scan still open. |
| tango_binomial_options | binomial_options | ✅ fix-bug, rename; sweep open | Double-buffered `shared_values` (`cur`/`nxt` swap per level); shared bytes doubled (~32 KB, under 48 KB). |
| npb_mg | mg | ✅ fix-bug (residual sign); fix-identity, methodology, sweep open | Residual now `rhs − A·u` (was `rhs + A·u`, = 2·rhs at fixed point) → V-cycle actually reduces its measure. Untimed `residual_kernel` in timed pass still open. |
| graph_cc | cc | ✅ fix-bug + verifier; rename done; sweep open | GPU: double-buffered (Jacobi) propagation → deterministic iteration count (rule 9). CPU reference `cc_cpu` rewritten to the same Jacobi double-buffer semantics so it faithfully mirrors the device (not just the shared fixed point). Verified H100: PASS, 5 iters every run. |
| npb_cg | cg | ✅ fix-identity (SPD + nontrivial + param); methodology, sweep open | Generator builds a symmetric, strictly diagonally-dominant matrix (⇒ SPD); RHS `b` is now random so the solution is nontrivial (with `b≡1` every row summed to 1, making `x≡1` a trivial exact solution → residual 0); edges are mirrored at `(nnz-1)/2` per row so `nonzeros_per_row` describes the actual average degree (was ~2× after mirroring). Verified H100: residual 2.1e-7, PASS. Verification re-runs the solve (methodology) still open. |
| polybench_fdtd2d | fdtd2d | ✅ fix-identity (verified H100: compiles + runs) | Restored the canonical PolyBench recurrence: `ey` carries the i-difference + `_fict_[t]=t` source, `ex` the j-difference, both `-=`. Now faithful. |
| altis_cfd | cfd | ✅ decision: document-only | Owner call: keep as-is. README discloses it computes Euler/Rusanov+RK over synthetic random connectivity (not a physical mesh). No code change. sweep open. |
| rodinia_gaussian | gaussian | ✅ decision: keep as-is | Owner call: correct Gaussian elimination; README notes the access pattern differs from Rodinia's uncoalesced Fan2 signature. No code change. sweep open. |
| lonestar_bh | bh | DEFERRED INDEFINITELY — issue [#57](https://github.com/sarchlab/gpu_benchmarks/issues/57) | GPU tree build (option b) deferred indefinitely 2026-07-12: results are correct/deterministic and honestly documented, so this is completeness not correctness. Accepted as a force-traversal + integrate benchmark. Its cold-cache/timing sub-issue folds into the methodology cluster. |
| lonestar_sssp | sssp | rename | Topology-driven Bellman-Ford; "inspired by" already disclosed. |
| shoc_scan | scan | methodology, rename | Verification re-runs the whole scan + a second warmer (untimed). |
| shoc_stencil2d | stencil2d | methodology, rename | 64×64 pre-timed launch warms I-cache/module state. |
| parboil_sgemm | sgemm | methodology, rename | N≤256 unbracketed verification relaunch. |
| polybench_gramschmidt | gramschmidt | methodology, sweep | `passes` (3, swept to 10) accumulate into one record — rule-4. |
| rodinia_srad | srad | methodology | Faithful otherwise; first-iter `hipMemcpy(J)` after warmer repopulates LLC. |
| npb_is | is | methodology, rename | Generic one-pass bucket sort; small documented untimed D2D copy (negligible). |
| parboil_spmv | spmv | fix-bug, sweep | `coo_to_jds` corrupts the matrix when the last row has exactly 1 nonzero (~1/159 combos): duplicate write + spurious FAIL. |

Mechanical **rename-only** (no algorithm/measurement work — just drop the prefix + provenance
line): the verbatim/faithful set listed below, plus the disclosed simpler-variant rewrites
(shoc_gemm/triad/devicememory/spmv, polybench_2mm/correlation/3dconv/jacobi2d/gemm/syr2k,
rodinia_bfs/hotspot3d, parboil_stencil, altis_gups/raytracing, cuda_scan_large). These are valid
instances of their workload once the suite name is dropped.

### Genuine kernel bugs (critical)
> **RESOLVED & VERIFIED 2026-07-12** (all 7). Compiled with nvcc 12.9 (`-arch=native`) and run on an
> NVIDIA H100 — all 7 build clean and pass verification: nbody PASS, particlefilter PASS, reduction
> PASS (rel err 1.4e-8), graph_cc PASS + deterministic (5 iters every run, rule 9), lbm PASS, mg
> `‖r‖/‖rhs‖`=2.0e-2 PASS (was a vacuous ~2), binomial_options PASS. Every one was a porting artifact
> (buffer collapse / dropped barrier / botched rewrite), not an upstream bug. Fixes: 1 nbody — restored
> position double buffer; 2 lbm — per-link bounce-back; 3 shoc_reduction — ping-pong the driver
> buffers; 4 particlefilter — restored `__syncthreads`; 5 binomial_options — double-buffered
> `shared_values`; 6 npb_mg — corrected residual sign; 7 graph_cc — Jacobi double-buffer. See the
> disposition table above for per-item detail. Descriptions below are the original diagnosis.

1. **cuda_nbody** (cuda_nbody.hip:75-116, same in .metal): integrator writes updated `pos[i]`
   into the array other blocks are still tiling from — unsynchronized global read/write race
   (upstream double-buffers). Nondeterministic results; verification can flake.
2. **parboil_lbm** (parboil_lbm.hip:112-125, same in .metal): boundary cells bounce-back into the
   same `f_dst` slots interior neighbors stream into — write-write race at every domain face —
   and slots whose upwind neighbor is a boundary cell are never written (stale ping-pong data,
   mass leak). The loose 0.5<ρ<2.0 verification passes anyway.
3. **shoc_reduction** (shoc_reduction.hip:125-136): multi-level loop launches
   `reduce_kernel(d_temp → d_temp)` in-place once `next_blocks > 1` twice — at the top declared
   size (268435456) block 1 writes `d_temp[1]` while block 0 reads `d_temp[0..511]`. UB/wrong sum
   at that sweep point; all sizes ≤ 134217728 safe.
4. **altis_particlefilter** (altis_particlefilter.hip:156-167): prefix-sum kernel missing
   `__syncthreads()` between smem load and first up-sweep add — cross-warp race, corrupt CDF,
   wrong resampling. (The Metal port has the barrier, confirming the omission.)
5. **tango_binomial_options** (tango_binomial_options.hip:101-110, same in .metal): in-place
   backward induction reads `shared_values[j+1]` while neighbor writes it, one barrier per level —
   race on every warp boundary (and intra-warp on sm_70+). Upstream double-buffers. 1%-tolerance
   verification often masks it.
6. **npb_mg** (npb_mg.hip:70 vs 156-161, same in .metal): residual computes `rhs + (6u − Σnb)`
   instead of `rhs − (6u − Σnb)` — at the smoother's fixed point the "residual" equals `2·rhs`;
   the V-cycle never reduces its own error measure. Not a functioning multigrid; `rel_res < 10`
   PASS threshold hides it. (Also: `residual_kernel` is launched inside the timed pass but never
   bracketed/emitted — an entire kernel's GPU time missing from `total_time_ms`, ~10-15%.)
7. **graph_cc** (graph_cc.hip:94-111): in-place label propagation — intra-launch visibility of
   other blocks' writes is scheduling-dependent, so iterations-to-convergence (= measured work)
   varies run to run on identical input (rule 9). lonestar_sssp has the same class of tension
   (atomicMin propagation), bounded but not fixed.

### Faithfulness to upstream (rule 1)
> Evidence for the disposition table above. Under the rename direction, the "rewrites/deviations"
> below are no longer rule-1 violations — they become `rename`-only, *except* the identity-broken
> few (fdtd2d, npb_cg, altis_cfd, rodinia_gaussian) already tagged `fix-identity`.

Verbatim/faithful (spot-verified quirks and constants): shoc_fft (Volkov radix-8 + twiddles),
shoc_sort (CUDPP 4-bit radix incl. quirks), rodinia backprop/hotspot (full pyramid)/kmeans/lud/
nw/pathfinder/srad (incl. documented AMD OOB fix)/lavamd, parboil cutcp (verbatim binning)/
histogram (incl. 8-bit saturation + overflow buffer)/spmv (JDS with documented __ldg
substitution), polybench 3mm/atax/bicg/mvt, cuda_transpose (all variants), heteromark_aes
(FIPS-197 verified: S-box, rcon, round structure), heteromark_fir/pagerank, tango_blackscholes
(A&S constants verified), npb_ep (randlc/vranlc + skip-ahead verbatim; reference sums match
published NPB digit-for-digit), graph_pr (genuinely pull-based), graph_tc (sorted-intersection
precondition genuinely holds), chai_hsto/cedd structure.

Rewrites or material deviations presented under the upstream name (README usually hedges
"derived from" but params/suite tables present them as the upstream workload):
- **SHOC:** devicememory (3 simple grid-stride kernels vs SHOC's unrolled/local/texture set),
  gemm (naive 16×16 tile vs cuBLAS N/T), triad (device-resident, drops SHOC's PCIe streaming
  pipeline), stencil2d (untiled vs shared-mem tiled), scan (recursive Blelloch vs vec4
  reduce/top/bottom), reduction (simplified), spmv (csr-scalar only).
- **PolyBench:** 2mm (tiled, different chain semantics), correlation (tiled full-matrix corr vs
  triangular naive), 3dconv (generic runtime-filter conv vs fixed sparse stencil; boundary
  semantics differ), fdtd2d (recurrence differs: swapped difference axes, flipped signs, `_fict_`
  source dropped — "PolyBench standard" claim false), jacobi2d (one sweep/step vs canonical two),
  gemm/syr2k (tiled but math-canonical, disclosed).
- **Rodinia:** bfs (mask arrays eliminated, frontier cleared differently, int vs bool masks —
  4× mask traffic), gaussian (Fan2 thread mapping inverted: coalesced where upstream is famously
  uncoalesced — the benchmark's signature behavior optimized away), hotspot3d (naive
  one-thread-per-cell vs z-marching register-reuse kernel).
- **Parboil:** lbm (generic D3Q19, no obstacles/Parboil layout — plus the race above), sgemm
  (row-major textbook tile vs mysgemmNT column-major register tile), stencil (different formula).
- **NPB:** is (generic one-pass bucket sort vs 10 ranking iterations + gaussian-ish keys),
  cg (generated matrix is **not symmetric** — CG on a non-SPD system is improper; weak
  `rel_res < 1.0` PASS hides it), mg (see bug above).
- **Altis:** cfd (2 kernels, fixed DT, random asymmetric connectivity — a random-gather
  workload, not euler3d), gups (atomicXor vs HPCC's racy stores — measures the atomic pipeline;
  disclosed), particlefilter (closed-form Gaussian likelihood vs video-frame gather; inter-block
  scan runs on the host mid-algorithm), raytracing (rewrite, mostly disclosed).
- **Lonestar:** bh (tree is a CPU-built 2D quadtree re-uploaded per step; only force+integrate on
  GPU — the irregular GPU tree-build the name promises is untimed host code), sssp
  (topology-driven Bellman-Ford vs worklist; "inspired by" disclosed).
- **CUDA SDK:** scan_large omits CONFLICT_FREE_OFFSET padding (correct results, different smem
  behavior); binomial_options prices American puts with per-node `powf` vs upstream European FMA
  induction (transcendental-heavy — different workload class).

### Methodology violations in the timed path (rule 4/6/8)
> **RESOLVED & VERIFIED 2026-07-13** (all but lonestar_bh). Compiled on an NVIDIA
> H100 PCIe (nvcc 12.9) and smoke-run — all pass. Fixes: shoc_scan / npb_cg /
> graph_cc / **npb_mg** — verification no longer re-runs the whole workload + a
> second warmer; it reads the timed output/residual directly. shoc_stencil2d —
> the 64×64 correctness check moved *after* the timed pass (no I-cache/module
> pre-warm). parboil_sgemm — verification reuses the timed `d_C` instead of an
> unbracketed relaunch. npb_mg — ALL device work inside the timed V-cycle is now
> bracketed and emitted: `residual_kernel` (was ~10-15% of GPU time missing from
> `total_time_ms`) and `coarse_zero` (the per-V-cycle coarse-grid zeroing that was
> also untimed); `total_time_ms` now equals the sum of the 5 emitted records.
> polybench_gramschmidt — dropped the `passes` multiplier so the run is one
> factorization (rule-4); `passes` removed from params/README. rodinia_srad — the
> pristine J is uploaded *before* the warmer so the first timed kernel reads cold.
> lonestar_bh remains deferred (issue #57). npb_is's tiny documented D2D copy is
> left as-is (negligible). Descriptions below are the original diagnosis.
- **shoc_scan** (shoc_scan.hip:294): verification re-runs the entire scan (plus a second warmer) —
  algorithm executes twice per process; second pass's launches unrecorded. Pointless (output
  already resident).
- **npb_cg** (npb_cg.hip:323): verification re-runs the whole solve including another warmer
  (results already in `d_x`).
- **shoc_stencil2d** (shoc_stencil2d.hip:205,147): launches the kernel-under-test on a 64×64 grid
  *before* the timed pass — unrecorded launch that pre-warms I-cache/module state the single-run
  design deliberately pays.
- **parboil_sgemm** (parboil_sgemm.hip:201-207): for N≤256 an unbracketed verification relaunch —
  NVBit traces would contain two launches for a one-record benchmark.
- **polybench_gramschmidt** (gramschmidt.hip:223-261, params.json): `passes` (default 3, swept to
  10) repeats the full factorization in-process, accumulating all passes into one record —
  direct rule-4 violation; warmer re-runs per pass.
- **lonestar_bh** (lonestar_bh.hip:445-471): per-timestep D2H/H2D tree re-upload (plus
  hipFree/hipMalloc churn) immediately before each timed force kernel — the working set transits
  L2 right before timing; cold-cache intent defeated.
- **rodinia_srad** (rodinia_srad.hip:412,428): first iteration's `hipMemcpy(J)` happens after
  `bench_pass_warm`, partially repopulating the LLC before the first timed kernel
  (architecture-dependent).
- **npb_mg**: untimed `residual_kernel` + `hipMemset` inside the timed pass (see above).
- **npb_is** (npb_is.hip:271): small untimed mid-algorithm D2D copy (documented, negligible).

### Sweep-quality problems (sweeping rules 1–4)
> **RESOLVED 2026-07-13** (in-scope items; ranges deliberately NOT extended).
> Owner call: the suite targets a cold-cache **simulator**, so pushing working
> sets past the 256 MB MALL is impractical/unsimulatable — the **LLC-extension
> group is DROPPED** (shoc_sort/scan_large/fir/hsto/gups/fft ranges unchanged).
> Fixes applied: **npb_mg** grid_size → [8,16,32,64,128,256,512] (the code rounds
> N up to a power of 2, so the 22 declared values were 6 distinct + 16 dupes; now
> 7 distinct, with a small simulator-tractable point at 8). **graph_cc**
> max_iterations → [100] (solve early-exits at convergence; 20 risked truncating
> to a wrong result). **pcie** transfer_type → ["all"] (one process emits all
> three mode records; drops the duplicate). **conv_separable / chai_cedd** — the
> false "square image" claim removed; `height` kept as an independent axis (owner
> wants non-square retained). **rodinia_hotspot3d** amb_temp → [80.0] (verified
> against the kernel: additive term only, performance-invariant). **tango_binomial_options**
> — `steps` fixed at 1024 and removed as a param (steps² work). The `options`
> scaling axis is kept: measured on the H100, the top value (1,048,576 options)
> runs in 3.24 s GPU / ~7 s wall and PASSes (only 10 options CPU-verified), so it
> is well under any timeout — no trim needed. **rodinia_pathfinder** — oversized
> configs are now rejected *before* allocation (grid > half of physical RAM →
> clean exit, not an overcommit OOM-kill; a post-malloc NULL check was
> insufficient — a 200 GB grid slips under a big-RAM host and SIGKILLs during page
> fault-in). **heteromark_fir** — num_taps>128 now errors cleanly (was shared-mem
> OOB). Also: **chai_cedd's**
> threshold verifier was found **already correct** (canny_cpu uses the swept
> thresholds). The **"too few scaling values"** items stay guidance-only per the
> S5 decision (validator warns, not errors). hotspot3d num_iterations, bh
> timesteps, gups num_updates, raytracing height/spheres kept as legitimate knobs.
> **STILL OPEN — parboil_cutcp bin saturation:** the params note now documents
> that at small `lattice_dim` (e.g. 64) the top `num_atoms` values saturate the
> bins (~46,656 atoms) and do near-identical work (rule-4). This is *documented,
> not resolved* — a proper fix needs constraining `num_atoms` by `lattice_dim` or
> trimming those combos. Left open pending an owner call.
> All changed .hip files compile + smoke-run clean on the H100 (nvcc 12.9).
- Never crosses the LLC (rule 2): shoc_sort (peak 67 MB), cuda_scan_large (top 32 MB, below its
  own binary default), heteromark_fir (32 MB), chai_hsto (8.4 MB), polybench_gramschmidt
  (n pinned 64 → ≤1.5 MB; dominant kernel launches 1 block at every m), altis_gups (max table
  134 MB < 256 MB MALL), shoc_fft (top = exactly 256 MB, never exceeds).
- Duplicate/no-op sweep values (rule 4): npb_mg rounds N up to powers of 2 — 22 declared values
  collapse to 6 distinct workloads (16 duplicate runs, records echo the rounded size);
  parboil_cutcp bin capacity saturates at 46,656 atoms — top ~5 scaling values (and everything
  ≥5000 at lattice_dim=64) measure near-identical work while dropping atoms (verification can't
  see it); graph_cc `max_iterations` [20,100,500] identical (converges <20); pcie `transfer_type=
  all` duplicates the three singles; `verify`/`precision` duplicates (S4).
- Second work axis hiding as non-scaling (rules 1/3): cuda_convolution_separable `height`
  (1024–8192, ×8 work; "square" claim false), chai_cedd `height` (same pattern),
  altis_raytracing `height`/`spheres`, lonestar_bh `timesteps` [1,5,10,20], altis_gups
  `num_updates`, rodinia_hotspot3d `num_iterations` [1,10,50] (×50 work; hotspot correctly pins
  its equivalent); hotspot3d `amb_temp` changes only data values (performance-identical runs).
- Too few scaling values: npb_ep (5 — fails validator), parboil_histogram (8 — fails validator),
  empty_kernel (11, coarse 8× jumps), shoc_fft/sort (13), rodinia_backprop (12), pathfinder cols
  (12), lavamd (17), kmeans (19).
- Hazardous declared combos: rodinia_pathfinder cols=5000000 × rows=10000 needs a ~200 GB host
  malloc (unchecked NULL); parboil_spmv's ported `coo_to_jds` quirk corrupts the matrix when the
  last row gets exactly 1 nonzero (~1/159 of combos at avg_nnz=80: duplicate write + spurious
  FAIL); heteromark_fir `num_taps>128` would OOB shared memory (declared values safe);
  tango_binomial_options top combo ≈ 2×10¹² `powf` node-updates — likely multi-minute, above the
  60 s default timeout.
- CPU-side work that can dwarf the GPU run (timeout risk, outside brackets so measurement is
  unaffected): polybench atax/bicg/mvt full CPU references at max sizes (minutes),
  cuda_convolution_separable full-image CPU reference (~35 GFLOP single-threaded), graph_tc
  always-on CPU verification, rodinia_gaussian 16k launches at N=8192, graph_cc verification
  re-runs the full GPU CC including another warmer.
- chai_cedd: CPU reference hardwires thresholds 50/100 while GPU defaults/sweeps differ —
  verification compares mismatched algorithms for most configs (absorbed by 5% tolerance).
- graph_cc generator: symmetric insertion makes actual degree ≈ 2× the `avg_degree` param
  (inconsistent with graph_pr/graph_tc semantics); graph_pr permits self-loops/duplicate edges
  (harmless, undocumented) and its 1e-4 absolute-tolerance verification is vacuous for N ≥ ~10⁵.

---

## Part D — Shared infrastructure (`common/*.h`)

Beyond S1 (clock boost) and the stale comments (S3):
- **Compat header is complete** — every `hip*` symbol used by all 77 `.hip` files is mapped in
  the NVCC branch; every benchmark can compile on NVIDIA. Shuffle macros verified correct for
  all call sites (explicit width, converged).
- **Event bracketing sound**, including pcie_bandwidth's two-stream bidir (legacy blocking
  streams serialize with the null stream, so the null-stream event pair correctly spans both
  copies).
- **JSON emission from HIP binaries is valid** (C locale, whitelisted strings, guarded event
  values).
- Warmer buffer values are small *normal* floats (0x01010101 ≈ 2.37e-38, exponent 2) — no
  denormal/FTZ concern; eviction + FP work claims hold per pass.
- [minor] 768 MB warmer allocation happens lazily *after* benchmark buffers: top-of-sweep sizes
  can pass with `vector_add_warmup=false` but OOM with it on (availability asymmetry), and the
  error misleadingly reads "HIP error (warmup)" for a benchmark-sizing problem
  (memory_bandwidth 2×4096 MB alone exceeds 8 GB cards).
- [minor] `hipGetLastError` in `bench_lap_end` could in principle attribute a prior async fault
  to the bracket, but no unsynced async work precedes any bracket in current callers.

## Part E — Harness (`cmd/gpubench`)

> **RESOLVED 2026-07-13** (all in-scope items). Critical + majors: DB write failures now
> propagate to run status and exit code (a run that can't be stored is counted failed, not a
> silent pass); timeout SIGKILLs the whole process group with a `WaitDelay` bound; trace mode
> forces `reps=1` (deterministic collection, no more N× NVBit overhead); validator gained
> type/duplicate-value/kernel-name checks (still 126/126, 0 errors). Minors: invalid
> `--reps`/`--timeout` now error instead of silently defaulting; `--scaling-values` requires
> `--benchmark` and de-dupes; empty-stdout-exit-0 recorded as `error` not `ok`; protocol `params`
> parsed and stored (new `kernel_results.params_json` column + migration); `error` severity
> parsed, non-fatal warnings surfaced/stored on success; missing `time_ms` flagged; non-JSON
> stdout warned; per-run `started_at` captured at launch; child cwd set to the benchmark dir;
> second Ctrl+C force-quits; measurement-param overrides coerced to declared type. Tests added
> for `parseOutput`, `buildEnv`, `validateParams`, `generateConfigs`, `coerceParamValue`.
> Already-resolved cross-refs verified: default timeout (S8), doc rot (S8), platform metadata (S7).
> `info` events remain unstored (needs a schema table; deliberately deferred).

Build/vet/tests all pass. Harness rules 1–5 are structurally compliant (no measurement math
anywhere; reps = fresh processes; `vector_add_warmup` excluded from sweeps, recorded per run,
forced `false` in trace mode with last-duplicate-wins verified). Findings:

- **[critical] Silent data loss** (commands.go:1136-1190): `db.Begin()` failure → run counted
  passed but never stored; kernel/metric insert errors discarded (`_, _ = tx.Exec`); `tx.Commit()`
  error only printed. Disk-full or db-locked mid-session → everything after executes, nothing
  stores, exit 0 "All runs passed."
- [major] Timeout kill is child-only (no process group / `WaitDelay`): a timed-out benchmark that
  spawned a subprocess holds the stdout pipe open — the harness blocks past its own timeout
  (tier5 llamacpp literally does this) and can leave the GPU occupied.
- [major] Default timeout 60 s (main.go:27) vs documented 300 s (cli_design.md:98) — top-of-sweep
  runs get killed and stored as `status='timeout'` by users trusting the doc.
- [major] Validator gaps + shipped repo fails validate (S5).
- [major] Trace mode runs `reps` (default 7) identical NVBit collections per config, each
  overwriting the same `TRACES_FOLDER` — 7× the 100–1000× overhead for zero data.
- [major] Discovery has no platform-support metadata: on macOS `compile` always exits 1 (S7) and
  `run` records error rows for binaries that can never exist.
- [minor] protocol.go vs docs/output_protocol.md drift: `info` dropped, kernel `params` never
  parsed/stored, warning-severity errors on exit-0 discarded, missing `time_ms` → 0.0, non-`{`
  garbage silently ignored; stderr discarded on success, truncated to 1000 chars on failure.
- [minor] `--scaling-values` with no `--benchmark` overrides every benchmark's axis; duplicates
  not deduped; bad `--reps`/`--timeout` silently fall back to defaults (`--reps 0` = exit 0,
  0 runs); `interrupted` bool has a benign data race; second Ctrl+C is swallowed (no force-quit
  short of kill -9 with the 3600 s trace timeout); `vector_add_warmup` stored as JSON bool when
  defaulted but string when set via flag (two types across sessions); runs-row `started_at` ≈
  `finished_at` (inserted post-hoc); child cwd is gpubench's, not the benchmark dir
  (cli_design.md shows otherwise — matters for tier4/5 scripts writing caches).
- [minor] Doc rot: `--clean` and the >10,000-run confirmation prompt documented but absent;
  `--jobs` default doc wrong; rocm detection dir-check vs documented hipcc-check.
- Test coverage: only trace helpers are tested; generateConfigs/parseOutput/buildEnv/
  validateParams untested.

---

## Part F — Prioritized fix list

**P0 — measurements invalid or data lost**
1. Warmer clock boost: loop `bench_warmer_run` (or use its dead `reps` arg) to a ~50–100 ms load
   target (S1). One-line-ish change in `common/`, affects all 77 benchmarks' validity.
2. Unpin `num_blocks` in the 6 tier-1 params.json (S6) — sweep it or default to the in-code
   saturating values.
3. fp16 multiplier → fp16-representable ≠1 (e.g. 1.001953125); int32 → `unsigned` and a
   runtime-loaded multiplier (Part B).
4. tensor_core_throughput: 64-thread blocks (or `warpSize`) on AMD; fp32 accumulate (or document);
   independent accumulator tiles; make `tf32` on unsupported platforms an error, not an empty
   success.
5. Kernel races/math: cuda_nbody double-buffer positions; parboil_lbm boundary streaming;
   shoc_reduction final-level guard (or always reduce into `d_result`); altis_particlefilter
   missing `__syncthreads`; tango_binomial_options double-buffer; npb_mg residual sign (+ time
   `residual_kernel`); graph_cc double-buffer labels (determinism).
6. cache_latency: prime the chain *after* the warmer (bench_pass_warm + explicit bracket).
7. Harness: propagate DB errors to run status/exit code; process-group kill + `WaitDelay` on
   timeout; align default timeout with docs.
8. macOS: make cache_bandwidth/empty_kernel Makefiles no-op (or skip) under metal; fix
   `PLATFORM=auto` forwarding; teach discovery/compile about per-platform support (S7).

**P1 — comparability and protocol**
9. Metal policy: fix the ~30 JSON comma bugs (mechanical), the 2 compile errors, the 48 KB
   threadgroup request, and the uninitialized-smem dispatches; then either adopt GPU timestamps +
   warmer + raw output, or stop recording `vector_add_warmup=true` for metal rows and mark the
   platform non-comparable (S2). Fix Metal ports that ignore the declared scaling param
   (shoc_fft, npb_ep, rodinia_lavamd, rodinia_hotspot, …).
10. Remove or wire up echo-only params (S4) — decide per benchmark; where kept, make them drive
    execution and validate ranges (chai_hsto `num_bins` OOB, heteromark_fir `num_taps`).
11. Make `gpubench validate` pass: extend npb_ep/parboil_histogram/triton_flash_attention value
    lists; then enforce in the validator what params_format.md promises (3–5 values, types,
    kernel-name cross-checks vs emitted output).
12. Sweep hygiene: extend ranges past the LLC where the metric warrants (shoc_sort, scan_large,
    fir, hsto, gramschmidt, gups); de-duplicate collapsing values (npb_mg rounding, cutcp bin
    saturation — emit the effective size/windows in records); single work axis (conv_separable/
    cedd height, hotspot3d iterations, bh timesteps).
13. `CUDA_ARCH` default: detect or document (S7); align README CUDA ≥ 11 claim.

**P2 — docs and robustness**
14. Regenerate per-benchmark READMEs from the current design (S3); fix root README counts (S8);
    fix bench_warmup.h stale comments.
15. Make verification failures affect exit status (S10); portable RNG everywhere (S9); validate
    `BENCH_PARAM_*` env values like argv (atoi → checked parse); bound CPU reference cost at
    large sizes (sample-based verification).
