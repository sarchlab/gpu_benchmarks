# Parametric Trace (`.ptrace`) — Implementation Plan

Actionable plan for replacing per-kernel Accel-Sim `.trace.zst` collection with a
gpubench-owned format that compresses across kernels/runs, carries the
instruction binary + assembly, and **generates large traces from small
executions**. Full design rationale: [`docs/ptrace_design.md`](docs/ptrace_design.md).

## Objective

Stop trace-collection wall-time from scaling with problem size. Instrument only a
few blocks of large runs, fit a per-instruction model, and synthesize the rest —
losslessly, gated by byte-equivalence verification.

## Locked decisions

- Dedicated gpubench tracer, **forked from the Accel-Sim `tracer_tool`** in
  bb_triosim (keep the NVBit front-end, replace the back-end serializer).
- **Lossless + verified synth**: a model may generate a trace only if it
  reproduces sampled real blocks byte-for-byte; otherwise record.
- **Optimize collection wall-time first** (synthesis path before disk tuning).
- **Compression**: custom binary columnar payload + zstd(trained dictionary).
  Not lz4, not Parquet for the payload (Parquet/Arrow only for the analysis sidecar).

## Address-model taxonomy (per static memory instruction)

A memory instruction's per-lane addresses are modeled as a function of
`(lane, warp, block, iter)`. Fitting tries models **in order of compactness** and
keeps the first that verifies. The `verify` gate makes every tier safe: a model
that mispredicts even one sampled block is rejected and we fall to the next tier.

| # | Model | Stored form | Typical kernels |
|---|-------|-------------|-----------------|
| 0 | **Constant / broadcast** | one address | uniform/scalar loads |
| 1 | **Affine / strided** | `base + a·lane + b·warp + c·block + d·iter` (zero residual ⇒ no address bytes) | coalesced dense access |
| 2 | **Piecewise affine** | small set of affine pieces + predicate on block coords / iter range | stencil & conv boundaries, halo regions |
| 3 | **Nested / tiled affine** | `base + Σ aₖ·ivₖ` over recovered loop induction vars | GEMM tiles, dense linear algebra |
| 4 | **PRNG-seeded** | generator id + seed + counter mapping (see below) | random init, Monte Carlo, GUPS, shuffles |
| 5 | **Indirect via recorded index** | `base + elem_size · index[]`, store only the index column | SpMV col-idx, gather/scatter, histogram bins |
| 6 | **Fully recorded** | columnar address stream (L1/L2 compression only) | data-dependent / unclassifiable |

Tiers 0–4 are synthesizable (extrapolate to new sizes). Tier 5 is *compact but
not extrapolatable* — the index stream is recorded at real size (still far
smaller than 32 per-lane addresses/instruction). Tier 6 is the safety net.

### PRNG-seeded models (the "find the pattern of random addresses" case)

Many benchmarks derive addresses or initialization from a PRNG. If the generator
and seed are known, the exact sequence regenerates losslessly. Two paths, in
priority order:

1. **Annotation-driven (reliable, recommended).** These are *our* benchmarks, so
   declare the generator in `params.json` / kernel metadata:
   `{"rng": {"gen": "curand_philox", "seed": 1234, "stream": "tid"}}`. The
   synthesizer reproduces the stream deterministically — no reverse engineering,
   and `verify` still confirms it.
2. **Blind fitting (best-effort stretch).** From an observed integer sequence,
   test common generators and solve their parameters:
   - LCG `x_{n+1} = (a·x_n + c) mod m` — solvable from a few consecutive values
     for standard moduli.
   - xorshift / xorshift128+ / PCG.
   - curand defaults: XORWOW and Philox 4x32 (the realistic CUDA targets).
   Accept only if the recovered recurrence reproduces the sampled stream exactly.

Because curand is the common case, P-phase work also adds an **optional curand
API hook** in the tracer to capture generator + seed directly, which makes path 1
automatic for curand users.

## Format layout (summary)

```
<benchmark>.ptrace/
  manifest.json          # benchmark, swept params, tool/driver/arch/cc versions, hashes
  modules/<hash>.{cubin,sass,isa.parquet}   # binary + asm + per-PC table, stored once
  kernels/<id>.json      # module ref, CFG, REGULAR|IRREGULAR
  invocations/<id>.json  # geometry, base addrs, ref to recipe|record
  recipes/<id>.json      # per-mem-inst model (taxonomy above) + CF template
  invocations/<id>.{cf,addr}.zst            # recorded columns (tiers 5–6)
  ground_truth/<id>@<size>/                 # one real Accel-Sim trace per kernel, for verify
```

Dynamic payload is columnar: `pc_idx` (varint delta), `active_mask` (bitpacked),
`addr` (model residual; empty when a model fits), `imm` (only if not in ISA
table). zstd with a dictionary shared across the whole `.ptrace`.

## Phases

### P0 — Analyzer (read-only, de-risk) — **DONE → GO**
Measure how synthesizable the suite actually is before building the back-end.
Results: [`docs/ptrace_p0_findings.md`](docs/ptrace_p0_findings.md).
- [x] 6 representative kernels traced on H100 (fp32, gemm, hotspot, fft, spmv, gups).
- [x] Offline fitter (`scripts/ptrace_analyze.py`, decoder `ptrace_decode.py`).
- [x] Per-kernel tier + byte coverage + extrapolation proof.
- **Outcome:** dense/GEMM/stencil/FFT = 100% synthesizable and **extrapolate
  exactly from 4/16 blocks**; spmv = 68% (tier-5 indirect); gups = tier-4 PRNG
  (and 70 MB vs ~0.2 MB regular → irregular dominates disk). **GO on P1–P3.**
- **Design lessons folded back in:** regress on separate `blockIdx.{x,y,z}` (not
  linearized); `list_all` is not the irregularity signal (use index-determined
  signature + affine base); PRNG needs tier-4; indirect is partial not all/none.

### P1 — Static section + columnar payload (lossless) — **DONE**
Implemented as a converter (`tracer/`, Go) ingesting the Accel-Sim traces
gpubench already collects; native NVBit emission deferred. Spec: `tracer/FORMAT.md`.
- [x] Static per-PC ISA table, deduped across relaunches by content hash.
      (cubin/SASS extraction deferred to when we fork the NVBit back-end.)
- [x] Columnar dynamic payload `(struct, pc_idx, mask, addr)`, full-window zstd.
- [x] `ptrace convert` / `expand`; expand reproduces Accel-Sim `kernelslist`,
      `stats`, and `.trace.zst` (BBSim-readable, dictID 1 = bb_triosim convention).
- [x] `ptrace verify` round-trip on all 6 P0 kernels: **byte-identical**;
      kernelslist + stats byte-identical too. Synthetic unit tests in CI.
- **Result:** lossless **9–16× on regular kernels** (fp32 15.6×, fft 14.1×, gemm
  9.8×, hotspot 9.4×); spmv 2.3× and gups 1.4× — high-entropy addresses dominate,
  confirming those need P2 modeling (PRNG-from-seed, recorded index) not codecs.
- **Deferred to P1b/P2:** fork NVBit back-end to emit `.ptrace` directly; add
  module cubin/SASS to the static section (goal #2, needs the binary at trace time).

### P2 — Recipe fit + `verify` — **DONE (tiers 0–3)**
Implemented in `tracer/ptrace/recipe.go` (exact `big.Rat` affine solver + gated,
self-verified fitter) and wired into the v2 dyn payload.
- [x] Per-mem-PC recipes: mode + constant tail + affine base in `(bx,by,bz,warp,occ)`.
- [x] Gating: index-determined classes only + tail reuse; data-dependent PCs stay
      recorded. Self-verify each class regenerates the exact base token.
- [x] `ptrace verify` reports modeled coverage; **byte-identical on all 6 P0
      kernels**. Coverage matches the P0 analyzer (spmv 68%) — cross-validated.
- **Result:** hotspot 9.4×→13.8×, gemm 9.8×→11.1×; gups stays 0% modeled
  (correctly recorded). Post-zstd size gain is moderate; the structural win
  (addresses → coefficients) is the substrate for P3 synthesis.
- **Deferred:** tier-4 PRNG-from-seed (rng annotation / curand hook) and tier-5
  recorded index column for gathers — the gups/spmv remainder.

### P3 — Sampled-block collection + `synth` (the wall-time payoff) — **DONE**
Implemented in `tracer/ptrace/sample.go` (sampled dyn payload v4 + corner sampler +
verified plan) and `synth.go` (`SynthBlock`, `ExtrapolateGrid`). CLI: `ptrace synth`.
- [x] **Sampled `.ptrace` (dyn v4):** store a 2×2×2 *corner sample* of blocks +
      recipes fit from it; synthesize the rest on decode. Recorded fraction is the
      collection-wall-time proxy. `Coverage.{Total,Recorded}Blocks` recorded per inv.
- [x] **Verify-or-fallback:** every candidate synth block is regenerated and
      compared byte-for-byte (`blockEqual`); any miss is demoted to recorded, so a
      sampled `.ptrace` is still losslessly equal to the original.
- [x] `ptrace synth <dir>` — sampled collect, byte-identical reconstruction check,
      reports instrumented-blocks fraction + projected speedup + size.
- [x] `ptrace synth --grid GxHxD <out>` — extrapolate verified recipes to a *new*
      (larger) grid (`ExtrapolateGrid`); labeled unverified (no ground truth at the
      new size). Blocks whose template is data-dependent are reported, not emitted.
- [x] Tests: sampled round-trip + synthesis, grid extrapolation, tier-5 residual
      codec, and a filesystem e2e (`sample_e2e_test.go`).
- **Plan encoding (the size win):** a synthesized block costs only one bitset
  flag + its zig-zag coordinate delta (both near-constant streams that zstd
  crushes), so a fully-synthesizable kernel's sampled `.ptrace` is **~constant
  size regardless of grid** — the ratio grows with the trace instead of plateauing.
- **Result (synthetic affine kernel, 8 warps, 6 insts/warp; sampled `.ptrace`
  stays ~1.7 KB across all sizes, byte-identical reconstruction):**

  | grid | blocks | accelsim | full (v3) | **sampled (v4)** | instrumented |
  |------|-------:|---------:|----------:|-----------------:|-------------:|
  | 16×16   | 256   | 39 KB  | 25×  | 24×    | 4/256 |
  | 64×64   | 4096  | 608 KB | 130× | **374×**  | 4/4096 |
  | 128×128 | 16384 | 2.4 MB | 153× | **1491×** | 4/16384 |
  | 256×256 | 65536 | 9.8 MB | 130× | **5981×** | 4/65536 |

  Full mode (every block stored, addresses modeled) plateaus ~130–150× because it
  is O(dynamic instructions); sampled mode is O(static program + grid descriptor),
  so its ratio scales with problem size. The earlier "9–16×" was a small-trace
  (112–572 KB P0 kernels) artifact dominated by fixed overhead.
- **Exit met:** large configs instrument only the corner sample; wall-time ~
  blocks-traced/total-blocks (here 4/65536 ≈ 16000×) for synthesizable kernels.
- **Deferred:** the *live* NVBit driver that actually skips uninstrumented blocks
  (needs the back-end fork, P1b). Sampled mode helps only synthesizable kernels;
  irregular blocks fall back to recorded and the ratio degrades toward full mode.

### P4 — (optional) native `.ptrace` reader in BBSim
Stream blocks directly from recipes/columns, skip materialization.

## Risks & mitigations

- **Silent synthesis error** → `verify` byte-equivalence gate + committed
  ground-truth trace per kernel; never ship an unverified recipe.
- **Blind PRNG recovery is brittle** → prefer annotation/curand-hook path; blind
  fitting is opportunistic only.
- **Base-address size dependence** → symbolize pointers as `argN+offset` (needs
  arg capture in the tracer) so recipes are size-independent.
- **Loop-induction recovery** (tier 3) → derive from SASS CFG; fall back to
  tier 5/6 if induction vars can't be recovered.

## Immediate next action

P0–P3 are implemented and verified on synthetic + the 6 P0 kernels. Next:
1. Run `ptrace synth` on the real H100 P0 traces (currently on `/scratch`) to get
   the **measured** instrumented-block fraction per kernel — turn the ~144×
   synthetic projection into real numbers across the suite.
2. P1b: fork the NVBit back-end so a *live* run instruments only the corner sample
   (the wall-time win is currently demonstrated on stored traces, not yet collected
   that way).
3. Tier-4 PRNG-from-seed + tier-5 indirect index column for the gups/spmv remainder.
