# tracer/ — gpubench `.ptrace` toolchain

gpubench-owned trace format and tooling. Replaces storing raw Accel-Sim
`.trace.zst` per kernel with `.ptrace`, which (1) compresses across kernels and
runs, (2) carries the static program (and, later, binary + assembly), and
(3) is the substrate for **synthesizing large traces from small executions**.

Design + plan:
[`../PTRACE_PLAN.md`](../PTRACE_PLAN.md), [`../docs/ptrace_design.md`](../docs/ptrace_design.md).
How it works (method of record): [`../docs/ptrace_method.md`](../docs/ptrace_method.md).
Format spec: [`FORMAT.md`](FORMAT.md).
Measurement that justified this: [`../docs/ptrace_p0_findings.md`](../docs/ptrace_p0_findings.md).

## Status: P3 (sampled-block synthesis + verified reconstruction)

`.ptrace` re-encodes the existing Accel-Sim format losslessly — `expand`
reproduces the original `.traceg` byte-for-byte, so **BBSim needs no changes**.
Wins come from (1) hoisting invariant opcode/registers/immediate into a shared,
deduped ISA table, (2) a single full-window zstd stream per invocation, and
(3) **address recipes**: memory instructions whose addresses are an exact affine
function of `(bx,by,bz,warp,occ)` store coefficients instead of per-instance
addresses. Recipe fitting is gated (index-determined classes only) and
self-verified, so data-dependent accesses (gather, PRNG) stay recorded — lossless
and honest. Tiers 0–3 (constant/affine/piecewise/tiled) are emitted; tiers 4
(PRNG-from-seed) and 5 (recorded index column) are next.

The NVBit C++ tracer (in [bb_triosim](https://github.com/sarchlab/bb_triosim)) is
unchanged; P1 ingests the traces gpubench already collects. Emitting `.ptrace`
directly from a forked NVBit back-end (skipping the Accel-Sim intermediate) is a
later step; defining and proving the format first de-risks it.

## CLI

```bash
go build -o ptrace ./tracer/cmd/ptrace

ptrace convert <accelsim-traces-dir> <out.ptrace>   # Accel-Sim -> .ptrace
ptrace expand  <in.ptrace> <out-traces-dir>         # .ptrace -> Accel-Sim (BBSim-ready)
ptrace verify  <accelsim-traces-dir>                # lossless round-trip + size report
ptrace stat    <in.ptrace>                          # invocations / ISA dedup / size
ptrace holdout <accelsim-traces-dir>                # fit on a sub-grid, synth the rest (measure)
ptrace synth   <accelsim-traces-dir>                # sampled collect + reconstruct; wall-time proxy
ptrace synth   <traces-dir> --grid GxHxD <out-dir>  # extrapolate recipes to a new (larger) grid
```

`verify` converts, expands, and asserts every kernel's decoded `.traceg` is
byte-identical to the original, then prints the size ratio. `synth` does the same
in sampled mode (only a corner of blocks recorded, the rest synthesized and
verified) and reports the recorded-block fraction — the collection-wall-time proxy.

## Measured results (H100 P0 traces, P2)

| benchmark | Accel-Sim | .ptrace | ratio | modeled | note |
|-----------|----------:|--------:|------:|--------:|------|
| fp32_throughput | 90.8 KB | 5.9 KB | 15.3× | 100% | compute micro |
| shoc_fft | 536.7 KB | 38.4 KB | 14.0× | 67% | 11 kernels, ISA dedup |
| rodinia_hotspot | 282.4 KB | 20.4 KB | **13.8×** | 100% | stencil; P1 was 9.4× |
| polybench_gemm | 147.6 KB | 13.3 KB | **11.1×** | 100% | dense GEMM; P1 was 9.8× |
| shoc_spmv | 218.1 KB | 86.9 KB | 2.5× | 68% | CSR reads modeled, gather recorded |
| altis_gups | 69.1 MB | 50.8 MB | 1.4× | 0% | PRNG — honestly recorded; needs tier-4 |

All round-trip byte-identical. "modeled" = share of memory instructions whose
addresses became coefficients; it matches the independent P0 Python analyzer
(e.g. spmv 68%), cross-validating the fitter. These P0 traces are small, so the
ratio is dominated by fixed overhead — full-mode compression **grows with trace
size** (≈130–150× at multi-MB) and **sampled mode grows without bound** for
synthesizable kernels. gups/spmv stay low because their high-entropy gather/PRNG
addresses are correctly left recorded; tiers 4–5 target those.

## Sampled mode (P3) — synthesis scaling

On a synthetic affine kernel the sampled `.ptrace` stays ~constant size (it stores
only the corner sample + recipes + a compressible plan), so the ratio scales with
problem size — all reconstructions byte-identical:

| grid | blocks | Accel-Sim | full (v3) | sampled (v4) | instrumented |
|------|-------:|----------:|----------:|-------------:|-------------:|
| 64×64   | 4096  | 608 KB | 130× | 374×  | 4/4096 |
| 128×128 | 16384 | 2.4 MB | 153× | 1491× | 4/16384 |
| 256×256 | 65536 | 9.8 MB | 130× | 5981× | 4/65536 |

Full mode is `O(dynamic instructions)`; sampled mode is `O(static program + grid
descriptor)`. See [`../docs/ptrace_method.md`](../docs/ptrace_method.md) §6, §8.

## Layout

```
tracer/
  README.md            this file
  FORMAT.md            .ptrace on-disk spec
  ptrace/              Go package
    container.go       Accel-Sim .trace.zst container reader/writer
    accelsim.go        .traceg line parse / emit; static/dynamic split
    ptrace.go          .ptrace format: ISA table, columnar dyn codec, manifest
    recipe.go          exact affine solver, recipe fitter (gated, self-verified)
    tier5.go           residual address column (recorded accesses, base + deltas)
    sample.go          P3 corner sampling, verify-or-fallback plan, sampled codec (v4)
    synth.go           SynthBlock, Holdout measurement, grid extrapolation
    convert.go         Accel-Sim traces dir -> .ptrace (Convert / ConvertSampled)
    expand.go          .ptrace -> Accel-Sim traces dir
    ptrace_test.go     synthetic round-trip + synthesis + residual tests
    sample_e2e_test.go filesystem convert->expand sampled round-trip
  sass/                offline SASS analyzer (classify before tracing)
    sass.go            cubin disasm parse, address/branch/determinism classifier
    sass_test.go       embedded-SASS classification tests
  cmd/ptrace/          CLI
  cmd/sassan/          SASS analyzer CLI (cuobjdump -sass -> per-kernel verdict)
```

## Offline SASS analysis (drives sampling)

`sassan <binary>` disassembles each kernel and, by backward-slicing address
registers and branch predicates, classifies **before any run**: each memory
access (affine-in-index / indirect-through-load / computed PRNG-or-hash), each
branch (uniform / index-determined / data-dependent, incl. loop back-edges), and
cross-launch determinism (no atomics/clock ⇒ relaunch-dedup safe). This tells the
Mode-2 sampler what it can synthesize from indices versus what it must capture.
It is a conservative classifier (the dynamic verify-or-fallback gate stays the
backstop). On this suite it independently reproduces the P0 dynamic split:

| kernel | static verdict |
|--------|----------------|
| fp32 / gemm / hotspot | REGULAR — affine from indices |
| shoc_fft | COMPUTED — `BREV` bit-reverse addressing (non-affine) |
| shoc_spmv | IRREGULAR — indirect gathers + data-dependent loop |
| altis_gups | COMPUTED — loop-carried xorshift PRNG |

## Next

- **P1b — live NVBit driver:** instrument only the sampled blocks at collection
  time (back-end fork), so wall-time — not just disk — drops with problem size.
- **Tiers 4–5:** PRNG-from-seed (annotation/curand) for gups-like kernels;
  recorded index column for gather/scatter (store `index[]`, not 32 addresses).
- **Validate on real traces:** run `ptrace synth` on the H100 P0 traces to turn
  the synthetic scaling above into measured per-kernel numbers.
