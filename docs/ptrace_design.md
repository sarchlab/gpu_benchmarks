# Parametric Trace (`.ptrace`) — Design

Status: **draft / proposal**
Owner: Yifan Sun
Goal: replace the current per-kernel Accel-Sim `.trace.zst` collection with a
gpubench-owned format that (1) compresses across kernels and runs, (2) carries
the instruction binary + assembly, and (3) lets us **generate large traces from
small executions** so trace-collection wall-time stops scaling with problem size.

## Decisions (locked)

- **Dedicated gpubench tracer**, forked from the Accel-Sim `tracer_tool` in
  bb_triosim. We keep the NVBit front-end and the SASS semantics, replace the
  back-end serializer.
- **Lossless + verified synthesis.** Recipes may generate/extrapolate traces,
  but only for kernels that pass byte-equivalence verification at sampled sizes.
  Anything else is recorded.
- **First iteration optimizes collection wall-time.** The synthesis path (trace
  a few blocks, generate the rest) is the priority, not disk size.
- **Compression: custom binary columnar payload + zstd(trained dictionary).**
  Not lz4 (ratio), not Parquet for the payload (streaming/dep). Parquet/Arrow is
  fine for the analysis sidecar.

## Why the current format is expensive

Accel-Sim format (today's output) is row-major **text** per dynamic instruction:
`[line] PC mask dest [regdst] opcode nsrc [regsrc] width [addrmode] [addrs] imm`,
one `.trace.zst` per kernel, zstd'd independently.

- **Addresses dominate** the bytes; opcode/regs/width are reprinted every line
  even though they are a pure function of the static PC.
- **SPMD redundancy is unexploited** across three axes:
  1. thread blocks within a kernel (near-identical control flow; addresses are an
     affine function of block/warp/lane index),
  2. relaunches within a run (iterative kernels relaunch the same static kernel),
  3. sweep points across runs (same static kernel, scaled grid + base addresses).
- **Collection time scales with total dynamic instructions** — i.e. with problem
  size — because every block is instrumented.

## Format layout

```
<benchmark>.ptrace/
  manifest.json                  # benchmark, swept params, tool/driver/arch/cc versions, hashes
  modules/
    <module_hash>.cubin          # raw device binary (stored once)            ── goal #2
    <module_hash>.sass           # nvdisasm disassembly text (stored once)    ── goal #2
    <module_hash>.isa.parquet    # per-PC static table: opcode, regs, pred, ctrl, width
  kernels/
    <static_kernel_id>.json      # module ref, entry PC, CFG, REGULAR|IRREGULAR
  invocations/
    <inv_id>.json                # grid/block dims, stream, arg base addrs, ref to recipe|record
    <inv_id>.cf.zst              # (irregular) recorded control-flow column stream
    <inv_id>.addr.zst            # (irregular) recorded address column stream
  recipes/
    <inv_id>.json                # (regular) per-mem-inst affine model + per-block divergence rules
  ground_truth/
    <static_kernel_id>@<size>/   # one real recorded Accel-Sim trace per kernel, for verify
```

### Static section (goal #2)

Per **module** (dedup by hash of the cubin), stored once regardless of how many
kernels/invocations/sweep points reference it:

- `*.cubin` — raw binary (`cuobjdump`/NVBit image).
- `*.sass` — `nvdisasm` text.
- `*.isa.parquet` — decoded per-PC table: `pc, opcode, dst_regs, src_regs,
  predicate, control_bits, mem_width, immediate`. The dynamic stream references
  rows here, so opcode/regs/width are never repeated per dynamic instruction.

### Dynamic section — columnar payload

Per warp, the dynamic stream is decomposed into columns, each pre-coded then
zstd'd with a dictionary shared across the whole `.ptrace`:

| column        | pre-coding                                  |
|---------------|---------------------------------------------|
| `pc_idx`      | varint delta (mostly sequential)            |
| `active_mask` | bitpacked                                    |
| `addr`        | per-static-inst affine residual; **zero residual ⇒ column empty** |
| `imm`         | only when not recoverable from ISA table    |

For a perfectly coalesced/strided access the entire address payload collapses to
4 coefficients `{base, lane_stride, warp_stride, block_stride}` — this is the
dominant size win and the basis for synthesis.

### Recipe (regular kernels)

```jsonc
{
  "kernel": "<static_kernel_id>",
  "cf_template": "single | piecewise(block_coords) | recorded",
  "mem_models": [
    { "pc": 1234, "base": "arg0", "lane_stride": 4, "warp_stride": 128,
      "block_stride": 16384, "loop": {"iv": "R12", "init":0, "step":4, "bound":"arg3"} }
  ],
  "verified_at": [ {"grid":[64,64,1]}, {"grid":[128,128,1]} ]
}
```

`ptrace synth --grid GxH` plugs new geometry + base addresses into the models
and emits a full trace **without running the big kernel**.

## CLI

```
gpubench trace <bench> ...            # collect → writes .ptrace directly
ptrace expand <b>.ptrace --inv <id>   # → kernelslist + .trace.zst (BBSim-ready today)
ptrace synth  <b>.ptrace --grid GxH   # extrapolate to a larger problem
ptrace verify <b>.ptrace              # byte-equiv recipe vs sampled real blocks (gate)
ptrace stat   <b>.ptrace              # size, regular/irregular split, model coverage
```

`expand` reproduces the **exact** Accel-Sim format, so BBSim needs no changes to
start consuming `.ptrace`. A native streaming reader in BBSim is a later option.

## Collection workflow (wall-time first)

For each benchmark with a scaling sweep:

1. **Smallest sweep point: full trace** (cheap). Build static section, fit
   recipe, classify REGULAR/IRREGULAR, save as `ground_truth`.
2. **Larger sweep points: instrument only a sampled set of blocks** (+ boundary
   blocks). `verify` the recipe against those blocks.
   - pass ⇒ `synth` the full trace offline; the big run never fully executed
     under instrumentation → wall-time ~ blocks-traced/total-blocks.
   - fail ⇒ fall back to full recording (irregular at scale); lossless preserved.

The NVBit front-end already supports block filtering and instruction intervals
(`should_trace_kernel`, `instr_begin/end_interval`), so sampled-block tracing is
a back-end + driver change, not new instrumentation.

## Regularity classification

A kernel is REGULAR (synthesizable) iff, across sampled blocks/iterations:

- **address regularity:** every static mem instruction fits
  `addr(lane,warp,block,iter) = base + a·lane + b·warp + c·block + d·iter` with
  zero residual; and
- **control-flow regularity:** all blocks share one PC sequence (pure), or
  divergence is a function of block coordinates only (piecewise, e.g. stencil
  boundaries).

Otherwise IRREGULAR (data-dependent: BFS, sort, SpMV, histogram, graph) → record.

Expected split for this suite: tier1 + dense tier2/tier3 (GEMM, stencil, conv,
FFT, BLAS) largely REGULAR and carry most of the byte volume; graph/sparse/sort
family IRREGULAR. **Action item: run the analyzer over a first batch of real
traces to measure the actual split before committing to synthesis scope.**

## Phased plan

- **P0 — Analyzer (read-only, de-risk).** On real H100 traces from a handful of
  representative kernels, fit affine models + CF templates and report the
  regular/irregular split and projected size/time savings. No format change yet.
- **P1 — Static section + columnar payload.** Fork the tracer back-end: emit
  module cubin/SASS/ISA once, dynamic stream as columnar `(pc_idx, mask, addr)`
  with shared zstd dictionary. `expand` round-trips to Accel-Sim. Lossless, full
  recording — immediate disk win, no synthesis yet.
- **P2 — Recipe fit + `verify`.** Classify kernels, emit recipes for REGULAR
  ones, `verify` gate against ground truth.
- **P3 — Sampled-block collection + `synth`.** Wire the driver to instrument
  only sampled blocks at large sizes and synth the rest — the wall-time payoff.
- **P4 (optional) — native `.ptrace` reader in BBSim.**

## Open questions

- Base-address symbolization: map raw arg pointers to `arg0+offset` so recipes
  are size-independent (needs arg capture in the tracer).
- Loop-induction recovery from SASS for the `loop` term in mem models.
- How many boundary-block classes a piecewise CF template needs in practice.
