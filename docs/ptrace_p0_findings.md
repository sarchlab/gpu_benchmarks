# P0 Findings — Trace Synthesizability Measurement

Read-only analysis to decide go/no-go on the synthesis path of
[`PTRACE_PLAN.md`](../PTRACE_PLAN.md). Six representative kernels were traced on
an **H100** with the existing bb_triosim NVBit tracer at small sizes, then the
analyzer ([`scripts/ptrace_analyze.py`](../scripts/ptrace_analyze.py),
decoder [`scripts/ptrace_decode.py`](../scripts/ptrace_decode.py)) fit the model
tiers per static memory instruction.

## Method

- `synth%` = fraction of dynamic memory accesses whose per-lane addresses are a
  deterministic, **index-determined** function of `(blockIdx.{x,y,z}, warp,
  occurrence)`: the intra-warp offset signature is constant within each
  index-selected class, and the per-instance base address is **exactly affine**
  in those indices (tested with exact rational arithmetic, not float).
- The address modes reported are the tracer's encodings:
  `base_stride` / `base_delta` / `list_all`.

## Results

| kernel | trace size | mem% | synth% | verdict |
|--------|-----------:|-----:|-------:|---------|
| fp32_throughput (compute micro) | 112 K | ~0 | 100 | TIER1 — synthesizable |
| polybench_gemm (dense GEMM)     | 168 K | 33 | 100 | **TIER1 affine — fully synthesizable** |
| rodinia_hotspot (2-D stencil)   | 340 K | 11 | 100 | **TIER1 affine — fully synthesizable** |
| shoc_fft (bit-reverse kernel)   | 572 K |  5 | 100 | TIER1/2 — synthesizable |
| shoc_spmv (sparse CSR)          | 240 K | 42 |  68 | TIER5 indirect — partial; record gather |
| altis_gups (random updates)     |  70 M |  1 |   0 | TIER4 — PRNG (xorshift64), or record |

### Extrapolation proof (the wall-time thesis)

Fitting the affine model on a **corner of the grid** and predicting the rest:

- polybench_gemm: trained on **4/16 blocks** → predicted **9408/9408** held-out
  base addresses **exactly (100%)**.
- rodinia_hotspot: trained on **4/16 blocks** → **556/556 exact (100%)**.

This is direct evidence that for regular kernels we can instrument a handful of
blocks of a large run and reconstruct the rest losslessly — i.e. collection
wall-time decouples from problem size.

### Disk-footprint signal

Regular kernels are 112–572 KB; the irregular **gups is 70 MB** — ~150× larger
at comparable launch scale. Irregular kernels dominate disk, so tier-5/6
compression (record the index column, not 32 addresses/instruction) is where the
size win concentrates, while tiers 0–4 give the wall-time win.

## Lessons that change the design

1. **Regress on separate block coordinates, not a linearized block id.** A 2-D
   grid's addresses are affine in `(bx, by)` separately; `4·by+bx` destroys
   affinity. The recipe's address model must keep `blockIdx.{x,y,z}` distinct.
   (This flipped hotspot from a false 0% to a true 100%.)
2. **`list_all` is not the irregularity signal.** The tracer's `base_delta`
   encodes *any* address set, including gups's random ones. The real test is
   whether the offset signature is index-determined and the base is affine —
   not the encoding mode.
3. **PRNG kernels need the tier-4 model, and it pays off.** gups is 0% affine
   but its addresses come from a declared xorshift64 — the annotation-driven
   PRNG path (and verify gate) is what makes it synthesizable; otherwise it is
   the dominant disk cost.
4. **Indirect (tier 5) is partial, not all-or-nothing.** spmv is 68%
   synthesizable (CSR row reads) with a 32% data-dependent gather — store the
   index column for the remainder rather than full per-lane addresses.

## Go / No-Go

**GO.** Across dense compute, GEMM, stencil, and FFT the synthesis path is not
just viable but exact, and extrapolation from a small block sample is verified.
The irregular cases fall cleanly into the planned tier-4 (PRNG) and tier-5
(indirect) buckets rather than defeating the approach. Recommend proceeding to
P1 (static section + columnar payload) and P2 (recipe fit + verify), carrying the
four design lessons above.

## Reproduce

```bash
# collect (example)
./gpubench run --benchmark polybench_gemm \
  --collect-trace .../tracer_tool.so --trace-output /scratch/.../p0_traces \
  --scaling-values 64 --sweep-non-scaling=false --reps 1
# analyze
python3 scripts/ptrace_analyze.py /scratch/.../p0_traces/*/*/traces
```
