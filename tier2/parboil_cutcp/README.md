# tier2/parboil_cutcp — Coulombic Potential with Cutoff (Parboil, cutoff binning)

Faithful port of the Parboil **CUTCP** benchmark: the device kernel
`cuda_cutoff_potential_lattice6overlap` and the host bin-construction driver are
the original (`parboil/benchmarks/cutcp/src/cuda/cutoff6overlap.cu`). Only the
host driver is restructured for the single-run timing harness and JSON output.

## Provenance & fidelity

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/) — the `cutcp` workload. Independent reimplementation, spot-verified to reproduce Parboil's algorithm faithfully (verbatim binning reproduced). See "What changed vs. upstream" below for harness-level differences and the bin-capacity limitation.

## Algorithm (cutoff binning — the faithful part)

- Atoms are geometrically hashed into a **4 Å lattice of bins**, `BIN_DEPTH=8`
  atoms per bin (atoms beyond depth overflow — see below).
- The potential lattice is decomposed into `8×8×8`-point **regions**. Each thread
  block (`blockDim = 8×2×8`, 128 threads, 4 lattice points/thread) computes one
  region, walking only the **neighboring bins within the cutoff** via a
  `__constant__` `NbrList` offset stencil, staging bins through shared memory
  (`AtomBinCache`) for coalesced reuse.
- The kernel is launched **once per x-y plane of regions (per z-slab)**; all
  launches of the single run are accumulated into one kernel record.
- Result regions are transposed back into the lattice on the host (untimed).

## What changed vs. upstream

- Host driver wrapped in the single-run timing harness; timers/`pb_*`/verbose
  output removed.
- Bin **overflow** atoms (bins with > `BIN_DEPTH` atoms) are dropped instead of
  being spilled to a CPU pass. They are also excluded from the CPU reference, so
  the GPU result still verifies.
- Synthetic atoms are generated in-process (no `.pdb`/dataset file).

### Bin-capacity ceiling on `num_atoms` (known limitation)

The bin lattice holds at most `BIN_DEPTH × ⌈lattice_dim·h/4⌉³` atoms in its
interior bins — **8 × 18³ ≈ 46,656 atoms at the default `lattice_dim=144`,
`h=0.5`**. Atoms hashed into a full bin are *dropped* (counted in the stderr
`overflow` field), so `num_atoms` sweep values above ~46k at the default
lattice measure near-identical GPU work: the kernel processes the (bounded)
binned population, not the requested atom count. To scale real work beyond
that, increase `lattice_dim` along with `num_atoms`. This is a known open
finding, not yet fixed.

## Measurement

The kernel sequence runs **exactly once** per process execution (see the root
README "Measurement Methodology"). With `BENCH_PARAM_vector_add_warmup`
(default `true`), a vector-add warmer boosts the GPU clock and evicts the
last-level cache immediately before the timed run, so it is measured
cold-cache at a boosted clock. Repetition = `gpubench run --reps N`.

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_num_atoms` | Number of charged atoms | 40000 |
| `BENCH_PARAM_lattice_dim` | Lattice side nx=ny=nz (multiple of 8) | 144 |
| `BENCH_PARAM_grid_spacing` | Lattice spacing h (Å); 0.5 keeps BIN_LENGTH/(8h)=1 | 0.5 |
| `BENCH_PARAM_cutoff_radius` | Interaction cutoff (Å) | 12.0 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

`grid_spacing` is fixed at 0.5 by the neighbor-list branch condition
(`BIN_LENGTH/(8h)` must be integral). `block_size` is not tunable — the region
scheme fixes it at `8×2×8` (compile-time).

## Build / Run

```bash
make                 # auto-detect platform (cuda / rocm)
./parboil_cutcp
BENCH_PARAM_num_atoms=100000 ./parboil_cutcp
```

## Output

**stdout** — one `kernel` record (`iter:0`) for
`cuda_cutoff_potential_lattice6overlap` with the GPU event time accumulated
over all z-slab launches of the single run, then a `summary` whose
`total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"cuda_cutoff_potential_lattice6overlap","iter":0,"time_ms":12.345678,"host_time_ms":12.501000,"params":{"num_atoms":40000,"lattice_dim":144,"grid_spacing":0.50,"cutoff_radius":12.0,"bin_depth":8,"nbrlist_len":275}}
{"type":"summary","total_time_ms":12.345678}
```

**stderr** — device info, geometry (including bin dims and the overflow /
excluded atom counts), timing, and PASS/FAIL (64 sampled lattice points vs. a
CPU reference over the binned atoms, rel. tol. 0.1%). On FAIL the process
exits non-zero.

Derived metrics (e.g. Ginteractions/sec) are computed downstream from `time_ms`
and params.
