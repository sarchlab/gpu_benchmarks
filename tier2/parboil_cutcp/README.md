# tier2/parboil_cutcp — Coulombic Potential with Cutoff (Parboil, cutoff binning)

Faithful port of the Parboil **CUTCP** benchmark: the device kernel
`cuda_cutoff_potential_lattice6overlap` and the host bin-construction driver are
the original (`parboil/benchmarks/cutcp/src/cuda/cutoff6overlap.cu`). Only the
host driver is restructured for the single-run timing harness and JSON output.

## Provenance & fidelity

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/) — the `cutcp` workload. Independent reimplementation, spot-verified to reproduce Parboil's algorithm faithfully (verbatim binning reproduced). See "What changed vs. upstream" below for harness-level differences and the bin-capacity handling.

## Algorithm (cutoff binning — the faithful part)

- Atoms are geometrically hashed into a **4 Å lattice of bins**, `BIN_DEPTH=16`
  atoms per bin (a config that would exceed this capacity is rejected — see below).
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
- The upstream benchmark spilled bins with more than `BIN_DEPTH` atoms to a
  separate CPU pass. This port has no CPU spill pass, so `BIN_DEPTH` is raised
  from 8 to **16** — enough to hold the Poisson tail at the design density
  (~7 atoms/bin) — and any config that *still* overflows a bin is **rejected
  before measurement** rather than silently dropping atoms (issue #63, C-02).
  See the bin-capacity section below.
- Synthetic atoms are generated in-process (no `.pdb`/dataset file).

### Bin-capacity constraint on `num_atoms` × `lattice_dim`

The bin lattice holds at most `BIN_DEPTH × ⌈lattice_dim·h/4⌉³` atoms across its
interior bins, and per-bin occupancy must stay within `BIN_DEPTH=16`. If any bin
would overflow, the run **fails before measurement** with a diagnostic naming
the offending config (non-zero exit, no `summary` emitted) instead of dropping
the excess and measuring a silently-truncated problem. Earlier behavior dropped
overflow atoms *and* excluded them from the CPU reference, so a truncated
problem still "verified" — that is now fixed (issue #63, C-02 / D-02(b)).

The declared sweep is constrained to configs that fit: `num_atoms` runs
5,000–45,000 (all within `BIN_DEPTH` at the default `lattice_dim=144`, peak
occupancy verified across every `num_atoms × lattice_dim` combination), and
`lattice_dim` is restricted to `≥144`. To scale real work beyond the top of the
sweep, increase `lattice_dim` alongside `num_atoms` so peak per-bin occupancy
stays `≤ BIN_DEPTH`.

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
BENCH_PARAM_num_atoms=45000 ./parboil_cutcp                        # top of the sweep at the default lattice
BENCH_PARAM_num_atoms=100000 BENCH_PARAM_lattice_dim=256 ./parboil_cutcp  # larger problem needs a larger lattice
```

A config whose peak per-bin occupancy exceeds `BIN_DEPTH` (e.g.
`BENCH_PARAM_num_atoms=100000` at the default `lattice_dim=144`) is rejected
before measurement: it prints an `ERROR` line, emits no `summary`, and exits
non-zero.

## Output

**stdout** — one `kernel` record (`iter:0`) for
`cuda_cutoff_potential_lattice6overlap` with the GPU event time accumulated
over all z-slab launches of the single run, then a `summary` whose
`total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"cuda_cutoff_potential_lattice6overlap","iter":0,"time_ms":12.345678,"host_time_ms":12.501000,"params":{"num_atoms":40000,"lattice_dim":144,"grid_spacing":0.50,"cutoff_radius":12.0,"bin_depth":16,"nbrlist_len":335}}
{"type":"summary","total_time_ms":12.345678}
```

**stderr** — device info, geometry (including bin dims and the overflow /
excluded atom counts), timing, and PASS/FAIL (64 sampled lattice points vs. a
CPU reference over the binned atoms — with overflow rejection this is the full
requested atom set, not a truncated subset; rel. tol. 0.1%). On FAIL, or on a
bin-overflow rejection, the process exits non-zero.

Derived metrics (e.g. Ginteractions/sec) are computed downstream from `time_ms`
and params.
