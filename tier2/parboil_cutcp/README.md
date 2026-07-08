# tier2/parboil_cutcp — Coulombic Potential with Cutoff (Parboil, cutoff binning)

Faithful port of the Parboil **CUTCP** benchmark: the device kernel
`cuda_cutoff_potential_lattice6overlap` and the host bin-construction driver are
the original (`parboil/benchmarks/cutcp/src/cuda/cutoff6overlap.cu`). Only the
host driver is restructured for the per-iteration timing harness and JSON output.

## Algorithm (cutoff binning — the faithful part)

- Atoms are geometrically hashed into a **4 Å lattice of bins**, `BIN_DEPTH=8`
  atoms per bin (atoms beyond depth overflow — see below).
- The potential lattice is decomposed into `8×8×8`-point **regions**. Each thread
  block (`blockDim = 8×2×8`, 128 threads, 4 lattice points/thread) computes one
  region, walking only the **neighboring bins within the cutoff** via a
  `__constant__` `NbrList` offset stencil, staging bins through shared memory
  (`AtomBinCache`) for coalesced reuse.
- The kernel is launched **once per x-y plane of regions (per z-slab)**; every
  launch within a pass is accumulated into one kernel record.
- Result regions are transposed back into the lattice on the host (untimed).

## What changed vs. upstream

- Host driver wrapped in the timing harness; timers/`pb_*`/verbose output removed.
- Bin **overflow** atoms (bins with > `BIN_DEPTH` atoms) are dropped instead of
  being spilled to a CPU pass. They are also excluded from the CPU reference, so
  the GPU result still verifies. Default density (~7 atoms/bin) keeps overflow small.
- Synthetic atoms are generated in-process (no `.pdb`/dataset file).

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_num_atoms` | Number of charged atoms | 40000 |
| `BENCH_PARAM_lattice_dim` | Lattice side nx=ny=nz (multiple of 8) | 144 |
| `BENCH_PARAM_grid_spacing` | Lattice spacing h (Å); 0.5 keeps BIN_LENGTH/(8h)=1 | 0.5 |
| `BENCH_PARAM_cutoff_radius` | Interaction cutoff (Å) | 12.0 |
| `BENCH_PARAM_num_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_num_iters` | Timed passes, each recorded | 7 |

`grid_spacing` is fixed at 0.5 by the neighbor-list branch condition
(`BIN_LENGTH/(8h)` must be integral). `block_size` is not tunable — the region
scheme fixes it at `8×2×8` (compile-time).

## Build / Run

```bash
make                 # auto-detect platform (cuda / rocm / metal)
./parboil_cutcp
BENCH_PARAM_num_atoms=100000 ./parboil_cutcp
```

## Output

**stdout** — one `kernel` record per timed pass for
`cuda_cutoff_potential_lattice6overlap` (accumulated GPU time of all z-slab
launches), then a `summary` whose `total_time_ms` is the sum of all recorded GPU
event times. **stderr** — device info, per-pass timing, PASS/FAIL (64 sampled
lattice points vs. a CPU reference over the binned atoms, rel. tol. 0.1%).

Derived metrics (e.g. Ginteractions/sec) are computed downstream from `time_ms`
and params.

> **Metal note:** `parboil_cutcp.metal` / `parboil_cutcp_metal.mm` still contain
> the earlier brute-force implementation and were not updated by this faithful
> port (HIP/CUDA only).
