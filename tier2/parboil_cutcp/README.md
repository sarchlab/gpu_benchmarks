# tier2/parboil_cutcp — Coulombic Potential with Cutoff (Parboil)

Benchmarks direct summation of electrostatic charge interactions on a 3D grid with a distance cutoff. Each GPU thread computes the potential at one grid point by iterating over all atoms within the cutoff radius, computing q/r contributions. Emits per-iteration kernel times; throughput in Ginteractions/sec is derived downstream.

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/parboil.aspx) CUTCP workload.

## Algorithm

- **num_atoms** charged atoms (default: 10,000) randomly placed in the domain
- Fixed 64×64×64 grid (262,144 grid points) with configurable spacing
- For each grid point, iterate over all atoms and accumulate Coulomb potential (q/r) for atoms within cutoff distance
- Cutoff radius limits the interaction range (default: 12.0 Å)
- Ginteractions/sec = (num_atoms × grid_points) / time_sec / 10⁹ (derived downstream from the emitted times)

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_num_atoms` | Number of charged atoms | 10000 |
| `BENCH_PARAM_block_size` | Threads per block | 128 |
| `BENCH_PARAM_grid_spacing` | Grid point spacing (Å) | 0.5 |
| `BENCH_PARAM_cutoff_radius` | Interaction cutoff (Å) | 12.0 |
| `BENCH_PARAM_num_warmup` | Warmup launches (untimed) | 3 |
| `BENCH_PARAM_num_iters` | Timed launches, each recorded | 7 |

## Files

| File | Description |
|------|-------------|
| `parboil_cutcp.hip` | HIP source — portable ROCm/CUDA implementation |
| `parboil_cutcp.metal` | Metal compute shader — potential kernel |
| `parboil_cutcp_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./parboil_cutcp                                    # defaults
BENCH_PARAM_num_atoms=50000 ./parboil_cutcp        # more atoms
BENCH_PARAM_cutoff_radius=6.0 ./parboil_cutcp      # smaller cutoff
```

## Output

**stdout** — JSON lines, one `kernel` record per timed iteration followed by a
`summary` whose `total_time_ms` is the sum of all iterations' GPU event times:
```json
{"type":"kernel","name":"cutcp_kernel","iter":0,"time_ms":12.345600,"host_time_ms":12.401000,"params":{"num_atoms":10000,"block_size":128,"grid_spacing":0.50,"cutoff_radius":12.0}}
...
{"type":"summary","total_time_ms":86.419200}
```

Derived metrics (e.g. Ginteractions/sec) are computed downstream from `time_ms` and params.

**stderr** — human-readable:
```
Device: Apple M1 Max
CUTCP  |  Atoms: 10000  |  Grid: 64^3 = 262144  |  Cutoff: 12.0  |  Spacing: 0.50  |  Iterations: 3 warmup + 7 timed

cutcp_kernel iter 0: gpu = 12.346 ms, host = 12.401 ms
...
Total GPU time: 86.419 ms over 7 iterations
PASS
```

## Verification

- 64 randomly-selected grid points are compared against CPU reference computation
- Reports PASS/FAIL with mismatch details (relative tolerance 0.1%)

## Platform Notes

- **CUDA/ROCm**: Uses float4 SoA for atom data for coalesced memory access
- **Metal**: Loads `parboil_cutcp.metal` shader at runtime from the same directory as the binary
