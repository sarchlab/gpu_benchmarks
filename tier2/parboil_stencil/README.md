# tier2/parboil_stencil — 3D Jacobi Stencil (Parboil)

7-point 3D Jacobi stencil computation over a cubic grid using ping-pong buffers. Each interior cell is updated from its six face-adjacent neighbors. Boundary cells remain fixed. A classic memory-bandwidth-bound benchmark.

## Provenance & fidelity

This implementation restores the operator in the [pinned Parboil source mirror](https://github.com/abduld/Parboil/tree/a4c63da28f45699e9a274cbc49af3e067162aa2b/benchmarks/stencil/src),
commit `a4c63da28f45699e9a274cbc49af3e067162aa2b`. Its CPU, `cuda_base`, and
optimized CUDA versions agree on the subtractive-center operator and float
coefficient definitions below. The current kernel retains an independent
16×16 direct-global launch layout; it does not reproduce the optimized
upstream kernel's shared-memory tiling or thread coarsening.

## Algorithm

For each interior cell `(x,y,z)`:
```
out = (in[z+1] + in[z-1] + in[y+1] + in[y-1] + in[x+1] + in[x-1]) * c1
      - in[center] * c0
c0 = 1.0f / 6.0f
c1 = 1.0f / 6.0f / 6.0f
```
The notation denotes six face-adjacent neighbors. The source preserves the
`cuda_base` neighbor order. Both buffers start with identical input boundaries;
the kernel leaves those boundaries fixed through exactly `num_timesteps`
updates, swapping source and destination after each launch. The final buffer
is selected by timestep parity.

The current cubic grid sweep, timestep sweep and seed 42 `rand()%1000/1000.0f`
input generator are suite configurations. Upstream reads external float grids;
this is not a claim to reproduce its supplied datasets or launch strategy.
Host-library RNG portability remains owner-deferred R-27. The old positive
averaging operator was a different workload; recollect its measurements.

## Measurement

The stencil (all `num_timesteps` launches) runs **exactly once** per process
execution (see the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`.

Ordinary runs verify every cell of the final measured state after timing,
using the original input and an independent double CPU reference for all
requested timesteps. Boundaries must match exactly; interiors use the qualified
`max(1e-7, 2e-5*abs(reference))` tolerance. Non-finite values fail. The final
ping-pong buffer is downloaded once into the existing host input allocation,
after the reference has consumed that input. No extra GPU stencil pass runs.
A mismatch returns exit code 1, even though timing records have already been
emitted. The kernel record identifies this policy as `full_cpu_v1`.

The full reference temporarily needs two double grids: 2 GiB at the largest
512³ configuration, in addition to the existing 512 MiB host input. One
reference scratch grid is released before the final 512 MiB device-to-host
copy. CPU work, reference storage and download are outside kernel timing but
count toward the process timeout; qualify capacity and runtime on the target
host before collection.

## Parameters

| Env var (or `--flag`) | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_grid_dim` / `--grid_dim` | Grid dimension N (N×N×N) | 64 |
| `BENCH_PARAM_num_timesteps` / `--num_timesteps` | Stencil time-steps per run | 10 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `parboil_stencil.hip` | HIP source — 3D stencil kernel + self-contained host utilities |
| `cpu_reference.h` | Full double CPU reference and finite comparison policy |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./parboil_stencil                                          # defaults: 64×64×64, 10 timesteps
./parboil_stencil --grid_dim 128
./parboil_stencil --grid_dim 64 --num_timesteps 20
```

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record (`iter:0`) carrying the stencil kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) accumulated over the run's `num_timesteps` launches, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"stencil3d","iter":0,"time_ms":1.234500,"host_time_ms":1.301100,"params":{"grid_dim":64,"num_timesteps":10,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":1.234500}
```

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Grid: 64×64×64  |  Timesteps/pass: 10

stencil3d gpu = 1.234 ms, host = 1.301 ms
Total GPU time: 1.234 ms (10 timesteps/pass)
Stencil verification: PASS (262144 cells, 0 mismatches)
```

## Performance Metrics (derived downstream from time_ms and params)

- **GFLOPS** = 8 × interior_cells × num_timesteps / time_s / 1e9
- **GB/s** = 8 × sizeof(float) × interior_cells × num_timesteps / time_s / 1e9
  (logical traffic: 7 reads + 1 write per interior cell per time-step; actual
  device traffic and instruction counts require target qualification)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `parboil_stencil.hip`
- Input data is generated synthetically — no external input files required

## Operator qualification

`make test PLATFORM=cuda` (or `rocm`) checks the actual production pass against
an independent dense double reference using the upstream float coefficient
values. Seven exact unit impulses isolate the center and six neighbor terms;
30 signed-grid cases cover rectangular shapes, partial tiles, one/two/odd/even
and 100-timestep sequences. The first destination interior is NaN-poisoned;
non-finite outputs fail, both buffers must preserve boundaries and allocation
guards, and the final buffer must match timestep parity. General outputs use
`max(1e-7, 2e-5*abs(reference))` tolerance. Unit impulses require exact float
outputs.

`BENCH_PARAM_vector_add_warmup=false ./bench_output_test --case N T` checks a
particular declared configuration; `--impulse 0` through `--impulse 6` isolate
the seven terms. CI compiles the test, while execution requires a GPU.
Main rejects grids without an interior, overflowing signed N³ indices, and
nonpositive timestep counts before timing.

`tests/stencil_reference_test.cpp` checks the complete CPU helper against an
independent dense transition matrix, including all 27 basis vectors of a 3³
grid, rectangular shapes, signed inputs, fixed boundaries and timestep parity.
It also checks finite-value rejection and both tolerance branches.

```bash
python3 tests/stencil_measured_result.py --platform cuda --work-dir /tmp/stencil-measured-result
```

Run this from the repository root with a new output directory (use `rocm` on
AMD). It compiles a disposable copy and corrupts the actual final device
buffer after timing. Its 212 cases include 16 controls and 196 faults: finite,
NaN and ±Inf values at interior and boundary positions, one-ULP boundary
changes, odd/even/100-step sequences, and 12 simultaneous mismatches to check
the diagnostic cap without truncating verification. CI runs the host oracle
and compiles this regression; a GPU is required to execute it. The existing
production-pass operator test keeps its separate CPU oracle.

R-19 AMD execution and R-23/R-24 trace qualification remain separate.
Functional profiling is not performance collection.
