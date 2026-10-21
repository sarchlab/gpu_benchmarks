# Parboil LBM: D3Q19 lid-driven cavity (native CUDA)

This benchmark restores Parboil's CUDA scatter collision/streaming operator,
cell-level obstacle bounce-back, acceleration planes, equilibrium initialization,
SoA populations and flag field. The upstream source is
[`abduld/Parboil` at `a4c63da28f45699e9a274cbc49af3e067162aa2b`](https://github.com/abduld/Parboil/tree/a4c63da28f45699e9a274cbc49af3e067162aa2b/benchmarks/lbm/src).
`UPSTREAM_LICENSE` retains its Illinois/NCSA license.

## Workload and upstream differences

The GPU collision arithmetic and D3Q19 ordering are ported from
`src/cuda/lbm_kernel.cu`. One block handles an x row; `block_size = grid_dim`.
The relaxation rate is fixed at upstream `omega = 1.95`. Both ping-pong buffers
start with equilibrium populations at all physical cells, zero padding, and
identical flags. All six enclosure faces are obstacles. The inner z planes
accelerate eligible cells to `(0.005, 0.002, 0)`.

The owner-selected input uses reproducible cubic grids and the upstream-supported
null obstacle-file option. It has **no additional input obstacles**, but retains
the enclosure walls. It differs from the upstream fixed 120×120×150 grid and its
published obstacle datasets. Padding stays at x+8 and z+4; two exterior z planes
provide the real pointer margin. Dimensions and strides are runtime values.
This is the upstream operator on our synthetic input, not a reproduction of the
published dataset results. Prior results from the former local BGK/link-bounce
implementation must be recollected.

The CPU reference retains the independent `src/cpu/lbm.c` FP32 trajectory and
its double-literal equilibrium coefficients. It uses AoS storage and bounded
host workers. Initialization is adapted to the CUDA zero-padding/physical-cell
contract: the original CPU driver's equilibrium-filled margins are not an
oracle for the CUDA driver. Flag reads use defined bit copies instead of
upstream type-punning. No CPU work is included in GPU timing.

## Inputs and measurement

| Environment variable | Meaning | Default |
|---|---|---|
| `BENCH_PARAM_grid_dim` | Cubic dimension | 64 |
| `BENCH_PARAM_num_timesteps` | Timesteps in the single measured trajectory | 100 |
| `BENCH_PARAM_vector_add_warmup` | Standard cold-cache clock warmer | true |

The old `BENCH_PARAM_tau` and `BENCH_PARAM_block_size` overrides are rejected;
these quantities are fixed or derived by the upstream operator. The manifest
retains 26 dimensions and three timestep counts (78 candidate combinations).
`256³ / 500 steps` is explicitly excluded before GPU access or timing, per the
owner's choice to retain the upstream accuracy requirement. Candidate membership
does not establish qualification: only complete successful runs within the
60-second process budget belong in performance collection.
The CLI currently generates the full Cartesian product, so an unfiltered sweep
also attempts the excluded point and reports a failed run/session at the end.
This expected rejection is not a qualified measurement. Manifest-level skipping
and platform-specific collection profiles remain separate work.

Dimensions must be at least 3 and steps positive. Padded storage must satisfy
`20 * (N+8) * N * (N+8) <= INT_MAX`, checked before multiplication or GPU access.
The actual device must support the upstream row launch. Environment integer
parsing retains `atoi` behavior; command-line flags are ignored. These admission
checks are not host/device memory-capacity guarantees.

Both buffers are initialized once, followed by the standard untimed warmer,
then exactly `num_timesteps` timed kernel launches. Pointer swaps are untimed.
One `lbm_collide_stream_kernel` JSON record carries their summed GPU and host
launch times; the summary repeats the GPU total. Repetitions use separate
`gpubench run --reps N` processes. MLUPS is derived from
`grid_dim^3 * num_timesteps / time_ms / 1000`.

## Complete verification

Every ordinary execution computes the entire independent CPU trajectory and
checks all physical cells, including every boundary and the final cell:

- All 19 populations must be finite and within absolute error `1e-5`.
- The velocity vector from the upstream FP32 output recipe must have L2 error
  at most the upstream `1e-5`; nonfinite velocity comparisons fail.
- All physical flag fields must equal the prescribed input flags.

The process prints checked-cell/population counts, mismatch count and maximum
errors to stderr, and exits nonzero on failure. Padding is workspace rather
than physical simulation output. Timing records emitted before a failed
verification do not qualify the run for collection.

The excluded 256³/500 assessment completed in about 42 seconds, but its maximum
velocity error was `1.0729538e-5`; the maximum population error was
`6.5863132e-6`. We retain the threshold instead of fitting it to that point.
These values are diagnostic qualification evidence, not published performance
measurements.

## Build and regression checks

```sh
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
BENCH_PARAM_grid_dim=64 BENCH_PARAM_num_timesteps=100 ./parboil_lbm
```

From the repository root:

```sh
python3 tests/lbm_input_ranges.py --compiler clang++ --work-dir /tmp/lbm-inputs
c++ -O1 -std=c++14 -pthread -fsanitize=address,undefined tests/lbm_host.cpp -o /tmp/lbm-host
/tmp/lbm-host
python3 tests/lbm_measured_result.py --platform cuda --work-dir /tmp/lbm-measured
```

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. The measured-result regression compiles test-only copies; production has no
fault-injection controls. It rejects corrupted tail populations, nonfinite
outputs/references, wrong flags and a missing timestep, with both warmer modes.
It also compares all allocated GPU entries and initialization against frozen
upstream CUDA sources for odd/even trajectories and multiple dimensions. Those
fixtures retain upstream code (whitespace normalized); only compile-time cubic
dimensions and the CUDA runtime wrapper are adapted. CPU host checks cover an
analytical first step, serial/parallel equivalence, and complete verifier
coverage. CI runs host tests and compiles GPU tests; actual GPU runs are
separate.