# polybench_jacobi2d (native CUDA)

FP32 Jacobi-2D from PolyBench/GPU 1.0, with CUDA launches and complete
measured-output verification.

## Source and workload

The numerical source is [cavazos-lab/PolyBench-ACC, commit 70ea4ca9047dc4ec9c3ca79b4c91b3b2cc1510e7, CUDA/stencils/jacobi-2d-imper/jacobi2D.cu](https://github.com/cavazos-lab/PolyBench-ACC/blob/70ea4ca9047dc4ec9c3ca79b4c91b3b2cc1510e7/CUDA/stencils/jacobi-2d-imper/jacobi2D.cu).
Its author and redistribution notices are in [UPSTREAM_LICENSE](UPSTREAM_LICENSE).

Each timestep runs **one stencil from A to B, then an interior copy from B to A**. This is the selected PolyBench/GPU variant; later PolyBench/C versions with two stencil sweeps describe a different workload. There are no buffer swaps. Both grids are initialized at every cell using the source formulas:

```cpp
A[i][j] = ((float)i * (j + 2) + 10) / n;
B[i][j] = ((float)(i - 4) * (j - 1) + 11) / n;
```

For every interior cell, each step evaluates the following FP32 expression in the source's center, left, right, down, up addition order, then copies the computed interior back to A:

```cpp
B[i][j] = 0.2f * (A[i][j] + A[i][j-1] + A[i][j+1]
                         + A[i+1][j] + A[i-1][j]);
A[i][j] = B[i][j];  // separate GPU copy kernel after the complete stencil
```

A and B keep their own distinct initialized boundaries for the entire execution. The production mapping assigns one interior cell per thread with a 16×16 block; the pin uses 32×8 and threads covering boundaries. Runtime sizing, mapping, event timing, admission and verification are adaptations. The numerical initialization, operation order and two-stage timestep follow the pin.

The manifest retains 20 sizes from 64 through 8192 and 10, 50 or 200 timesteps: **60 suite-selected workloads**. These use the pinned algorithm but are not named upstream presets (whose header selects 20 timesteps). The default is 1024×1024, 50 timesteps. `tsteps` is a workload multiplier, not statistical repetition. Older results from the random-input, zero-boundary, swap implementation have a different workload and should be kept separate.

## Build and run

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./polybench_jacobi2d --n 1024 --tsteps 50
```

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. `BENCH_PARAM_n` and `BENCH_PARAM_tsteps` override the corresponding CLI flags. Repeated CLI flags keep their first value. Malformed values, unknown flags, `n < 3`, `n > 46340`, and nonpositive or out-of-int-range timesteps fail before device initialization. The cell-index bound is `n*n <= INT_MAX`; device geometry and allocation checks may impose tighter limits. Manually admitted sizes outside the manifest are not covered by the declared H100 collection domain.

One entry vector-add warmer runs after input uploads and before the complete algorithm. Set `BENCH_PARAM_vector_add_warmup=false` to disable it. It is intended cache/clock preparation, without a cache-residency or clock guarantee. Each process executes the requested algorithm once; use the harness's `--reps` for separate executions.

## Timing and interpretation

Both the stencil and copy are timed GPU kernels. Standard output contains two accumulated kernel records at `iter=0`, followed by a summary equal to the sum of their GPU event times. Each record includes `n`, `tsteps`, actual block/grid dimensions, padded `threads_per_launch`, `interior_cells`, `kernel_launches=tsteps`, `updated_cells=(n-2)^2*tsteps`, and these identifiers:

- `precision=fp32`, `boundary=fixed_initial_values`
- `input=polybench_gpu_jacobi_v1`
- `algorithm=polybench_gpu_jacobi_copyback_v1`
- `verification=full_cpu_exact_v1`

Input generation, uploads, the entry warmer, downloads and CPU verification are outside the reported kernel total. GPU events and host timing follow the shared timing contract. Per-kernel metadata counts that record's work: there are `2*tsteps` timed launches in the complete algorithm.

Use elapsed time or interior updates per second for performance comparisons. Source-level traffic is five float loads plus one store per stencil update, and one load plus one store per copy update. Multiplying those counts by four bytes gives **logical** traffic (24 and 8 bytes per update); it does not measure DRAM traffic because caches reuse values. Compare results only with matching algorithm, input, size, steps and warmer settings.

## Verification and qualification

After timing, the program downloads **both complete measured grids**, including every boundary. An independently regenerated CPU reference executes the full requested recurrence and copy-back. Every GPU and CPU value must be finite and numerically exactly equal. There is no small auxiliary GPU run or sampled verification. Exact equality is justified by the matching FP32 per-cell expression order, without a reduction or fused multiply-add in the stencil. Compiler options that reassociate arithmetic change this contract.

A mismatch, unavailable reference, allocation failure or device error exits nonzero. Mismatch diagnostics show at most ten locations while counting all errors. Timing records can precede a failed verification; the harness must retain their failed run status and exclude them from accepted performance results.

Host regression tests compare initialization and complete trajectories against frozen pinned CPU bodies, check argument bounds and poison both sides of the comparator. The measured-result suite runs every declared workload, partial blocks and odd/even steps, then changes actual device output at boundaries, interiors and the largest grids' tails. Finite one-ULP changes and NaN/infinities must fail. A separate GPU test compares every input/output byte with the pinned GPU kernels under their original 32×8 layout.

```bash
# [TODO for cuda] these tests exercise ../hip/; not yet ported
python3 tests/jacobi_host.py --compiler clang++
python3 tests/jacobi_host.py --compiler g++ --all-inputs
python3 tests/jacobi_measured_result.py --platform cuda --work-dir /tmp/jacobi-check
make -C tier2/polybench_jacobi2d/cuda test
```

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. Run the Python commands from the repository root. CI runs host tests and
compiles GPU tests; H100 execution evidence is recorded in the repair PR.
Trace/capture/replay qualification remains deferred.