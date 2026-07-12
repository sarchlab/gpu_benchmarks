# tier2/cuda_transpose — Shared-Memory Tiled Matrix Transpose

Benchmarks matrix transpose for a square N×N single-precision matrix (default 4096×4096). Compares naive vs. optimized (bank-conflict-free) transpose kernels, each timed exactly once. Effective memory bandwidth in GB/s is derived downstream.

## Provenance & fidelity

Faithful reproduction of the classic [CUDA Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) optimization pattern: all transpose variants (naive + shared-memory tiled, bank-conflict-free) are reproduced.

## Algorithm

Two transpose variants are benchmarked:

| Variant | Description |
|---------|-------------|
| **Naive** | Each thread reads one element (coalesced) and writes one element (strided). Simple but suffers from uncoalesced writes. |
| **Optimized** | Uses shared memory with `TILE_DIM+1` padding to avoid bank conflicts. Both reads and writes are coalesced. |

- **TILE_DIM** = 32, **BLOCK_ROWS** = 8 — each threadblock handles a 32×32 tile, with 32×8 = 256 threads per block (compile-time constants)
- **Effective bandwidth** = 2 × N × N × sizeof(float) / time\_sec / 1e9 (GB/s) — derived downstream from the emitted times

## Measurement

Each variant launches **exactly once** per process execution (see the root
README "Measurement Methodology"). With `BENCH_PARAM_vector_add_warmup`
(default `true`), a vector-add warmer boosts the GPU clock and evicts the
last-level cache immediately before each timed launch. Repetition =
`gpubench run --reps N`.

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_size` / `--size` | Matrix dimension N (N×N) | 4096 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `cuda_transpose.hip` | HIP source — naive + optimized transpose kernels |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./cuda_transpose                # default: 4096×4096
./cuda_transpose --size 2048    # custom matrix size
```

## Output

**stdout** — JSON lines: one `kernel` record (`iter:0`) per variant
(`transpose_naive`, `transpose_optimized`), then a `summary` whose
`total_time_ms` is the sum of both recorded GPU event times:
```
{"type":"kernel","name":"transpose_naive","iter":0,"time_ms":1.234500,"host_time_ms":1.301000,"params":{"size":4096}}
{"type":"kernel","name":"transpose_optimized","iter":0,"time_ms":0.567800,"host_time_ms":0.621000,"params":{"size":4096}}
{"type":"summary","total_time_ms":1.802300}
```

Derived metrics (GB/s per variant) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 4096×4096

transpose_naive: gpu = 1.234 ms, host = 1.301 ms
  Verification: PASS
transpose_optimized: gpu = 0.568 ms, host = 0.621 ms
  Verification: PASS
Total GPU time: 1.802 ms
```

Each variant's output is verified against a CPU transpose (first 1000
elements); on FAIL the process exits non-zero.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_transpose.hip`
