# tier2/cuda_transpose — Shared-Memory Tiled Matrix Transpose

Benchmarks matrix transpose using shared memory tiling for a square N×N single-precision matrix (default 4096×4096). Compares naive vs. optimized (bank-conflict-free) transpose kernels. Emits per-iteration kernel times; effective memory bandwidth in GB/s is derived downstream.

Based on the classic [CUDA Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) optimization pattern.

## Algorithm

Two transpose variants are benchmarked:

| Variant | Description |
|---------|-------------|
| **Naive** | Each thread reads one element (coalesced) and writes one element (strided). Simple but suffers from uncoalesced writes. |
| **Optimized** | Uses shared memory (threadgroup memory on Metal) with `TILE_DIM+1` padding to avoid bank conflicts. Both reads and writes are coalesced. |

- **TILE_DIM** = 32, **BLOCK_ROWS** = 8 — each threadblock handles a 32×32 tile, with 32×8 = 256 threads per block
- **Effective bandwidth** = 2 × N × N × sizeof(float) / time\_sec / 1e9 (GB/s) — derived downstream from the emitted times

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_size` | Matrix dimension N (N×N) | 4096 |
| `BENCH_PARAM_block_size` | Threads per block (descriptor) | 256 |
| `BENCH_PARAM_tile_dim` | Tile dimension (descriptor) | 32 |
| `BENCH_PARAM_num_warmup` | Warmup launches per variant (untimed) | 3 |
| `BENCH_PARAM_num_iters` | Timed launches per variant, each recorded | 7 |

## Files

| File | Description |
|------|-------------|
| `cuda_transpose.hip` | HIP source — naive + optimized transpose kernels |
| `cuda_transpose.metal` | Metal compute shaders — both variants using threadgroup memory |
| `cuda_transpose_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./cuda_transpose                # default: 4096×4096
./cuda_transpose --size 2048    # custom matrix size
```

## Output

**stdout** — JSON lines: one `kernel` record per timed iteration of each
variant (`transpose_naive`, `transpose_optimized`), then a `summary` whose
`total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"transpose_naive","iter":0,"time_ms":1.234500,"host_time_ms":1.301000,"params":{"size":4096,"block_size":256,"tile_dim":32}}
...
{"type":"kernel","name":"transpose_optimized","iter":0,"time_ms":0.567800,"host_time_ms":0.621000,"params":{"size":4096,"block_size":256,"tile_dim":32}}
...
{"type":"summary","total_time_ms":12.616100}
```

Derived metrics (GB/s per variant) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: Apple M2 (id=0)
Matrix size: 4096×4096  |  Warmup: 3  |  Timed: 7

transpose_naive iter 0: gpu = 1.234 ms, host = 1.301 ms
...
  Verification: PASS
transpose_optimized iter 0: gpu = 0.568 ms, host = 0.621 ms
...
  Verification: PASS
Total GPU time: 12.616 ms over 7 iterations per variant
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_transpose.hip`
- **Metal**: compiled with `clang++` from `cuda_transpose_metal.mm`; the Metal shader (`cuda_transpose.metal`) is loaded and compiled at runtime from the same directory as the binary
