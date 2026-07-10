# tensor_core_throughput

Measures Tensor Core throughput (TOPS — Tera Operations Per Second) using
hardware matrix multiply-accumulate instructions.

## What it measures

- **NVIDIA (CUDA):** Uses `nvcuda::wmma` API with 16×16×16 FP16 tiles and
  optionally 16×16×8 TF32 tiles (Ampere / sm_80+).
- **AMD (ROCm):** Uses `rocwmma` library with 16×16×16 FP16 tiles.
- **Apple (Metal):** Uses `simdgroup_matrix_multiply_accumulate` with 8×8
  half-precision tiles.

The benchmark performs a full M×N×K GEMM (default 4096×4096×4096) using
tiled WMMA/simdgroup operations and records per-iteration kernel times;
throughput is derived downstream.

## Build

```bash
make                    # auto-detect platform
make PLATFORM=cuda      # NVIDIA CUDA (requires sm_70+)
make PLATFORM=rocm      # AMD ROCm (requires CDNA / rocWMMA)
make PLATFORM=metal     # Apple Metal (requires M1+)
```

## Run

```bash
./tensor_core_throughput          # default 4096×4096×4096
./tensor_core_throughput 2048     # custom M=N=K=2048
```

## Output

Human-readable progress goes to stderr; stdout carries JSON lines — one
record per timed iteration of each precision that runs, plus a final summary:

```
{"type":"kernel","name":"wmma_fp16_gemm","iter":0,"time_ms":12.345678,"host_time_ms":12.456789,"params":{"matrix_dim":4096,"precision":"fp16"}}
...
{"type":"kernel","name":"wmma_tf32_gemm","iter":0,"time_ms":18.234567,"host_time_ms":18.345678,"params":{"matrix_dim":4096,"precision":"all"}}   # precision=all/tf32, CUDA sm_80+ only
...
{"type":"summary","total_time_ms":214.061717}
```

Every timed iteration is emitted individually — no averaging; the summary
`total_time_ms` is the sum of all iterations' GPU event times. TOPS is not
printed by the binary.

## Methodology

- `BENCH_PARAM_warmup` (default 3) warmup iterations followed by
  `BENCH_PARAM_iteration` (default 7) timed iterations, each recorded
  individually.
- TOPS is derived downstream from each record's `time_ms` and `matrix_dim`:
  `TOPS = 2 × M × N × K / time_s / 10¹²`
- Each warp/simdgroup computes one output tile; the grid covers the full
  M×N output matrix.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_matrix_dim` (or argv[1]) | 4096 | Square matrix dimension M=N=K (multiple of 16) |
| `BENCH_PARAM_precision` | fp16 | WMMA fragment precision (`fp16`, `tf32`, `all`) |
| `BENCH_PARAM_warmup` | 3 | Warm-up kernel launches (not timed) |
| `BENCH_PARAM_iteration` | 7 | Timed kernel launches, each recorded individually |

## Requirements

| Platform | Requirement |
|----------|-------------|
| CUDA     | sm_70+ (Volta or later) for FP16 WMMA; sm_80+ for TF32 |
| ROCm     | CDNA architecture with rocWMMA support |
| Metal    | Apple Silicon M1 or later |
