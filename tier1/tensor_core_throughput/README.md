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
tiled WMMA/simdgroup operations and reports the achieved throughput.

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

Human-readable summary plus CSV lines to stdout:

```
tensor_core_throughput,fp16,<time_ms>,<TOPS>
tensor_core_throughput,tf32,<time_ms>,<TOPS>    # CUDA sm_80+ only
```

## Methodology

- 5 warmup iterations followed by 5 timed iterations (report average).
- `TOPS = 2 × M × N × K / time_s / 10¹²`
- Each warp/simdgroup computes one output tile; the grid covers the full
  M×N output matrix.

## Requirements

| Platform | Requirement |
|----------|-------------|
| CUDA     | sm_70+ (Volta or later) for FP16 WMMA; sm_80+ for TF32 |
| ROCm     | CDNA architecture with rocWMMA support |
| Metal    | Apple Silicon M1 or later |
