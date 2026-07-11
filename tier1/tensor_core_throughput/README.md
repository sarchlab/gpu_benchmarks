# tensor_core_throughput

Measures Tensor Core throughput (TFLOPS — Tera Floating-Point Operations Per
Second) using hardware matrix multiply-accumulate instructions.

## What it measures

- **NVIDIA (CUDA):** Uses `nvcuda::wmma` API with 16×16×16 FP16 tiles and
  optionally 16×16×8 TF32 tiles (Ampere / sm_80+).
- **AMD (ROCm):** Uses `rocwmma` library with 16×16×16 FP16 tiles.

The benchmark performs a full M×N×K GEMM (default 4096×4096×4096) using
tiled WMMA operations and records the raw kernel time; throughput is derived
downstream.

## Build

```bash
make                    # auto-detect platform
make PLATFORM=cuda      # NVIDIA CUDA (requires sm_70+)
make PLATFORM=rocm      # AMD ROCm (requires CDNA / rocWMMA)
```

## Run

```bash
./tensor_core_throughput          # default 4096×4096×4096
./tensor_core_throughput 2048     # custom M=N=K=2048
```

## Output

Human-readable progress goes to stderr; stdout carries JSON lines — one
record per precision kernel that runs (timed exactly once), plus a final
summary:

```
{"type":"kernel","name":"wmma_fp16_gemm","iter":0,"time_ms":12.345678,"host_time_ms":12.456789,"params":{"matrix_dim":4096,"precision":"fp16"}}
{"type":"kernel","name":"wmma_tf32_gemm","iter":0,"time_ms":18.234567,"host_time_ms":18.345678,"params":{"matrix_dim":4096,"precision":"tf32"}}   # precision=tf32, CUDA sm_80+ only
{"type":"summary","total_time_ms":30.580245}
```

The summary `total_time_ms` is the sum of the emitted records' GPU event
times. TFLOPS is not printed by the binary.

## Methodology

- Each execution launches and times the selected kernel **exactly once** —
  there is no in-program warmup or iteration loop. Unless
  `BENCH_PARAM_vector_add_warmup` is set to `false` (default `true`), a
  vector-add warmer runs ~100 ms of sustained load immediately before the
  timed launch, boosting the GPU clock from idle and evicting the last-level
  cache, so the kernel is measured cold-cache at a boosted clock. Repeat
  measurements come from fresh process re-executions: `gpubench run --reps N`.
- TFLOPS is derived downstream from each record's `time_ms` and `matrix_dim`:
  `TFLOPS = 2 × M × N × K / time_s / 10¹²`
- Each warp computes one output tile; the grid covers the full
  M×N output matrix.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_matrix_dim` (or argv[1]) | 4096 | Square matrix dimension M=N=K (multiple of 16) |
| `BENCH_PARAM_precision` | fp16 | WMMA fragment precision (`fp16`, `tf32`) |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer (clock boost + LLC eviction) before the timed launch; set `false` to time the kernel as-is |

## Requirements

| Platform | Requirement |
|----------|-------------|
| CUDA     | sm_70+ (Volta or later) for FP16 WMMA; sm_80+ for TF32 |
| ROCm     | CDNA architecture with rocWMMA support |
