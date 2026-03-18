# Triton GEMM Benchmark

Triton kernel benchmark: matrix multiplication C = A × B,
M = N = K = 4096, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64. Reports TFLOPS.

## Requirements

- Python 3.8+
- PyTorch 2.x
- Triton (for CUDA/ROCm): `pip3 install triton`
- CUDA GPU (NVIDIA) or ROCm GPU (AMD) or Apple Silicon (Metal/MPS)

## Build & Run

### CUDA (NVIDIA)
```
make run PLATFORM=cuda
```

### ROCm (AMD)
```
make run PLATFORM=rocm
```

### Metal (Apple)
```
make run PLATFORM=metal
```

## Output Format

CSV: `benchmark_name,size,time_ms,metric`

Example:
```
triton_gemm,4096,5.123,107.45
```

## Metric Formula

TFLOPS = 2 × M × N × K ÷ (time_ms ÷ 1000) ÷ 1e12
