# Triton Layer Norm Benchmark

Triton kernel benchmark: fused layer normalization on a 4096×4096 float32 matrix,
eps=1e-5. Reports memory bandwidth in GB/s.

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
triton_layer_norm,4096x4096,1.567,337.22
```

## Metric Formula

GB/s = 2 × M × N × 4 bytes ÷ (time_ms ÷ 1000) ÷ 1e9

Dominant data movement: read input + write output.
