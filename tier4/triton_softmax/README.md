# Triton Softmax Benchmark

Triton kernel benchmark: fused row-wise softmax on a 4096×4096 float32 matrix.
Reports memory bandwidth in GB/s.

## Requirements

- Python 3.8+
- PyTorch 2.x
- Triton (for CUDA/ROCm): `pip3 install triton`
- CUDA GPU (NVIDIA) or ROCm GPU (AMD)

## Build & Run

### CUDA (NVIDIA)
```
make run PLATFORM=cuda
```

### ROCm (AMD)
```
make run PLATFORM=rocm
```

## Output Format

CSV: `benchmark_name,size,time_ms,metric`

Example:
```
triton_softmax,4096x4096,1.234,431.56
```

## Metric Formula

GB/s = 2 × M × N × 4 bytes ÷ (time_ms ÷ 1000) ÷ 1e9

2 passes: 1 read (input) + 1 write (output).
