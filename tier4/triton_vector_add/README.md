# Triton Vector Add Benchmark

Triton kernel benchmark: element-wise vector addition C = A + B,
N = 2^26 = 67,108,864 float32 elements. Reports memory bandwidth in GB/s.

## Requirements

- Python 3.8+
- PyTorch 2.x
- Triton (for CUDA/ROCm): `pip3 install triton`
- CUDA GPU (NVIDIA) or ROCm GPU (AMD)

## Build & Run

### CUDA (NVIDIA)
```
make run PLATFORM=cuda
# or just: make run
```

### ROCm (AMD)
```
make run PLATFORM=rocm
```

## Output Format

CSV: `benchmark_name,size,time_ms,metric`

Example:
```
vector_add,67108864,0.512,1567.23
```

## Metric Formula

GB/s = 3 × N × 4 bytes ÷ (time_ms ÷ 1000) ÷ 1e9

3 arrays accessed: 2 reads (A, B) + 1 write (C).
