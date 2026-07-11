# Triton Dropout Benchmark

Triton kernel benchmark: dropout with p=0.5 on N=2^26=67,108,864 float32 elements.
Uses tl.rand for random number generation. Reports memory bandwidth in GB/s.

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
triton_dropout,67108864,0.789,684.32
```

## Metric Formula

GB/s = 2 × N × 4 bytes ÷ (time_ms ÷ 1000) ÷ 1e9

2 arrays: 1 read (input) + 1 write (output).
