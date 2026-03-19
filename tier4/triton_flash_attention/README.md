# Triton Flash Attention Benchmark

Triton kernel benchmark: Flash Attention forward pass,
B=2, H=16, S=1024, D=64 (batch, heads, seq_len, head_dim), FP16.
Uses online softmax with tiled Q/K/V blocks. Reports TFLOPS.

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

CSV: `benchmark_name,config,time_ms,metric`

Example:
```
triton_flash_attention,2x16x1024x64,1.234,5.67
```

## Metric Formula

TFLOPS = 4 × B × H × S × S × D ÷ (time_ms ÷ 1000) ÷ 1e12
