# Scaled Dot Product Attention Benchmark

PyTorch benchmark: `torch.nn.functional.scaled_dot_product_attention`,
B=8, H=16, S=1024, D=64, float16 (CUDA/ROCm) / float32 (MPS).
5 warmup + 50 timed iterations. Reports TFLOPS.

## Requirements

- Python 3.8+
- PyTorch 2.x (scaled_dot_product_attention requires PyTorch >= 2.0)
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
sdpa_bench,1024,0.456,23.1234
```

## Metric Formula

TFLOPS = 2 × B × H × S × S × D / (time_ms / 1000) / 1e12

Where B=8 (batch), H=16 (heads), S=1024 (seq_len), D=64 (head_dim).
