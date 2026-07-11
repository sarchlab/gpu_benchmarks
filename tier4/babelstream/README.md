# BabelStream Benchmark

C++ STREAM benchmark implementing copy/scale/add/triad/dot kernels on
N = 32×1024×1024 float32 elements. Supports CUDA (NVIDIA) and HIP (AMD ROCm).
Reports memory bandwidth in GB/s for each kernel.

## Files

| File | Description |
|------|-------------|
| `babelstream.hip` | CUDA/HIP implementation |
| `Makefile` | Auto-detecting build system |

## Requirements

- CUDA: NVIDIA GPU + CUDA toolkit
- ROCm/HIP: AMD GPU + ROCm toolkit

## Build & Run

### CUDA (NVIDIA)
```
make PLATFORM=cuda
./babelstream
```

### ROCm (AMD)
```
make PLATFORM=rocm
./babelstream
```

## Output Format

CSV: `benchmark_name,size,time_ms,GB/s`

Example:
```
babelstream_copy,33554432,0.543,494.23
babelstream_scale,33554432,0.547,490.62
babelstream_add,33554432,0.821,490.15
babelstream_triad,33554432,0.819,491.34
babelstream_dot,33554432,0.537,499.87
```

## GB/s Formulas

- copy:  GB/s = 2 × N × 4 ÷ time_s ÷ 1e9
- scale: GB/s = 2 × N × 4 ÷ time_s ÷ 1e9
- add:   GB/s = 3 × N × 4 ÷ time_s ÷ 1e9
- triad: GB/s = 3 × N × 4 ÷ time_s ÷ 1e9
- dot:   GB/s = 2 × N × 4 ÷ time_s ÷ 1e9
