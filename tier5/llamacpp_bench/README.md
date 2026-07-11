# llama.cpp Inference Benchmark

Subprocess wrapper benchmark for [llama.cpp](https://github.com/ggerganov/llama.cpp) inference.
Builds llama.cpp from source, downloads a tiny GGUF model (SmolLM2-135M Q4_K_M),
and measures token generation throughput via the `llama-cli` binary.

## Requirements

- C/C++ compiler (gcc/clang)
- CMake 3.14+
- Git
- curl
- GPU: CUDA (NVIDIA) or ROCm (AMD)

## Setup

```
make setup
```

This will:
1. Clone llama.cpp repository
2. Build from source with GPU support auto-detected
3. Download a tiny GGUF model (~100MB)

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
llamacpp_bench,128,1523.456,84.05
```

## Metric Formula

tokens/sec as reported by llama.cpp eval timing.
Time is wall-clock time for the full inference subprocess call.
