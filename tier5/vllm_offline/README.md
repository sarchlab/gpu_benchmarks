# vLLM Offline Batch Inference Benchmark

Benchmark for vLLM offline batch inference using PagedAttention.
Runs a batch of 10 prompts through facebook/opt-125m with max_tokens=128,
measuring throughput in tokens/sec.

## Requirements

- Python 3.8+
- vLLM (`pip install vllm`)
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
vllm_offline,128,1234.567,8901.23
```

- `size`: max_tokens per request (128)
- `time_ms`: average batch inference time in milliseconds
- `metric`: tokens per second

## Metric Formula

tokens/sec = total_generated_tokens / (time_ms / 1000)
