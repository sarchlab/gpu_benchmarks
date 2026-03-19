# GPT-2 Generate Benchmark

PyTorch benchmark: GPT-2 style transformer autoregressive text generation (inference),
n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=50257.
Generates 128 new tokens from a 32-token random prompt.
Reports tokens/sec.

Model defined entirely inline — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
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

CSV: `benchmark_name,num_tokens,time_ms,tokens_per_sec`

Example:
```
gpt2_generate,128,456.789,280.11
```

## Metric Formula

tokens/sec = 128 / (avg_time_ms / 1000)
