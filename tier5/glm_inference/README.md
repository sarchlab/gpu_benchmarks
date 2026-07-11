# GLM Inference Benchmark

PyTorch benchmark: ChatGLM-style model autoregressive text generation (inference),
using GPT2Config as the base architecture with GLM-like parameters:
n_layer=6, n_embd=512, n_head=8, n_inner=2048, vocab_size=32000.
Generates 128 new tokens from a 32-token random prompt.
Reports tokens/sec.

Model instantiated from GPT2Config with random weights — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
- HuggingFace Transformers
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

CSV: `benchmark_name,num_tokens,time_ms,tokens_per_sec`

Example:
```
glm_inference,128,456.789,280.11
```

## Metric Formula

tokens/sec = 128 / (avg_time_ms / 1000)
