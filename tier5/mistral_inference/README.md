# Mistral Inference Benchmark

PyTorch benchmark: Mistral model autoregressive text generation (inference),
num_hidden_layers=6, hidden_size=512, num_attention_heads=8, num_key_value_heads=4,
intermediate_size=2048, vocab_size=32000.
Uses Grouped Query Attention (GQA) with 8 query heads and 4 KV heads.
Generates 128 new tokens from a 32-token random prompt.
Reports tokens/sec.

Model instantiated from MistralConfig with random weights — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
- HuggingFace Transformers
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
mistral_inference,128,456.789,280.11
```

## Metric Formula

tokens/sec = 128 / (avg_time_ms / 1000)
