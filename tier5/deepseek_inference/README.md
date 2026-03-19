# DeepSeek Inference Benchmark

PyTorch benchmark: DeepSeek-style transformer autoregressive text generation (inference).
Uses LlamaForCausalLM with DeepSeek-like config: num_hidden_layers=6, hidden_size=512,
num_attention_heads=8, num_key_value_heads=4 (GQA), intermediate_size=1376, vocab_size=102400.
Generates 128 new tokens from a 32-token random prompt.
Reports tokens/sec.

Random weights — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
- Transformers
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
deepseek_inference,128,456.789,280.11
```

## Metric Formula

tokens/sec = 128 / (avg_time_ms / 1000)
