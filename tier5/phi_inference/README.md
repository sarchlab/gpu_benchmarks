# Phi Inference Benchmark

PyTorch benchmark: Microsoft Phi model autoregressive text generation (inference),
num_hidden_layers=6, hidden_size=512, num_attention_heads=8, intermediate_size=2048,
vocab_size=51200. Features Grouped Query Attention (GQA).
Generates 128 new tokens from a 32-token random prompt.
Reports tokens/sec.

Model instantiated from PhiConfig with random weights — no external model downloads required.

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
phi_inference,128,456.789,280.11
```

## Metric Formula

tokens/sec = 128 / (avg_time_ms / 1000)
