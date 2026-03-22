# GPT-J Inference Benchmark

PyTorch benchmark: GPT-J style autoregressive inference with a proxy transformer model.
Uses parallel attention+MLP blocks (GPT-J architecture), n_layer=12, n_head=16, n_embd=4096,
vocab_size=50257. Generates 64 tokens from a random prompt using greedy decoding.
Reports tokens/sec throughput.

## Model Architecture

- Token embedding + positional embedding
- N transformer blocks with parallel attention + MLP (GPT-J style)
  - Single LayerNorm shared by attention and MLP branches
  - Causal self-attention
  - MLP with GELU activation (4x expansion)
- Final LayerNorm + linear head (lm_head)
- Xavier-initialized weights, no pretrained weights

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

### Custom Parameters
```
BENCH_PARAM_seq_len=128 BENCH_PARAM_gen_tokens=64 BENCH_PARAM_n_layer=12 make run
```

## Output Format

JSON-lines to stdout:

```json
{"type": "kernel", "name": "forward", "time_ms": 1234.56, "params": {...}}
{"type": "summary", "total_time_ms": 1234.56, "metrics": [{"name": "tokens_per_sec", "value": 51.84}]}
```

## Metric Formula

tokens/sec = gen_tokens / (time_ms / 1000)
