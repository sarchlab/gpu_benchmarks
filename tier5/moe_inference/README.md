# MoE Inference Benchmark

PyTorch benchmark: Mixture of Experts (MoE) transformer autoregressive text generation
(inference), with all model components defined inline (no external model downloads).

Architecture: Embedding + positional embedding + 4 TransformerBlocks + LayerNorm + output head.
Each TransformerBlock uses LayerNorm → MultiheadAttention → residual → LayerNorm → MoELayer → residual.
MoELayer routes each token to top-2 of 4 experts via a learned gating network.
Config: hidden_size=512, num_experts=4, top_k=2, num_layers=4, num_heads=8, vocab_size=50257.
Generates 64 new tokens from a 256-token random prompt with batch_size=4.
Reports tokens/sec.

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

CSV: `benchmark_name,seq_len,time_ms,tokens_per_sec`

Example:
```
moe_inference,256,1234.567,51.84
```

## Metric Formula

tokens/sec = 64 / (avg_time_ms / 1000)
