# GPT-NeoX Training Benchmark

PyTorch benchmark: GPTNeoXForCausalLM training (forward + backward + optimizer step),
num_hidden_layers=6, hidden_size=512, num_attention_heads=8, intermediate_size=2048,
rotary_pct=0.25, vocab_size=50257, max_position_embeddings=2048.
Simulates causal language model training with random input_ids used as labels.
Batch=4, seq_len=256.
Reports tokens/sec.

Model instantiated from GPTNeoXConfig with random weights — no external model downloads required.

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

CSV: `benchmark_name,batch_size,time_ms,tokens_per_sec`

Example:
```
gptneox_train,4,120.456,8510.33
```

## Metric Formula

tokens/sec = (batch_size * seq_len) / (avg_time_ms / 1000)
