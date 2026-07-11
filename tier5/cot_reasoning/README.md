# Chain-of-Thought Reasoning Benchmark

PyTorch benchmark: Chain-of-Thought (CoT) reasoning using a GPT-2 style transformer,
num_layers=6, num_heads=6, hidden_size=384, max_seq_len=1024, vocab_size=50257.
Simulates multi-step CoT reasoning: 3 reasoning steps (64 tokens each) + final answer (32 tokens) = 224 tokens generated from a 32-token random prompt.
Reports tokens/sec.

Model defined entirely inline — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
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

CSV: `benchmark_name,total_tokens,time_ms,tokens_per_sec`

Example:
```
cot_reasoning,224,1234.567,181.44
```

## Metric Formula

tokens/sec = 224 / (avg_time_ms / 1000)
