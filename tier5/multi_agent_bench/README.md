# Multi-Agent LLM Benchmark

PyTorch benchmark: multi-agent LLM pipeline with 3 independent GPT-2-style agents
passing tokens between each other. Each agent has 4 layers, hidden_size=256, 4 heads.
All models defined inline (no external model downloads required).

## Pipeline

1. **Agent A**: Takes random prompt (32 tokens), generates 64 tokens
2. **Agent B**: Takes Agent A's last 64 tokens as input, generates 64 tokens
3. **Agent C**: Takes concatenation of Agent A's last 32 + Agent B's last 32 tokens, generates 64 tokens

Total tokens generated per iteration: 192

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

CSV: `benchmark_name,size,time_ms,metric`

Example:
```
multi_agent_bench,3x64,1234.567,155.52
```

## Metric Formula

tokens/sec = 192 / (avg_ms / 1000)
