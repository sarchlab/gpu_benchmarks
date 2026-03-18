# NanoGPT Training Benchmark

PyTorch benchmark: GPT-2 style transformer training step (forward + backward + AdamW),
n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=50257, batch=4.
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

CSV: `benchmark_name,size,time_ms,metric`

Example:
```
nanogpt_train,1024,123.456,82943.00
```

## Metric Formula

tokens/sec = batch * block_size * 10 / total_time_s
