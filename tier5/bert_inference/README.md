# BERT Inference Benchmark

PyTorch benchmark: BERT-base-uncased inference (forward pass),
batch=16, seq_len=128, random input_ids, model.eval(), torch.no_grad(), reports sequences/sec.

Uses HuggingFace transformers with random weights (no model download required).

## Requirements

- Python 3.8+
- PyTorch 2.x
- transformers (`pip install transformers`)
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
bert_inference,16,25.123,637.00
```

## Metric Formula

sequences/sec = batch_size / (time_ms / 1000)
