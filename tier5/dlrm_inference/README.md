# DLRM Inference Benchmark

PyTorch benchmark: DLRM (Deep Learning Recommendation Model) inference (forward pass only).
Uses embedding tables (nn.EmbeddingBag) + bottom MLP + dot-product feature interaction + top MLP.
Synthetic tabular data with dense and sparse categorical features, Xavier-initialized weights,
model.eval(), torch.no_grad(). Reports samples/sec throughput.

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

### Custom Parameters
```
BENCH_PARAM_batch_size=128 BENCH_PARAM_num_embeddings=1000000 make run
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 64 | Number of samples per inference batch (scaling) |
| num_embeddings | 100000 | Number of rows per embedding table |
| embedding_dim | 64 | Dimension of each embedding vector |
| num_dense_features | 13 | Number of continuous/dense input features |
| num_sparse_features | 26 | Number of categorical/sparse embedding tables |
| precision | fp32 | Floating-point precision (fp32, fp16, amp) |

## Output Format

JSON-lines to stdout:
```
{"type": "kernel", "name": "forward", "time_ms": 1.234, "params": {...}}
{"type": "summary", "total_time_ms": 1.234, "metrics": [{"name": "samples_per_sec", "value": 51890.0}]}
```

## Metric Formula

samples/sec = batch_size / (time_ms / 1000)
