# GNN Inference Benchmark

PyTorch benchmark: Graph Neural Network inference using an R-GAT (Relational Graph Attention) proxy model.
Multi-head attention over edges with scatter-based message passing on a synthetic random graph.
Pure PyTorch implementation — no PyG or DGL dependencies.
Default: 10000 nodes, 50000 edges, hidden_dim=128, 4 attention heads, 3 layers.
Reports nodes/sec throughput.

## Model

The R-GAT proxy implements multi-layer Graph Attention Networks:
- Each layer computes multi-head attention weights per edge
- Softmax normalization per destination node via scatter operations
- Weighted aggregation of source node features
- LayerNorm + ELU activation between layers
- All weights are Xavier-initialized (no pretrained weights)

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
BENCH_PARAM_num_nodes=20000 BENCH_PARAM_num_edges=100000 make run
```

## Output Format

JSON-lines to stdout:

```json
{"type": "kernel", "name": "forward", "time_ms": 12.34, "params": {...}}
{"type": "summary", "total_time_ms": 12.34, "metrics": [{"name": "nodes_per_sec", "value": 810372.77}]}
```

## Metric Formula

nodes/sec = num_nodes / (time_ms / 1000)
