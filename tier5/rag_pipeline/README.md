# RAG Pipeline Benchmark

PyTorch benchmark: Retrieval-Augmented Generation (RAG) pipeline with inline
Encoder (BERT-like), Retriever (cosine similarity top-k), and Generator
(GPT-2-like with cross-attention). All model components are defined as
nn.Module classes with random weights (no model download required).

Configuration: encoder/generator with 4 layers, hidden=256, 4 heads,
vocab_size=50257. 1000 precomputed document embeddings, top-5 retrieval,
batch=4 queries (query_len=32), autoregressive generation of 64 tokens.

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
rag_pipeline,1000,150.234,26.63
```

## Metric Formula

queries/sec = batch_size / (avg_time_ms / 1000)
