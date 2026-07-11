# BERT Fine-Tuning Benchmark

PyTorch benchmark: BertForSequenceClassification fine-tuning (training),
num_hidden_layers=6, hidden_size=384, num_attention_heads=6, intermediate_size=1536,
vocab_size=30522, num_labels=2.
Simulates binary classification fine-tuning: forward + backward + optimizer step.
Batch=16, seq_len=128, random input_ids + random labels.
Reports samples/sec.

Model instantiated from BertConfig with random weights — no external model downloads required.

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

CSV: `benchmark_name,batch_size,time_ms,samples_per_sec`

Example:
```
bert_finetune,16,45.678,350.22
```

## Metric Formula

samples/sec = batch_size / (avg_time_ms / 1000)
