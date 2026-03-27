import json
import os
import sys
import time
import torch
from transformers import BertForSequenceClassification, BertConfig


def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '16'))
    seq_len = int(os.environ.get('BENCH_PARAM_seq_len', '128'))
    num_hidden_layers = int(os.environ.get('BENCH_PARAM_num_hidden_layers', '6'))
    hidden_size = int(os.environ.get('BENCH_PARAM_hidden_size', '384'))
    num_labels = int(os.environ.get('BENCH_PARAM_num_labels', '2'))
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    config = BertConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=max(1, hidden_size // 64),
        intermediate_size=hidden_size * 4,
        vocab_size=30522,
        num_labels=num_labels,
    )
    model = BertForSequenceClassification(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len),
                              device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long,
                                device=device)
    labels = torch.randint(0, config.num_labels, (batch_size,), device=device)

    print(f"BERT fine-tune benchmark (MPS) — {num_hidden_layers} layers, "
          f"hidden={hidden_size}, batch={batch_size}, seq_len={seq_len}",
          file=sys.stderr)

    # Warmup
    for i in range(num_warmup):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"  warmup {i+1}/{num_warmup}", file=sys.stderr)
    if num_warmup > 0:
        _ = loss.item()  # sync

    # Timed
    total_time = 0.0
    for i in range(10):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        _ = loss.item()  # force sync
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        print(f"  run {i+1}/10: {elapsed_ms:.3f} ms", file=sys.stderr)

    avg_ms = total_time / 10
    samples_per_sec = batch_size / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {samples_per_sec:.2f} samples/sec",
          file=sys.stderr)

    params = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "num_labels": num_labels,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "forward_backward_step",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "samples_per_sec", "value": samples_per_sec},
        ],
    }))


if __name__ == '__main__':
    benchmark()
