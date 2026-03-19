import sys
import time
import torch
from transformers import BertForSequenceClassification, BertConfig


def benchmark():
    device = torch.device('mps')
    config = BertConfig(
        num_hidden_layers=6,
        hidden_size=384,
        num_attention_heads=6,
        intermediate_size=1536,
        vocab_size=30522,
        num_labels=2,
    )
    model = BertForSequenceClassification(config).to(device)
    model.train()

    batch_size = 16
    seq_len = 128
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len),
                              device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long,
                                device=device)
    labels = torch.randint(0, config.num_labels, (batch_size,), device=device)

    print(f"BERT fine-tune benchmark — {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, heads={config.num_attention_heads}",
          file=sys.stderr)
    print(f"Batch={batch_size}, seq_len={seq_len}, "
          f"forward+backward+optimizer step", file=sys.stderr)

    # Warmup
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"  warmup {i+1}/5", file=sys.stderr)
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
    print(f'bert_finetune,{batch_size},{avg_ms:.3f},{samples_per_sec:.2f}')


if __name__ == '__main__':
    benchmark()
