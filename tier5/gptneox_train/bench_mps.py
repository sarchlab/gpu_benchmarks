import json
import os
import sys
import time
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig


def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '4'))
    seq_len = int(os.environ.get('BENCH_PARAM_seq_len', '256'))
    num_hidden_layers = int(os.environ.get('BENCH_PARAM_num_hidden_layers', '6'))
    hidden_size = int(os.environ.get('BENCH_PARAM_hidden_size', '512'))
    num_attention_heads = int(os.environ.get('BENCH_PARAM_num_attention_heads', '8'))

    config = GPTNeoXConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        rotary_pct=0.25,
        vocab_size=50257,
        max_position_embeddings=2048,
    )
    model = GPTNeoXForCausalLM(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len),
                              device=device)
    labels = input_ids.clone()

    print(f"GPT-NeoX training benchmark (MPS) — {num_hidden_layers} layers, "
          f"hidden={hidden_size}, heads={num_attention_heads}, "
          f"batch={batch_size}, seq_len={seq_len}",
          file=sys.stderr)

    # Warmup
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
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
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        _ = loss.item()  # force sync
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        print(f"  run {i+1}/10: {elapsed_ms:.3f} ms", file=sys.stderr)

    avg_ms = total_time / 10
    tokens_per_sec = (batch_size * seq_len) / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec",
          file=sys.stderr)

    params = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
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
            {"name": "tokens_per_sec", "value": tokens_per_sec},
        ],
    }))


if __name__ == '__main__':
    benchmark()
