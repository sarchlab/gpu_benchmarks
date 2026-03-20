import json
import os
import sys
import torch
from transformers import BertModel, BertConfig
import time

def benchmark():
    device = torch.device('cuda')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '16'))
    seq_len = int(os.environ.get('BENCH_PARAM_seq_len', '128'))
    hidden_size = int(os.environ.get('BENCH_PARAM_hidden_size', '768'))
    num_hidden_layers = int(os.environ.get('BENCH_PARAM_num_hidden_layers', '12'))
    num_attention_heads = int(os.environ.get('BENCH_PARAM_num_attention_heads', '12'))

    print(f"Running bert_inference with batch_size={batch_size}, seq_len={seq_len}, "
          f"hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}, "
          f"num_attention_heads={num_attention_heads}", file=sys.stderr)

    config = BertConfig(
        vocab_size=30522,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
    )
    model = BertModel(config).to(device)
    model.eval()

    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()

    num_iters = 10

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / num_iters
    sequences_per_sec = batch_size / (avg_ms / 1000)

    params = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "forward",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "sequences_per_sec", "value": sequences_per_sec},
        ],
    }))

if __name__ == '__main__':
    benchmark()
