import json
import os
import sys
import torch
import torch.nn.functional as F


def benchmark():
    device = torch.device('cuda')

    # Read parameters from environment variables
    seq_len = int(os.environ.get('BENCH_PARAM_seq_len', '1024'))
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '8'))
    num_heads = int(os.environ.get('BENCH_PARAM_num_heads', '16'))
    head_dim = int(os.environ.get('BENCH_PARAM_head_dim', '64'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float16')
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    dtype = dtype_map.get(dtype_str, torch.float16)

    print(f"SDPA benchmark — B={batch_size}, H={num_heads}, S={seq_len}, "
          f"D={head_dim}, dtype={dtype_str}", file=sys.stderr)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Timed
    num_iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        _ = F.scaled_dot_product_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / num_iters
    # TFLOPS: 2 * B * H * S * S * D per attention computation
    tflops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim / (avg_ms / 1000) / 1e12

    print(f"  avg: {avg_ms:.3f} ms, {tflops:.4f} TFLOPS", file=sys.stderr)

    params = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype": dtype_str,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "scaled_dot_product_attention",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "tflops", "value": tflops},
        ],
    }))

if __name__ == '__main__':
    benchmark()
