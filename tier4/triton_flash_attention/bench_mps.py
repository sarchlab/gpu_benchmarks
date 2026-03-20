import json
import os
import sys
import torch
import torch.nn.functional as F
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    s = int(os.environ.get('BENCH_PARAM_s', '1024'))
    b = int(os.environ.get('BENCH_PARAM_b', '2'))
    h = int(os.environ.get('BENCH_PARAM_h', '16'))
    d = int(os.environ.get('BENCH_PARAM_d', '64'))
    block_m = int(os.environ.get('BENCH_PARAM_block_m', '128'))
    block_n = int(os.environ.get('BENCH_PARAM_block_n', '64'))

    print(f"Running flash_attention (MPS) with b={b}, h={h}, s={s}, d={d}, block_m={block_m}, block_n={block_n}", file=sys.stderr)

    Q = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    K = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    V = torch.randn(b, h, s, d, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        out = F.scaled_dot_product_attention(Q, K, V)
    _ = out.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = F.scaled_dot_product_attention(Q, K, V)
        _ = out.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)

    tflops = 4 * b * h * s * s * d / (avg_ms / 1000) / 1e12

    params = {
        "s": s,
        "b": b,
        "h": h,
        "d": d,
        "block_m": block_m,
        "block_n": block_n,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "flash_attention_fwd",
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
