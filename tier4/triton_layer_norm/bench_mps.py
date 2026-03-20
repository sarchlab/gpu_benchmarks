import json
import os
import sys
import torch
import torch.nn.functional as F
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '4096'))
    m = int(os.environ.get('BENCH_PARAM_m', '4096'))
    eps = float(os.environ.get('BENCH_PARAM_eps', '1e-5'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    print(f"Running layer_norm (MPS) with m={m}, n={n}, eps={eps}, dtype={dtype_str}", file=sys.stderr)

    x = torch.rand((m, n), device=device, dtype=dtype)
    weight = torch.ones(n, device=device, dtype=dtype)
    bias = torch.zeros(n, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        y = F.layer_norm(x, (n,), weight, bias, eps=eps)
    _ = y.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = F.layer_norm(x, (n,), weight, bias, eps=eps)
        _ = y.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)

    # GB/s: read x + weight + bias + write y
    # Simplified: 2 * m * n * bytes_per_elem (dominant term)
    gb_s = 2 * m * n * bytes_per_elem / (avg_ms / 1000) / 1e9

    params = {
        "n": n,
        "m": m,
        "eps": eps,
        "dtype": dtype_str,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "layer_norm_kernel",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "bandwidth_gbps", "value": gb_s},
        ],
    }))

if __name__ == '__main__':
    benchmark()
