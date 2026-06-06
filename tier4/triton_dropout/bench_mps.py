import json
import os
import sys
import torch
import torch.nn.functional as F
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '67108864'))
    block_size = int(os.environ.get('BENCH_PARAM_block_size', '1024'))
    p = float(os.environ.get('BENCH_PARAM_p', '0.5'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')
    num_warmup = 5

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    print(f"Running dropout (MPS) with n={n}, block_size={block_size}, p={p}, dtype={dtype_str}", file=sys.stderr)

    x = torch.rand(n, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        y = F.dropout(x, p=p, training=True)
    if num_warmup > 0:
        _ = y.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = F.dropout(x, p=p, training=True)
        _ = y.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)

    # GB/s: read input + write output = 2 * n * bytes_per_elem
    gb_s = 2 * n * bytes_per_elem / (avg_ms / 1000) / 1e9

    params = {
        "n": n,
        "block_size": block_size,
        "p": p,
        "dtype": dtype_str,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "dropout_kernel",
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
