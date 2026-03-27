import json
import os
import sys
import torch
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '4096'))
    m = int(os.environ.get('BENCH_PARAM_m', '4096'))
    num_warps = int(os.environ.get('BENCH_PARAM_num_warps', '4'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    print(f"Running triton_softmax (MPS) with m={m}, n={n}, dtype={dtype_str}, num_warps={num_warps}", file=sys.stderr)

    x = torch.rand((m, n), device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        y = torch.softmax(x, dim=1)
    if num_warmup > 0:
        _ = y.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = torch.softmax(x, dim=1)
        _ = y.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)

    # GB/s: read input + write output = 2 * m * n * bytes_per_elem
    gb_s = 2 * m * n * bytes_per_elem / (avg_ms / 1000) / 1e9

    params = {
        "n": n,
        "m": m,
        "dtype": dtype_str,
        "num_warps": num_warps,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "softmax_kernel",
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
