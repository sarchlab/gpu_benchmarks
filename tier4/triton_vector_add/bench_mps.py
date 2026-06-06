import json
import os
import sys
import torch
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '67108864'))
    block_size = int(os.environ.get('BENCH_PARAM_block_size', '1024'))
    num_warps = int(os.environ.get('BENCH_PARAM_num_warps', '4'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')
    num_warmup = 5

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    print(f"Running vector_add (MPS) with n={n}, block_size={block_size}, dtype={dtype_str}, num_warps={num_warps}", file=sys.stderr)

    a = torch.rand(n, device=device, dtype=dtype)
    b = torch.rand(n, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        c = torch.add(a, b)
    if num_warmup > 0:
        _ = c.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        c = torch.add(a, b)
        _ = c.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)

    # GB/s: 3 arrays (2 reads + 1 write) * n * bytes_per_elem
    gb_s = 3 * n * bytes_per_elem / (avg_ms / 1000) / 1e9

    params = {
        "n": n,
        "block_size": block_size,
        "dtype": dtype_str,
        "num_warps": num_warps,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "vector_add_kernel",
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
