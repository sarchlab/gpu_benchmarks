import json
import os
import sys
import torch
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    m = int(os.environ.get('BENCH_PARAM_m', '4096'))
    block_m = int(os.environ.get('BENCH_PARAM_block_m', '128'))
    block_n = int(os.environ.get('BENCH_PARAM_block_n', '256'))
    block_k = int(os.environ.get('BENCH_PARAM_block_k', '64'))
    group_size_m = int(os.environ.get('BENCH_PARAM_group_size_m', '8'))
    num_warmup = 5

    # Square matrices: M = N = K
    M_dim, N_dim, K_dim = m, m, m

    print(f"Running triton_gemm (MPS) with m={m}, block_m={block_m}, block_n={block_n}, block_k={block_k}, group_size_m={group_size_m}", file=sys.stderr)

    a = torch.rand((M_dim, K_dim), device=device, dtype=torch.float32)
    b = torch.rand((K_dim, N_dim), device=device, dtype=torch.float32)

    # Warmup
    for _ in range(num_warmup):
        c = torch.mm(a, b)
    if num_warmup > 0:
        _ = c.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        c = torch.mm(a, b)
        _ = c.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)

    tflops = 2 * M_dim * N_dim * K_dim / (avg_ms / 1000) / 1e12

    params = {
        "m": m,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "group_size_m": group_size_m,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "matmul_kernel",
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
