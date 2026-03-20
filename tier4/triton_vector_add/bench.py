import json
import os
import sys
import torch
import triton
import triton.language as tl
import time

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def benchmark():
    device = 'cuda'

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '67108864'))
    block_size = int(os.environ.get('BENCH_PARAM_block_size', '1024'))
    num_warps = int(os.environ.get('BENCH_PARAM_num_warps', '4'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    print(f"Running vector_add with n={n}, block_size={block_size}, dtype={dtype_str}, num_warps={num_warps}", file=sys.stderr)

    a = torch.rand(n, device=device, dtype=dtype)
    b = torch.rand(n, device=device, dtype=dtype)
    c = torch.empty(n, device=device, dtype=dtype)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # Warmup
    for _ in range(5):
        vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / 5

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
