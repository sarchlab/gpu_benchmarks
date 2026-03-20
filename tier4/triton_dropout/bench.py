import json
import os
import sys
import torch
import triton
import triton.language as tl

@triton.jit
def dropout_kernel(
    x_ptr, output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def benchmark():
    device = 'cuda'

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '67108864'))
    block_size = int(os.environ.get('BENCH_PARAM_block_size', '1024'))
    p = float(os.environ.get('BENCH_PARAM_p', '0.5'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    seed = 42

    print(f"Running dropout with n={n}, block_size={block_size}, p={p}, dtype={dtype_str}", file=sys.stderr)

    x = torch.rand(n, device=device, dtype=dtype)
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # Warmup
    for _ in range(5):
        dropout_kernel[grid](x, output, n, p, seed, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        dropout_kernel[grid](x, output, n, p, seed, BLOCK_SIZE=block_size)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / 5

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
