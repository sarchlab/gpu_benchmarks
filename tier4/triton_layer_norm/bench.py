import json
import os
import sys
import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N,
    eps,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride
    y_ptr = y_ptr + row * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    y = (x_centered * rstd) * w + b
    tl.store(y_ptr + col_offsets, y, mask=mask)

def benchmark():
    device = 'cuda'

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '4096'))
    m = int(os.environ.get('BENCH_PARAM_m', '4096'))
    eps = float(os.environ.get('BENCH_PARAM_eps', '1e-5'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')
    num_warmup = 5

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    BLOCK_SIZE = triton.next_power_of_2(n)

    print(f"Running layer_norm with m={m}, n={n}, eps={eps}, dtype={dtype_str}", file=sys.stderr)

    x = torch.rand((m, n), device=device, dtype=dtype)
    weight = torch.ones(n, device=device, dtype=dtype)
    bias = torch.zeros(n, device=device, dtype=dtype)
    y = torch.empty_like(x)

    grid = (m,)

    # Warmup
    for _ in range(num_warmup):
        layer_norm_kernel[grid](
            x, weight, bias, y,
            n, eps,
            x.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        layer_norm_kernel[grid](
            x, weight, bias, y,
            n, eps,
            x.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / 5

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
