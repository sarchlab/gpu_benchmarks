import json
import os
import sys
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def benchmark():
    device = 'cuda'

    # Read parameters from environment variables
    n = int(os.environ.get('BENCH_PARAM_n', '4096'))
    m = int(os.environ.get('BENCH_PARAM_m', '4096'))
    num_warps = int(os.environ.get('BENCH_PARAM_num_warps', '4'))
    dtype_str = os.environ.get('BENCH_PARAM_dtype', 'float32')
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(dtype_str, torch.float32)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    BLOCK_SIZE = triton.next_power_of_2(n)

    print(f"Running triton_softmax with m={m}, n={n}, dtype={dtype_str}, num_warps={num_warps}", file=sys.stderr)

    x = torch.rand((m, n), device=device, dtype=dtype)
    y = torch.empty_like(x)

    grid = (m,)

    # Warmup
    for _ in range(num_warmup):
        softmax_kernel[grid](
            y, x,
            x.stride(0), y.stride(0),
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        softmax_kernel[grid](
            y, x,
            x.stride(0), y.stride(0),
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / 5

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
