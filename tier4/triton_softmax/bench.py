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
    M, N = 4096, 4096
    BLOCK_SIZE = triton.next_power_of_2(N)

    x = torch.rand((M, N), device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    grid = (M,)

    # Warmup
    for _ in range(5):
        softmax_kernel[grid](
            y, x,
            x.stride(0), y.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
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
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 5

    # GB/s: read input + write output = 2 * M * N * 4 bytes
    gb_s = 2 * M * N * 4 / (ms / 1000) / 1e9
    print(f'triton_softmax,{M}x{N},{ms:.3f},{gb_s:.2f}')

if __name__ == '__main__':
    benchmark()
