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
    M, N = 4096, 4096
    eps = 1e-5
    BLOCK_SIZE = triton.next_power_of_2(N)

    x = torch.rand((M, N), device=device, dtype=torch.float32)
    weight = torch.ones(N, device=device, dtype=torch.float32)
    bias = torch.zeros(N, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    grid = (M,)

    # Warmup
    for _ in range(5):
        layer_norm_kernel[grid](
            x, weight, bias, y,
            N, eps,
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
            N, eps,
            x.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 5

    # GB/s: read x + weight + bias + write y
    # Simplified: 2 * M * N * 4 bytes (dominant term)
    gb_s = 2 * M * N * 4 / (ms / 1000) / 1e9
    print(f'triton_layer_norm,{M}x{N},{ms:.3f},{gb_s:.2f}')

if __name__ == '__main__':
    benchmark()
