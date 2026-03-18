import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024

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
    N = 1 << 26  # 67108864
    p = 0.5
    seed = 42

    x = torch.rand(N, device=device, dtype=torch.float32)
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Warmup
    for _ in range(5):
        dropout_kernel[grid](x, output, N, p, seed, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        dropout_kernel[grid](x, output, N, p, seed, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 5

    # GB/s: read input + write output = 2 * N * 4 bytes
    gb_s = 2 * N * 4 / (ms / 1000) / 1e9
    print(f'triton_dropout,{N},{ms:.3f},{gb_s:.2f}')

if __name__ == '__main__':
    benchmark()
