import torch
import triton
import triton.language as tl
import time

BLOCK_SIZE = 1024

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
    N = 1 << 26  # 67108864
    a = torch.rand(N, device=device, dtype=torch.float32)
    b = torch.rand(N, device=device, dtype=torch.float32)
    c = torch.empty(N, device=device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Warmup
    for _ in range(5):
        vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 5

    # GB/s: 3 arrays (2 reads + 1 write) * N * 4 bytes
    gb_s = 3 * N * 4 / (ms / 1000) / 1e9
    print(f'vector_add,{N},{ms:.3f},{gb_s:.2f}')

if __name__ == '__main__':
    benchmark()
