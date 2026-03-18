import torch
import torch.nn.functional as F
import time

def benchmark():
    device = torch.device('mps')
    B, H, S, D = 8, 16, 1024, 64
    # MPS doesn't support float16 for all ops; use float32
    dtype = torch.float32

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        out = F.scaled_dot_product_attention(q, k, v)
    _ = out.sum().item()  # sync

    # Timed
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        out = F.scaled_dot_product_attention(q, k, v)
        _ = out.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    ms_per_iter = sum(times) / len(times)
    tflops = 2 * B * H * S * S * D / (ms_per_iter / 1000) / 1e12
    print(f'sdpa_bench,{S},{ms_per_iter:.3f},{tflops:.4f}')

if __name__ == '__main__':
    benchmark()
