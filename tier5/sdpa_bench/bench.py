import torch
import torch.nn.functional as F
import time

def benchmark():
    device = torch.device('cuda')
    B, H, S, D = 8, 16, 1024, 64
    dtype = torch.float16

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        _ = F.scaled_dot_product_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    ms_per_iter = total_ms / 50
    # TFLOPS: 2 * B * H * S * S * D per attention computation
    tflops = 2 * B * H * S * S * D / (ms_per_iter / 1000) / 1e12
    print(f'sdpa_bench,{S},{ms_per_iter:.3f},{tflops:.4f}')

if __name__ == '__main__':
    benchmark()
