import torch
import torch.nn.functional as F
import time

def benchmark():
    device = torch.device('mps')
    B, H, S, D = 2, 16, 1024, 64

    Q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        out = F.scaled_dot_product_attention(Q, K, V)
    _ = out.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = F.scaled_dot_product_attention(Q, K, V)
        _ = out.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    ms = sum(times) / len(times)

    tflops = 4 * B * H * S * S * D / (ms / 1000) / 1e12
    print(f'triton_flash_attention,{B}x{H}x{S}x{D},{ms:.3f},{tflops:.2f}')

if __name__ == '__main__':
    benchmark()
