import torch
import time

def benchmark():
    device = torch.device('mps')
    M, N, K = 4096, 4096, 4096

    a = torch.rand((M, K), device=device, dtype=torch.float32)
    b = torch.rand((K, N), device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        c = torch.mm(a, b)
    _ = c.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        c = torch.mm(a, b)
        _ = c.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    ms = sum(times) / len(times)

    tflops = 2 * M * N * K / (ms / 1000) / 1e12
    print(f'triton_gemm,{M},{ms:.3f},{tflops:.2f}')

if __name__ == '__main__':
    benchmark()
