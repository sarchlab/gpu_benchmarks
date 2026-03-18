import torch
import time

def benchmark():
    device = torch.device('mps')
    M, N = 4096, 4096

    x = torch.rand((M, N), device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        y = torch.softmax(x, dim=1)
    _ = y.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = torch.softmax(x, dim=1)
        _ = y.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    ms = sum(times) / len(times)

    gb_s = 2 * M * N * 4 / (ms / 1000) / 1e9
    print(f'triton_softmax,{M}x{N},{ms:.3f},{gb_s:.2f}')

if __name__ == '__main__':
    benchmark()
