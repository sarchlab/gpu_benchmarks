import torch
import time

def benchmark():
    device = torch.device('mps')
    N = 1 << 26  # 67108864
    a = torch.rand(N, device=device, dtype=torch.float32)
    b = torch.rand(N, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        c = torch.add(a, b)
    _ = c.item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        c = torch.add(a, b)
        _ = c.item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    ms = sum(times) / len(times)

    gb_s = 3 * N * 4 / (ms / 1000) / 1e9
    print(f'vector_add,{N},{ms:.3f},{gb_s:.2f}')

if __name__ == '__main__':
    benchmark()
