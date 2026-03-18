import torch
import torch.nn.functional as F
import time

def benchmark():
    device = torch.device('mps')
    N = 1 << 26  # 67108864
    p = 0.5

    x = torch.rand(N, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        y = F.dropout(x, p=p, training=True)
    _ = y.sum().item()

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = F.dropout(x, p=p, training=True)
        _ = y.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    ms = sum(times) / len(times)

    gb_s = 2 * N * 4 / (ms / 1000) / 1e9
    print(f'triton_dropout,{N},{ms:.3f},{gb_s:.2f}')

if __name__ == '__main__':
    benchmark()
