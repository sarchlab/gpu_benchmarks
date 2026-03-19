import torch
import torchvision.models as models
import time
import sys


def bench_variant(model, x, batch, label):
    """Run 5 warmup + 5 timed iterations, return avg ms."""
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            out = model(x)
    _ = out.sum().item()  # sync

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.perf_counter()
            out = model(x)
            _ = out.sum().item()  # force sync
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    ms = sum(times) / len(times)
    images_per_sec = batch / (ms / 1000)
    print(f'torch_compile_bench,{label},{ms:.3f},{images_per_sec:.2f}')
    return ms


def benchmark():
    device = torch.device('mps')
    batch = 32

    model = models.resnet50(weights=None).to(device)
    model.eval()

    x = torch.randn(batch, 3, 224, 224, device=device)

    # Eager variant
    eager_ms = bench_variant(model, x, batch, 'eager')

    # Compiled variant — torch.compile may not support MPS backend
    compiled_ms = None
    try:
        compiled_model = torch.compile(model)
        compiled_ms = bench_variant(compiled_model, x, batch, 'compiled')
    except Exception as e:
        print(f'NOTE: torch.compile not supported on MPS backend, '
              f'skipping compiled variant ({e})', file=sys.stderr)

    # Speedup
    if compiled_ms is not None and compiled_ms > 0:
        speedup = eager_ms / compiled_ms
        print(f'speedup: {speedup:.2f}x', file=sys.stderr)
    else:
        print('speedup: N/A (compiled variant skipped)', file=sys.stderr)


if __name__ == '__main__':
    benchmark()
