import torch
import torchvision.models as models
import sys


def bench_variant(model, x, batch, label):
    """Run 5 warmup + 5 timed iterations, return avg ms."""
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    ms = total_ms / 5
    images_per_sec = batch / (ms / 1000)
    print(f'torch_compile_bench,{label},{ms:.3f},{images_per_sec:.2f}')
    return ms


def benchmark():
    device = torch.device('cuda')
    batch = 32

    model = models.resnet50(weights=None).to(device)
    model.eval()

    x = torch.randn(batch, 3, 224, 224, device=device)

    # Eager variant
    eager_ms = bench_variant(model, x, batch, 'eager')

    # Compiled variant
    compiled_model = torch.compile(model)
    compiled_ms = bench_variant(compiled_model, x, batch, 'compiled')

    # Speedup
    speedup = eager_ms / compiled_ms
    print(f'speedup: {speedup:.2f}x', file=sys.stderr)


if __name__ == '__main__':
    benchmark()
