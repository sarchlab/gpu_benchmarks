import torch
import torchvision.models as models
import time

def benchmark():
    device = torch.device('mps')
    model = models.resnet50(weights=None).to(device)
    model.eval()

    batch = 64
    x = torch.randn(batch, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            out = model(x)
    _ = out.sum().item()  # sync

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            out = model(x)
            _ = out.sum().item()  # force sync
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    ms = sum(times) / len(times)
    images_per_sec = batch / (ms / 1000)
    print(f'resnet50_infer,{batch},{ms:.3f},{images_per_sec:.2f}')

if __name__ == '__main__':
    benchmark()
