import torch
import torchvision.models as models
import time

def benchmark():
    device = torch.device('cuda')
    model = models.vit_b_16(weights=None).to(device)
    model.eval()

    batch = 32
    x = torch.randn(batch, 3, 224, 224, device=device)

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
        for _ in range(10):
            _ = model(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 10
    images_per_sec = batch / (ms / 1000)
    print(f'vit_inference,{batch},{ms:.3f},{images_per_sec:.2f}')

if __name__ == '__main__':
    benchmark()
