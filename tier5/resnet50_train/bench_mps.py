import torch
import torchvision.models as models
import time

def benchmark():
    device = torch.device('mps')
    model = models.resnet50(weights=None).to(device)
    model.train()

    batch = 32
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (batch,), device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    _ = loss.item()  # sync

    # Timed
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        _ = loss.item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    ms = sum(times) / len(times)
    images_per_sec = batch / (ms / 1000)
    print(f'resnet50_train,{batch},{ms:.3f},{images_per_sec:.2f}')

if __name__ == '__main__':
    benchmark()
