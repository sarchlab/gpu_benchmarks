import json
import os
import sys
import torch
import time


class UNet3D(torch.nn.Module):
    """3D U-Net for volumetric medical image segmentation."""

    def __init__(self, in_channels=1, num_classes=3, base_filters=32):
        super().__init__()
        bf = base_filters

        # Encoder path
        self.enc1 = self._double_conv(in_channels, bf)
        self.pool1 = torch.nn.MaxPool3d(2)
        self.enc2 = self._double_conv(bf, bf * 2)
        self.pool2 = torch.nn.MaxPool3d(2)
        self.enc3 = self._double_conv(bf * 2, bf * 4)
        self.pool3 = torch.nn.MaxPool3d(2)
        self.enc4 = self._double_conv(bf * 4, bf * 8)
        self.pool4 = torch.nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._double_conv(bf * 8, bf * 16)

        # Decoder path
        self.up4 = torch.nn.ConvTranspose3d(bf * 16, bf * 8, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(bf * 16, bf * 8)
        self.up3 = torch.nn.ConvTranspose3d(bf * 8, bf * 4, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(bf * 8, bf * 4)
        self.up2 = torch.nn.ConvTranspose3d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(bf * 4, bf * 2)
        self.up1 = torch.nn.ConvTranspose3d(bf * 2, bf, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(bf * 2, bf)

        # Output
        self.out_conv = torch.nn.Conv3d(bf, num_classes, kernel_size=1)

        # Xavier initialization
        self._init_weights()

    def _double_conv(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


def benchmark():
    device = torch.device('cuda')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '2'))
    volume_depth = int(os.environ.get('BENCH_PARAM_volume_depth', '128'))
    volume_height = int(os.environ.get('BENCH_PARAM_volume_height', '128'))
    volume_width = int(os.environ.get('BENCH_PARAM_volume_width', '128'))
    in_channels = int(os.environ.get('BENCH_PARAM_in_channels', '1'))
    num_classes = int(os.environ.get('BENCH_PARAM_num_classes', '3'))
    precision = os.environ.get('BENCH_PARAM_precision', 'fp32')

    print(f"Running unet3d_inference with batch_size={batch_size}, "
          f"volume={volume_depth}x{volume_height}x{volume_width}, "
          f"in_channels={in_channels}, num_classes={num_classes}, "
          f"precision={precision}", file=sys.stderr)

    model = UNet3D(in_channels=in_channels, num_classes=num_classes).to(device)
    model.eval()

    if precision == 'fp16':
        model = model.half()

    x = torch.randn(batch_size, in_channels, volume_depth, volume_height,
                     volume_width, device=device)
    if precision == 'fp16':
        x = x.half()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            if precision == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
    torch.cuda.synchronize()

    num_iters = 5

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(num_iters):
            if precision == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / num_iters
    volumes_per_sec = batch_size / (avg_ms / 1000)

    params = {
        "batch_size": batch_size,
        "volume_depth": volume_depth,
        "volume_height": volume_height,
        "volume_width": volume_width,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "precision": precision,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "forward",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "volumes_per_sec", "value": volumes_per_sec},
        ],
    }))

if __name__ == '__main__':
    benchmark()
