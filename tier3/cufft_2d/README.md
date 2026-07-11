# tier3/cufft_2d — 2D Complex FFT via cuFFT / hipFFT

A multi-platform benchmark for single-precision complex 2D FFT,
wrapping vendor-optimised FFT libraries:

| Platform | Library                                           |
|----------|---------------------------------------------------|
| NVIDIA   | **cuFFT** (`cufftExecC2C` with `cufftPlan2d`)    |
| AMD      | **hipFFT** (`hipfftExecC2C` with `hipfftPlan2d`) |

## Algorithm

Computes a **forward 2D complex-to-complex (C2C) FFT** on an NX×NY
single-precision complex grid:

- **NX = NY = 2048** (total: 4,194,304 complex samples)
- Input: NX×NY interleaved complex floats
- Transform: forward 2D FFT,
  X[m][n] = Σ_{i,j} x[i][j] · exp(−2πi·(m·i/NX + n·j/NY))

## Input and Verification

The input signal is a separable 2D sinusoid at discrete frequencies
(kx=3, ky=5):

```
x[i][j].real = sin(2π · (3·i/NX + 5·j/NY)),   x[i][j].imag = 0
```

After a 2D forward FFT, the DFT of the sinusoidal input concentrates at:

```
|X[3][5]|     ≈ NX·NY / 2 = 2,097,152
|X[NX-3][NY-5]| ≈ NX·NY / 2
```

The benchmark checks that the relative error at bin `[3][5]` is < 1%.

## Files

| File                    | Description                                                        |
|-------------------------|--------------------------------------------------------------------|
| `cufft_2d.hip`          | CUDA/ROCm source (`#ifdef USE_CUFFT` / `#elif USE_HIPFFT`)        |
| `Makefile`              | Auto-detecting build system                                        |
| `README.md`             | This file                                                          |

## Build

Auto-detect (recommended):

```bash
make
```

Explicit platform:

```bash
make PLATFORM=cuda    # NVIDIA CUDA + cuFFT
make PLATFORM=rocm    # AMD ROCm/HIP + hipFFT
```

Clean:

```bash
make clean
```

Manual build commands:

```bash
# CUDA
nvcc -O3 -std=c++14 -x cu -DUSE_CUFFT  cufft_2d.hip -lcufft  -o cufft_2d

# ROCm
hipcc -O3 -std=c++14     -DUSE_HIPFFT cufft_2d.hip -lhipfft -o cufft_2d
```

## Run

```bash
./cufft_2d
```

## Output Format

**CSV (stdout)** — machine-parseable:

```
cufft_2d,2048,<time_ms>,<GBs>
```

Example:

```
cufft_2d,2048,12.457,53.82
```

**Human-readable (stderr):**

```
Verification: bin[3][5] mag=2097152.0  expected=2097152.0  rel_err=0.0000  PASS
Row-0 bin magnitudes [0..15]:
  [0][ 0] = 0.00
  ...
cufft_2d  NX=2048 NY=2048  time_ms=12.457  GB/s=53.82
```

## Performance Metric

```
GB/s = 2 × NX × NY × 2 × sizeof(float) / time_s / 1e9
```

The factor `2 × NX × NY × 2 × sizeof(float)` accounts for:
- Reading NX×NY complex floats (input)
- Writing NX×NY complex floats (output)
- Each complex float = 2 × 4 bytes = 8 bytes

## Timing

- **5 warmup** iterations (discarded)
- **5 timed** iterations; the reported time is the **average**

On CUDA/ROCm, timing uses `cudaEvent` / `hipEvent` (GPU-side timer).

## Platform Notes

### CUDA / ROCm (cufft_2d.hip)

- Uses interleaved complex format: `cufftComplex` / `hipfftComplex` (`float2`)
- Out-of-place transform: separate input (`d_in`) and output (`d_out`) buffers
- Plan created once with `cufftPlan2d(NX, NY)` / `hipfftPlan2d(NX, NY)` and reused
