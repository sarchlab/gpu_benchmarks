# tier3/cufft_2d — 2D Complex FFT via cuFFT / hipFFT / Accelerate vDSP

A multi-platform benchmark for single-precision complex 2D FFT,
wrapping vendor-optimised FFT libraries:

| Platform | Library                                           |
|----------|---------------------------------------------------|
| NVIDIA   | **cuFFT** (`cufftExecC2C` with `cufftPlan2d`)    |
| AMD      | **hipFFT** (`hipfftExecC2C` with `hipfftPlan2d`) |
| Apple    | **Accelerate vDSP** (row + column 1D FFT passes)  |

## Algorithm

Computes a **forward 2D complex-to-complex (C2C) FFT** on an NX×NY
single-precision complex grid:

- **NX = NY = 2048** (total: 4,194,304 complex samples)
- Input: NX×NY interleaved complex floats (CUDA/ROCm) or split-complex floats (Apple)
- Transform: forward 2D FFT,
  X[m][n] = Σ_{i,j} x[i][j] · exp(−2πi·(m·i/NX + n·j/NY))

### Apple Implementation Note

Apple's Accelerate framework does not expose a `C2C` 2D complex FFT directly.
The benchmark implements 2D FFT via **two sequential passes of 1D FFTs**:

1. Apply `vDSP_fft_zip` (forward) to each of the NX rows (NY-point transform).
2. Apply `vDSP_fft_zip` (forward) to each of the NY columns (NX-point transform,
   via a temporary contiguous buffer to handle non-unit stride).

By the separability of the 2D DFT, this is mathematically identical to a
direct 2D transform.

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
| `cufft_2d_metal.mm`     | Objective-C++ host using `vDSP_fft_zip` row+column passes         |
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
make PLATFORM=metal   # Apple Accelerate vDSP
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

# Apple
clang++ -O2 -std=c++17 -fobjc-arc \
  -framework Foundation -framework Accelerate \
  cufft_2d_metal.mm -o cufft_2d
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
On Apple, timing uses `mach_absolute_time()` around the row+column FFT passes
(data initialisation is not included in the timed region).

## Platform Notes

### CUDA / ROCm (cufft_2d.hip)

- Uses interleaved complex format: `cufftComplex` / `hipfftComplex` (`float2`)
- Out-of-place transform: separate input (`d_in`) and output (`d_out`) buffers
- Plan created once with `cufftPlan2d(NX, NY)` / `hipfftPlan2d(NX, NY)` and reused

### Apple (cufft_2d_metal.mm)

- Uses split-complex format: separate `float *realp` and `float *imagp` arrays
- 2D FFT implemented as row FFTs followed by column FFTs using `vDSP_fft_zip`
- Column FFTs use a temporary contiguous buffer to handle non-unit memory stride
- `FFTSetup` created once with `vDSP_create_fftsetup(11, FFT_RADIX2)` (2^11 = 2048)
- Input is re-initialised before the timed iterations (warmup is in-place)
