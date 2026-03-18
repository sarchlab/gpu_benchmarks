# tier3/cufft_1d — 1D Complex FFT via cuFFT / hipFFT / Accelerate vDSP

A multi-platform benchmark for single-precision complex 1D FFT,
wrapping vendor-optimised FFT libraries:

| Platform | Library                                       |
|----------|-----------------------------------------------|
| NVIDIA   | **cuFFT** (`cufftExecC2C`)                    |
| AMD      | **hipFFT** (`hipfftExecC2C`)                  |
| Apple    | **Accelerate vDSP** (`vDSP_fft_zip`)          |

## Algorithm

Computes a **forward 1D complex-to-complex (C2C) FFT** on N single-precision
complex samples:

- **N = 4194304 (1 << 22)**
- Input: N interleaved complex floats (CUDA/ROCm) or split-complex floats (Apple)
- Transform: forward FFT, X[k] = Σ x[n] · exp(−2πi·n·k/N)

## Input and Verification

The input signal is a pure sinusoid at discrete frequency k₀ = 3:

```
x[n].real = sin(2π · 3 · n / N),   x[n].imag = 0
```

After a forward FFT, the DFT concentration theorem predicts:

```
|X[3]|  ≈ N / 2  ≈ 2097152
|X[N-3]| ≈ N / 2
```

The benchmark checks that the relative error at bin 3 is < 1%.

**Note on vDSP scaling:** `vDSP_fft_zip` computes the standard DFT without
extra normalisation. For the sinusoidal input above, the expected magnitude at
bin 3 is N/2.

## Files

| File                    | Description                                                   |
|-------------------------|---------------------------------------------------------------|
| `cufft_1d.hip`          | CUDA/ROCm source (`#ifdef USE_CUFFT` / `#elif USE_HIPFFT`)   |
| `cufft_1d_metal.mm`     | Objective-C++ host using `vDSP_fft_zip` (Accelerate)         |
| `Makefile`              | Auto-detecting build system                                   |
| `README.md`             | This file                                                     |

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
nvcc -O3 -std=c++14 -x cu -DUSE_CUFFT  cufft_1d.hip -lcufft  -o cufft_1d

# ROCm
hipcc -O3 -std=c++14     -DUSE_HIPFFT cufft_1d.hip -lhipfft -o cufft_1d

# Apple
clang++ -O2 -std=c++17 -fobjc-arc \
  -framework Foundation -framework Accelerate \
  cufft_1d_metal.mm -o cufft_1d
```

## Run

```bash
./cufft_1d
```

## Output Format

**CSV (stdout)** — machine-parseable:

```
cufft_1d,4194304,<time_ms>,<GBs>
```

Example:

```
cufft_1d,4194304,8.231,40.71
```

**Human-readable (stderr):**

```
Verification: bin3 mag=2097152.0  expected=2097152.0  rel_err=0.0000  PASS
First 16 bin magnitudes:
  bin[ 0] = 0.00
  bin[ 1] = 0.00
  bin[ 2] = 0.00
  bin[ 3] = 2097152.00
  ...
cufft_1d  N=4194304  time_ms=8.231  GB/s=40.71
```

## Performance Metric

```
GB/s = 2 × N × 2 × sizeof(float) / time_s / 1e9
```

The factor of `2 × N × 2 × sizeof(float)` accounts for:
- Reading N complex floats (input)
- Writing N complex floats (output)
- Each complex float = 2 × 4 bytes = 8 bytes

## Timing

- **5 warmup** iterations (discarded)
- **5 timed** iterations; the reported time is the **average**

On CUDA/ROCm, timing uses `cudaEvent` / `hipEvent` (GPU-side timer).  
On Apple, timing uses `mach_absolute_time()` around `vDSP_fft_zip`.

## Platform Notes

### CUDA / ROCm (cufft_1d.hip)

- Uses interleaved complex format: `cufftComplex` / `hipfftComplex` (`float2`)
- Out-of-place transform: separate input (`d_in`) and output (`d_out`) buffers
- Plan created once with `cufftPlan1d` / `hipfftPlan1d` and reused across iterations

### Apple (cufft_1d_metal.mm)

- Uses split-complex format: separate `float *realp` and `float *imagp` arrays
- In-place transform via `vDSP_fft_zip` with stride = 1
- Input is refreshed before each timed iteration to ensure consistent results
- `FFTSetup` created once with `vDSP_create_fftsetup(22, FFT_RADIX2)`
