# tier2/altis_raytracing — Simple Ray Tracing (Altis)

Benchmarks a simple ray tracer on the GPU. Each thread casts one ray from a camera through an image plane, tests all spheres for intersection, and shades the nearest hit with Phong lighting. Emits per-iteration kernel times; throughput in Mrays/sec is derived downstream.

Derived from the [Altis benchmark suite](https://github.com/utcs-scea/altis) ray tracing workload.

## Algorithm

- **W×H** image (default: 1024×1024 = 1M pixels/rays)
- **N** spheres randomly placed in the scene (default: 64)
- For each pixel:
  1. Compute ray direction from camera through pixel
  2. Test ray against all N spheres (ray-sphere intersection via quadratic formula)
  3. Shade nearest hit using Phong lighting model:
     - Ambient component (0.15)
     - Diffuse (Lambertian) shading
     - Specular highlight (exponent ~8)
  4. Write RGBA pixel to output
- Camera at (0, 0, 5), looking along -Z, 90° FOV
- Light direction: normalized (1, 1, 1)
- Mrays/sec = W × H / time\_sec / 10⁶ (derived downstream from the emitted times)

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_width` | Image width in pixels | 1024 |
| `BENCH_PARAM_height` | Image height in pixels | 1024 |
| `BENCH_PARAM_spheres` | Number of spheres | 64 |
| `BENCH_PARAM_warmup` | Warmup launches (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed launches, each recorded | 7 |

## Files

| File | Description |
|------|-------------|
| `altis_raytracing.hip` | HIP source — ray tracing kernel, CPU reference, verification |
| `altis_raytracing.metal` | Metal compute shader — ray tracing kernel |
| `altis_raytracing_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./altis_raytracing                                # defaults: 1024x1024, 64 spheres
./altis_raytracing --width 512 --height 512       # smaller image
./altis_raytracing --spheres 128                  # more spheres
```

## Output

**stdout** — JSON lines, one `kernel` record per timed iteration followed by a
`summary` whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"raytrace_kernel","iter":0,"time_ms":2.345600,"host_time_ms":2.401000,"params":{"width":1024,"height":1024,"spheres":64}}
...
{"type":"summary","total_time_ms":16.419200}
```

Derived metrics (e.g. Mrays/sec) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: Apple M2
Ray Tracing  |  Image: 1024x1024  |  Spheres: 64  |  Iterations: 3 warmup + 7 timed

raytrace_kernel iter 0: gpu = 2.346 ms, host = 2.401 ms
...
Total GPU time: 16.419 ms over 7 iterations
PASS
```

## Verification

- 1024 randomly-selected pixels are compared against a CPU reference implementation
- Allows ±1 per channel for floating-point rounding differences
- Reports PASS/FAIL with mismatch details

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `altis_raytracing.hip`
- **Metal**: compiled with `clang++` from `altis_raytracing_metal.mm`; the Metal shader (`altis_raytracing.metal`) is loaded and compiled at runtime from the same directory as the binary
