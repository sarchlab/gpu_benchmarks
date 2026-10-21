# tier2/altis_raytracing/hip — Simple Ray Tracing (Altis)

Benchmarks a simple ray tracer on the GPU. Each thread casts one ray from a camera through an image plane, tests all spheres for intersection, and shades the nearest hit with Phong lighting. The kernel runs exactly once per execution; its time is emitted and throughput in Mrays/sec is derived downstream.

## Provenance & fidelity

Derived from the [Altis benchmark suite](https://github.com/utcs-scea/altis) — the `raytracing` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- A rewrite of the raytracing workload (mostly disclosed) — the scene setup, intersection, and shading are our own code, not the upstream kernel.

Read this as "a ray-tracing benchmark in the spirit of Altis," not Altis's `raytracing`.

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
| `BENCH_PARAM_vector_add_warmup` | Cold-cache calibration: boost the clock and evict the LLC before the timed launch; set `false` to time as-is | true |

Width and height are independent — setting `width` does not change `height`.

## Files

| File | Description |
|------|-------------|
| `altis_raytracing.hip` | HIP source — ray tracing kernel, CPU reference, verification |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./altis_raytracing                                # defaults: 1024x1024, 64 spheres
./altis_raytracing --width 512 --height 512       # smaller image
./altis_raytracing --spheres 128                  # more spheres
```

## Output

**stdout** — JSON lines, one `kernel` record for the single timed launch
followed by a `summary` whose `total_time_ms` equals that launch's GPU event
time:
```
{"type":"kernel","name":"raytrace_kernel","iter":0,"time_ms":2.345600,"host_time_ms":2.401000,"params":{"width":1024,"height":1024,"spheres":64}}
{"type":"summary","total_time_ms":2.345600}
```

Derived metrics (e.g. Mrays/sec) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Ray Tracing  |  Image: 1024x1024  |  Spheres: 64

raytrace_kernel: gpu = 2.346 ms, host = 2.401 ms
Total GPU time: 2.346 ms
PASS
```

## Verification

- 1024 randomly-selected pixels are compared against a CPU reference implementation
- Allows ±1 per channel for floating-point rounding differences
- Reports PASS/FAIL with mismatch details; on FAIL the binary exits with a non-zero status

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `altis_raytracing.hip`
