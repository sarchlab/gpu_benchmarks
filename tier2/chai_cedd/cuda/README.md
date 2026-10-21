# Chai CEDD (native CUDA)

One frame of the four-stage byte-valued CEDD workload from
[Chai CUDA-D/CEDD, commit 61db988](https://github.com/chai-benchmarks/chai/tree/61db988662cb993fa108429a13759dbdc83ab9be/CUDA-D/CEDD).
The [GPU](https://github.com/chai-benchmarks/chai/blob/61db988662cb993fa108429a13759dbdc83ab9be/CUDA-D/CEDD/kernel.cu)
and [CPU](https://github.com/chai-benchmarks/chai/blob/61db988662cb993fa108429a13759dbdc83ab9be/CUDA-D/CEDD/kernel.cpp)
sources agree on the operations below. The port retains their arithmetic and
16×16 shared-memory tiles, with bounded cooperative halo loading for partial
tiles. The upstream copyright and license are retained in the source and
[UPSTREAM_LICENSE](UPSTREAM_LICENSE).

## Operator and boundaries

All images, intermediate magnitudes and directions use unsigned bytes. One
process executes these four kernels once, in this order:

1. `gaussian_kernel`: 3×3 weights `{{1,2,1},{2,4,2},{1,2,1}}/16`.
   The upstream accumulator is an **integer**: each weighted term is truncated
   when added. Computing a float sum and rounding once gives a different result.
2. `sobel_kernel`: integer Sobel weights accumulated into floats; `hypot`
   magnitude truncated to an integer and clamped to 255; `atan2` direction
   quantized to 0°, 45°, 90° or 135° using the pinned inclusive angular bounds.
3. `non_max_supp_kernel`: keep the magnitude only when it is **strictly greater**
   than both directional neighbors; ties are suppressed.
4. `hyst_kernel`: pinned low=10/high=70 branches, with their midpoint=40 for
   intermediate magnitudes. This is independent per-pixel thresholding,
   equivalent to output 255 when magnitude≥40 and zero otherwise. It does not
   propagate edge connectivity or inspect neighboring strong edges.

The buffers follow the original dependency chain `A → B → A → B → A`.
Each stage writes only the interior. A starts with the input image, so its
outer ring retains those input bytes, including in the final output. B and the
angle buffer start at zero. This explicitly defines upstream's uninitialized
scratch border, which can affect pixels near the image edge. Comparisons with
upstream must initialize the same borders; undefined upstream bytes are not a
reference result. Original GPU wrappers floor the tile count and omit partial
tiles; this port covers every interior pixel with bounds-checked halo loads and
output guards after the shared-memory barrier.

The original driver handles a collection of image frames across CPU/GPU workers.
This benchmark's unit of work is one GPU frame with deterministic synthetic
values from the existing `srand(42); rand()%256` sequence. These values are now
stored directly as bytes, rather than their previous exact float representation.
This is not the upstream image dataset. Libc RNG sequences still require input
provenance checks for comparisons across systems.

## Build and run

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./chai_cedd                   # 2048×2048
./chai_cedd W=4096 H=1024
make test      # GPU stage, guard and production-pass tests
```

| Control | Default | Meaning |
|---|---|---|
| `W=` / `BENCH_PARAM_width` | 2048 | Width; environment takes precedence |
| `H=` / `BENCH_PARAM_height` | 2048 | Height; environment takes precedence |
| `BENCH_PARAM_vector_add_warmup` | true | Shared entry warmer; false leaves the incoming state as-is |

Both dimensions must be integers ≥3 and their product must fit signed kernel
indexing. The manifest preserves 22 widths (64…16384) and four heights
(512, 1024, 2048, 4096): **88 configurations**, or **616 processes** at seven
repetitions under the runner's standard warmer policy. The previous low/high
threshold sweep is removed because the pinned operator fixes these constants.
Any old threshold environment variable or command-line override is rejected;
it cannot silently label a different workload. Other unknown arguments also fail.

This replaces the previous five-stage float pipeline, 5×5 Gaussian, non-strict
suppression and one-hop threshold rule. Old measurements are not comparable
and must be recollected. Fixed `operator`, `boundary`, `input_generator` and
`verification` tags are emitted in each kernel's `params` alongside width/height.

## Timing and verification

Upload A and zero scratch buffers before the single entry warmer. Each of the
four stages has one GPU-event interval and one host launch interval. No resets,
transfers, warmers or verification work intervene between stages. Standard output
contains four kernel JSON records and one summary whose GPU time is their sum.
Downstream Mpixels/s uses width×height divided by that sum; the active interior
of each stage is `(width−2)×(height−2)` pixels.

After timing, copy the actual final A to the host and compare **every byte**,
including the input-valued border, with an independent integer/double CPU
pipeline. The reference evaluates Gaussian term truncation with integer division,
Sobel with integer differences and double magnitude/direction classification,
and the final threshold with its equivalent midpoint predicate. There is no
mismatch quota or floating tolerance on the byte output. A mismatch prints its
index, returns nonzero and makes the runner reject the measurement. The fixed
policy is `full_cpu_byte_v1`. No separate GPU verification pipeline is launched.

Verification adds O(width×height) CPU work outside the reported intervals.
Device buffers total three bytes per pixel; host input/output total two bytes
per pixel, and the CPU reference adds three more while active. At 16384×4096,
these are 192 MiB device and 320 MiB peak host array storage, plus the shared
warmer's independent memory requirement and runtime overhead. Capacity and
process-time budgets must be qualified for the target.