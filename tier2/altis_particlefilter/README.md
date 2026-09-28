# tier2/altis_particlefilter — Altis video particle filter

Runs the Altis/Rodinia video particle filter in double precision for ten updates.
Every process measures one full GPU trajectory, then verifies every particle of
that actual run against a complete CPU trajectory. It does not rerun a smaller
GPU problem or accept a mean-position check.

## Provenance and input

Ported from [`ex_particle_CUDA_float_seq.cu`](https://github.com/utcs-scea/altis/blob/042e2926d1fe9a5ea5b3e2d659b75bf925b152cc/src/cuda/level2/particlefilter/ex_particle_CUDA_float_seq.cu)
at Altis commit `042e2926d1fe9a5ea5b3e2d659b75bf925b152cc`, originally from Rodinia.
Despite its `float` filename, this GPU library uses **double** positions, weights,
likelihoods, CDFs and random arithmetic. See `UPSTREAM_LICENSE`.

The owner-approved input is a fixed 128 × 128 × 11 video with a radius-five
moving disk, background 100, foreground 228, and Gaussian pixel noise. Frames
1 through 10 are filtered. The 69 disk neighbors, motion, integer truncation,
flattened absolute gather indices and out-of-range fallback to video element 0
follow the pinned code. The source's `roundDouble` actually truncates because it
returns a postincremented local variable; this port preserves that behavior.
The source's two x/y motion conventions are also retained, not reinterpreted.
The existing 27 particle counts (500 through 10,000,000) remain in `params.json`.

These deliberate input and safety changes are **not** claims of identical
execution of the defective upstream driver:

- Seed each particle with the defined signed interpretation of `12345 * i`
  modulo 2^32, replacing the wall-clock seed. Video noise advances particle 0's
  RNG before upload, as upstream does.
- Define the intended 32-bit LCG wrap without signed overflow; retain signed
  remainder modulo `INT_MAX` and absolute normalization. For Box–Muller only,
  replace a zero uniform with `1 / INT_MAX`, retaining both RNG draws.
- Initialize the entire video and dilation background. Convert finite noise to
  signed integer before unsigned-byte wrapping. Upstream reads uninitialized
  pixels and converts negative floating-point noise directly to unsigned byte.
- Use the approved square domain; the original driver can index out of bounds
  for rectangular inputs. This fixed domain is different from Altis's presets.
- Split upstream's racy combined normalization/CDF/threshold kernel into three
  globally ordered launches. Block barriers cannot order different blocks.
  A separate scalar holds the common threshold origin, avoiding concurrent
  reads and writes of `u[0]`.

Portable, cross-libm input generation and AMD hardware qualification remain
separate work. Input noise uses the host's double-precision math library.

## Timed GPU work

Each frame launches these six phases, in order:

1. `likelihood_kernel`: draw motion noise; update positions; gather 69 video
   pixels; compute likelihood and exponential weights; reduce 512-thread blocks.
2. `sum_kernel`: ordered, sequential sum of block partials.
3. `normalize_weights_kernel`: divide all weights by the total.
4. `cdf_kernel`: ordered, sequential CDF and one particle-0 uniform draw.
5. `thresholds_kernel`: systematic-resampling thresholds for every particle.
6. `find_index_kernel`: **linear search** from the first CDF entry for each
   particle, with the original last-particle fallback.

The quadratic GPU search is intentional upstream work. It has not been replaced
with a binary search or a different resampling algorithm. Only the untimed CPU
oracle uses a binary search for the same first qualifying CDF entry.

Initialization, uploads, allocation, CPU verification and downloads are untimed.
The vector-add warmer runs once after setup and before frame 1; all ten frames
then keep their natural inter-kernel data reuse. There is no host scan or
verification download in the measured trajectory. stdout contains six accumulated
kernel records (ten launches each), followed by their GPU-time sum. Records carry
particle count, seed, video shape, precision, algorithm/input/verification IDs,
particle updates and allocated device bytes; derived rates belong downstream.

## Verification and limits

The CPU starts from pristine video, seeds and centered particles, recomputes all
ten frames, models the specified 512-lane reduction order, and uses double math
with explicit fused motion updates. GPU outputs never initialize the reference.
After timing, verification checks all final predicted/resampled x/y positions,
likelihoods, normalized weights, CDF entries, thresholds, RNG states, all 69
indices per particle, block sums and the common threshold origin: exactly
`78 * N + ceil(N / 512) + 1` values. Downloads use bounded chunks.

Integer seeds and gather indices must match exactly. Double values must be finite
and differ by at most `1e-11 * max(1, abs(reference))`. Weights and partial sums
instead use `4e-300 + 1e-10 * abs(reference)`, so tiny weights cannot silently be
zeroed by an absolute tolerance for positions. A mismatch, unavailable reference,
allocation failure or invalid input exits nonzero. Timings emitted before a
verification failure remain available to the harness as failed measurements.
The host reference uses at most 16 workers and handles partial thread-creation
failure by finishing the work with the remaining workers and caller.

Sizes must be positive decimal integers within the signed `69 * N` gather-index
range. The driver checks launch limits and device memory including warmer space
before allocating. Allocation/reference failures never become PASS. This input
range does not imply that every size is qualified for collection.

## H100 collection profile

`h100_profile.json` records the checked source/binary hashes, toolchain, physical
H100 UUID and result of **every** declared size with a 60-second whole-process
budget, including setup, warmer and full CPU verification. Only rows marked
`qualified` belong in H100 collection. A timeout or complete trajectory mismatch
is explicitly unsupported; infrastructure/protocol failures abort qualification.
These runs establish readiness, not publication-quality performance statistics.
The recorded H100 profile qualifies 24 counts through 3,000,000 particles;
5,000,000, 7,500,000 and 10,000,000 exceeded the 60-second process budget.

Generate fresh evidence and a filtered harness workspace on an H100 host:

```bash
python3 tests/particlefilter_h100_profile.py \
  --device-uuid GPU-... --work-dir /tmp/particlefilter-qualification
```

Use a new output directory. After `profile.json` reports `complete: true`, use
the generated `collection/` workspace with its exact qualified binary and
narrowed manifest. It fixes the warmer to `true`. Copy the native `gpubench`
executable into that workspace: the harness locates benchmarks beside its own
executable before consulting the current directory. Running the CLI from another
checkout can select that checkout's full candidate manifest instead.

```bash
cd /tmp/particlefilter-qualification/collection
cp /absolute/path/to/gpubench ./gpubench
CUDA_VISIBLE_DEVICES=GPU-YOUR-H100-UUID ./gpubench run \
  --benchmark altis_particlefilter --platform cuda --reps 7 --timeout 60 \
  --db results.sqlite
```

Keep the complete profile and raw results together. Do not compile inside this
collection workspace: it intentionally contains the already-qualified binary.
The general repository manifest still enumerates all 27 candidate counts and
does not enforce the restricted profile. Requalify after changing relevant
sources, compiler, hardware or collection input.
CI checks that this committed profile's source hashes and candidate coverage
remain current. A change to a benchmark input or any of the three shared headers
requires fresh H100 qualification; updating hashes alone is not requalification.
Run `python3 tests/h100_profile_identity.py --benchmark altis_particlefilter`
from the repository to check the source/configuration identity without a GPU.

## Build and run

```bash
make PLATFORM=cuda CUDA_ARCH=sm_90
# Or: make PLATFORM=rocm (AMD hardware qualification is pending)
./altis_particlefilter --size 100000
BENCH_PARAM_size=50000 ./altis_particlefilter
```

Environment size overrides `--size`; default is 100,000. Seed, video shape,
precision, block size and ten updates are fixed, not tunable controls.
`BENCH_PARAM_vector_add_warmup=false` disables only the measurement warmer and is
not included in the qualified collection profile.

Regression tests cover frozen upstream input/RNG equality (with the documented
safety repairs), complete references, worker fallback, strict input admission,
frozen GPU algorithm bodies, every final output family, omitted stages, malformed
qualification evidence and actual measured-result corruption. The frozen original
multi-block normalization kernel is not used as a correctness oracle.
