# lonestar_bh — Barnes-Hut N-body Simulation

**Suite:** Lonestar GPU  
**Tier:** 2 — Established Benchmark Suites

## Provenance & fidelity

Derived from the [Lonestar GPU](https://iss.oden.utexas.edu/?p=projects/galois/lonestar) suite — the `bh` (Barnes-Hut) workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- The 2D quadtree is **BUILT ON THE CPU** and re-uploaded every timestep; only force-traversal and integration run on the GPU. The irregular GPU tree-build that Barnes-Hut is known for is **untimed host code** — what is measured is tree traversal + integrate, **not** the tree build.

Read this as "a Barnes-Hut force-traversal benchmark in the spirit of Lonestar," not Lonestar's `bh`.

## Algorithm

The Barnes-Hut algorithm is an O(N log N) approximation to the N-body gravitational
force calculation. Instead of computing all N² pairwise interactions, it builds a
spatial tree (quadtree in 2D) and approximates distant groups of bodies as a single
point mass.

### Implementation

This is a simplified 2D Barnes-Hut implementation:

1. **Tree build (CPU):** Construct a 2D quadtree from body positions. Each leaf
   holds a single body; internal nodes store center-of-mass and total mass.
2. **Force calculation (GPU):** Each thread computes the force on one body by
   traversing the quadtree with a stack. The opening criterion uses θ = 0.5:
   if `(node_width / distance) < θ`, the node's center-of-mass approximation
   is used; otherwise, the node's children are opened.
3. **Integration (GPU):** Update velocities and positions using Euler integration.
4. **Repeat** for 10 timesteps.

Note that the quadtree is built **on the CPU** at every timestep (and the
flattened tree re-uploaded to the device); only the force calculation and the
Euler integration run on the GPU, and only those two kernels are timed. The
simulation runs exactly once per execution; repetition across fresh processes
is the harness's job (`gpubench run --reps N`).

### Parameters

| Parameter    | Default | Description                     |
|--------------|---------|---------------------------------|
| `--bodies`   | 32768   | Number of bodies                |
| `--timesteps`| 10      | Simulation timesteps per run    |
| `--theta`    | 0.5     | Opening angle for BH criterion  |

Environment variables (`BENCH_PARAM_bodies`, `BENCH_PARAM_timesteps`,
`BENCH_PARAM_theta`) override the CLI flags, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed run; set `false` to time as-is |

## Build

```bash
make                    # auto-detect platform
make PLATFORM=cuda      # NVIDIA CUDA
make PLATFORM=rocm      # AMD ROCm
```

## Run

```bash
./lonestar_bh
./lonestar_bh --bodies 65536 --timesteps 20
```

## Output

**JSON lines (stdout):** for the single timed run, one `{"type":"kernel",...}`
record per kernel (`bh_force_kernel`, `integrate_kernel`) with that kernel's
accumulated GPU event time (`time_ms`, summed over the timesteps) and host
wall-clock time (`host_time_ms`), followed by a
`{"type":"summary","total_time_ms":...}` line whose value is the sum of all
recorded GPU event times. Derived metrics
(GFLOPS) are computed downstream from `time_ms` and the params.

**Human-readable (stderr):**
```
Device: AMD Radeon RX 7900 XTX (id=0)
Barnes-Hut N-body  |  Bodies: 32768  |  Theta: 0.50  |  Timesteps: 10
bh_force gpu = 42.123 ms, integrate gpu = 0.456 ms
Total GPU time: 42.579 ms
Max relative error: 0.1234 (over 64 bodies)
PASS
```

## Verification

Forces computed by the BH approximation are compared against direct O(N²)
computation on a subset of 64 bodies. Since BH is an approximation, a relative
error threshold of 50% is used (typical BH error with θ=0.5 is ~10-30%).
On FAIL the binary exits with a non-zero status.

## References

- Barnes, J., Hut, P. (1986). "A hierarchical O(N log N) force-calculation algorithm"
- Burtscher, M., Pingali, K. (2011). "An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm" (Lonestar GPU)
