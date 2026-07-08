# lonestar_bh — Barnes-Hut N-body Simulation

**Suite:** Lonestar GPU  
**Tier:** 2 — Established Benchmark Suites

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
4. **Repeat** for 10 timesteps (tree is rebuilt on CPU each step).

### Parameters

| Parameter    | Default | Description                     |
|--------------|---------|---------------------------------|
| `--bodies`   | 32768   | Number of bodies                |
| `--timesteps`| 10      | Simulation timesteps per run    |
| `--theta`    | 0.5     | Opening angle for BH criterion  |
| `--iterations`| 5      | Legacy parameter (echoed in params; pass count comes from `BENCH_PARAM_num_iters`) |

Environment variables (`BENCH_PARAM_bodies`, `BENCH_PARAM_timesteps`,
`BENCH_PARAM_theta`, `BENCH_PARAM_iterations`) override the CLI flags, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_num_warmup` | 3 | Untimed warmup passes |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded individually |

## Build

```bash
make                    # auto-detect (Metal on macOS)
make PLATFORM=cuda      # NVIDIA CUDA
make PLATFORM=rocm      # AMD ROCm
make PLATFORM=metal     # Apple Metal
```

## Run

```bash
./lonestar_bh
./lonestar_bh --bodies 65536 --timesteps 20
```

## Output

**JSON lines (stdout):** per timed pass, one `{"type":"kernel",...}` record per
kernel (`bh_force_kernel`, `integrate_kernel`) with that pass's accumulated GPU
event time (`time_ms`, summed over the timesteps) and host wall-clock time
(`host_time_ms`), followed by a `{"type":"summary","total_time_ms":...}` line
whose value is the sum of all recorded GPU event times. Derived metrics
(GFLOPS) are computed downstream from `time_ms` and the params.

**Human-readable (stderr):**
```
Device: Apple M2
Barnes-Hut N-body  |  Bodies: 32768  |  Theta: 0.50  |  Timesteps: 10  |  Iterations: 3 warmup + 7 timed
pass 0: bh_force gpu = 42.123 ms, integrate gpu = 0.456 ms
...
Total GPU time: 300.123 ms over 7 passes
Max relative error: 0.1234 (over 64 bodies)
PASS
```

## Verification

Forces computed by the BH approximation are compared against direct O(N²)
computation on a subset of 64 bodies. Since BH is an approximation, a relative
error threshold of 50% is used (typical BH error with θ=0.5 is ~10-30%).

## References

- Barnes, J., Hut, P. (1986). "A hierarchical O(N log N) force-calculation algorithm"
- Burtscher, M., Pingali, K. (2011). "An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm" (Lonestar GPU)
