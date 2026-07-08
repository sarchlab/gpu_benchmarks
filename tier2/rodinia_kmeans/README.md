# rodinia_kmeans — Rodinia K-Means Clustering Benchmark

GPU implementation of the K-Means clustering algorithm, derived from the
[Rodinia benchmark suite](https://rodinia.cs.virginia.edu/).

K-Means partitions **N** D-dimensional data points into **K** clusters by
iteratively assigning each point to the nearest cluster center and
recomputing the centers.

## Algorithm

1. **Initialize** N random points in [0,1)^D; use the first K as initial centers.
2. **Assign** — GPU kernel assigns each point to its nearest cluster center
   (one thread per point; computes squared Euclidean distance to all K centers).
3. **Update** — GPU kernel accumulates per-cluster point sums using atomics;
   host divides to get new centers.
4. **Repeat** until fewer than 0.01% of assignments change or `max_iter` is reached.
5. **Verify** — run one extra assign pass and check final delta < 1%.

## Files

| File | Description |
|------|-------------|
| `rodinia_kmeans.hip` | HIP/CUDA implementation (two GPU kernels + host loop) |
| `rodinia_kmeans.metal` | Metal compute shaders (assign + update kernels) |
| `rodinia_kmeans_metal.mm` | Objective-C++ Metal host code |
| `Makefile` | Platform auto-detect build system |
| `README.md` | This file |

## Build

```bash
# Auto-detect platform (Darwin → metal, /opt/rocm → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal

# Clean
make clean
```

## Run

```bash
# Default: N=16384 points, D=16 dims, K=8 clusters, max_iter=10
./rodinia_kmeans

# Custom parameters
./rodinia_kmeans --points 65536 --dims 32 --clusters 16 --max_iter 20
```

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `BENCH_PARAM_points` | 16384 | Number of data points (overrides `--points`) |
| `BENCH_PARAM_dims` | 16 | Dimensions per point (overrides `--dims`) |
| `BENCH_PARAM_clusters` | 8 | Number of clusters (overrides `--clusters`) |
| `BENCH_PARAM_max_iter` | 10 | Max k-means iterations per pass (overrides `--max_iter`) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup clustering passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed clustering passes, each recorded |

## Output

Per-iteration convergence info is printed to **stderr**:

```
  iter  1: changed=12345   delta=0.752930
  iter  2: changed=1234    delta=0.075317
  ...
Final stability check: changed=0  delta=0.000000  PASS
```

Timing records are printed to **stdout** as JSON lines: per timed clustering
pass, one kernel record per kernel (`kmeans_assign_kernel`,
`kmeans_update_kernel`) carrying that kernel's GPU event time (`time_ms`)
and host wall-clock time (`host_time_ms`) accumulated over the pass's
k-means iterations, then a summary whose `total_time_ms` is the sum of all
recorded GPU event times. Centroids and membership are reset to their
initial state before every pass (untimed), so each pass converges over
identical work:

```
{"type":"kernel","name":"kmeans_assign_kernel","iter":0,"time_ms":8.123400,"host_time_ms":8.500000,"params":{"points":16384,"dims":16,"clusters":8,"max_iter":10}}
{"type":"kernel","name":"kmeans_update_kernel","iter":0,"time_ms":4.222200,"host_time_ms":4.610000,"params":{...}}
...
{"type":"summary","total_time_ms":86.419200}
```

Derived metrics such as points/sec are computed downstream from `time_ms`
and the params.

## Platform Notes

- **CUDA/ROCm**: Uses `hipEvent_t` for timing; atomicAdd for center updates.
- **Metal**: Uses `atomic_fetch_add_explicit` with `memory_order_relaxed` on
  device-side `atomic_float` / `atomic_int`. Shader source is loaded at
  runtime from the same directory as the binary.
- The Metal `.metal` file must be present alongside the binary at runtime.

## Reference

Rodinia: A Benchmark Suite for Heterogeneous Computing.
S. Che et al., IEEE IISWC 2009.
