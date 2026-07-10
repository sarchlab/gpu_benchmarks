# rodinia_kmeans — Rodinia K-Means Clustering Benchmark

GPU K-means clustering. This is a **faithful port** of Rodinia kmeans: the
device kernels `invert_mapping` and `kmeansPoint` are the original upstream
kernels (`rodinia/cuda/kmeans/kmeans_cuda_kernel.cu`). The defining faithful
characteristic is the **inverted (feature-major) feature layout** for coalesced
loads — `invert_mapping` transforms the row-major `[point][feature]` array into a
column-major `[feature][point]` array, and `kmeansPoint` indexes it as
`point_id + j*npoints`. That layout is preserved exactly.

Derived from the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/).

## Portability substitution (texture → `__ldg`)

The original `kmeansPoint` reads the inverted features through a bound 1D texture
(`tex1Dfetch(t_features, addr)`). The HIP/CUDA compatibility layer does not map
CUDA texture references, so the fetch is replaced by `__ldg(&features[addr])`
(the read-only data cache — the closest portable equivalent). **The data layout,
which is the faithful part, is unchanged**; only the fetch mechanism differs.
Cluster centers stay in `__constant__` memory (`c_clusters`), exactly as
upstream, and centroid sums are done on the CPU each iteration (the upstream
default `CPU_CENTER_REDUCE`/`CPU_DELTA_REDUCE`; the GPU-reduction paths are
compiled out upstream and are omitted here).

## Algorithm

1. Generate `N` random points, each with `D` features (row-major); first `K`
   points seed the initial centers.
2. `invert_mapping` (once) builds the feature-major array for coalesced access.
3. Per iteration: upload centers to constant memory; `kmeansPoint` assigns each
   point to its nearest center (one thread per point, squared Euclidean distance
   to all `K` centers); copy membership back and recompute centers as per-cluster
   means on the CPU.
4. Repeat for `max_iter` iterations.

## Files

| File | Description |
|------|-------------|
| `rodinia_kmeans.hip` | HIP/CUDA source — original `invert_mapping`/`kmeansPoint` kernels + timing driver |
| `rodinia_kmeans.metal` | Metal compute shaders (separate reimplementation, not the upstream kernels) |
| `rodinia_kmeans_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Platform auto-detect build system |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect (Darwin→metal, /opt/rocm→rocm, else→cuda)
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
make clean
```

## Run

```bash
./rodinia_kmeans                                              # default: 100000 points, 34 features, 5 clusters, max_iter 10
./rodinia_kmeans --points 262144 --clusters 5 --max_iter 20
BENCH_PARAM_points=65536 ./rodinia_kmeans
```

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `BENCH_PARAM_points` | 100000 | Number of data points |
| `BENCH_PARAM_features` | 34 | Features per point (constant-memory limit: `clusters*features ≤ 1088`) |
| `BENCH_PARAM_clusters` | 5 | Number of clusters (K) |
| `BENCH_PARAM_max_iter` | 10 | K-means iterations per pass |
| `BENCH_PARAM_warmup` | 3 | Warmup clustering passes (untimed) |
| `BENCH_PARAM_iteration` | 7 | Timed clustering passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel
(`invert_mapping`, `kmeansPoint`) with that kernel's accumulated GPU event time
(`time_ms`) and host time (`host_time_ms`), then a summary whose `total_time_ms`
is the sum of all recorded GPU event times. Cluster centers are re-initialized to
the pristine initial centers at the start of every pass (untimed), so each pass
converges over identical work:

```json
{"type":"kernel","name":"invert_mapping","iter":0,"time_ms":0.412,"host_time_ms":0.480,"params":{"points":100000,"features":34,"clusters":5,"max_iter":10,"threads_per_block":256}}
{"type":"kernel","name":"kmeansPoint","iter":0,"time_ms":8.123,"host_time_ms":9.500,"params":{...}}
{"type":"summary","total_time_ms":86.419}
```

`invert_mapping` reads immutable features and produces the same result every
pass; it is re-run once per pass only so that it emits a timing record.

**stderr** — human-readable per-pass timing.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_kmeans.hip` (the faithful upstream kernels; texture fetch → `__ldg`).
- **Metal**: a separate reimplementation in `rodinia_kmeans_metal.mm`; it is not the upstream kernel.
- Input data is generated synthetically — no external input files required.

## Reference

Rodinia: A Benchmark Suite for Heterogeneous Computing. S. Che et al., IEEE IISWC 2009.
