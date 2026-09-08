# rodinia_kmeans — Rodinia K-Means Clustering Benchmark

GPU K-means clustering using adapted Rodinia GPU kernels and the suite's
portable measurement driver.

## Provenance & fidelity

The source reference is Rodinia 3.1, commit
`366b283456506ef2fe2c2b7dc5e83e941ffc5524`:
[GPU kernels](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/kmeans/kmeans_cuda_kernel.cu),
[GPU host driver](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/kmeans/kmeans_cuda.cu),
[clustering loop](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/kmeans/kmeans_clustering.c), and
[input driver](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/kmeans/kmeans.c).

The `invert_mapping` body retains the pinned row-major `[point][feature]` to
feature-major `[feature][point]` mapping. The default CPU-reduction
`kmeansPoint` body retains its nearest-center arithmetic, strict `<` comparison
(first center wins equal computed distances), constant-memory centers and
point indexing. Default-disabled GPU reduction branches are omitted. The
assignment launch retains 256 threads and the rounded square 2D grid; inversion
uses the corresponding total block count in a 1D grid.

The pinned texture read `tex1Dfetch(t_features,addr)` becomes
`__ldg(&features[addr])`. This preserves the feature address mapping and uses a
different memory-access mechanism. Texture/cache behavior, generated
instructions and performance equivalence require target-specific qualification.
This is an adapted port with the following driver differences:

- The local driver generates synthetic [0,1) floats using an LCG seeded with
  42; the pinned driver reads an ASCII or binary dataset. For the declared
  sizes and five centers, both initialize from the first five input points.
- The local driver performs exactly `max_iter` assignments (default 10).
  The pinned clustering loop stops on membership-change convergence, with a
  loop-count guard. These control different amounts of work.
- Both use CPU centroid sums in ascending point order and retain the previous
  center for an empty cluster. The local driver omits the pinned default path's
  unused membership and global cluster-buffer uploads. Texture binding setup is
  omitted because the feature texture access is replaced by `__ldg`. Centers
  used by the assignment kernel are still uploaded to constant memory.
- The local driver adds the entry warmer, single-pass execution and accumulated
  per-kernel JSON timings, including feature inversion. These records describe
  the local measured work, not the pinned program's end-to-end runtime.

Source comparison does not establish full measured-result or AMD qualification;
those remain open in [the readiness tracker](https://github.com/sarchlab/gpu_benchmarks/issues/71).

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
| `rodinia_kmeans.hip` | HIP/CUDA source — adapted `invert_mapping`/`kmeansPoint` kernels + timing driver |
| `Makefile` | Platform auto-detect build system |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect (/opt/rocm→rocm, else→cuda)
make PLATFORM=cuda
make PLATFORM=rocm
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
| `BENCH_PARAM_max_iter` | 10 | K-means assignment iterations in the run |
| `BENCH_PARAM_vector_add_warmup` | `true` | Cold-cache calibration before the timed run |

The clustering (all `max_iter` iterations) runs **exactly once**; repetition
is the harness's job (`gpubench run --reps N`).

## Output

**stdout** — JSON lines: one `kernel` record per kernel
(`invert_mapping`, `kmeansPoint`) with that kernel's accumulated GPU event time
(`time_ms`) and host time (`host_time_ms`) over the run, then a summary whose
`total_time_ms` is the sum of all recorded GPU event times:

```json
{"type":"kernel","name":"invert_mapping","iter":0,"time_ms":0.412,"host_time_ms":0.480,"params":{"points":100000,"features":34,"clusters":5,"max_iter":10,"threads_per_block":256}}
{"type":"kernel","name":"kmeansPoint","iter":0,"time_ms":8.123,"host_time_ms":9.500,"params":{...}}
{"type":"summary","total_time_ms":8.535}
```

**stderr** — human-readable timing.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_kmeans.hip` (adapted kernels; texture fetch → `__ldg`).
- Input data is generated synthetically — no external input files required.

## Reference

Rodinia: A Benchmark Suite for Heterogeneous Computing. S. Che et al., IEEE IISWC 2009.
