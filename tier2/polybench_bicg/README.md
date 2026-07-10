# polybench_bicg — PolyBench BiCG Sub-kernel Benchmark

Benchmarks the BiCG sub-kernel of the BiCGStab iterative linear solver on GPU.

## Algorithm

Computes two matrix-vector products simultaneously:

```
s = A^T * r   (transpose matrix-vector product)
q = A   * p   (matrix-vector product)
```

where:
- **A** is an M×N matrix (default M=N=4096)
- **p** is an N-vector (input)
- **r** is an M-vector (input)
- **s** is an N-vector (output)
- **q** is an M-vector (output)

### GPU Kernels

| Kernel | Operation | Parallelism |
|--------|-----------|-------------|
| `bicg_kernel1` | `s[j] = Σᵢ A[i,j] * r[i]` | One thread per column j |
| `bicg_kernel2` | `q[i] = Σⱼ A[i,j] * p[j]` | One thread per row i |

## Files

| File | Description |
|------|-------------|
| `polybench_bicg.hip` | HIP/CUDA implementation (NVIDIA + AMD) |
| `polybench_bicg.metal` | Metal compute shaders (Apple GPU) |
| `polybench_bicg_metal.mm` | Metal host code (Objective-C++) |
| `Makefile` | Build system with platform auto-detection |

## Build

```bash
# Auto-detect platform (Darwin→metal, /opt/rocm→rocm, else→cuda)
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
# Default: M=N=4096
./polybench_bicg

# Custom size and iteration counts
BENCH_PARAM_iteration=10 ./polybench_bicg --m 8192 --n 8192
```

### Environment variables

`BENCH_PARAM_m`, `BENCH_PARAM_n`, `BENCH_PARAM_block_size` override the flags.
In addition:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_warmup` | 3 | Warmup passes (not recorded) |
| `BENCH_PARAM_iteration`  | 7 | Timed passes, each recorded  |

## Output

**stderr** — device info, correctness check, timing details:
```
Device: Apple M2 Pro
Matrix size: 4096×4096

Correctness check (first 8 s, 8 q elements): PASS

pass 0: bicg_kernel1 gpu = 0.161 ms, bicg_kernel2 gpu = 0.159 ms
...
Total GPU time: 2.241 ms over 7 passes
```

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel with
its GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`),
then a summary whose `total_time_ms` is the sum of all recorded GPU event
times:
```json
{"type":"kernel","name":"bicg_kernel1","iter":0,"time_ms":0.161,"host_time_ms":0.180,"params":{"m":4096,"n":4096,"block_size":256}}
{"type":"kernel","name":"bicg_kernel2","iter":0,"time_ms":0.159,"host_time_ms":0.175,"params":{"m":4096,"n":4096,"block_size":256}}
{"type":"summary","total_time_ms":2.241}
```

## Performance Metric (derived downstream)

Memory bandwidth (GB/s), computed downstream from `time_ms` and params — not
reported by the binary:

```
GB/s = (2 × M × N × 4 bytes) / time_s / 1e9
```

Matrix A is read once by each kernel, totalling 2 reads.

## Correctness Verification

GPU results for `s` and `q` are compared against a CPU reference. The first 8 elements of each output vector are checked. Maximum relative error threshold: **1e-4**.

## References

- [PolyBench/GPU](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/)
- BiCGStab iterative solver: Barrett et al., *Templates for the Solution of Linear Systems*, SIAM 1994.
