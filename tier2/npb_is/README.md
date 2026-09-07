# tier2/npb_is — Integer Sort (NAS IS)

Faithful GPU port of the **NAS Parallel Benchmarks IS** (Integer Sort) kernel:
ten `rank()` iterations over near-gaussian keys, computing the rank of every
key with a full-key-range counting sort. The workload runs exactly once per
execution and emits raw per-kernel times accumulated across the ten
iterations; keys-ranked/sec is derived downstream.

## Provenance & fidelity

Ported from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html)
IS benchmark; reference source is the NPB 2.3/3.0 serial C `is.c`
(M. Yarrow, OpenMP C version S. Satoh), non-bucket variant. The upstream
algorithm is preserved exactly:

- **Key model**: keys generated with NPB's 46-bit `randlc` LCG
  (seed 314159265, multiplier 1220703125); each key is `MAX_KEY/4` times the
  sum of four uniform draws — the benchmark's defining near-gaussian
  distribution.
- **Iteration structure**: 10 timed `rank(iteration)` calls; each first sets
  `key_array[iteration] = iteration` and
  `key_array[iteration+10] = MAX_KEY - iteration`, then re-ranks all keys.
- **Ranking**: clear the `MAX_KEY`-wide work array, histogram every key over
  the full `[0, MAX_KEY)` range (copying the keys into `key_buff2` in the
  same pass, as upstream does), then an inclusive prefix sum yields each
  key's rank.
- **Verification**: per-iteration partial verification of five test-key
  ranks, and an untimed `full_verify` (rank-directed permutation +
  sortedness check) after the timed loop.

GPU-mapping departures (re-parallelizations of the same algorithm, permitted
under methodology Rule 1):

- The `MAX_KEY`-wide inclusive scan is decomposed into three kernels
  (per-block Blelloch scan → block-sum scan → offset add).
- The per-iteration two-key perturbation is a one-thread kernel.
- Partial-verify expected ranks are precomputed on the host before the timed
  loop. The measured scan-add kernel captures five ranks per iteration;
  comparison runs on the host afterward. Capture runs in both verification
  modes, so they execute identical GPU work with no extra launches.
- The upstream free/warmup `rank(1)` is omitted under the repository's
  one-execution measurement policy. Its key perturbations are idempotent
  with the first measured `rank(1)`, so this preserves the ranked inputs.
- Upstream's optional `USE_BUCKETS` cache optimization is not used (matching
  the reference's default).

## Problem sizes

`size` (the number of keys, N) is the scaling parameter. The NPB class sizes
map exactly to published configurations:

| Class | size (N) | MAX_KEY |
|-------|----------|---------|
| S | 65,536 = 2¹⁶ | 2¹¹ |
| W | 1,048,576 = 2²⁰ | 2¹⁶ |
| A | 8,388,608 = 2²³ (default) | 2¹⁹ |
| B | 33,554,432 = 2²⁵ | 2²¹ |
| C | 134,217,728 = 2²⁷ | 2²³ |

At class sizes the published partial-verification tables are checked in
addition to the host mirror. Non-class sizes use `MAX_KEY = N/16` (the
W/A/B/C ratio) with test positions derived from the Class A indices modulo
N; expected ranks then come from the host mirror alone.

The minimum accepted size is **176**: `MAX_KEY = N/16` must exceed 10,
the largest key value inserted by the ten ranking iterations. Smaller sizes
fail before GPU setup with an error record and no timing records.

## Files

| File | Description |
|------|-------------|
| `npb_is.hip` | HIP source — rank kernels + host mirror/verification |
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
./npb_is                              # default: Class A (N=8388608)
./npb_is --size 1048576               # Class W
BENCH_PARAM_size=1048576 ./npb_is     # env vars override argv
```

Environment variables: `BENCH_PARAM_size` (default 8388608),
`BENCH_PARAM_block_size` (default 256; must be a power of two ≥ 32), and
`BENCH_PARAM_vector_add_warmup` (default true — cold-cache calibration:
boosts the clock and evicts the LLC once after setup and before the timed
loop). When disabled, there is no GPU warmup or primer.

`BENCH_PARAM_verify=false` (also `0`/`skip`) skips the CPU reference and
result comparisons — an undeclared debug knob, not a sweep parameter in
`params.json`. Snapshot indices are prepared and the scan-add kernel captures
the same ranks in both modes; capture cost is included in its event time.
Every run has exactly 60 measured algorithm launches: six kernels per
iteration. Trace collection disables the shared warmer and therefore sees
exactly those launches, with verification either on or off.

## Output

**stdout** — JSON lines: one kernel record per rank-phase kernel
(`modify_keys_kernel`, `clear_kernel`, `histogram_kernel`,
`scan_blocks_kernel`, `scan_sums_kernel`, `scan_add_kernel`), each carrying
that kernel's GPU event time (`time_ms`) and host time (`host_time_ms`)
**accumulated across the 10 timed rank iterations**, then a summary whose
`total_time_ms` is the sum of the six accumulated GPU event times:
```
{"type":"kernel","name":"modify_keys_kernel","iter":0,"time_ms":0.021,"host_time_ms":0.19,"params":{"size":8388608,"block_size":256}}
{"type":"kernel","name":"clear_kernel","iter":0,"time_ms":0.11,"host_time_ms":0.24,"params":{...}}
{"type":"kernel","name":"histogram_kernel","iter":0,"time_ms":14.9,"host_time_ms":15.1,"params":{...}}
{"type":"kernel","name":"scan_blocks_kernel","iter":0,"time_ms":0.33,"host_time_ms":0.45,"params":{...}}
{"type":"kernel","name":"scan_sums_kernel","iter":0,"time_ms":0.09,"host_time_ms":0.21,"params":{...}}
{"type":"kernel","name":"scan_add_kernel","iter":0,"time_ms":0.18,"host_time_ms":0.30,"params":{...}}
{"type":"summary","total_time_ms":15.63}
```

Derived metrics such as Mkeys-ranked/sec (upstream reports
`MAX_ITERATIONS * N / time`) are computed downstream from `time_ms` and the
params.

**stderr** — human-readable results:
```
Device: NVIDIA H100 (id=0)
NAS IS: 8388608 keys (2^23), MAX_KEY=524288, 10 iterations  [Class A]

modify gpu = 0.021 ms, clear gpu = 0.110 ms, histogram gpu = 14.900 ms,
scan_blocks gpu = 0.330 ms, scan_sums gpu = 0.090 ms, scan_add gpu = 0.180 ms
Total GPU time: 15.631 ms
Verification: 50/50 partial + full_verify ok + rank array ok => PASS
```

## Verification

- **Partial verify** (five test keys × 10 iterations): expected ranks come
  from a bit-exact host mirror of the upstream serial algorithm, computed
  before the timed loop; the device ranks are captured by the measured
  scan-add kernel and compared after the loop. Each sample is read by its
  owning tile after a block barrier makes that tile's updates visible;
  there are no cross-block reads or extra capture launches. At class sizes
  the mirror is additionally cross-checked against the published NPB `test_rank_array`
  tables (with the upstream per-class iteration shifts). Because the
  expensive host mirror and all readbacks sit outside the timed section, the
  measured kernels are never separated by a host pause that could let the GPU
  clock droop.
- **full_verify** (untimed, after the timed loop): the device's rank array
  permutes the device's key copy exactly as upstream
  (`key_array[--rank[key]] = key`); the result must be fully sorted. The
  device rank array is also compared bit-for-bit against the host mirror.
- Upstream's acceptance criterion (5 × 10 partial verifications + full
  verify all passing) must hold; on failure the binary exits non-zero.

Run behavioral regressions after building:

```bash
make test PLATFORM=cuda                    # GPU result/protocol and bounds checks
python3 test_npb_is.py --sweep              # include all 27 declared sizes
python3 test_npb_is.py --trace              # NVIDIA nsys: assert exact launch sequence
python3 test_npb_is.py --invalid-only       # early rejection; no GPU needed
```

The trace check covers verification on/off, the minimum size, and a
multi-chunk block-sum scan. It exports Nsight Systems CUDA activity to
SQLite and checks the actual sequence: each of the six kernels exactly
ten times, with no warmup or verification-only kernels.
`make test PLATFORM=rocm` uses the same result checks; `--trace` requires
NVIDIA Nsight Systems. Python 3 is required for these tests.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `npb_is.hip`; six kernel
  launches per rank iteration, 10 timed iterations per run.
