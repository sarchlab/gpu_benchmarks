# atomic_operations — fixed-attempt atomic microbenchmark (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

One timed kernel has all threads contend on one unsigned 32-bit counter,
initially zero. Each thread makes exactly `ops_per_thread` attempts. Repetition
belongs to `gpubench --reps`; there is no retry-until-success loop.

| Mode | Work per attempt | Result verification |
| --- | --- | --- |
| `atomicAdd` | `atomicAdd(counter, 1)` | Exact total attempts modulo 2^32 |
| `atomicCAS` | Atomic read via `atomicAdd(counter, 0)`, then one `atomicCAS(old, old+1)` | Every thread's success count is downloaded; their full sum modulo 2^32 must equal the actual counter |
| `atomicMin` | `atomicMin(counter, tid+i)` with unsigned arithmetic | Counter remains exactly zero |

CAS succeeds according to thread interleaving. The return value identifies a
successful update; each thread accumulates these outcomes in a register and
writes one count to its own array slot inside the measured kernel. Verification
checks every count is in `[0, ops_per_thread]` and the full success total is in
`[ops_per_thread, attempted_operations]`. This additional bookkeeping is part of
the measured CAS workload, including its final per-thread store. It adds no
atomic operation beyond the value-preserving atomic read and the CAS itself.
Unsigned counter wrap is intentional, including when the sweep exceeds 2^32
attempts. The atomic read replaces the former racy ordinary load.

The minimum mode measures an unchanged-value workload. A zero candidate can
compare equal; this does **not** prove which physical compare/write path the GPU
uses. The result is neither successful-update throughput nor a claim of peak
hardware atomic throughput. Likewise, CAS attempt throughput is not CAS success
throughput, and includes the cost of the atomic read and success bookkeeping.

## Build and run

```
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./atomic_operations        # 1024 attempts/thread, 2048 blocks, 256 threads/block
BENCH_PARAM_atomic_op=atomicCAS ./atomic_operations 4096
```

`BENCH_PARAM_ops_per_thread` overrides the optional positional argument; when
set, the positional value is ignored and is not validated.
`BENCH_PARAM_num_blocks`, `BENCH_PARAM_threads_per_block`, and
`BENCH_PARAM_atomic_op` override their defaults. Unknown modes,
malformed/nonpositive effective integers, more than one positional argument,
unsafe thread-index products, and unsupported device geometry fail before
measurement. Integer parsing does not silently truncate or fall back.

Uploads and allocations precede the entry warmer. By default
`BENCH_PARAM_vector_add_warmup=true` runs the shared vector-add warmer immediately
before the selected kernel. This targets cold cache at a raised clock; achieving
that state requires hardware qualification. `false` disables the warmer.
All downloads and host verification follow timing. Verification or allocation
failure returns nonzero, even if timing records have already been emitted.

## Records and interpretation

Stdout contains one kernel record and a summary with the same GPU event time.
Host wall-clock time is also retained. Stderr reports the verified actual counter,
number of CAS counts checked, and successful CAS total. JSON params include:

- Requested controls and effective `num_threads`.
- `attempted_operations = num_threads * ops_per_thread`.
- `atomic_read_operations`: the same count for CAS, zero otherwise.
- `total_atomic_operations`: twice the attempt count for CAS, once otherwise.
- `success_count_stores`: one per thread for CAS, zero otherwise.
- `counter_bits=32`, `initial_counter=0`, `workload=fixed_attempts_v1` and
  `verification=full_counter_and_cas_counts_v1`.

Counts describe source-level operations, not an assumed number of GPU
instructions. Derive attempt or total-operation throughput downstream from the
appropriate count and raw time; no derived throughput is emitted. CAS results
can differ between processes while attempted work remains fixed.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. `tests/atomic_host.py` checks integer admission, count bounds and modulo behavior.
`tests/atomic_measured_result.py` qualifies every declared H100 configuration,
checks emitted work counts, and corrupts actual device results to prove rejection.
