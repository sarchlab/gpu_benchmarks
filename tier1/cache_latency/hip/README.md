# cache_latency — dependent pointer-chase latency

A single GPU thread follows a random cycle of `uint32_t` indices. Each load
supplies the next address, exposing dependent-load latency. One working-set size
is measured per execution. Derive average ns/access downstream as
`time_ms * 1e6 / num_accesses`; the recorded event interval also includes loop,
address-calculation, dispatch and final-store costs.

## Primed entry policy

This benchmark is an explicit exception to the suite's default cold-cache entry:

1. Run the optional vector-add clock/cache warmer.
2. Run an untimed pointer-chase primer and wait for it to finish.
3. Time one pointer-chase launch with GPU events. No warmer runs between the
   primer and this launch.

Setting `BENCH_PARAM_vector_add_warmup=false` disables only step 1. The primer
still runs. Each repetition is a fresh process; the measured chase executes once.
Both the warmer and primer are outside the GPU and host timing brackets.

The chain has `M = floor(array_bytes / cacheline_bytes)` nodes for the declared,
word-aligned spacing. The primer traverses `min(M, 256 MiB / cacheline_bytes)`
nodes. This is one full cycle within the fixed cap and a prefix for larger
chains. `warmup_accesses` records that actual count; the cap bounds setup cost
and is not a discovered cache size. The timed count is
`clamp(measure_laps * M, 131072, 16000000)`, unless `num_accesses` overrides it.

Priming targets resident-cache measurements, but **primed does not guarantee a
particular hit level or hit rate**. Cache capacity, replacement, compiled load
policy, and scheduling determine the result. A separate primer does not guarantee
that the timed block runs on the same SM/CU or retains its L1 state. CUDA permits
blocks to be assigned among available SMs, and AMD's vector L1 is local to a CU.
See the [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
and [AMD compute-unit description](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/cdna/compute-unit.html).

Consequently, a timed launch can still pay an initial L1 fill; larger chains can
have capacity misses, and a capped primer can leave nodes untouched. Do not call
these results "pure L1 hits" or "all-miss DRAM latency" based on size alone.
Qualify the generated loads and cache behavior on each target before assigning
cache-level labels. CUDA evidence does not establish AMD behavior (R-19).

## Chain and workload identity

The host shuffles node ordinals with `std::mt19937` and `std::shuffle`, then links
them into one cycle. Node slots are separated by `cacheline_bytes / 4` words.
The historical parameter name `cacheline_bytes` denotes configured node spacing;
it does not discover the physical line/sector size. Distinct nodes can share a
physical line on a target with larger lines. The declared sweep uses 64-byte
spacing and aligned array sizes. Portable cross-toolchain input identity remains
the owner-deferred R-27 work.

Output retains the requested size, spacing, seed and lap count, plus effective
`num_accesses` and `warmup_accesses`. `cache_entry="primed"` and
`primer_launches=1` identify the entry protocol. These descriptors do not certify
hardware residency. Older results collected with an intervening warmer represent
a different entry policy and should be recollected rather than pooled with these.

## Timing and trace replay

Stdout contains one `pointer_chase_kernel` record for the timed launch and one
summary; the primer has no timing record. The summary equals the timed record's
GPU time. `host_time_ms` spans event submission, launch, checking and synchronization.

Trace collection currently forces the clock warmer off. It must retain both
pointer-chase launches, in order: first the primer, then the timed chase. A
simulator must replay the primer to establish cache state and exclude its work
from the reported latency; replaying only the final chase from a cold cache
changes the experiment. Do not reset caches between the two launches, discard
the first launch as a generic warmup, or include its accesses in `num_accesses`.
The raw capture has two launches even though timing output has one kernel record.
General trace-region identification and artifact/replay qualification remain
R-23/R-24; this benchmark's ordering alone does not qualify a simulator.

## Build and run

```sh
make PLATFORM=cuda       # nvcc; use PLATFORM=rocm for hipcc
./cache_latency         # default: 2 MiB
./cache_latency 16384    # positional array_bytes
BENCH_PARAM_array_bytes=16384 BENCH_PARAM_vector_add_warmup=false ./cache_latency
```

Environment values take priority over the positional size:

| Parameter | Default | Meaning |
|---|---:|---|
| `BENCH_PARAM_array_bytes` | 2097152 | Array size; 26 declared sizes from 4 KiB to 4 GiB |
| `BENCH_PARAM_cacheline_bytes` | 64 | Configured spacing between nodes in bytes |
| `BENCH_PARAM_measure_laps` | 4 | Requested timed laps, subject to access-count bounds |
| `BENCH_PARAM_num_accesses` | computed | Explicit timed-access override |
| `BENCH_PARAM_rng_seed` | 42 | Chain permutation seed |
| `BENCH_PARAM_vector_add_warmup` | true | Clock/cache warmer before the retained primer |

Array size and node spacing must be multiples of a `uint32_t` word (4 bytes),
with at least two nodes. The node count and last node's word index must fit
`uint32_t`; invalid layouts fail before allocation or device access. An aligned
array may retain an unused tail smaller than the node spacing. These checks
establish representable layout, not available host/device memory.

`num_accesses` is a positive, unswept direct-binary override. The inherited
environment can also apply it to harness runs. It replaces the computed timed
count, is recorded in every kernel result, and leaves the untimed primer count
unchanged. Zero effective accesses are rejected. Compare or derive latency using
the recorded count, including when an override differs from the requested laps.

Illustrative JSON for a 16 KiB chain (times are not performance claims):

```json
{"type":"kernel","name":"pointer_chase_kernel","iter":0,"time_ms":3.000000,"host_time_ms":3.100000,"params":{"array_bytes":16384,"num_accesses":131072,"warmup_accesses":256,"measure_laps":4,"cacheline_bytes":64,"rng_seed":42,"cache_entry":"primed","primer_launches":1}}
{"type":"summary","total_time_ms":3.000000}
```

On an NVIDIA GPU with Nsight Systems and Python installed, check actual launch
and event ordering using a new output directory:

```sh
python3 ../../../tests/cache_latency_order.py --binary "$PWD/cache_latency" --work-dir /tmp/cache-entry-check
```

The check covers warmer on/off at 4 KiB and just above the 256 MiB primer cap.
It retains profiler databases and checks the real CUDA timeline, not source text.
It is a functional protocol check; timings under profiling are not performance
measurements. Equivalent ROCm runtime/cache qualification remains R-19.
