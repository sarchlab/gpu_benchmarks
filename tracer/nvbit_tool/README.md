# nvbit_tool/ — gpubench live-sampling NVBit tracer (P1b)

Fork of the Accel-Sim NVBit tracer from
[bb_triosim](https://github.com/sarchlab/bb_triosim) (`tracer/tracer_tool/`),
extended with **live block sampling**: the injected device function early-exits
for thread blocks outside the sample plan, so unsampled blocks skip the
address-gather (warp shuffles) and channel push entirely. This turns the
sampled-`.ptrace` instrumented-block fraction (P3, `tracer/ptrace/sample.go`)
from a storage ratio into an actual collection wall-time ratio.

bb_triosim's tracer is unchanged and remains the byte-faithful reference for
`verify` ground truth.

## Build

Requires an NVBit release (1.7.x) and CUDA ≥ 12. Either run
`./install_nvbit.sh` to fetch NVBit into `../nvbit_release`, or point at an
existing copy:

```bash
make NVBIT_PATH=/path/to/bb_triosim/tracer/nvbit_release/core ARCH=sm_90
```

## Usage

Same as the upstream tracer (`LD_PRELOAD=.../nvbit_tool.so <app>`, output
controlled by `TRACES_FOLDER`, kernel selection by `DYNAMIC_KERNEL_RANGE`),
plus:

| Env var | Meaning |
|---|---|
| `PTRACE_SAMPLE` (unset) | Trace every block — identical behavior to upstream. |
| `PTRACE_SAMPLE=corner` | Record only blocks with `ctaid.{x,y,z} < 2` (≤ 8 blocks) — matches the Go-side `cornerSample()`, which fits affine address recipes from exactly these blocks. |
| `PTRACE_SAMPLE=corner:K` | Corner width K (2–255). Use `corner:3` for kernels with a boundary block at the origin (e.g. FIR head-of-stream) so the fitter still sees two regular blocks per axis; `ptrace gen` records the boundary block verbatim as an exception. |
| `PTRACE_SAMPLE=x,y,z;x,y,z` | Record an explicit block list (max 64; combinable: `corner;4,0,0`). Use to re-collect blocks that failed recipe verification. |
| `PTRACE_SAMPLE=count` | Counting-only profile: record nothing, count every warp instruction. `stats.csv` carries per-kernel totals — the measured side of the generation gate (`scripts/collect_counts.py` uses this when ncu counters are admin-restricted). |
| `PTRACE_SAMPLE_COUNT_SKIPPED=1` | Also count skipped blocks' warp instructions (one atomic per warp-inst — slower; off by default, so stats.csv undercounts under sampling). |
| `PTRACE_DEDUP=K` | Trace only the first K launches per (kernel name, grid, block, shmem) signature; later relaunches run uninstrumented and their `kernelslist` entry references the first instance's trace, so the replay sequence stays complete. Iterative benchmarks (jacobi2d: 200 launches → 2 traces) lose their dominant collection multiplier. Caveat: ping-pong buffer pointers are folded onto the first instance (strides/footprint unaffected). |

## NVBit call-arg constraint (read before touching the signature)

NVBit 1.7.6 marshals injected-call args into consecutive 32-bit register
units; a 64-bit arg at an **odd** unit offset arrives corrupted on the device
(dereferencing such a pointer → `cudaErrorIllegalAddress`). Every 64-bit
parameter of `instrument_inst` must sit at an even 32-bit offset from the
first parameter — that is why `psample_list` precedes the 32-bit
`sample_mode`, and why the three upstream counter/flag pointers are
consolidated behind one `tool_ctrl_t*`. Also: do not touch `__managed__`
memory from `nvbit_at_init` (no CUDA context → breaks NVBit's interception)
or from context-init callbacks; publish device-visible state at the first
kernel launch, after `cudaDeviceSynchronize`, as done for the sample list.

## Downstream workflow

```bash
# Collect (suite-wide, via gpubench):
./gpubench run --collect-trace tracer/nvbit_tool/nvbit_tool.so \
  --trace-sample corner --trace-sizes 2 --reps 1 --continue-on-error

# Regenerate full-grid traces from the sampled containers:
ptrace gen <sampled-traces-dir> <out-dir>      # exceptions + majority-fit + synth

# Gate the (locally unverified) generation:
ptrace diff <full-traces-dir> <gen-dir>        # same-size structural check
ptrace predict <gen-dir> pred.json
scripts/collect_counts.py -o meas.json -- <benchmark cmd>
ptrace countcheck pred.json meas.json          # counts fingerprint, 5% default
```

The `.trace.zst` container format is unchanged; a sampled run simply contains
fewer blocks (the writer reports `sampled N/M blocks`). Downstream
`ptrace convert` / recipe fitting consume it as-is.

The early-exit guard is block-uniform, so warps exit together without
divergence; the kernel itself still executes all blocks at native speed
(memory side-effects for subsequent kernels are preserved). The residual
overhead on unsampled blocks is the bare injected-call prologue
(ctaid read + compare).

## Measured results (H100 PCIe, CUDA 12.9, polybench_gemm, TILE=16)

| Config | Kernel time | Wall | Max RSS | Recorded blocks |
|---|---:|---:|---:|---:|
| N=256 native | 0.088 ms | 3.5 s | 0.1 GB | — |
| N=256 full trace | 4,150 ms | 7.8 s | 0.65 GB | 256/256 |
| N=256 `corner` | 337 ms | 3.9 s | 0.12 GB | 4/256 |
| N=1024 native | 4.3 ms | 0.4 s | 0.1 GB | — |
| N=1024 full trace | 253,603 ms | 258 s | 31.7 GB | 4096/4096 |
| N=1024 `corner` | 1,674 ms | 2.5 s | 0.16 GB | 4/4096 |

Sampling speedup grows with grid size (the recorded set is constant): 12× at
256 blocks, **151× kernel / 104× wall** at 4096 blocks, with **200× less
RAM** (the full-trace in-memory block buffering is what makes large kernels
untraceable; sampling removes it).

Verification: every sampled block compared token-for-token against the
same-coordinate block of a full-trace run — 0 structural diffs at both sizes;
all differing tokens are memory addresses explained by a single run-specific
allocation-base delta (cudaMalloc layout shift between processes), the same
class P6 scaling validation accepts.
