# tier2/heteromark_aes — AES-256 ECB Encryption (HeteroMark)

Benchmarks AES-256 encryption in ECB mode on the GPU. Each thread encrypts one 16-byte block through 14 rounds. Emits the raw kernel time of a single launch; throughput (GB/s) is derived downstream.

## Provenance & fidelity

The AES-256 ECB algorithm follows HeteroMark at revision
[`aa685e02705f4be380cba215c6df10b42262a9cd`](https://github.com/NUCAR-DEV/Hetero-Mark/tree/aa685e02705f4be380cba215c6df10b42262a9cd):
the [CPU implementation](https://github.com/NUCAR-DEV/Hetero-Mark/blob/aa685e02705f4be380cba215c6df10b42262a9cd/src/aes/aes_benchmark.cc)
and [CUDA kernel](https://github.com/NUCAR-DEV/Hetero-Mark/blob/aa685e02705f4be380cba215c6df10b42262a9cd/src/aes/cuda/aes_cuda_benchmark.cu)
expand a 32-byte key and encrypt independent 16-byte blocks through 14 rounds,
omitting MixColumns in the final round. The local byte-based key schedule and
MixColumns expression preserve those operations; one GPU thread owns each block.

The local driver uses a fixed key (`key[i] = (i*0x13 + 0x37) mod 256`) and
plaintext (`data[i] = i mod 256`). Upstream reads the key and plaintext from hex
files, so inputs are different. The local default launch has 256 threads per
block and guards partial blocks; upstream uses 64. The local driver times one
encryption launch with the suite's entry protocol. These are documented input,
launch and measurement adaptations, not source or input identity.

The regression checks both the existing CPU cipher and actual GPU ciphertext
against the four published [NIST SP 800-38A, F.1.5 AES-256 ECB
vectors](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38a.pdf).

## Algorithm

- **N** 16-byte plaintext blocks (default: N=1,048,576 = 1M blocks = 16 MB)
- AES-256 ECB encryption (14 rounds per block):
  1. **SubBytes** — S-box substitution for each byte
  2. **ShiftRows** — cyclic row shifts in the 4×4 state matrix
  3. **MixColumns** — column mixing via GF(2⁸) multiplication (rounds 1–13 only)
  4. **AddRoundKey** — XOR state with round key
- 256-bit key → 15 round keys (240 bytes) via key expansion
- Each GPU thread processes one independent block (embarrassingly parallel)
- GB/s = N × 16 / time\_sec / 10⁹ (derived downstream from the emitted times and params)

## Measurement

The encryption kernel launches **exactly once** per process execution (see the
root README "Measurement Methodology"). With `BENCH_PARAM_vector_add_warmup`
(default `true`), a vector-add warmer boosts the GPU clock and evicts the
last-level cache immediately before the timed launch (the plaintext upload is
untimed). Repetition = `gpubench run --reps N`. The workload is fixed AES-256
(14 rounds).

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_size` / `--size` | Number of 16-byte blocks to encrypt | 1048576 |
| `BENCH_PARAM_block_size` / `--block_size` | Threads per block | 256 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `heteromark_aes.hip` | HIP source — AES-256 ECB kernel, key expansion, verification |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./heteromark_aes                              # defaults: N=1048576
./heteromark_aes --size 65536                 # fewer blocks
BENCH_PARAM_size=65536 ./heteromark_aes       # env vars override argv
```

## Output

**stdout** — JSON lines: one `kernel` record (`iter:0`) with the GPU event
time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary
whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"aes256_encrypt_kernel","iter":0,"time_ms":8.543200,"host_time_ms":8.601000,"params":{"size":1048576,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":8.543200}
```

Derived metrics such as GB/s are computed downstream from `time_ms` and the
params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
AES-256 ECB  |  Blocks: 1048576  |  Data: 16.00 MB

aes256_encrypt_kernel: gpu = 8.543 ms, host = 8.601 ms
Total GPU time: 8.543 ms
AES verification: PASS (1048576 blocks, 16777216 bytes, 0 byte mismatches)
```

## Verification

Every byte of the actual measured ciphertext must exactly match CPU AES-256
encryption of the pristine input with the same expanded key. Verification uses
the existing CPU cipher and a 16-byte temporary reference block, with no second
GPU encryption. Download, CPU calculation and comparison are untimed.

The `full_cpu_v1` descriptor identifies this complete comparison. The diagnostic
reports the total block/byte count and all byte mismatches, showing at most ten
examples. Any mismatch produces a nonzero exit. The NIST fixtures are confined
to separate test binaries; ordinary measurements retain the synthetic input/key.

From the repository root, using new output directories:

```bash
python3 tests/aes_verification.py --host-only --work-dir /tmp/aes-host
python3 tests/aes_verification.py --platform cuda --work-dir /tmp/aes-gpu --sweep
```

The GPU regression checks known answers, partial blocks, actual ciphertext
corruption at the old cutoff and final byte, every byte position within a block,
complete mismatch counts and both warmer settings. `--sweep` also runs all 27
declared configurations with the production binary and warmer enabled, requiring
complete verification within 60 seconds per process.

On the NVIDIA H100 PCIe with CUDA on 2026-09-28, all 61 production/fault
cases passed, including all 27 declared configurations. The largest case
(8,388,608 blocks, 128 MiB) completed in 2.538 seconds including full verification.
Four Compute Sanitizer tools and six actual CLI/SQLite success/failure checks
passed, and the complete compiled AES kernel SASS matches the baseline. These
shared-GPU checks establish correctness and observed process budget, not
publication performance or AMD qualification.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `heteromark_aes.hip`; S-box in `__constant__` memory
