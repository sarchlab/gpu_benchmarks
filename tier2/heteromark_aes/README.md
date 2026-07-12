# tier2/heteromark_aes — AES-256 ECB Encryption (HeteroMark)

Benchmarks AES-256 encryption in ECB mode on the GPU. Each thread encrypts one 16-byte block through 14 rounds. Emits the raw kernel time of a single launch; throughput (GB/s) is derived downstream.

## Provenance & fidelity

Faithful reproduction of the [HeteroMark benchmark suite](https://github.com/NUCAR-DEV/Hetero-Mark) AES workload: the cipher is FIPS-197 verified (S-box, rcon, and round structure).

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
{"type":"kernel","name":"aes256_encrypt_kernel","iter":0,"time_ms":8.543200,"host_time_ms":8.601000,"params":{"size":1048576,"block_size":256}}
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
PASS
```

## Verification

- First 256 blocks are compared byte-by-byte against a CPU reference implementation
- Reports PASS/FAIL with mismatch details; on FAIL the process exits non-zero

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `heteromark_aes.hip`; S-box in `__constant__` memory
