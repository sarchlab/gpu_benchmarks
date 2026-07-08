# tier2/heteromark_aes — AES-256 ECB Encryption (HeteroMark)

Benchmarks AES-256 encryption in ECB mode on the GPU. Each thread encrypts one 16-byte block through 14 rounds. Emits raw per-iteration kernel times; throughput (GB/s) is derived downstream.

Derived from the [HeteroMark benchmark suite](https://github.com/NUCAR-DEV/Hetero-Mark) AES workload.

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

## Files

| File | Description |
|------|-------------|
| `heteromark_aes.hip` | HIP source — AES-256 ECB kernel, key expansion, verification |
| `heteromark_aes.metal` | Metal compute shader — AES-256 ECB kernel |
| `heteromark_aes_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./heteromark_aes                              # defaults: N=1048576
./heteromark_aes --size 65536                 # fewer blocks
BENCH_PARAM_size=65536 ./heteromark_aes       # env vars override argv
```

Environment variables: `BENCH_PARAM_size`, `BENCH_PARAM_block_size`,
`BENCH_PARAM_key_length`, plus `BENCH_PARAM_num_warmup` (warmup passes,
default 3) and `BENCH_PARAM_num_iters` (timed passes, each recorded,
default 7).

## Output

**stdout** — JSON lines, one kernel record per timed pass with the GPU event
time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary
whose `total_time_ms` is the sum of all recorded GPU event times (the
per-pass plaintext re-upload is untimed):
```
{"type":"kernel","name":"aes256_encrypt_kernel","iter":0,"time_ms":8.543200,"host_time_ms":8.601000,"params":{"size":1048576,"block_size":256,"key_length":"256"}}
...
{"type":"summary","total_time_ms":59.802400}
```

Derived metrics such as GB/s are computed downstream from `time_ms` and the
params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
AES-256 ECB  |  Blocks: 1048576  |  Data: 16.00 MB  |  Iterations: 3 warmup + 7 timed

aes256_encrypt_kernel iter 0: gpu = 8.543 ms, host = 8.601 ms
...
Total GPU time: 59.802 ms over 7 iterations
PASS
```

## Verification

- First 256 blocks are compared byte-by-byte against a CPU reference implementation
- Reports PASS/FAIL with mismatch details

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `heteromark_aes.hip`; S-box in `__constant__` memory
- **Metal**: compiled with `clang++` from `heteromark_aes_metal.mm`; the Metal shader (`heteromark_aes.metal`) is loaded and compiled at runtime from the same directory as the binary; S-box in `constant` address space
