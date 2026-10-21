# Rodinia backprop

One complete hybrid training epoch of an input → 16 → 1 neural network. The two
original GPU kernels are connected by the CPU forward/error stages that produce
the hidden deltas used by the weight update.

## Pinned implementation

The reference is Rodinia 3.1, mirrored at
[`HPC-FAIR/rodinia_3.1`, commit `366b283`](https://github.com/HPC-FAIR/rodinia_3.1/tree/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/backprop):

- `backprop_cuda.cu`: the hybrid training driver and partial-sum reduction.
- `backprop_cuda_kernel.cu`: the two GPU kernels and their 16×16 tiling.
- `backprop.c`/`backprop.h`: sigmoid, output/hidden errors, weight updates and constants.
- `facetrain.c`/`imagenet.c`: seed, dimensions, target and input initialization.

The input→hidden and hidden→output weights are initialized in that order with
`rand()` after `srand(7)`, then the non-bias input units are drawn. Momentum starts
at zero, the output target is 0.1, and the bias activation is 1. The unused weight
column zero is retained to preserve the reference's RNG draw order. Inputs and
weights are floats; `ETA` and `MOMENTUM` retain the upstream **double literals**
`0.3`, including their arithmetic promotions in the update kernels. They are
fixed constants, not sweep parameters.

This is a portable single-file implementation of the pinned algorithm. It does
not promise a portable `rand()` sequence across host libraries (R-27), identical
rounding across compilers/devices, or completed AMD qualification (R-19). The
suite's default input size is 1,000,000; the upstream example size 65,536 is also
in the declared sweep.

## Complete epoch and timing

1. Upload inputs, weights and initial momentum, then run the optional entry warmer.
2. Time `bpnn_layerforward_CUDA`: each 16×16 block computes 16 input rows' partial sums.
3. Download the actual partial sums; on the CPU, sum them, add input bias weights,
   and apply sigmoid to obtain hidden activations.
4. On the CPU, compute the output activation, output delta and hidden deltas.
   Hidden deltas use the output weights **before** their update. Update the
   output-layer weights and momentum on the CPU, as in Rodinia.
5. Upload the derived hidden deltas and restore the input weights clobbered by
   the original forward kernel.
6. Time `bpnn_adjust_weights_cuda`, including bias and momentum updates.
7. Download the actual updated input weights and momentum outside timing.

There is one entry warmer and no reconditioning before the dependent update.
The intervening CPU stages and transfers can change cache residency and clocks.
Their time is excluded from the sum of the two GPU event times, which is **GPU
kernel time**, not complete epoch wall time. Repetition means a fresh process
(`gpubench run --reps N`), not another epoch inside the binary.

The positive upstream inputs often saturate hidden activations at large sizes,
so derived hidden deltas can legitimately be zero. The update kernel still
executes its original work. Qualification also uses non-saturating signed inputs
and nonzero momentum to expose broken dependencies that saturated cases conceal.

Earlier versions synthesized hidden deltas, used a different initialization
order and float learning-rate literals. **Recollect their performance results.**

## Parameters and build

| Parameter | Control | Default | Meaning |
|---|---|---|---|
| `input` | `BENCH_PARAM_input` or `--input N` | 1000000 | Positive input count; rounded down to a multiple of 16, with minimum 16 |
| `vector_add_warmup` | `BENCH_PARAM_vector_add_warmup` | true | Optional entry warmer; false runs as-is |

The environment overrides the command line. Hidden/output sizes are fixed at
16/1. Launch geometry is `grid(1, input/16)`, `threads(16,16)`. Requests exceeding
the device's grid limit or the kernel's signed weight-index range fail before
allocation and timing. The declared 12-size sweep is unchanged.

```sh
make PLATFORM=cuda             # or PLATFORM=rocm
./rodinia_backprop --input 65536
make test PLATFORM=cuda        # requires a GPU
```

Stdout contains one record per GPU kernel and a summary:

```text
{"type":"kernel","name":"bpnn_layerforward_CUDA","iter":0,"time_ms":0.1,"host_time_ms":0.2,"params":{"input":65536,"hidden":16,"height":16,"width":16,"verification":"full_epoch_cpu_v1"}}
{"type":"kernel","name":"bpnn_adjust_weights_cuda","iter":0,"time_ms":0.2,"host_time_ms":0.3,"params":{"input":65536,"hidden":16,"height":16,"width":16,"verification":"full_epoch_cpu_v1"}}
{"type":"summary","total_time_ms":0.3}
```

Times above are illustrative. Stderr reports the device, raw timing summary and
verification outcome.

## Qualification

[`backprop_output_test.hip`](../../../tests/backprop_output_test.hip) calls the same
complete production epoch as `main`. It checks every updated input/output
weight and momentum value, all activations and deltas, and unchanged inputs
against independent dense double-precision calculations. The tolerance is
`max(2e-5, 2e-5 * abs(reference))` for full weights and activations. Deltas,
momentum and actual weight changes use the tighter absolute floor `1e-7` with
the same relative tolerance, so the magnitude of an original weight cannot hide
an incorrect small update. This allows float reduction/activation rounding;
non-finite values fail. Eighteen cases include non-saturating signed inputs,
nonzero momentum, multiple tiles, the largest declared size and pinned setup.
Use `bench_output_test --upstream N` for a specific declared size. CI compiles
this test; running it requires a GPU.

Ordinary timing runs now independently verify the actual measured epoch after
both kernel timings have finished. An initial host-network snapshot is taken
before the entry warmer. The already-downloaded forward partials are retained
and checked against independent double dot products over each 16-row tile;
final activations, deltas, weights, momentum and weight changes are checked
against an independent full CPU epoch. Unused weight/momentum slots and host
inputs must remain unchanged. No extra GPU transfer, kernel or verification
work is inserted between the two training stages.

Checking the partials matters: the standard positive inputs saturate hidden
activations, so a wrong forward sum can leave final weights unchanged. Each
partial uses tolerance `1e-6 * max(1, sum(abs(input * initial_weight)))`; final
state uses the tolerances described above, including the tight check on actual
weight changes. Non-finite values fail with a diagnostic and nonzero exit.
The fixed `verification=full_epoch_cpu_v1` tag is stored in both kernel records;
there is no verification-disable control. Failed verification can follow valid
timing records; the harness marks the complete run as an error.

Verification requires O(input × 16) CPU work. The snapshot copies the input and
two full weight/momentum arrays, about 140 MiB at the largest declared size;
retaining the partials keeps about another 4 MiB live after the epoch. These
costs are outside GPU and host launch timing, but the complete process must fit
the target's host memory and collection timeout. Target qualification must
additionally check the pinned original implementation and actual GPU timelines. With the warmer disabled, a raw run
has two GPU kernels, a partial-sum download and two uploads between them, and
result downloads afterward. General trace-region/artifact qualification remains
R-23/R-24. AMD execution and performance collection remain separate from these
functional checks.
