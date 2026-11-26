# MI300X collection checklist

What to do, besides running the benchmarks, when collecting MI300X kernel
timings that will be used as a calibration reference. Everything below was
observed on AMD Developer Cloud "AMD Instinct MI300X VF" instances with ROCm
10.2.0 and amdgpu 7.1.3 on 2026-11-26; check it again on another machine or
driver.

## Why the clock matters

The v1.1 reference was collected with the GPU core clock at about 1680 MHz.
The same instance type runs at 2100 MHz by default. Compute-bound kernels are
exactly 1.25 times slower in v1.1 than at 2100 MHz, and memory-bound kernels are
the same within a few percent. Nothing in the v1.1 data records this.

Left at its default, the clock also changes with load:

- about 130 MHz when idle, 2098-2105 MHz with one wavefront running;
- 1915-1952 MHz with the whole GPU busy, when power reaches its 750 W cap;
- after the suite's vector-add warmer, the next kernel is slower for roughly
  its first millisecond (a 1 ms kernel by 15-25%, a 6 ms kernel by about 3%).

Capping the clock below the point where power limits it removes these effects.
The last one was not checked again under a cap.

## Before the collection

1. Nothing else may use the GPU, and only one benchmark runs at a time.
2. Record what is being measured:

   ```bash
   git rev-parse HEAD
   ```

   ```bash
   amd-smi version
   ```

   ```bash
   rocprofv3 --version
   ```

   ```bash
   amd-smi metric --clock
   ```

3. Cap the core clock (needs root), then confirm that every `GFX_n` block of
   `amd-smi metric --clock` shows `MAX_CLK: 1680 MHz`:

   ```bash
   amd-smi set -L sclk max 1680
   ```

   Use 1680 MHz to stay comparable with v1.1. Setting a minimum
   (`amd-smi set -L sclk min 2100`) reports success but does not hold the clock
   up: it still drops to about 130 MHz when idle.

4. Log the clock and power for the whole collection. `freq1_input` is in Hz
   and `power1_input` in microwatts:

   ```bash
   while sleep 0.1; do echo "$(date +%s.%N),$(cat /sys/class/drm/card*/device/hwmon/hwmon*/freq1_input),$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_input)"; done > clock-monitor.csv &
   ```

5. Make sure no `BENCH_PARAM_*` variable is left in the shell. A benchmark reads
   every one of them, including parameters the sweep does not set, and a stray
   value changes the workload without changing the kernel name or launch shape:

   ```bash
   env | grep BENCH_PARAM_
   ```

## During the collection

- Time kernels on the GPU side with `rocprofv3 --kernel-trace -f csv`; a
  dispatch's duration is `End_Timestamp - Start_Timestamp` in the
  `*_kernel_trace.csv` it writes. The benchmark's own event time is about
  30 microseconds longer per launch on this machine.
- Not every dispatch in a trace is the measurement:
  - `bench_warmer_kernel` is the suite's clock warmer;
  - `__amd_rocclr_fillBufferUnAligned` is the runtime filling a buffer;
  - `cache_latency` launches `pointer_chase_kernel` twice with the same launch
    shape, an untimed primer and then the timed chase. Take the last one.
- Use a fresh process per repetition and keep the per-repetition dispatch rows,
  so that a selection rule can be corrected later without measuring again.
- Record whether the warmer was on (`BENCH_PARAM_vector_add_warmup`). The v1.1
  collection did not.

## After the collection

1. Stop the clock log, then lift the cap and put the performance level back,
   which `amd-smi set` leaves at `manual`:

   ```bash
   amd-smi set -L sclk max 2100
   ```

   ```bash
   echo auto > /sys/class/drm/card1/device/power_dpm_force_performance_level
   ```

   The card number can differ; it is the `card*` directory that has
   `device/hwmon`.

2. Check the clock log: while kernels ran, the clock should sit at the cap and
   never above it.
3. Compare a few compute-bound kernels with v1.1. At 1680 MHz these reproduce
   it to within 0.3%:

   | Benchmark | Configuration | v1.1 |
   |---|---|---|
   | `fp32_throughput` | 1 block, 256 threads, 1,048,576 FMAs | 8.107 ms |
   | `empty_kernel` | 2,097,152 blocks of 256 threads | 1.259 ms |

4. Keep the version record, the clock log and the cap value with the data.

## Counting cycles instead of time

`rocprofv3 --pmc` reads hardware counters on this instance type
(`rocprofv3 -L` lists them), which gives cycle counts that do not depend on
the clock. Collect them in runs separate from timing runs.

- `SQ_BUSY_CYCLES` divided by the loop count is the number of clock cycles per
  loop iteration of a single-wavefront kernel.
- `SQ_ACTIVE_INST_VALU`, `SQ_ACTIVE_INST_SCA`, `SQ_ACTIVE_INST_MISC` and
  `SQ_WAIT_ANY` are in units of four cycles.
- `GRBM_COUNT` is summed over the eight XCDs.

With these, every instruction of the `instruction_cost` kernels is active for
four cycles, and a taken loop branch adds twenty cycles of waiting.
