// parboil_histogram.hip — faithful port of Parboil histo:
// the four device kernels (histo_prescan_kernel, histo_intermediates_kernel,
// histo_main_kernel, histo_final_kernel) are the original, and the host driver
// reproduces the original main.cu geometry. Only the host driver is
// restructured for the per-iteration timing harness and JSON output.
//
// Provenance (kernels verbatim):
//   histo/src/cuda/histo_prescan.cu       (histo_prescan_kernel)
//   histo/src/cuda/histo_intermediates.cu (histo_intermediates_kernel)
//   histo/src/cuda/histo_main.cu          (histo_main_kernel + device helpers)
//   histo/src/cuda/histo_final.cu         (histo_final_kernel)
//   host geometry: histo/src/cuda/main.cu ; constants: histo/src/cuda/util.h
//
// Algorithm (the faithful part): a 4-kernel privatized pipeline building a
// histo_width x histo_height 2D histogram with 8-bit SATURATING bins.
//   1. prescan       — sample the input, estimate avg/stddev, derive the active
//                      histogram-block range [sm_range_min, sm_range_max].
//   2. intermediates — decompose each input value into its (block, hi, lo,
//                      offset) bin coordinate (uchar4 sm_mappings).
//   3. main          — each threadblock owns one 24KB histogram block; it keeps
//                      a per-block sub-histogram in shared memory with atomic
//                      8-bit saturating increments (overflow tracked globally).
//   4. final         — saturating merge of the BLOCK_X sub-histograms + the
//                      global/overflow buffers into the final 8-bit histogram.
//
// No textures in this kernel set. Vector types (uchar4/uint2/uint4/ushort4) and
// atomicMin/atomicMax/atomicAdd/min are portable across hipcc and nvcc -x cu.
//
// Environment variables (override argv):
//   BENCH_PARAM_img_width, BENCH_PARAM_img_height,
//   BENCH_PARAM_histo_width, BENCH_PARAM_histo_height
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line for EACH
// of the four kernels carrying that kernel's accumulated GPU event time, then a
// summary whose total_time_ms is the SUM of all recorded GPU event times.
// Output (stderr): human-readable results + verification PASS/FAIL

#include "../../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Constants — verbatim from Parboil histo/src/cuda/util.h
// ---------------------------------------------------------------------------
#define KB                      24
#define BINS_PER_BLOCK          (KB * 1024)
#define BLOCK_X                 14
#define PRESCAN_THREADS         512
#define PRESCAN_BLOCKS_X        64
#define THREADS                 768        /* KB == 24 */
#define UNROLL                  16

// ---------------------------------------------------------------------------
// Kernel 1: prescan — verbatim from histo_prescan.cu
// ---------------------------------------------------------------------------
__global__ void histo_prescan_kernel(unsigned int* input, int size, unsigned int* minmax)
{
    __shared__ float Avg[PRESCAN_THREADS];
    __shared__ float StdDev[PRESCAN_THREADS];

    int stride = size / gridDim.x;
    int addr = blockIdx.x * stride + threadIdx.x;
    int end = blockIdx.x * stride + stride / 8; // Only sample 1/8th of the input data

    float avg = 0.0;
    unsigned int count = 0;
    while (addr < end) {
        avg += input[addr];
        count++;
        addr += blockDim.x;
    }
    avg /= count;
    Avg[threadIdx.x] = avg;

    int addr2 = blockIdx.x * stride + threadIdx.x;
    float stddev = 0;
    while (addr2 < end) {
        stddev += (input[addr2] - avg) * (input[addr2] - avg);
        addr2 += blockDim.x;
    }
    stddev /= count;
    StdDev[threadIdx.x] = sqrtf(stddev);

#define SUM(stride__)\
if(threadIdx.x < stride__){\
    Avg[threadIdx.x] += Avg[threadIdx.x+stride__];\
    StdDev[threadIdx.x] += StdDev[threadIdx.x+stride__];\
}

#if (PRESCAN_THREADS >= 32)
    for (int stride = PRESCAN_THREADS / 2; stride >= 32; stride = stride >> 1) {
        __syncthreads();
        SUM(stride);
    }
#endif
#if (PRESCAN_THREADS >= 16)
    SUM(16);
#endif
#if (PRESCAN_THREADS >= 8)
    SUM(8);
#endif
#if (PRESCAN_THREADS >= 4)
    SUM(4);
#endif
#if (PRESCAN_THREADS >= 2)
    SUM(2);
#endif

    if (threadIdx.x == 0) {
        float avg = Avg[0] + Avg[1];
        avg /= PRESCAN_THREADS;
        float stddev = StdDev[0] + StdDev[1];
        stddev /= PRESCAN_THREADS;

        atomicMin(minmax, ((unsigned int)(avg - 10 * stddev)) / (KB * 1024));
        atomicMax(minmax + 1, ((unsigned int)(avg + 10 * stddev)) / (KB * 1024));
    }
}
#undef SUM

// ---------------------------------------------------------------------------
// Kernel 2: intermediates — verbatim from histo_intermediates.cu
// ---------------------------------------------------------------------------
__device__ void calculateBin(const unsigned int bin, uchar4 *sm_mapping)
{
    unsigned char offset  =  bin        %   4;
    unsigned char indexlo = (bin >>  2) % 256;
    unsigned char indexhi = (bin >> 10) %  KB;
    unsigned char block   =  bin / BINS_PER_BLOCK;

    offset *= 8;

    uchar4 sm;
    sm.x = block;
    sm.y = indexhi;
    sm.z = indexlo;
    sm.w = offset;

    *sm_mapping = sm;
}

__global__ void histo_intermediates_kernel(uint2 *input,
                                           unsigned int height,
                                           unsigned int width,
                                           unsigned int input_pitch,
                                           uchar4 *sm_mappings)
{
    unsigned int line = UNROLL * blockIdx.x; // 16 is the unroll factor;

    uint2 *load_bin = input + line * input_pitch + threadIdx.x;

    unsigned int store = line * width + threadIdx.x;
    bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

    #pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        uint2 bin_value = *load_bin;

        calculateBin(bin_value.x, &sm_mappings[store]);

        if (!skip) calculateBin(bin_value.y, &sm_mappings[store + blockDim.x]);

        load_bin += input_pitch;
        store += width;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: main — verbatim from histo_main.cu
// ---------------------------------------------------------------------------
__device__ void testIncrementGlobal(unsigned int *global_histo,
                                     unsigned int sm_range_min,
                                     unsigned int sm_range_max,
                                     const uchar4 sm)
{
    const unsigned int range = sm.x;
    const unsigned int indexhi = sm.y;
    const unsigned int indexlo = sm.z;
    const unsigned int offset  = sm.w;

    if (range < sm_range_min || range > sm_range_max)
    {
        const unsigned int bin = range * BINS_PER_BLOCK + offset / 8 + (indexlo << 2) + (indexhi << 10);
        const unsigned int bin_div2 = bin / 2;
        const unsigned int bin_offset = (bin % 2 == 1) ? 16 : 0;

        unsigned int old_val = global_histo[bin_div2];
        unsigned short old_bin = (old_val >> bin_offset) & 0xFFFF;

        if (old_bin < 255)
        {
            atomicAdd(&global_histo[bin_div2], 1 << bin_offset);
        }
    }
}

__device__ void testIncrementLocal(unsigned int *global_overflow,
                                    unsigned int smem[KB][256],
                                    const unsigned int myRange,
                                    const uchar4 sm)
{
    const unsigned int range = sm.x;
    const unsigned int indexhi = sm.y;
    const unsigned int indexlo = sm.z;
    const unsigned int offset  = sm.w;

    if (range == myRange)
    {
        unsigned int add = (unsigned int)(1 << offset);
        unsigned int prev = atomicAdd(&smem[indexhi][indexlo], add);

        unsigned int prev_bin_val = (prev >> offset) & 0x000000FF;

        if (prev_bin_val == 0x000000FF)
        {
            const unsigned int bin =
                range * BINS_PER_BLOCK +
                offset / 8 + (indexlo << 2) + (indexhi << 10);

            bool can_overflow_to_bin_plus_1 = (offset < 24) ? true : false;
            bool can_overflow_to_bin_plus_2 = (offset < 16) ? true : false;
            bool can_overflow_to_bin_plus_3 = (offset <  8) ? true : false;

            bool overflow_into_bin_plus_1 = false;
            bool overflow_into_bin_plus_2 = false;
            bool overflow_into_bin_plus_3 = false;

            unsigned int prev_bin_plus_1_val = (prev >> (offset +  8)) & 0x000000FF;
            unsigned int prev_bin_plus_2_val = (prev >> (offset + 16)) & 0x000000FF;
            unsigned int prev_bin_plus_3_val = (prev >> (offset + 24)) & 0x000000FF;

            if (can_overflow_to_bin_plus_1 &&        prev_bin_val == 0x000000FF) overflow_into_bin_plus_1 = true;
            if (can_overflow_to_bin_plus_2 && prev_bin_plus_1_val == 0x000000FF) overflow_into_bin_plus_2 = true;
            if (can_overflow_to_bin_plus_3 && prev_bin_plus_2_val == 0x000000FF) overflow_into_bin_plus_3 = true;

            unsigned int bin_plus_1_add;
            unsigned int bin_plus_2_add;
            unsigned int bin_plus_3_add;

            if (overflow_into_bin_plus_1) bin_plus_1_add = (prev_bin_plus_1_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
            if (overflow_into_bin_plus_2) bin_plus_2_add = (prev_bin_plus_2_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
            if (overflow_into_bin_plus_3) bin_plus_3_add = (prev_bin_plus_3_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;

                                          atomicAdd(&global_overflow[bin],   256);
            if (overflow_into_bin_plus_1) atomicAdd(&global_overflow[bin+1], bin_plus_1_add);
            if (overflow_into_bin_plus_2) atomicAdd(&global_overflow[bin+2], bin_plus_2_add);
            if (overflow_into_bin_plus_3) atomicAdd(&global_overflow[bin+3], bin_plus_3_add);
        }
    }
}

__device__ void clearMemory(unsigned int smem[KB][256])
{
    for (int i = threadIdx.x; i < BINS_PER_BLOCK / 4; i += blockDim.x)
    {
        ((unsigned int*)smem)[i] = 0;
    }
}

__device__ void copyMemory(unsigned int *dst, unsigned int src[KB][256])
{
    for (int i = threadIdx.x; i < BINS_PER_BLOCK / 4; i += blockDim.x)
    {
        dst[i] = ((unsigned int*)src)[i];
    }
}

__global__ void histo_main_kernel(uchar4 *sm_mappings,
                                  unsigned int num_elements,
                                  unsigned int sm_range_min,
                                  unsigned int sm_range_max,
                                  unsigned int histo_height,
                                  unsigned int histo_width,
                                  unsigned int *global_subhisto,
                                  unsigned int *global_histo,
                                  unsigned int *global_overflow)
{
    __shared__ unsigned int sub_histo[KB][256];

    unsigned int local_scan_range = sm_range_min + blockIdx.y;
    unsigned int local_scan_load = blockIdx.x * blockDim.x + threadIdx.x;

    clearMemory(sub_histo);
    __syncthreads();

    if (blockIdx.y == 0)
    {
        while (local_scan_load < num_elements)
        {
            uchar4 sm = sm_mappings[local_scan_load];
            local_scan_load += blockDim.x * gridDim.x;

            testIncrementLocal(global_overflow, sub_histo, local_scan_range, sm);
            testIncrementGlobal(global_histo, sm_range_min, sm_range_max, sm);
        }
    }
    else
    {
        while (local_scan_load < num_elements)
        {
            uchar4 sm = sm_mappings[local_scan_load];
            local_scan_load += blockDim.x * gridDim.x;

            testIncrementLocal(global_overflow, sub_histo, local_scan_range, sm);
        }
    }

    unsigned int store_index = blockIdx.x * (histo_height * histo_width / 4) + (local_scan_range * BINS_PER_BLOCK / 4);

    __syncthreads();
    copyMemory(&(global_subhisto[store_index]), sub_histo);
}

// ---------------------------------------------------------------------------
// Kernel 4: final — verbatim from histo_final.cu
// ---------------------------------------------------------------------------
__global__ void histo_final_kernel(unsigned int sm_range_min,
                                   unsigned int sm_range_max,
                                   unsigned int histo_height,
                                   unsigned int histo_width,
                                   unsigned int *global_subhisto,
                                   unsigned int *global_histo,
                                   unsigned int *global_overflow,
                                   unsigned int *final_histo)
{
    unsigned int start_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min + 1) * BINS_PER_BLOCK;

    for (unsigned int i = start_offset; i < size_low_histo / 4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        ((ushort4*)global_histo)[i] = zero_short;

        uchar4 final_histo_data = {
            (unsigned char)min((unsigned int)global_histo_data.x, 255u),
            (unsigned char)min((unsigned int)global_histo_data.y, 255u),
            (unsigned char)min((unsigned int)global_histo_data.z, 255u),
            (unsigned char)min((unsigned int)global_histo_data.w, 255u)
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }

    for (unsigned int i = (size_low_histo / 4) + start_offset; i < (size_low_histo + size_mid_histo) / 4; i += gridDim.x * blockDim.x)
    {
        uint4 global_histo_data = ((uint4*)global_overflow)[i];
        ((uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        #pragma unroll
        for (int j = 0; j < BLOCK_X; j++)
        {
            unsigned int bin4in = ((unsigned int*)global_subhisto)[i + j * histo_height * histo_width / 4];
            internal_histo_data.x += (bin4in >>  0) & 0xFF;
            internal_histo_data.y += (bin4in >>  8) & 0xFF;
            internal_histo_data.z += (bin4in >> 16) & 0xFF;
            internal_histo_data.w += (bin4in >> 24) & 0xFF;
        }

        internal_histo_data.x = min(internal_histo_data.x, 255u);
        internal_histo_data.y = min(internal_histo_data.y, 255u);
        internal_histo_data.z = min(internal_histo_data.z, 255u);
        internal_histo_data.w = min(internal_histo_data.w, 255u);

        uchar4 final_histo_data = {
            (unsigned char)internal_histo_data.x,
            (unsigned char)internal_histo_data.y,
            (unsigned char)internal_histo_data.z,
            (unsigned char)internal_histo_data.w
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }

    for (unsigned int i = ((size_low_histo + size_mid_histo) / 4) + start_offset; i < (histo_height * histo_width) / 4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        ((ushort4*)global_histo)[i] = zero_short;

        uchar4 final_histo_data = {
            (unsigned char)min((unsigned int)global_histo_data.x, 255u),
            (unsigned char)min((unsigned int)global_histo_data.y, 255u),
            (unsigned char)min((unsigned int)global_histo_data.z, 255u),
            (unsigned char)min((unsigned int)global_histo_data.w, 255u)
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

/* Approximately-normal deviate in [-6,6] via sum of 12 uniforms. */
static float randn(unsigned int* state)
{
    float s = 0.0f;
    for (int i = 0; i < 12; ++i) {
        *state = *state * 1664525u + 1013904223u;
        s += (float)((*state >> 8) & 0xFFFF) / 65535.0f;
    }
    return s - 6.0f;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Standard Parboil histo 2D histogram shape: 256 x 8192 = 2,097,152 bins
    // (256 * 8192 / BINS_PER_BLOCK ~= 85 histogram blocks). Input image sized
    // to ~1M elements (img_width == histo_width as in the reference dataset, so
    // the img_width-based allocations cover the histo_width-based indexing).
    int img_width    = 256;
    int img_height   = 4096;
    int histo_width  = 256;
    int histo_height = 8192;

    img_width    = parseIntParam(argc, argv, "--img_width",    img_width);
    img_height   = parseIntParam(argc, argv, "--img_height",   img_height);
    histo_width  = parseIntParam(argc, argv, "--histo_width",  histo_width);
    histo_height = parseIntParam(argc, argv, "--histo_height", histo_height);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_img_width");    if (env_val) img_width = atoi(env_val);
    env_val = getenv("BENCH_PARAM_img_height");   if (env_val) img_height = atoi(env_val);
    env_val = getenv("BENCH_PARAM_histo_width");  if (env_val) histo_width = atoi(env_val);
    env_val = getenv("BENCH_PARAM_histo_height"); if (env_val) histo_height = atoi(env_val);

    const unsigned int num_bins = (unsigned int)histo_width * histo_height;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Histo  |  img %dx%d (%d px)  |  histo %dx%d (%u bins, ~%u blocks)\n\n",
            img_width, img_height, img_width * img_height,
            histo_width, histo_height, num_bins, num_bins / BINS_PER_BLOCK);

    // ---- Host input: values drawn from a narrow (near-normal) band so the
    // prescan-derived active range stays well inside [0, num_bins/BINS_PER_BLOCK).
    // This mimics the spatially-coherent value distribution of the real image
    // dataset (whose central histogram region is narrow). ----
    unsigned int* img = (unsigned int*)malloc((size_t)img_width * img_height * sizeof(unsigned int));
    unsigned char* histo = (unsigned char*)calloc((size_t)num_bins, sizeof(unsigned char));

    unsigned int center = num_bins / 2;                 // block ~ (num_bins/2)/24576
    float stddev = (float)(num_bins) / 28.0f;           // ~6-sigma stays in-range
    unsigned int seed = 42;
    for (int p = 0; p < img_width * img_height; ++p) {
        long v = (long)(center + randn(&seed) * stddev);
        if (v < 0) v = 0;
        if (v >= (long)num_bins) v = num_bins - 1;
        img[p] = (unsigned int)v;
    }

    // ---- Device allocations (original main.cu geometry) ----
    int even_width = ((img_width + 1) / 2) * 2;
    unsigned int  *input;
    unsigned int  *ranges;
    uchar4        *sm_mappings;
    unsigned int  *global_subhisto;
    unsigned short*global_histo;
    unsigned int  *global_overflow;
    unsigned char *final_histo;

    HIP_CHECK(hipMalloc((void**)&input,           (size_t)even_width * (((img_height + UNROLL - 1) / UNROLL) * UNROLL) * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&ranges,          2 * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&sm_mappings,     (size_t)img_width * img_height * sizeof(uchar4)));
    HIP_CHECK(hipMalloc((void**)&global_subhisto, (size_t)BLOCK_X * img_width * histo_height * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&global_histo,    (size_t)img_width * histo_height * sizeof(unsigned short)));
    HIP_CHECK(hipMalloc((void**)&global_overflow, (size_t)img_width * histo_height * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&final_histo,     (size_t)img_width * histo_height * sizeof(unsigned char)));

    HIP_CHECK(hipMemset(final_histo, 0, (size_t)img_width * histo_height * sizeof(unsigned char)));

    // upload image rows into the padded (even_width) input buffer (untimed)
    for (int y = 0; y < img_height; y++) {
        HIP_CHECK(hipMemcpy(&(((unsigned int*)input)[(size_t)y * even_width]),
                            &img[(size_t)y * img_width],
                            img_width * sizeof(unsigned int), hipMemcpyHostToDevice));
    }

    char params[256];
    snprintf(params, sizeof(params),
             "\"img_width\":%d,\"img_height\":%d,\"histo_width\":%d,"
             "\"histo_height\":%d,\"bins_per_block\":%d,\"block_x\":%d",
             img_width, img_height, histo_width, histo_height, BINS_PER_BLOCK, BLOCK_X);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_pre, acc_int, acc_main, acc_fin;

    // One recorded pass = the full 4-kernel pipeline. Each kernel is launched
    // once per pass and recorded into its own accumulator. State that mutates
    // across passes (ranges, sub-histogram, global/overflow buffers) is re-init
    // at the start of each pass, UNTIMED. final_histo is fully overwritten by
    // the final kernel, so it needs no reset.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_pre);
        bench_lap_reset(&acc_int);
        bench_lap_reset(&acc_main);
        bench_lap_reset(&acc_fin);

        unsigned int ranges_h[2] = {0xFFFFFFFFu, 0u};
        HIP_CHECK(hipMemcpy(ranges, ranges_h, 2 * sizeof(unsigned int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(global_subhisto, 0, (size_t)img_width * histo_height * sizeof(unsigned int)));
        HIP_CHECK(hipMemset(global_histo, 0, (size_t)img_width * histo_height * sizeof(unsigned short)));
        HIP_CHECK(hipMemset(global_overflow, 0, (size_t)img_width * histo_height * sizeof(unsigned int)));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        // Kernel 1: prescan
        bench_lap_add(&acc_pre, bench_time_op(&timer, [&]() {
            histo_prescan_kernel<<<dim3(PRESCAN_BLOCKS_X), dim3(PRESCAN_THREADS)>>>(
                (unsigned int*)input, img_height * img_width, ranges);
        }));

        // read back the derived active range (untimed)
        HIP_CHECK(hipMemcpy(ranges_h, ranges, 2 * sizeof(unsigned int), hipMemcpyDeviceToHost));

        // Kernel 2: intermediates
        bench_lap_add(&acc_int, bench_time_op(&timer, [&]() {
            histo_intermediates_kernel<<<dim3((img_height + UNROLL - 1) / UNROLL), dim3((img_width + 1) / 2)>>>(
                (uint2*)(input),
                (unsigned int)img_height,
                (unsigned int)img_width,
                (img_width + 1) / 2,
                (uchar4*)(sm_mappings));
        }));

        // Kernel 3: main (gridDim.y from the prescan range)
        bench_lap_add(&acc_main, bench_time_op(&timer, [&]() {
            histo_main_kernel<<<dim3(BLOCK_X, ranges_h[1] - ranges_h[0] + 1), dim3(THREADS)>>>(
                (uchar4*)(sm_mappings),
                img_height * img_width,
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow));
        }));

        // Kernel 4: final
        bench_lap_add(&acc_fin, bench_time_op(&timer, [&]() {
            histo_final_kernel<<<dim3(BLOCK_X * 3), dim3(512)>>>(
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow),
                (unsigned int*)(final_histo));
        }));

        // catch any launch-config / launch error this pass
        HIP_CHECK(hipGetLastError());
    };

    run_pass();
    double total_gpu_ms = 0.0;
    bench_emit_iter("histo_prescan_kernel",       0, &acc_pre,  params);
    bench_emit_iter("histo_intermediates_kernel", 0, &acc_int,  params);
    bench_emit_iter("histo_main_kernel",          0, &acc_main, params);
    bench_emit_iter("histo_final_kernel",         0, &acc_fin,  params);
    total_gpu_ms += acc_pre.gpu_ms + acc_int.gpu_ms + acc_main.gpu_ms + acc_fin.gpu_ms;
    fprintf(stderr, "prescan=%.3f inter=%.3f main=%.3f final=%.3f ms\n",
            acc_pre.gpu_ms, acc_int.gpu_ms, acc_main.gpu_ms, acc_fin.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // ---- Verification: final_histo[value] = min(count(value), 255) ----
    HIP_CHECK(hipMemcpy(histo, final_histo, (size_t)num_bins * sizeof(unsigned char), hipMemcpyDeviceToHost));
    unsigned int* ref = (unsigned int*)calloc((size_t)num_bins, sizeof(unsigned int));
    for (int p = 0; p < img_width * img_height; ++p) ref[img[p]]++;

    long errors = 0;
    for (unsigned int b = 0; b < num_bins; ++b) {
        unsigned char expect = (unsigned char)(ref[b] > 255 ? 255 : ref[b]);
        if (histo[b] != expect) {
            if (errors < 5)
                fprintf(stderr, "MISMATCH bin[%u]: gpu=%u cpu=%u\n", b, histo[b], expect);
            errors++;
        }
    }
    if (errors)
        fprintf(stderr, "FAIL: %ld mismatched bins\n", errors);
    else
        fprintf(stderr, "PASS (all %u bins)\n", num_bins);

    HIP_CHECK(hipFree(input));
    HIP_CHECK(hipFree(ranges));
    HIP_CHECK(hipFree(sm_mappings));
    HIP_CHECK(hipFree(global_subhisto));
    HIP_CHECK(hipFree(global_histo));
    HIP_CHECK(hipFree(global_overflow));
    HIP_CHECK(hipFree(final_histo));
    free(img); free(histo); free(ref);

    return (errors > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
