/**
 * nccl_sendrecv.hip — Point-to-point SendRecv benchmark using RCCL (NCCL-compatible).
 *
 * Measures unidirectional point-to-point bandwidth between GPU 0 and GPU 1.
 * GPU 0 sends `message_size` bytes to GPU 1 using ncclSend/ncclRecv
 * inside a ncclGroupStart/ncclGroupEnd block. Only uses 2 GPUs regardless
 * of the num_gpus setting.
 *
 * Parameters read from BENCH_PARAM_* environment variables.
 * Output: JSON-lines to stdout, human-readable info to stderr.
 *
 * Build:
 *   hipcc -O3 -std=c++14 nccl_sendrecv.hip -lrccl -o nccl_sendrecv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#define HIP_CHECK(x) do {                                                 \
    hipError_t _e = (x);                                                  \
    if (_e != hipSuccess) {                                               \
        fprintf(stderr, "HIP error %s:%d: %s\n",                         \
                __FILE__, __LINE__, hipGetErrorString(_e));               \
        exit(1);                                                           \
    }                                                                      \
} while (0)

#define NCCL_CHECK(x) do {                                                 \
    ncclResult_t _e = (x);                                                 \
    if (_e != ncclSuccess) {                                               \
        fprintf(stderr, "NCCL error %s:%d: %s\n",                         \
                __FILE__, __LINE__, ncclGetErrorString(_e));               \
        exit(1);                                                           \
    }                                                                      \
} while (0)

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    /* ------------------------------------------------------------------ */
    /* Parameters                                                          */
    /* ------------------------------------------------------------------ */
    long long message_size = 1048576; /* 1 MB default */
    int num_gpus = 0;
    int warmup = 0;
    const char *dtype_str = "fp32";
    const char *env_val;

    env_val = getenv("BENCH_PARAM_message_size");
    if (env_val) message_size = atoll(env_val);
    env_val = getenv("BENCH_PARAM_num_gpus");
    if (env_val) num_gpus = atoi(env_val);
    env_val = getenv("BENCH_PARAM_dtype");
    if (env_val) dtype_str = env_val;

    /* Determine dtype */
    ncclDataType_t nccl_dtype = ncclFloat;
    int elem_size = 4;
    if (strcmp(dtype_str, "fp16") == 0) {
        nccl_dtype = ncclHalf;
        elem_size = 2;
    } else if (strcmp(dtype_str, "bf16") == 0) {
        nccl_dtype = ncclBfloat16;
        elem_size = 2;
    }
    /* fp32 is default */

    /* Auto-detect GPU count if not specified */
    if (num_gpus <= 0) {
        HIP_CHECK(hipGetDeviceCount(&num_gpus));
    }
    if (num_gpus < 2) {
        fprintf(stderr, "NCCL benchmarks require >= 2 GPUs (found %d). Skipping.\n", num_gpus);
        printf("{\"type\":\"error\",\"message\":\"requires >= 2 GPUs\"}\n");
        return 0;
    }

    /* SendRecv only uses 2 GPUs regardless of num_gpus setting */
    int used_gpus = 2;

    long long count = message_size / elem_size;

    fprintf(stderr, "NCCL SendRecv\n");
    fprintf(stderr, "  GPUs:         %d (using 2 for point-to-point)\n", num_gpus);
    fprintf(stderr, "  Message size: %lld bytes (%lld elements)\n", message_size, count);
    fprintf(stderr, "  Dtype:        %s\n", dtype_str);
    fprintf(stderr, "  Warmup:       %d\n\n", warmup);

    /* ------------------------------------------------------------------ */
    /* Initialize NCCL (only 2 GPUs)                                       */
    /* ------------------------------------------------------------------ */
    ncclComm_t comms[2];
    hipStream_t streams[2];
    void *sendbuf = NULL;
    void *recvbuf = NULL;
    int devs[2] = {0, 1};

    NCCL_CHECK(ncclCommInitAll(comms, used_gpus, devs));

    /* GPU 0: sender */
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamCreate(&streams[0]));
    HIP_CHECK(hipMalloc(&sendbuf, message_size));
    HIP_CHECK(hipMemset(sendbuf, 1, message_size));

    /* GPU 1: receiver */
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipStreamCreate(&streams[1]));
    HIP_CHECK(hipMalloc(&recvbuf, message_size));
    HIP_CHECK(hipMemset(recvbuf, 0, message_size));

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int w = 0; w < warmup; w++) {
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(sendbuf, count, nccl_dtype, 1, comms[0], streams[0]));
        NCCL_CHECK(ncclRecv(recvbuf, count, nccl_dtype, 0, comms[1], streams[1]));
        NCCL_CHECK(ncclGroupEnd());
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamSynchronize(streams[0]));
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipStreamSynchronize(streams[1]));
    }

    /* ------------------------------------------------------------------ */
    /* Timed run                                                           */
    /* ------------------------------------------------------------------ */
    HIP_CHECK(hipSetDevice(0));
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start, streams[0]));

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(sendbuf, count, nccl_dtype, 1, comms[0], streams[0]));
    NCCL_CHECK(ncclRecv(recvbuf, count, nccl_dtype, 0, comms[1], streams[1]));
    NCCL_CHECK(ncclGroupEnd());

    HIP_CHECK(hipEventRecord(ev_stop, streams[0]));
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(streams[0]));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipStreamSynchronize(streams[1]));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Bus bandwidth: for SendRecv, bus bw = message_size (unidirectional) */
    double elapsed_s = elapsed_ms / 1000.0;
    double algbw_gbps = (double)message_size / elapsed_s / 1e9;
    double busbw_gbps = algbw_gbps;

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    fprintf(stderr, "nccl_sendrecv  msg=%lld  time_ms=%.3f  algbw=%.2f GB/s  busbw=%.2f GB/s\n",
            message_size, (double)elapsed_ms, algbw_gbps, busbw_gbps);

    printf("{\"type\":\"kernel\",\"name\":\"ncclSendRecv\",\"time_ms\":%.6f,"
           "\"params\":{\"message_size\":%lld,\"num_gpus\":%d,"
           "\"dtype\":\"%s\",\"warmup\":%d}}\n",
           (double)elapsed_ms, message_size, num_gpus, dtype_str, warmup);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":["
           "{\"name\":\"algbw_gbps\",\"value\":%.2f},"
           "{\"name\":\"busbw_gbps\",\"value\":%.2f}"
           "]}\n",
           (double)elapsed_ms, algbw_gbps, busbw_gbps);

    /* ------------------------------------------------------------------ */
    /* Cleanup                                                             */
    /* ------------------------------------------------------------------ */
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipFree(sendbuf));
    HIP_CHECK(hipStreamDestroy(streams[0]));
    ncclCommDestroy(comms[0]);

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipFree(recvbuf));
    HIP_CHECK(hipStreamDestroy(streams[1]));
    ncclCommDestroy(comms[1]);

    return 0;
}
