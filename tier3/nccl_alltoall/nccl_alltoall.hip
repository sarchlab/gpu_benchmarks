/**
 * nccl_alltoall.hip — AllToAll benchmark using RCCL (NCCL-compatible).
 *
 * Measures AllToAll bandwidth across multiple GPUs.
 * NCCL does not have a native AllToAll, so this uses ncclSend/ncclRecv
 * inside a ncclGroupStart/ncclGroupEnd block. Each GPU i sends a chunk
 * of message_size/num_gpus bytes to every GPU j, and receives a chunk
 * from every GPU j.
 *
 * Parameters read from BENCH_PARAM_* environment variables.
 * Output: JSON-lines to stdout, human-readable info to stderr.
 *
 * Build:
 *   hipcc -O3 -std=c++14 nccl_alltoall.hip -lrccl -o nccl_alltoall
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

    long long chunk_size = message_size / num_gpus;
    long long chunk_count = chunk_size / elem_size;

    fprintf(stderr, "NCCL AllToAll\n");
    fprintf(stderr, "  GPUs:         %d\n", num_gpus);
    fprintf(stderr, "  Message size: %lld bytes (total per GPU)\n", message_size);
    fprintf(stderr, "  Chunk size:   %lld bytes (%lld elements per peer)\n", chunk_size, chunk_count);
    fprintf(stderr, "  Dtype:        %s\n", dtype_str);
    fprintf(stderr, "  Warmup:       %d\n\n", warmup);

    /* ------------------------------------------------------------------ */
    /* Initialize NCCL                                                     */
    /* ------------------------------------------------------------------ */
    ncclComm_t *comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
    hipStream_t *streams = (hipStream_t *)malloc(num_gpus * sizeof(hipStream_t));
    void **sendbuf = (void **)malloc(num_gpus * sizeof(void *));
    void **recvbuf = (void **)malloc(num_gpus * sizeof(void *));
    int *devs = (int *)malloc(num_gpus * sizeof(int));

    for (int i = 0; i < num_gpus; i++) devs[i] = i;

    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));

    for (int i = 0; i < num_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));
        HIP_CHECK(hipMalloc(&sendbuf[i], message_size));
        HIP_CHECK(hipMalloc(&recvbuf[i], message_size));
        HIP_CHECK(hipMemset(sendbuf[i], 1, message_size));
        HIP_CHECK(hipMemset(recvbuf[i], 0, message_size));
    }

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int w = 0; w < warmup; w++) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            for (int j = 0; j < num_gpus; j++) {
                NCCL_CHECK(ncclSend((char *)sendbuf[i] + j * chunk_size,
                                    chunk_count, nccl_dtype, j,
                                    comms[i], streams[i]));
                NCCL_CHECK(ncclRecv((char *)recvbuf[i] + j * chunk_size,
                                    chunk_count, nccl_dtype, j,
                                    comms[i], streams[i]));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        for (int i = 0; i < num_gpus; i++) {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamSynchronize(streams[i]));
        }
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
    for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < num_gpus; j++) {
            NCCL_CHECK(ncclSend((char *)sendbuf[i] + j * chunk_size,
                                chunk_count, nccl_dtype, j,
                                comms[i], streams[i]));
            NCCL_CHECK(ncclRecv((char *)recvbuf[i] + j * chunk_size,
                                chunk_count, nccl_dtype, j,
                                comms[i], streams[i]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    HIP_CHECK(hipEventRecord(ev_stop, streams[0]));
    for (int i = 0; i < num_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Bus bandwidth: for AllToAll, bus bw = (n-1)/n * message_size */
    double elapsed_s = elapsed_ms / 1000.0;
    double algbw_gbps = (double)message_size / elapsed_s / 1e9;
    double busbw_gbps = algbw_gbps * (num_gpus - 1) / num_gpus;

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    fprintf(stderr, "nccl_alltoall  msg=%lld  time_ms=%.3f  algbw=%.2f GB/s  busbw=%.2f GB/s\n",
            message_size, (double)elapsed_ms, algbw_gbps, busbw_gbps);

    printf("{\"type\":\"kernel\",\"name\":\"ncclAllToAll\",\"time_ms\":%.6f,"
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
    for (int i = 0; i < num_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(sendbuf[i]));
        HIP_CHECK(hipFree(recvbuf[i]));
        HIP_CHECK(hipStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
    free(comms); free(streams); free(sendbuf); free(recvbuf); free(devs);
    return 0;
}
