/**
 * nccl_broadcast.cu — Broadcast benchmark using NCCL.
 *
 * Measures Broadcast bandwidth across multiple GPUs.
 * Root GPU 0 sends `message_size` bytes to all other GPUs.
 *
 * Parameters read from BENCH_PARAM_* environment variables.
 * Output: JSON-lines to stdout, human-readable info to stderr.
 *
 * Build:
 *   nvcc -O3 -std=c++14 -arch=native nccl_broadcast.cu -lnccl -o nccl_broadcast
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(x) do {                                                 \
    cudaError_t _e = (x);                                                  \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(_e));               \
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
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    }
    if (num_gpus < 2) {
        fprintf(stderr, "NCCL benchmarks require >= 2 GPUs (found %d). Skipping.\n", num_gpus);
        printf("{\"type\":\"error\",\"message\":\"requires >= 2 GPUs\"}\n");
        return 0;
    }

    long long count = message_size / elem_size;

    fprintf(stderr, "NCCL Broadcast\n");
    fprintf(stderr, "  GPUs:         %d\n", num_gpus);
    fprintf(stderr, "  Message size: %lld bytes (%lld elements)\n", message_size, count);
    fprintf(stderr, "  Dtype:        %s\n", dtype_str);
    fprintf(stderr, "  Warmup:       %d\n\n", warmup);

    /* ------------------------------------------------------------------ */
    /* Initialize NCCL                                                     */
    /* ------------------------------------------------------------------ */
    ncclComm_t *comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
    cudaStream_t *streams = (cudaStream_t *)malloc(num_gpus * sizeof(cudaStream_t));
    void **sendbuf = (void **)malloc(num_gpus * sizeof(void *));
    void **recvbuf = (void **)malloc(num_gpus * sizeof(void *));
    int *devs = (int *)malloc(num_gpus * sizeof(int));

    for (int i = 0; i < num_gpus; i++) devs[i] = i;

    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&sendbuf[i], message_size));
        CUDA_CHECK(cudaMalloc(&recvbuf[i], message_size));
        /* Only root needs sendbuf initialized */
        if (i == 0) {
            CUDA_CHECK(cudaMemset(sendbuf[i], 1, message_size));
        }
        CUDA_CHECK(cudaMemset(recvbuf[i], 0, message_size));
    }

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int w = 0; w < warmup; w++) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            NCCL_CHECK(ncclBroadcast(sendbuf[i], recvbuf[i], count,
                                     nccl_dtype, 0, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    /* ------------------------------------------------------------------ */
    /* Timed run                                                           */
    /* ------------------------------------------------------------------ */
    CUDA_CHECK(cudaSetDevice(0));
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, streams[0]));

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        NCCL_CHECK(ncclBroadcast(sendbuf[i], recvbuf[i], count,
                                 nccl_dtype, 0, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    CUDA_CHECK(cudaEventRecord(ev_stop, streams[0]));
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Bus bandwidth: for Broadcast, bus bw = message_size (no scaling factor) */
    double elapsed_s = elapsed_ms / 1000.0;
    double algbw_gbps = (double)message_size / elapsed_s / 1e9;
    double busbw_gbps = algbw_gbps;

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    fprintf(stderr, "nccl_broadcast  msg=%lld  time_ms=%.3f  algbw=%.2f GB/s  busbw=%.2f GB/s\n",
            message_size, (double)elapsed_ms, algbw_gbps, busbw_gbps);

    printf("{\"type\":\"kernel\",\"name\":\"ncclBroadcast\",\"time_ms\":%.6f,"
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
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(sendbuf[i]));
        CUDA_CHECK(cudaFree(recvbuf[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
    free(comms); free(streams); free(sendbuf); free(recvbuf); free(devs);
    return 0;
}
