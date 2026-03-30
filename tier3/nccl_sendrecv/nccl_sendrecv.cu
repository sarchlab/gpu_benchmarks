/**
 * nccl_sendrecv.cu — Point-to-point SendRecv benchmark using NCCL.
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
 *   nvcc -O3 -std=c++14 -arch=native nccl_sendrecv.cu -lnccl -o nccl_sendrecv
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
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) warmup = atoi(env_val);
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
    cudaStream_t streams[2];
    void *sendbuf = NULL;
    void *recvbuf = NULL;
    int devs[2] = {0, 1};

    NCCL_CHECK(ncclCommInitAll(comms, used_gpus, devs));

    /* GPU 0: sender */
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaMalloc(&sendbuf, message_size));
    CUDA_CHECK(cudaMemset(sendbuf, 1, message_size));

    /* GPU 1: receiver */
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));
    CUDA_CHECK(cudaMalloc(&recvbuf, message_size));
    CUDA_CHECK(cudaMemset(recvbuf, 0, message_size));

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int w = 0; w < warmup; w++) {
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(sendbuf, count, nccl_dtype, 1, comms[0], streams[0]));
        NCCL_CHECK(ncclRecv(recvbuf, count, nccl_dtype, 0, comms[1], streams[1]));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
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
    NCCL_CHECK(ncclSend(sendbuf, count, nccl_dtype, 1, comms[0], streams[0]));
    NCCL_CHECK(ncclRecv(recvbuf, count, nccl_dtype, 0, comms[1], streams[1]));
    NCCL_CHECK(ncclGroupEnd());

    CUDA_CHECK(cudaEventRecord(ev_stop, streams[0]));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

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
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(sendbuf));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    ncclCommDestroy(comms[0]);

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(recvbuf));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    ncclCommDestroy(comms[1]);

    return 0;
}
