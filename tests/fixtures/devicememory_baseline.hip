// Frozen local kernel bodies at d589cb261621ceeb00c77c48062349e23fc2bf9c.
// No SHOC-upstream attribution; only function names changed.
__global__ void baseline_readOnly_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = gid; i < N; i += totalThreads) {
        sum += input[i];
    }
    if (gid < N) {
        output[gid] = sum;
    }
}

__global__ void baseline_writeOnly_kernel(float* __restrict__ output,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        output[i] = (float)i * 0.001f;
    }
}

__global__ void baseline_readWrite_kernel(float* __restrict__ data,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        data[i] = data[i] * 1.0001f;
    }
}
