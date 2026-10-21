// Frozen functions from utcs-scea/altis 042e2926d1fe9a5ea5b3e2d659b75bf925b152cc
// src/cuda/level2/particlefilter/ex_particle_CUDA_float_seq.cu
// Original file SHA256: 1e3eff2700178cb79957bd6244d7a73b4bc41ba2554530c422a2f31b66045ad4
// Copyright and terms: ../../tier2/altis_particlefilter/hip/UPSTREAM_LICENSE.
#pragma once
// d_randu uses defined wrap; d_randn repairs only log(0). All other frozen
// function bodies below are unchanged except trailing whitespace. The regression driver decomposes the
// racy normalize_weights_kernel into globally ordered operations; no test
// claims that the original multi-block normalize kernel is correct.
namespace pf_upstream_gpu {
const int threads_per_block=512;
__device__ double calcLikelihoodSum(unsigned char * I, int * ind, int numOnes, int index) {
    double likelihoodSum = 0.0;
    int x;
    for (x = 0; x < numOnes; x++)
        likelihoodSum += (pow((double) (I[ind[index * numOnes + x]] - 100), 2) - pow((double) (I[ind[index * numOnes + x]] - 228), 2)) / 50.0;
    return likelihoodSum;
}

__device__ void cdfCalc(double * CDF, double * weights, int Nparticles) {
    int x;
    CDF[0] = weights[0];
    for (x = 1; x < Nparticles; x++) {
        CDF[x] = weights[x] + CDF[x - 1];
    }
}

__device__ double d_randu(int * seed, int index) {

    int M = INT_MAX;
    int A = 1103515245;
    int C = 12345;
    unsigned bits = unsigned(A) * unsigned(seed[index]) + unsigned(C);
    int num = int(bits <= unsigned(INT_MAX) ? (long long)bits : (long long)bits - 4294967296LL);
    seed[index] = num % M;

    return fabs(seed[index] / ((double) M));
}

__device__ double d_randn(int * seed, int index) {
    //Box-Muller algortihm
    double pi = 3.14159265358979323846;
    double u = d_randu(seed, index);
    double v = d_randu(seed, index);
    double cosine = cos(2 * pi * v);
    double rt = -2 * log(u == 0.0 ? 1.0 / INT_MAX : u);
    return sqrt(rt) * cosine;
}

__device__ double dev_round_double(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}

__global__ void find_index_kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, double * weights, int Nparticles) {
    int block_id = blockIdx.x;
    int i = blockDim.x * block_id + threadIdx.x;

    if (i < Nparticles) {

        int index = -1;
        int x;

        for (x = 0; x < Nparticles; x++) {
            if (CDF[x] >= u[i]) {
                index = x;
                break;
            }
        }
        if (index == -1) {
            index = Nparticles - 1;
        }

        xj[i] = arrayX[index];
        yj[i] = arrayY[index];

        //weights[i] = 1 / ((double) (Nparticles)); //moved this code to the beginning of likelihood kernel

    }
    __syncthreads();
}

__global__ void sum_kernel(double* partial_sums, int Nparticles) {
    int block_id = blockIdx.x;
    int i = blockDim.x * block_id + threadIdx.x;

    if (i == 0) {
        int x;
        double sum = 0.0;
        int num_blocks = ceil((double) Nparticles / (double) threads_per_block);
        for (x = 0; x < num_blocks; x++) {
            sum += partial_sums[x];
        }
        partial_sums[0] = sum;
    }
}

__global__ void likelihood_kernel(double * arrayX, double * arrayY, double * xj, double * yj, double * CDF, int * ind, int * objxy, double * likelihood, unsigned char * I, double * u, double * weights, int Nparticles, int countOnes, int max_size, int k, int IszY, int Nfr, int *seed, double* partial_sums) {
    int block_id = blockIdx.x;
    int i = blockDim.x * block_id + threadIdx.x;
    int y;

    int indX, indY;
    __shared__ double buffer[512];
    if (i < Nparticles) {
        arrayX[i] = xj[i];
        arrayY[i] = yj[i];

        weights[i] = 1 / ((double) (Nparticles)); //Donnie - moved this line from end of find_index_kernel to prevent all weights from being reset before calculating position on final iteration.

        arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i);
        arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i);

    }

    __syncthreads();

    if (i < Nparticles) {
        for (y = 0; y < countOnes; y++) {
            //added dev_round_double() to be consistent with roundDouble
            indX = dev_round_double(arrayX[i]) + objxy[y * 2 + 1];
            indY = dev_round_double(arrayY[i]) + objxy[y * 2];

            ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
            if (ind[i * countOnes + y] >= max_size)
                ind[i * countOnes + y] = 0;
        }
        likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

        likelihood[i] = likelihood[i] / countOnes;

        weights[i] = weights[i] * exp(likelihood[i]); //Donnie Newell - added the missing exponential function call

    }

    buffer[threadIdx.x] = 0.0;

    __syncthreads();

    if (i < Nparticles) {

        buffer[threadIdx.x] = weights[i];
    }

    __syncthreads();

    //this doesn't account for the last block that isn't full
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            buffer[threadIdx.x] += buffer[threadIdx.x + s];
        }

        __syncthreads();

    }
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = buffer[0];
    }

    __syncthreads();


}

}
