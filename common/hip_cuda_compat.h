/**
 * hip_cuda_compat.h
 *
 * Single-source HIP/CUDA compatibility layer.
 *
 * Benchmarks are written once against the HIP API. On AMD they compile with
 * hipcc as-is. On NVIDIA the same file is fed to nvcc (via `-x cu`), and the
 * macros below rewrite the hip* names to their cuda* equivalents at
 * preprocessing time — zero wrappers, zero runtime overhead: nvcc sees
 * exactly the code a native CUDA implementation would contain.
 *
 * Do NOT define __HIP_PLATFORM_AMD__ here: benchmarks use
 * `#ifdef __HIP_PLATFORM_AMD__` for AMD-specific tuning (e.g. wavefront
 * size 64), and hipcc defines it as a builtin when targeting AMD.
 */

#ifndef HIP_CUDA_COMPAT_H
#define HIP_CUDA_COMPAT_H

#if defined(__NVCC__) && !__has_include(<hip/hip_runtime.h>)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* Runtime API */
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipHostMalloc           cudaMallocHost
#define hipHostFree             cudaFreeHost
#define hipHostMallocDefault    cudaHostAllocDefault
#define hipMemcpy               cudaMemcpy
#define hipMemcpyAsync          cudaMemcpyAsync
#define hipMemcpyToSymbol       cudaMemcpyToSymbol
#define hipMemset               cudaMemset
#define hipMemsetAsync          cudaMemsetAsync
#define hipMemGetInfo           cudaMemGetInfo
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetDevice            cudaGetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetLastError         cudaGetLastError
#define hipGetErrorString       cudaGetErrorString

/* Events and streams */
#define hipEvent_t              cudaEvent_t
#define hipEventCreate          cudaEventCreate
#define hipEventDestroy         cudaEventDestroy
#define hipEventRecord          cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipStream_t             cudaStream_t
#define hipStreamCreate         cudaStreamCreate
#define hipStreamDestroy        cudaStreamDestroy
#define hipStreamSynchronize    cudaStreamSynchronize

/* Types */
#define hipDeviceProp_t         cudaDeviceProp
#define hipError_t              cudaError_t
#define hipSuccess              cudaSuccess

/* Legacy hip* thread/block index aliases */
#define hipBlockIdx_x           blockIdx.x
#define hipBlockIdx_y           blockIdx.y
#define hipBlockIdx_z           blockIdx.z
#define hipThreadIdx_x          threadIdx.x
#define hipThreadIdx_y          threadIdx.y
#define hipThreadIdx_z          threadIdx.z
#define hipBlockDim_x           blockDim.x
#define hipBlockDim_y           blockDim.y
#define hipBlockDim_z           blockDim.z
#define hipGridDim_x            gridDim.x
#define hipGridDim_y            gridDim.y
#define hipGridDim_z            gridDim.z

#else /* HIP (AMD, or HIP-on-NVIDIA if installed) */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#endif

#endif /* HIP_CUDA_COMPAT_H */
