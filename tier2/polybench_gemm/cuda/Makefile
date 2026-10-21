# Makefile for tier2/polybench_gemm/cuda — PolyBench GEMM benchmark (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := polybench_gemm
SRC       := polybench_gemm.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH) -Xcompiler -pthread

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) cpu_reference.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
