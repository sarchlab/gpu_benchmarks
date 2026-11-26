# Makefile for tier1/code_fetch_latency/cuda — code_fetch_latency microbenchmark (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := code_fetch_latency
SRC       := code_fetch_latency.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(wildcard ../../../common_cuda/*.h)
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
