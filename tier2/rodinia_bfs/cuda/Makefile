# Makefile for tier2/rodinia_bfs/cuda — Rodinia Breadth-First Search (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := rodinia_bfs
SRC       := rodinia_bfs.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH)

COMMON := ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) cpu_reference.h $(COMMON)
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
