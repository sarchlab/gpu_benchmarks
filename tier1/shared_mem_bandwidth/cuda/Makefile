# Makefile for tier1/shared_mem_bandwidth/cuda — shared_mem_bandwidth microbenchmark (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := shared_mem_bandwidth
SRC       := shared_mem_bandwidth.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test test-pattern

all: $(TARGET)

$(TARGET): $(SRC) shared_mem_pattern.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

HOST_CXX ?= c++
bench_pattern_test: ../../../tests/cuda/shared_mem_pattern_test.cpp shared_mem_pattern.h
	$(HOST_CXX) -O2 -std=c++14 $< -o $@

test-pattern: bench_pattern_test
	./bench_pattern_test

bench_output_test: ../../../tests/cuda/shared_mem_output_test.cu $(SRC) shared_mem_pattern.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

test: test-pattern bench_output_test
	./bench_output_test

clean:
	rm -f $(TARGET) bench_pattern_test bench_output_test
