# Makefile for tier2/rodinia_hotspot/cuda — Rodinia Hotspot benchmark (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := rodinia_hotspot
SRC       := rodinia_hotspot.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH) -Xcompiler -pthread

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) hotspot_reference.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/cuda/hotspot_original_gpu.cu $(SRC) hotspot_reference.h ../../../tests/upstream/hotspot_physics.h ../../../tests/cuda/upstream/hotspot_gpu.cu ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test --grid_size 17 --total_iterations 3 --pyramid_height 7

clean:
	rm -f $(TARGET) bench_output_test
