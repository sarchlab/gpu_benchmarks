# Makefile for tier2/rodinia_hotspot3d/cuda — Rodinia HotSpot3D benchmark (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := rodinia_hotspot3d
SRC       := rodinia_hotspot3d.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) hotspot3d_reference.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/cuda/hotspot3d_original_gpu.cu $(SRC) hotspot3d_reference.h ../../../tests/upstream/hotspot3d_cpu.h ../../../tests/cuda/upstream/hotspot3d_gpu.cu ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test --grid_size 17 --num_iterations 3

clean:
	rm -f $(TARGET) bench_output_test
