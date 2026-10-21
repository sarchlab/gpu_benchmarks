# Makefile for tier2/polybench_jacobi2d/cuda — PolyBench Jacobi 2D stencil benchmark (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := polybench_jacobi2d
SRC       := polybench_jacobi2d.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) jacobi_reference.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/cuda/jacobi_original_gpu.cu $(SRC) jacobi_reference.h ../../../tests/fixtures/polybench_jacobi_original.h ../../../tests/fixtures/polybench_jacobi_original_gpu.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h
	$(NVCC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test --n 65 --tsteps 20

clean:
	rm -f $(TARGET) bench_output_test
