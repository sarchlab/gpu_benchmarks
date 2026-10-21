# Makefile for tier2/npb_mg/cuda — NPB Multi-Grid (native CUDA)
#
# Usage:
#   make                       # build for the GPU in this machine (-arch=native)
#   make CUDA_ARCH=sm_80       # A100
#   make CUDA_ARCH=sm_90       # H100
#   make clean
#
# -arch=native needs CUDA >= 11.5 and a visible GPU at build time; pass an
# explicit CUDA_ARCH when building on a machine without one.

TARGET    := npb_mg
SRC       := npb_mg.cu
NVCC      ?= nvcc
CUDA_ARCH ?= native
CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH) -Xcompiler -pthread

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

COMMON := nas_mg.h mg_reference.h mg_kernels.h ../../../common_cuda/bench_timing.h ../../../common_cuda/bench_warmup.h

$(TARGET): $(SRC) $(COMMON)
	$(NVCC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/cuda/algorithmic_zero_output.cu $(SRC) $(COMMON)
	$(NVCC) $(CFLAGS) -DTEST_MG_ZERO $< -o $@

test: bench_output_test
	./bench_output_test

clean:
	rm -f $(TARGET) bench_output_test
