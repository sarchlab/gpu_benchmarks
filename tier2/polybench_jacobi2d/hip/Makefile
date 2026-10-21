# Makefile for tier2/polybench_jacobi2d — PolyBench Jacobi 2D stencil benchmark
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA  (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
#
# Usage:
#   make                   # auto-detect platform
#   make PLATFORM=cuda
#   make PLATFORM=rocm
#   make clean

# ---------------------------------------------------------------------------
# Auto-detect platform
# ---------------------------------------------------------------------------
PLATFORM ?= auto
ifeq ($(PLATFORM),auto)
  ifneq ($(wildcard /opt/rocm),)
    override PLATFORM := rocm
  else
    override PLATFORM := cuda
  endif
endif

# ---------------------------------------------------------------------------
# Platform-specific settings
# ---------------------------------------------------------------------------
TARGET := polybench_jacobi2d

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := polybench_jacobi2d.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := polybench_jacobi2d.hip
  CFLAGS := -O3 -std=c++14

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) jacobi_reference.h ../../../common/bench_timing.h ../../../common/bench_warmup.h ../../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/jacobi_original_gpu.hip $(SRC) jacobi_reference.h ../../../tests/fixtures/polybench_jacobi_original.h ../../../tests/fixtures/polybench_jacobi_original_gpu.h ../../../common/bench_timing.h ../../../common/bench_warmup.h ../../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test --n 65 --tsteps 20

clean:
	rm -f $(TARGET) bench_output_test
