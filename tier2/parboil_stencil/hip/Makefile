# Makefile for tier2/parboil_stencil — Parboil 3D Stencil benchmark
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
TARGET := parboil_stencil

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := parboil_stencil.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := parboil_stencil.hip
  CFLAGS := -O3 -std=c++14

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

COMMON := ../../../common_hip/bench_timing.h ../../../common_hip/bench_warmup.h ../../../common_hip/hip_cuda_compat.h

$(TARGET): $(SRC) cpu_reference.h $(COMMON)
	$(CC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/stencil_output_test.hip $(SRC) cpu_reference.h $(COMMON)
	$(CC) $(CFLAGS) $< -o $@

test: bench_output_test
	BENCH_PARAM_vector_add_warmup=false ./bench_output_test

clean:
	rm -f $(TARGET) bench_output_test
