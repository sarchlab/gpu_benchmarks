# Makefile for shared_mem_bandwidth microbenchmark
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
#
# Usage:
#   make                        # auto-detect platform
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
TARGET := shared_mem_bandwidth

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := shared_mem_bandwidth.hip
  CUDA_ARCH ?= native
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := shared_mem_bandwidth.hip
  CFLAGS  := -O3 -std=c++14

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test test-pattern

all: $(TARGET)

$(TARGET): $(SRC) shared_mem_pattern.h ../../../common_hip/bench_timing.h ../../../common_hip/bench_warmup.h ../../../common_hip/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

HOST_CXX ?= c++
bench_pattern_test: ../../../tests/shared_mem_pattern_test.cpp shared_mem_pattern.h
	$(HOST_CXX) -O2 -std=c++14 $< -o $@

test-pattern: bench_pattern_test
	./bench_pattern_test

bench_output_test: ../../../tests/shared_mem_output_test.hip $(SRC) shared_mem_pattern.h ../../../common_hip/bench_timing.h ../../../common_hip/bench_warmup.h ../../../common_hip/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

test: test-pattern bench_output_test
	./bench_output_test

clean:
	rm -f $(TARGET) bench_pattern_test bench_output_test
