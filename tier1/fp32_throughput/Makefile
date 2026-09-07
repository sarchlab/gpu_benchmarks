# Makefile for fp32_throughput microbenchmark
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
TARGET := fp32_throughput

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := fp32_throughput.hip
  CUDA_ARCH ?= native
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := fp32_throughput.hip
  CFLAGS  := -O3 -std=c++14

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

# Optional CUDA/ROCm device test of the production kernel's per-thread output.
bench_output_test: ../../tests/fma_output_test.hip $(SRC) ../../common/bench_timing.h ../../common/bench_warmup.h ../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) -DTEST_FP32 $< -o $@

test: bench_output_test
	./bench_output_test

clean:
	rm -f $(TARGET) bench_output_test
