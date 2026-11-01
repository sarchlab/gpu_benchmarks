# Makefile for cache_bandwidth microbenchmark
#
# Measures per-cache-level read bandwidth (L1/L2/last-level/DRAM) in two modes:
#   mode=shared  — one buffer read by all blocks (global / device-wide view)
#   mode=private — each block reads its own buffer (per-unit view)
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
TARGET := cache_bandwidth

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := cache_bandwidth.hip
  CUDA_ARCH ?= native
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := cache_bandwidth.hip
  CFLAGS  := -O3 -std=c++14

else
  $(error PLATFORM=$(PLATFORM) not supported by cache_bandwidth. Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) ../../../common_hip/bench_timing.h ../../../common_hip/bench_warmup.h ../../../common_hip/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
