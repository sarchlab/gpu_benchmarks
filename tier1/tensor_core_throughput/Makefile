# Makefile for tensor_core_throughput microbenchmark
#
# Measures Tensor Core throughput (TOPS) using WMMA API on CUDA/ROCm
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
TARGET := tensor_core_throughput

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := tensor_core_throughput.hip
  CUDA_ARCH ?= native
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu -DUSE_CUDA

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := tensor_core_throughput.hip
  CFLAGS  := -O3 -std=c++17 -DUSE_ROCM

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
