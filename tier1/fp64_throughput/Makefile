# Makefile for fp64_throughput microbenchmark
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
#
#       measures FP32 FMA throughput as a proxy.  See README.md for details.
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
TARGET := fp64_throughput

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := fp64_throughput.hip
  CUDA_ARCH ?= native
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := fp64_throughput.hip
  CFLAGS  := -O3 -std=c++14

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
