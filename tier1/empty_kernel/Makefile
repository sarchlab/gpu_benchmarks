# Makefile for empty_kernel launch-overhead microbenchmark
#
# Measures kernel launch / dispatch / grid-scheduling overhead (empty kernel)
# across grid geometry (num_blocks x block_size).
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
ifndef PLATFORM
  ifeq ($(shell uname -s),Darwin)
    PLATFORM := metal
  else ifneq ($(wildcard /opt/rocm),)
    PLATFORM := rocm
  else
    PLATFORM := cuda
  endif
endif

# ---------------------------------------------------------------------------
# Platform-specific settings
# ---------------------------------------------------------------------------
TARGET := empty_kernel

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := empty_kernel.hip
  CUDA_ARCH ?= sm_90
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := empty_kernel.hip
  CFLAGS  := -O3 -std=c++14

else
  $(error PLATFORM=$(PLATFORM) not supported by empty_kernel. Use cuda or rocm.)
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
