# Makefile for tensor_core_throughput microbenchmark
#
# Measures Tensor Core throughput (TOPS) using WMMA API on CUDA/ROCm
# and simdgroup_matrix_multiply_accumulate on Metal.
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
#   PLATFORM=metal  — Apple Metal (default on Darwin)
#
# Usage:
#   make                        # auto-detect platform
#   make PLATFORM=cuda
#   make PLATFORM=rocm
#   make PLATFORM=metal
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
TARGET := tensor_core_throughput

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := tensor_core_throughput.hip
  CUDA_ARCH ?= sm_90
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu -DUSE_CUDA

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := tensor_core_throughput.hip
  CFLAGS  := -O3 -std=c++17 -DUSE_ROCM

else ifeq ($(PLATFORM),metal)
  CC      := clang++
  SRC     := tensor_core_throughput_metal.mm
  CFLAGS  := -O2 -std=c++17 \
             -framework Metal \
             -framework Foundation \
             -framework MetalKit \
             -fobjc-arc

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda, rocm, or metal.)
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
