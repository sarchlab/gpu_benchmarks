# Makefile for tier2/rodinia_hotspot — Rodinia Hotspot benchmark
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
TARGET := rodinia_hotspot

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := rodinia_hotspot.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -Xcompiler -pthread -x cu

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := rodinia_hotspot.hip
  CFLAGS := -O3 -std=c++14 -pthread

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) hotspot_reference.h ../../../common_hip/bench_timing.h ../../../common_hip/bench_warmup.h ../../../common_hip/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/hotspot_original_gpu.hip $(SRC) hotspot_reference.h ../../../tests/upstream/hotspot_physics.h ../../../tests/upstream/hotspot_gpu.hip ../../../common_hip/bench_timing.h ../../../common_hip/bench_warmup.h ../../../common_hip/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test --grid_size 17 --total_iterations 3 --pyramid_height 7

clean:
	rm -f $(TARGET) bench_output_test
