# Makefile for tier2/polybench_2mm — Two Matrix Multiplications (PolyBench 2MM)
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
TARGET := polybench_2mm

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := polybench_2mm.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := polybench_2mm.hip
  CFLAGS := -O3 -std=c++14

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

COMMON := ../../../common/bench_timing.h ../../../common/bench_warmup.h ../../../common/hip_cuda_compat.h

$(TARGET): $(SRC) cpu_reference.h $(COMMON)
	$(CC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/two_mm_output_test.hip $(SRC) cpu_reference.h $(COMMON)
	$(CC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test

clean:
	rm -f $(TARGET) bench_output_test
