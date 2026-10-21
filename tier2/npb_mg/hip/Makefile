# Makefile for tier2/npb_mg — NPB Multi-Grid
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
TARGET := npb_mg

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := npb_mg.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -Xcompiler -pthread -x cu

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := npb_mg.hip
  CFLAGS := -O3 -std=c++14 -pthread

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

COMMON := nas_mg.h mg_reference.h mg_kernels.h ../../../common/bench_timing.h ../../../common/bench_warmup.h ../../../common/hip_cuda_compat.h

$(TARGET): $(SRC) $(COMMON)
	$(CC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/algorithmic_zero_output.hip $(SRC) $(COMMON)
	$(CC) $(CFLAGS) -DTEST_MG_ZERO $< -o $@

test: bench_output_test
	./bench_output_test

clean:
	rm -f $(TARGET) bench_output_test
