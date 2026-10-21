# Makefile for tier2/altis_cfd — Computational Fluid Dynamics (Euler solver)
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
TARGET := altis_cfd

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := altis_cfd.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu -Xcompiler -pthread

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := altis_cfd.hip
  CFLAGS := -O3 -std=c++14 -pthread

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) cfd_reference.h cfd_kernels.h ../../../common/bench_timing.h ../../../common/bench_warmup.h ../../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

bench_output_test: ../../../tests/cfd_upstream_gpu.hip ../../../tests/fixtures/cfd_upstream.h $(SRC) cfd_reference.h cfd_kernels.h ../../../common/bench_timing.h ../../../common/bench_warmup.h ../../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test --size 503

clean:
	rm -f $(TARGET) bench_output_test
