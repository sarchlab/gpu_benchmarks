# Makefile for tier2/parboil_cutcp — Coulombic Potential with cutoff
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
TARGET := parboil_cutcp

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := parboil_cutcp.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu -Xcompiler -pthread

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := parboil_cutcp.hip
  CFLAGS := -O3 -std=c++14 -pthread

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) cpu_reference.h ../../common/bench_timing.h ../../common/bench_warmup.h ../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
