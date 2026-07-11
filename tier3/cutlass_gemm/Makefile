# Makefile for tier3/cutlass_gemm — CUTLASS-style tiled FP32 GEMM
#
# Custom tiled GEMM kernel with shared-memory staging and register tiling.
# Self-contained — no external library dependencies.
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
TARGET := cutlass_gemm

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cutlass_gemm.cu
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH)
  LIBS   :=

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := cutlass_gemm.hip
  CFLAGS := -O3 -std=c++14 -DUSE_HIP
  LIBS   :=

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	rm -f $(TARGET)
