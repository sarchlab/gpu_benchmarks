# Makefile for tier3/cutlass_gemm — CUTLASS-style tiled FP32 GEMM
#
# Custom tiled GEMM kernel with shared-memory staging and register tiling.
# Self-contained — no external library dependencies.
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA  (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
#   PLATFORM=metal  — Apple Metal  (default on Darwin)
#
# Usage:
#   make                   # auto-detect platform
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
TARGET := cutlass_gemm

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cutlass_gemm.cu
  CFLAGS := -O3 -std=c++14
  LIBS   :=

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := cutlass_gemm.hip
  CFLAGS := -O3 -std=c++14 -DUSE_HIP
  LIBS   :=

else ifeq ($(PLATFORM),metal)
  CC     := clang++
  SRC    := cutlass_gemm_metal.mm
  CFLAGS := -O2 -std=c++17 -fobjc-arc \
             -framework Metal \
             -framework Foundation
  LIBS   :=

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda, rocm, or metal.)
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
