# Makefile for tier3/cublas_dgemm — FP64 GEMM benchmark
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA  (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
#   PLATFORM=metal  — Apple (uses Accelerate cblas_dgemm, CPU-only; Metal has no FP64)
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
TARGET := cublas_dgemm

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cublas_dgemm.cu
  CFLAGS := -O3 -std=c++14
  LIBS   := -lcublas

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := cublas_dgemm.hip
  CFLAGS := -O3 -std=c++14 -DUSE_ROCBLAS
  LIBS   := -lrocblas

else ifeq ($(PLATFORM),metal)
  CC     := clang++
  SRC    := cublas_dgemm_metal.mm
  CFLAGS := -O2 -std=c++17 -fobjc-arc \
             -framework Foundation \
             -framework Accelerate
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
