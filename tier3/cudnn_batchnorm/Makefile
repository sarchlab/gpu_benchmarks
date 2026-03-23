# Makefile for tier3/cudnn_batchnorm — Batch Normalization via cuDNN / MIOpen / MPS
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA + cuDNN (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP + MIOpen (default when /opt/rocm exists)
#   PLATFORM=metal  — Apple Metal + MPS  (default on Darwin)
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
TARGET := cudnn_batchnorm

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cudnn_batchnorm.cu
  CFLAGS := -O3 -std=c++14
  LIBS   := -lcudnn

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := cudnn_batchnorm.hip
  CFLAGS := -O3 -std=c++14 -DUSE_MIOPEN
  LIBS   := -lMIOpen

else ifeq ($(PLATFORM),metal)
  CC     := clang++
  SRC    := cudnn_batchnorm_metal.mm
  CFLAGS := -O2 -std=c++17 -fobjc-arc \
             -framework Metal \
             -framework MetalPerformanceShaders \
             -framework Foundation
  LIBS   :=

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda, rocm, or metal.)
endif

# ---------------------------------------------------------------------------
# cuDNN availability check (CUDA platform only)
# ---------------------------------------------------------------------------
ifeq ($(PLATFORM),cuda)
  CUDNN_FOUND := $(or $(wildcard /usr/include/cudnn.h),$(wildcard /usr/local/cuda/include/cudnn.h),$(wildcard $(CUDA_HOME)/include/cudnn.h))
  ifeq ($(CUDNN_FOUND),)
    $(error cuDNN not found. Install nvidia-cuda-toolkit with cuDNN support or set CUDA_HOME/CUDNN_PATH to your cuDNN installation)
  endif
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
