# Makefile for tier3/cufft_2d — 2D complex FFT via cuFFT / hipFFT / Accelerate vDSP
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
TARGET := cufft_2d

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cufft_2d.cu
  CFLAGS := -O3 -std=c++14 -arch=native
  LIBS   := -lcufft

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := cufft_2d.hip
  CFLAGS := -O3 -std=c++14 -DUSE_HIPFFT
  LIBS   := -lhipfft

else ifeq ($(PLATFORM),metal)
  CC     := clang++
  SRC    := cufft_2d_metal.mm
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
