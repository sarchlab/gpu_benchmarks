# Makefile for tier3/nccl_sendrecv — collective benchmark
#
# Uses NCCL on NVIDIA/CUDA and RCCL (the API-compatible AMD equivalent) on ROCm.
#
# Usage:
#   make                  # auto-detect platform
#   make PLATFORM=cuda
#   make PLATFORM=rocm
#   make clean

# ---------------------------------------------------------------------------
# Auto-detect platform
# ---------------------------------------------------------------------------
ifndef PLATFORM
  ifneq ($(wildcard /opt/rocm),)
    PLATFORM := rocm
  else
    PLATFORM := cuda
  endif
endif

TARGET := nccl_sendrecv

ifeq ($(PLATFORM),cuda)
  CC        := nvcc
  SRC       := nccl_sendrecv.cu
  CUDA_ARCH ?= native
  CFLAGS    := -O3 -std=c++14 -arch=$(CUDA_ARCH)
  LIBS      := -lnccl

else ifeq ($(PLATFORM),rocm)
  CC        := hipcc
  SRC       := nccl_sendrecv.hip
  CFLAGS    := -O3 -std=c++14
  LIBS      := -lrccl

else
  $(error NCCL/RCCL collective benchmarks support PLATFORM=cuda or rocm only (got $(PLATFORM)))
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	rm -f $(TARGET)
