# Makefile for tier3/nccl_alltoall — AllToAll benchmark using NCCL
#
# NCCL is NVIDIA-specific; only CUDA is supported.
#
# Usage:
#   make
#   make clean

TARGET := nccl_alltoall
CC     := nvcc
SRC    := nccl_alltoall.cu
CUDA_ARCH ?= sm_90
CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH)
LIBS   := -lnccl

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	rm -f $(TARGET)
