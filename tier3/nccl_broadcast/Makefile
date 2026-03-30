# Makefile for tier3/nccl_broadcast — Broadcast benchmark using NCCL
#
# NCCL is NVIDIA-specific; only CUDA is supported.
#
# Usage:
#   make
#   make clean

TARGET := nccl_broadcast
CC     := nvcc
SRC    := nccl_broadcast.cu
CFLAGS := -O3 -std=c++14 -arch=native
LIBS   := -lnccl

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	rm -f $(TARGET)
