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
LDFLAGS :=

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cudnn_batchnorm.cu
  CFLAGS := -O3 -std=c++14

  # Optional override for non-standard installs.
  # If CUDNN_PATH is set, search only under this path.
  ifneq ($(strip $(CUDNN_PATH)),)
    CUDNN_INCLUDE_CANDIDATES := $(CUDNN_PATH) $(CUDNN_PATH)/include
    CUDNN_LIB_CANDIDATES     := $(CUDNN_PATH) $(CUDNN_PATH)/lib $(CUDNN_PATH)/lib64 $(CUDNN_PATH)/lib/x64
  else
    CUDA_ROOT_CANDIDATES     := $(strip $(CUDA_HOME) $(CUDA_PATH) /usr/local/cuda)
    CUDNN_INCLUDE_CANDIDATES := /usr/include /usr/include/x86_64-linux-gnu $(addsuffix /include,$(CUDA_ROOT_CANDIDATES))
    CUDNN_LIB_CANDIDATES     := /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib $(addsuffix /lib64,$(CUDA_ROOT_CANDIDATES)) $(addsuffix /targets/x86_64-linux/lib,$(CUDA_ROOT_CANDIDATES))
  endif

  CUDNN_HEADER_FILE := $(firstword $(foreach d,$(CUDNN_INCLUDE_CANDIDATES),$(wildcard $(d)/cudnn.h)))
  CUDNN_LIB_FILE    := $(firstword $(foreach d,$(CUDNN_LIB_CANDIDATES),$(wildcard $(d)/libcudnn.so) $(wildcard $(d)/libcudnn.so.*) $(wildcard $(d)/libcudnn_static.a)))

  CUDNN_INCLUDE_DIR := $(patsubst %/cudnn.h,%,$(CUDNN_HEADER_FILE))
  CUDNN_LIB_DIR     := $(patsubst %/,%,$(dir $(CUDNN_LIB_FILE)))

  ifneq ($(CUDNN_INCLUDE_DIR),)
    CFLAGS += -I$(CUDNN_INCLUDE_DIR)
  endif
  ifneq ($(CUDNN_LIB_DIR),)
    LDFLAGS += -L$(CUDNN_LIB_DIR)
  endif

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
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean check-deps

all: check-deps $(TARGET)

check-deps:
ifeq ($(PLATFORM),cuda)
	@if [ -z "$(CUDNN_HEADER_FILE)" ]; then \
		echo "ERROR: $(TARGET) requires cuDNN, but cudnn.h was not found." 1>&2; \
		echo "       searched include paths: $(CUDNN_INCLUDE_CANDIDATES)" 1>&2; \
		echo "       hint: install cuDNN or set CUDNN_PATH=<cuDNN root>." 1>&2; \
		exit 2; \
	fi
	@if [ -z "$(CUDNN_LIB_FILE)" ]; then \
		echo "ERROR: $(TARGET) requires cuDNN, but libcudnn was not found." 1>&2; \
		echo "       searched library paths: $(CUDNN_LIB_CANDIDATES)" 1>&2; \
		echo "       hint: install cuDNN or set CUDNN_PATH=<cuDNN root>." 1>&2; \
		exit 2; \
	fi
endif

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< $(LDFLAGS) $(LIBS) -o $@

clean:
	rm -f $(TARGET)
