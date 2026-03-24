# Makefile for tier3/cudnn_conv — Forward 2D Convolution via cuDNN / MIOpen / MPS
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
TARGET := cudnn_conv
LDFLAGS :=

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cudnn_conv.cu
  CFLAGS := -O3 -std=c++14

  UNAME_M := $(shell uname -m)
  ifeq ($(UNAME_M),x86_64)
    CUDA_TARGET_SUBDIRS    := x86_64-linux aarch64-linux
    CUDNN_ARCH_INCLUDE_DIRS := /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu
    CUDNN_ARCH_LIB_DIRS     := /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  else ifneq ($(filter aarch64 arm64,$(UNAME_M)),)
    CUDA_TARGET_SUBDIRS    := aarch64-linux x86_64-linux
    CUDNN_ARCH_INCLUDE_DIRS := /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
    CUDNN_ARCH_LIB_DIRS     := /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
  else
    CUDA_TARGET_SUBDIRS    := x86_64-linux aarch64-linux
    CUDNN_ARCH_INCLUDE_DIRS := /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu
    CUDNN_ARCH_LIB_DIRS     := /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  endif

  # Optional override for non-standard installs.
  # If CUDNN_PATH is set, search only under this path.
  ifneq ($(strip $(CUDNN_PATH)),)
    CUDNN_INCLUDE_CANDIDATES := \
      $(CUDNN_PATH) \
      $(CUDNN_PATH)/include \
      $(foreach t,$(CUDA_TARGET_SUBDIRS),$(CUDNN_PATH)/targets/$(t)/include)
    CUDNN_LIB_CANDIDATES := \
      $(CUDNN_PATH) \
      $(CUDNN_PATH)/lib \
      $(CUDNN_PATH)/lib64 \
      $(CUDNN_PATH)/lib/x64 \
      $(CUDNN_PATH)/lib/x86_64-linux-gnu \
      $(CUDNN_PATH)/lib/aarch64-linux-gnu \
      $(foreach t,$(CUDA_TARGET_SUBDIRS),$(CUDNN_PATH)/targets/$(t)/lib)
  else
    CUDA_ROOT_CANDIDATES := $(strip $(CUDA_HOME) $(CUDA_PATH) /usr/local/cuda)
    CUDNN_INCLUDE_CANDIDATES := \
      /usr/include \
      $(CUDNN_ARCH_INCLUDE_DIRS) \
      $(addsuffix /include,$(CUDA_ROOT_CANDIDATES)) \
      $(foreach t,$(CUDA_TARGET_SUBDIRS),$(addsuffix /targets/$(t)/include,$(CUDA_ROOT_CANDIDATES)))
    CUDNN_LIB_CANDIDATES := \
      $(CUDNN_ARCH_LIB_DIRS) \
      /usr/lib64 \
      /usr/lib \
      $(addsuffix /lib64,$(CUDA_ROOT_CANDIDATES)) \
      $(foreach t,$(CUDA_TARGET_SUBDIRS),$(addsuffix /targets/$(t)/lib,$(CUDA_ROOT_CANDIDATES)))
  endif

  CUDNN_HEADER_FILE := $(firstword $(foreach d,$(CUDNN_INCLUDE_CANDIDATES),$(wildcard $(d)/cudnn.h)))
  CUDNN_LIB_FILE    := $(firstword $(foreach d,$(CUDNN_LIB_CANDIDATES),$(wildcard $(d)/libcudnn.so) $(wildcard $(d)/libcudnn.so.*) $(wildcard $(d)/libcudnn_static.a)))
  CUDNN_AVAILABLE   := $(if $(and $(CUDNN_HEADER_FILE),$(CUDNN_LIB_FILE)),1,0)

  CUDNN_INCLUDE_DIR := $(patsubst %/cudnn.h,%,$(CUDNN_HEADER_FILE))
  CUDNN_LIB_DIR     := $(patsubst %/,%,$(dir $(CUDNN_LIB_FILE)))

  ifeq ($(CUDNN_AVAILABLE),1)
    CFLAGS += -I$(CUDNN_INCLUDE_DIR)
    LDFLAGS += -L$(CUDNN_LIB_DIR)
    LIBS   := -lcudnn
  else
    LIBS   :=
  endif

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := cudnn_conv.hip
  CFLAGS := -O3 -std=c++14 -DUSE_MIOPEN
  LIBS   := -lMIOpen

else ifeq ($(PLATFORM),metal)
  CC     := clang++
  SRC    := cudnn_conv_metal.mm
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
ifeq ($(CUDNN_AVAILABLE),1)
	@echo "INFO: $(TARGET) using cuDNN header: $(CUDNN_HEADER_FILE)"
	@echo "INFO: $(TARGET) using cuDNN library: $(CUDNN_LIB_FILE)"
else
	@echo "WARNING: $(TARGET) cuDNN not found; building diagnostic fallback executable."
	@echo "         searched include paths: $(CUDNN_INCLUDE_CANDIDATES)"
	@echo "         searched library paths: $(CUDNN_LIB_CANDIDATES)"
	@echo "         hint: install cuDNN or set CUDNN_PATH=<cuDNN root>."
endif
endif

$(TARGET): $(SRC)
ifeq ($(PLATFORM),cuda)
ifeq ($(CUDNN_AVAILABLE),1)
	$(CC) $(CFLAGS) $< $(LDFLAGS) $(LIBS) -o $@
else
	@echo "WARNING: $(TARGET) fallback executable created (cuDNN unavailable)."
	@printf '%s\n' '#!/usr/bin/env bash' > $@
	@printf '%s\n' 'echo "$(TARGET): built without cuDNN support (diagnostic fallback)."' >> $@
	@printf '%s\n' 'echo "Install cuDNN and rebuild. Optional hint: CUDNN_PATH=<cuDNN root>."' >> $@
	@printf '%s\n' 'exit 1' >> $@
	@chmod +x $@
endif
else
	$(CC) $(CFLAGS) $< $(LDFLAGS) $(LIBS) -o $@
endif

clean:
	rm -f $(TARGET)
