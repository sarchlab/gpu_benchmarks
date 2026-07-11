#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA with cuSPARSE  (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP with rocSPARSE (default when /opt/rocm exists)
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
TARGET := cusparse_spmv

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := cusparse_spmv.cu
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH)
  LIBS   := -lcusparse

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := cusparse_spmv.hip
  CFLAGS := -O3 -std=c++14 -DUSE_ROCSPARSE
  LIBS   := -lrocsparse

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
