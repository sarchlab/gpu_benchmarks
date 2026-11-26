# Makefile for code_fetch_latency microbenchmark
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
#
# Usage:
#   make                        # auto-detect platform
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
TARGET := code_fetch_latency

ifeq ($(PLATFORM),cuda)
  CC      := nvcc
  SRC     := code_fetch_latency.hip
  CUDA_ARCH ?= native
  CFLAGS  := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC      := hipcc
  SRC     := code_fetch_latency.hip
  CFLAGS  := -O3 -std=c++14
  # Clang's loop passes recurse once per chained operation, so the largest
  # kernels overflow the default 8 MB compiler stack: raise it (4 GB, else the
  # hard limit) for this compile. nvcc compiles them within the default.
  RAISE_STACK := ulimit -s 4194304 2>/dev/null || ulimit -s `ulimit -H -s` 2>/dev/null || true;

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(wildcard ../../../common_hip/*.h)
	$(RAISE_STACK) $(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
