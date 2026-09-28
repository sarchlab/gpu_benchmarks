# Local global-memory microbenchmarks (historical shoc_devicememory ID)
#
# Supported platforms:
#   PLATFORM=cuda   — NVIDIA CUDA  (default on Linux without ROCm)
#   PLATFORM=rocm   — AMD ROCm/HIP (default when /opt/rocm exists)
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
TARGET := shoc_devicememory

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := shoc_devicememory.hip
  CUDA_ARCH ?= native
  CFLAGS := -O3 -std=c++14 -arch=$(CUDA_ARCH) -x cu

else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := shoc_devicememory.hip
  CFLAGS := -O3 -std=c++14

else
  $(error Unknown PLATFORM=$(PLATFORM). Use cuda or rocm.)
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) cpu_reference.h ../../common/bench_timing.h ../../common/bench_warmup.h ../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

bench_output_test: ../../tests/devicememory_baseline_gpu.hip ../../tests/fixtures/devicememory_baseline.hip $(SRC) cpu_reference.h ../../common/bench_timing.h ../../common/bench_warmup.h ../../common/hip_cuda_compat.h
	$(CC) $(CFLAGS) $< -o $@

test: bench_output_test
	./bench_output_test --array_size_mb 1

clean:
	rm -f $(TARGET) bench_output_test
