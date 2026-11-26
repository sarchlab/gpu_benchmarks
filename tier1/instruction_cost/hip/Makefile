# Makefile for instruction_cost microbenchmark
#
# The timed kernels are hand-written gfx942 assembly, so this benchmark builds
# on AMD ROCm only. PLATFORM=cuda builds nothing.
#
# Usage:
#   make                        # auto-detect platform
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
# Settings
# ---------------------------------------------------------------------------
TARGET  := instruction_cost
SRC     := instruction_cost.hip
KERNELS := instruction_cost_kernels

CC     := hipcc
CFLAGS := -O3 -std=c++14

# The kernels are assembled and linked with the LLVM tools ROCm ships.
ROCM_PATH ?= /opt/rocm
ROCM_LLVM ?= $(ROCM_PATH)/lib/llvm/bin

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all clean

ifeq ($(PLATFORM),rocm)
all: $(TARGET)

# One rule: assemble the kernels into a code object ($(KERNELS).hsaco), wrap
# it in a host object, and link it into the host program, which loads it with
# hipModuleLoadData. The object goes straight to the linker (-Wl) because
# hipcc treats every input file as HIP source.
$(TARGET): $(SRC) $(KERNELS).S $(KERNELS).h ../../../common_hip/bench_timing.h ../../../common_hip/bench_warmup.h ../../../common_hip/hip_cuda_compat.h
	$(ROCM_LLVM)/clang -x assembler-with-cpp --target=amdgcn-amd-amdhsa -mcpu=gfx942 -c $(KERNELS).S -o $(KERNELS).o
	$(ROCM_LLVM)/ld.lld -shared $(KERNELS).o -o $(KERNELS).hsaco
	ld -r -b binary -z noexecstack $(KERNELS).hsaco -o $(KERNELS)_blob.o
	$(CC) $(CFLAGS) $(SRC) -Wl,$(KERNELS)_blob.o -o $@
else
all $(TARGET):
	@echo "instruction_cost: gfx942 assembly kernels, ROCm only; nothing to build for PLATFORM=$(PLATFORM)"
endif

clean:
	rm -f $(TARGET) $(KERNELS).o $(KERNELS).hsaco $(KERNELS)_blob.o
