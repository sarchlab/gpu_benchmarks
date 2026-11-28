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
# One code object per file. instruction_cost_kernels is the code object of
# GPU Benchmarks v1.2, unchanged; the bodies added since are in the other.
KERNELS := instruction_cost_kernels instruction_cost_other_kernels
COMMON  := ../../../common_hip

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

# One rule: assemble each kernel file into a code object (<file>.hsaco), wrap
# it in a host object, and link the objects into the host program, which loads
# one of them with hipModuleLoadData. The objects go straight to the linker
# (-Wl) because hipcc treats every input file as HIP source.
$(TARGET): $(SRC) $(KERNELS:=.S) instruction_cost_kernels.h $(COMMON)/gfx942_loop_kernels.h $(COMMON)/gfx942_loop_host.h $(COMMON)/bench_timing.h $(COMMON)/bench_warmup.h $(COMMON)/hip_cuda_compat.h
	set -e; for k in $(KERNELS); do \
		$(ROCM_LLVM)/clang -x assembler-with-cpp --target=amdgcn-amd-amdhsa -mcpu=gfx942 -c $$k.S -o $$k.o; \
		$(ROCM_LLVM)/ld.lld -shared $$k.o -o $$k.hsaco; \
		ld -r -b binary -z noexecstack $$k.hsaco -o $${k}_blob.o; \
	done
	$(CC) $(CFLAGS) $(SRC) $(KERNELS:%=-Wl,%_blob.o) -o $@
else
all $(TARGET):
	@echo "instruction_cost: gfx942 assembly kernels, ROCm only; nothing to build for PLATFORM=$(PLATFORM)"
endif

clean:
	rm -f $(TARGET) $(KERNELS:=.o) $(KERNELS:=.hsaco) $(KERNELS:=_blob.o)
