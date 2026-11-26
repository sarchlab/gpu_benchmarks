# Tier 1 — Microbenchmarks
# Each benchmark keeps one source tree per platform:
#   <bench>/hip/   original HIP sources (hipcc, or nvcc via hip_cuda_compat.h)
#   <bench>/cuda/  native CUDA port (nvcc)
# PLATFORM=cuda builds every cuda/ tree (falling back to hip/ for a benchmark
# without one); PLATFORM=rocm builds every hip/ tree. This matches the
# variant gpubench selects for each platform.

BENCHMARKS := memory_bandwidth fp32_throughput fp64_throughput fp16_throughput int32_throughput cache_latency code_fetch_latency cache_bandwidth pcie_bandwidth mixbench shared_mem_bandwidth atomic_operations warp_shuffle empty_kernel

PLATFORM ?= auto
ifeq ($(PLATFORM),auto)
  ifneq ($(wildcard /opt/rocm),)
    override PLATFORM := rocm
  else
    override PLATFORM := cuda
  endif
endif

ifeq ($(PLATFORM),rocm)
  SUBDIRS := $(foreach b,$(BENCHMARKS),$(b)/hip)
else
  SUBDIRS := $(foreach b,$(BENCHMARKS),$(if $(wildcard $(b)/cuda/Makefile),$(b)/cuda,$(b)/hip))
endif

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ PLATFORM=$(PLATFORM)

clean:
	for b in $(SUBDIRS); do \
		$(MAKE) -C $$b clean PLATFORM=$(PLATFORM); \
	done
