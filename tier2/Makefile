# Tier 2 — Application Benchmarks
# Each benchmark keeps one source tree per platform:
#   <bench>/hip/   original HIP sources (hipcc, or nvcc via hip_cuda_compat.h)
#   <bench>/cuda/  native CUDA port (nvcc)
# PLATFORM=cuda builds every cuda/ tree (falling back to hip/ for a benchmark
# without one); PLATFORM=rocm builds every hip/ tree. This matches the
# variant gpubench selects for each platform.

PLATFORM ?= auto
ifeq ($(PLATFORM),auto)
  ifneq ($(wildcard /opt/rocm),)
    override PLATFORM := rocm
  else
    override PLATFORM := cuda
  endif
endif

BENCHES  := $(sort $(patsubst %/,%,$(dir $(wildcard */hip/Makefile */cuda/Makefile))))
BENCHES  := $(sort $(patsubst %/hip,%,$(patsubst %/cuda,%,$(BENCHES))))

ifeq ($(PLATFORM),rocm)
  SUBDIRS := $(foreach b,$(BENCHES),$(if $(wildcard $(b)/hip/Makefile),$(b)/hip))
else
  SUBDIRS := $(foreach b,$(BENCHES),$(if $(wildcard $(b)/cuda/Makefile),$(b)/cuda,$(b)/hip))
endif

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ PLATFORM=$(PLATFORM)

clean:
	for b in $(SUBDIRS); do \
		$(MAKE) -C $$b clean PLATFORM=$(PLATFORM); \
	done
