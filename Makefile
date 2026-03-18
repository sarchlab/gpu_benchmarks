# GPU Benchmarks — Top-Level Makefile
#
# Usage:
#   make                        # Build everything (auto-detect platform)
#   make all                    # Same as above
#   make tier1                  # Build all Tier 1 microbenchmarks
#   make clean                  # Clean all build artifacts
#   make PLATFORM=cuda          # Force CUDA/NVIDIA build
#   make PLATFORM=rocm          # Force ROCm/AMD build
#   make PLATFORM=metal         # Force Metal/Apple build
#
# Platform Detection:
#   PLATFORM=auto (default) — detects the platform at build time:
#     - Darwin   → metal
#     - /opt/rocm present → rocm
#     - Otherwise → cuda

PLATFORM ?= auto

ifeq ($(PLATFORM), auto)
  ifeq ($(shell uname), Darwin)
    PLATFORM = metal
  else ifneq ($(wildcard /opt/rocm/bin/hipcc),)
    PLATFORM = rocm
  else
    PLATFORM = cuda
  endif
endif

export PLATFORM

.PHONY: all tier1 tier2 tier3 tier4 tier5 clean

all: tier1 tier2 tier3 tier4 tier5

tier1:
	$(MAKE) -C tier1 PLATFORM=$(PLATFORM)

tier2:
	$(MAKE) -C tier2 PLATFORM=$(PLATFORM)

tier3:
	$(MAKE) -C tier3 PLATFORM=$(PLATFORM)

tier4:
	$(MAKE) -C tier4 PLATFORM=$(PLATFORM)

tier5:
	$(MAKE) -C tier5 PLATFORM=$(PLATFORM)

clean:
	$(MAKE) -C tier1 clean PLATFORM=$(PLATFORM)
	$(MAKE) -C tier2 clean PLATFORM=$(PLATFORM)
	$(MAKE) -C tier3 clean PLATFORM=$(PLATFORM)
	$(MAKE) -C tier4 clean PLATFORM=$(PLATFORM)
	$(MAKE) -C tier5 clean PLATFORM=$(PLATFORM)
