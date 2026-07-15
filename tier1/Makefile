# Tier 1 — Microbenchmarks
# Invokes make in each benchmark subdirectory.

BENCHMARKS := memory_bandwidth fp32_throughput fp64_throughput fp16_throughput int32_throughput cache_latency cache_bandwidth pcie_bandwidth mixbench shared_mem_bandwidth atomic_operations warp_shuffle empty_kernel

# Pass PLATFORM down to sub-makes
PLATFORM ?= auto

.PHONY: all clean $(BENCHMARKS)

all: $(BENCHMARKS)

$(BENCHMARKS):
	$(MAKE) -C $@ PLATFORM=$(PLATFORM)

clean:
	for b in $(BENCHMARKS); do \
		$(MAKE) -C $$b clean PLATFORM=$(PLATFORM); \
	done
