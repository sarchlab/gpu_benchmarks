# Tier 5 — ML Workload Benchmarks (PyTorch)
# Invokes make in each benchmark subdirectory that has a Makefile.

PLATFORM ?= auto
ifeq ($(PLATFORM),auto)
  ifneq ($(wildcard /opt/rocm),)
    override PLATFORM := rocm
  else
    override PLATFORM := cuda
  endif
endif

SUBDIRS := $(patsubst %/Makefile,%,$(wildcard */Makefile))

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ PLATFORM=$(PLATFORM)

clean:
	for b in $(SUBDIRS); do \
		$(MAKE) -C $$b clean PLATFORM=$(PLATFORM); \
	done
