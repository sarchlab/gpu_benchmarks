# Tier 3 — Vendor Library Benchmarks
# Invokes make in each benchmark subdirectory that has a Makefile.

# Platform auto-detection (mirrors root Makefile and individual benchmark Makefiles)
PLATFORM ?= auto
ifeq ($(PLATFORM),auto)
  ifeq ($(shell uname -s),Darwin)
    PLATFORM := metal
  else ifneq ($(wildcard /opt/rocm),)
    PLATFORM := rocm
  else
    PLATFORM := cuda
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
