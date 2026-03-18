# Tier 2 — Application Benchmarks
# Invokes make in each benchmark subdirectory that has a Makefile.

# Pass PLATFORM down to sub-makes
PLATFORM ?= auto

SUBDIRS := $(patsubst %/Makefile,%,$(wildcard */Makefile))

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ PLATFORM=$(PLATFORM)

clean:
	for b in $(SUBDIRS); do \
		$(MAKE) -C $$b clean PLATFORM=$(PLATFORM); \
	done
