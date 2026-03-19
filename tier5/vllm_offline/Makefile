ifndef PLATFORM
  ifeq ($(shell uname -s),Darwin)
    PLATFORM := metal
  else ifneq ($(wildcard /opt/rocm),)
    PLATFORM := rocm
  else
    PLATFORM := cuda
  endif
endif

.PHONY: all run setup clean

all: run

run:
ifeq ($(PLATFORM),metal)
	python3 bench_mps.py
else
	python3 bench.py
endif

setup:
	pip3 install vllm

clean:
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
