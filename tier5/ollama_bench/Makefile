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
	@echo "=== Ollama Setup ==="
	@echo ""
	@echo "1. Install ollama:"
	@echo "   macOS:  brew install ollama"
	@echo "   Linux:  curl -fsSL https://ollama.com/install.sh | sh"
	@echo ""
	@echo "2. Start the ollama server:"
	@echo "   ollama serve"
	@echo ""
	@echo "3. Pull a model:"
	@echo "   ollama pull tinyllama"
	@echo ""
	@echo "4. Install Python dependency:"
	@echo "   pip install requests"
	@echo ""
	pip install requests

clean:
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
