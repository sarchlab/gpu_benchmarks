ifndef PLATFORM
  ifeq ($(shell uname -s),Darwin)
    PLATFORM := metal
  else ifneq ($(wildcard /opt/rocm),)
    PLATFORM := rocm
  else
    PLATFORM := cuda
  endif
endif

LLAMA_DIR := llama.cpp
MODEL_DIR := models
MODEL_URL := https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M.Q4_K_M.gguf

.PHONY: all run setup clean

all: run

run:
ifeq ($(PLATFORM),metal)
	python3 bench_mps.py
else
	python3 bench.py
endif

setup:
	@echo "=== Cloning llama.cpp ==="
	if [ ! -d "$(LLAMA_DIR)" ]; then \
		git clone --depth 1 https://github.com/ggerganov/llama.cpp.git $(LLAMA_DIR); \
	fi
	@echo "=== Building llama.cpp ==="
	cd $(LLAMA_DIR) && mkdir -p build && cd build && cmake .. && cmake --build . --config Release -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@echo "=== Downloading model ==="
	mkdir -p $(MODEL_DIR)
	if [ ! -f "$(MODEL_DIR)/SmolLM2-135M.Q4_K_M.gguf" ]; then \
		curl -L -o $(MODEL_DIR)/SmolLM2-135M.Q4_K_M.gguf $(MODEL_URL); \
	fi
	@echo "=== Setup complete ==="

clean:
	rm -rf $(LLAMA_DIR) $(MODEL_DIR)
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
