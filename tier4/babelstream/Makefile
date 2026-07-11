PLATFORM ?= auto
ifeq ($(PLATFORM),auto)
  ifneq ($(wildcard /opt/rocm),)
    override PLATFORM := rocm
  else
    override PLATFORM := cuda
  endif
endif

TARGET := babelstream

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := babelstream.cu
  CUDA_ARCH ?= native
  CFLAGS := -O3 -arch=$(CUDA_ARCH)
  LIBS   :=
else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := babelstream.hip
  CFLAGS := -O3 -DUSE_HIP
  LIBS   :=
else
  $(error Unknown PLATFORM=$(PLATFORM))
endif

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
