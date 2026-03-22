ifndef PLATFORM
  ifeq ($(shell uname -s),Darwin)
    PLATFORM := metal
  else ifneq ($(wildcard /opt/rocm),)
    PLATFORM := rocm
  else
    PLATFORM := cuda
  endif
endif

TARGET := babelstream

ifeq ($(PLATFORM),cuda)
  CC     := nvcc
  SRC    := babelstream.cu
  CFLAGS := -O3
  LIBS   :=
else ifeq ($(PLATFORM),rocm)
  CC     := hipcc
  SRC    := babelstream.hip
  CFLAGS := -O3 -DUSE_HIP
  LIBS   :=
else ifeq ($(PLATFORM),metal)
  CC     := clang++
  SRC    := babelstream_metal.mm
  CFLAGS := -O2 -std=c++17 -fobjc-arc \
            -framework Metal -framework Foundation
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
