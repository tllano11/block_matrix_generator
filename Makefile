CXX = mpicxx
CXXFLAGS = -std=c++11 -Wall
LDFLAGS = -lcudart -L/opt/cuda/lib64 -I/opt/cuda/include
NVCCFLAGS = -Wno-deprecated-gpu-targets -Xcompiler -fPIC
NVCC = nvcc
DEBUG = -g -D DEBUG
TARGET = matrix_generator-v2
GPU = gpu_mul

all: $(TARGET)

debug: CXXFLAGS += $(DEBUG)
debug: all

$(TARGET): $(TARGET).o $(GPU).o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(TARGET).o $(GPU).o

$(TARGET).o: $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp

$(GPU).o: $(GPU).cu
	$(NVCC) $(NVCCFLAGS) -c $(GPU).cu

clean:
	rm -f *~ *.o $(TARGET)
