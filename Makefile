MPICXX = mpicxx
CXX = g++
CXXFLAGS = -std=c++11 -Wall
CUDAFLAGS = -lcudart -L/opt/cuda/lib64 -I/opt/cuda/include
LDFLAGS = -I./SLAE_Solver
NVCCFLAGS = -x cu -Wno-deprecated-gpu-targets -Xcompiler -fPIC -std=c++11
NVCC = nvcc
DEBUG = -g -D DEBUG
OUT_PATH = ./bin
TARGET = matrix_generator
SRC_GEN = matrix_generator-v2
SRC_GPU_MUL = gpu_mul
SRC_GPU_JACOBI = jacobi
SRC_SOLVER = solver

all: init $(TARGET)

debug: CXXFLAGS += $(DEBUG)
debug: all

$(TARGET): $(OUT_PATH)/$(SRC_GEN).o $(OUT_PATH)/$(SRC_SOLVER).o $(OUT_PATH)/$(SRC_GPU_JACOBI).o
	$(MPICXX) $(CXXFLAGS) $(CUDAFLAGS) -o $@ $^

$(OUT_PATH)/$(SRC_GEN).o: $(SRC_GEN).cpp
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) -c $^ -o $@

$(OUT_PATH)/$(SRC_SOLVER).o: SLAE_Solver/$(SRC_SOLVER).cpp
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

$(OUT_PATH)/$(SRC_GPU_JACOBI).o: SLAE_Solver/$(SRC_GPU_JACOBI).cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

# $(OUT_PATH)/$(SRC_GPU_MUL).o: $(SRC_GPU_MUL).cu
# 	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

init:
	mkdir -p $(OUT_PATH)

clean:
	rm -f *~ *.o $(TARGET)
	rm -rf $(OUT_PATH)
