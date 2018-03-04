CXX = g++
CXXFLAGS = -std=c++11 -Wall -pthread
CUDAFLAGS = -lcublas -lcudart -L/opt/cuda/lib64 -I/opt/cuda/include \
-I/opt/include/ ${LAPACK_INCLUDE} -lmkl_intel_lp64 -lmkl_sequential \
-lmkl_core -lpthread -lm
LDFLAGS = -I./SLAE_Solver -I./src/
NVCCFLAGS = -lineinfo -x cu -Wno-deprecated-gpu-targets -std=c++11 \
-Xcompiler -fPIC
NVCC = nvcc
DEBUG = -g -DDEBUG
OUT_PATH = ./bin
TARGET = matrix_generator
SRC_GEN = matrix_generator-v3
SRC_GPU_JACOBI = jacobi
SRC_SOLVER = solver

# all: library.cpp main.cpp
# In this case:
#
# $@ evaluates to all
# $< evaluates to library.cpp
# $^ evaluates to library.cpp main.cpp

all: init $(TARGET)

debug: CXXFLAGS += $(DEBUG)
debug: NVCCFLAGS += $(foreach opt, $(DEBUG), -Xcompiler $(opt))
debug: all

$(TARGET): $(OUT_PATH)/$(SRC_GEN).o $(OUT_PATH)/$(SRC_SOLVER).o \
$(OUT_PATH)/$(SRC_GPU_JACOBI).o
	$(CXX) $(CXXFLAGS) $(CUDAFLAGS) -o $@ $^

$(OUT_PATH)/$(SRC_GEN).o: $(SRC_GEN).cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $^ -o $@

$(OUT_PATH)/$(SRC_SOLVER).o: SLAE_Solver/$(SRC_SOLVER).cpp
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) -c $^ -o $@

$(OUT_PATH)/$(SRC_GPU_JACOBI).o: SLAE_Solver/$(SRC_GPU_JACOBI).cu
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) -c $^ -o $@

init:
	mkdir -p $(OUT_PATH)

clean:
	rm -f *~ *.o $(TARGET)
	rm -rf $(OUT_PATH)
