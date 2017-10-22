#include<solver.h>

using namespace solver;

template <class T> T* solver::cuda_allocate (int size) {
  T* device_ptr;
  cudaMalloc(&device_ptr, size*sizeof(T));
  return device_ptr;
}

void solver::solve(long double* A, long double* b,
		   long double* x_c, uint32_t niter,
		   float tol, float rel){
  int tpb = 32;
  int bpg = bpg = len(A) + (tpb - 1) / tpb;

  int vector_size = sizeof(x_c)/sizeof(long double*);
  long double* x_n = new long double[vector_size];
  long double* x_e = new long double[vector_size];
  long double* gpu_x_n = cuda_allocate<long double>(vector_size);
  long double* gpu_x_e = cuda_allocate<long double>(vector_size);
}
