#pragma once

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "mkl_lapacke.h"
#include "mkl.h"
#include "jacobi.h"
#include <iomanip>

extern int rows_A, cols_A;
extern double rel;
extern int bpg;

inline void gpu_assert(cudaError_t err,
		       const char *file,
		       int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n",
	    cudaGetErrorString( err ),
	    file, line );
    exit(EXIT_FAILURE);
  }
}
#define gassert(err) (gpu_assert( err, __FILE__, __LINE__ ))

template <class T> T* cuda_allocate (int size);

template <class T> T* to_device(T* src, int size);

void launch_jacobi(double* A, double* gpu_A, double* gpu_b,
		   double* gpu_x_c, double* gpu_x_n, double* gpu_x_e,
		   int rows_gpu, int total_iters);

void solve(double* A, double* b, double* x, int niter, double tol);

void solve_mkl(double* A, double* b, int n, double* x);
