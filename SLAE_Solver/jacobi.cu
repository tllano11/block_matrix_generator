#include "jacobi.h"

using namespace solver;

__device__ long double jacobi::abs(long double number) {
  if (number < 0) {
    return -number;
  } else {
    return number;
  }
}

/**
   Use Iterative Jacobi to compute an answer for the
   system of linear algebraic equations A.

   @param A           Coefficient matrix.
   @param b           Linearly independent vector.
   @param x_c         Current answer's approximation.
   @param x_n         Vector in which to store new answer.
   @param n           Coefficient matrix size.
   @param rel         Relaxation coefficient.
*/
__global__ void jacobi::solve(long double* A, long double* b,
			      long double* x_c, long double* x_n,
			      uint32_t n, float rel) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    long double sigma = 0;
    //Indicates which row must be computed by the current thread.
    uint32_t index = idx * n;
    for (uint32_t j = 0; j < n; ++j) {
      //Ensures not to use a diagonal value when computing.
      if (idx != j) {
	sigma += A[index + j] * x_c[j];
      }
    }
    x_n[idx] = (b[idx] - sigma) / A[index + idx];
    x_n[idx] = (long double)rel * x_n[idx] + (long double)(1 - rel) * x_c[idx];
  }
}

/**
   Calculates jacobi's maximum error.

   @param x_c Pointer to list representing current
   approximation for vector x in a system Ax = b.
   @param x_n    Pointer to list representing new
   approximation for vector x in a system Ax = b.
   @param x_e   Pointer to list in which an error for each
   approximation will be stored.
   @param n      Coefficient matrix size.

   @return None
*/
__global__ void jacobi::compute_error (long double* x_c, long double* x_n,
				       long double* x_e, uint32_t n) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x_e[idx] = abs(x_n[idx] - x_c[idx]);
  }
}
