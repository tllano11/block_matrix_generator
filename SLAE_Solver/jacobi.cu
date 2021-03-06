#include "jacobi.h"
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
__global__ void run_jacobi(double* A, double* b,
			   double* x_c, double* x_n, double* x_e,
			   int rows, int cols,
			   int first_row_block, double rel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < rows) {
    int current_row = first_row_block + idx;

    double sigma = 0.0;
    //Indicates which row must be computed by the current thread.
    int index = idx * cols;
    for (int j = 0; j < cols; ++j) {
      //Ensures not to use a diagonal value when computing.
      if (current_row != j) {
	sigma += A[index + j] * x_c[j];
      }
    }

    x_n[current_row] = (b[current_row]- sigma) / A[index + current_row];
    //printf("Sigma [%i]: %f\n", current_row, sigma);

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
__global__ void compute_error (double* x_c, double* x_n,
			       double* x_e, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x_e[idx] = x_n[idx] - x_c[idx];
    //    printf("Err [%i]: %.16f - %.16f = %.16f\n", idx, x_n[idx], x_c[idx], x_e[idx]);
  }
}