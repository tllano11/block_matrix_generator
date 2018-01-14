__device__ double abs(double number) {
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
__global__ void run_jacobi(double* A, double* b,
			      double* x_c, double* x_n,
			      uint32_t rows, uint32_t cols,
			      uint32_t first_row_block, double rel) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t current_row = first_row_block + idx;
  if (idx < rows) {
    double sigma = 0;
    //Indicates which row must be computed by the current thread.
    uint32_t index = idx * cols;
    for (uint32_t j = 0; j < cols; ++j) {
      //Ensures not to use a diagonal value when computing.
      if (current_row != j) {
	sigma += A[index + j] * x_c[j];
      }
    }
    x_n[idx] = (b[idx] - sigma) / A[index + current_row];
    x_n[idx] = rel * x_n[idx] + (double)(1.0 - rel) * x_c[idx];
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
				       double* x_e, uint32_t n) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x_e[idx] = abs(x_n[idx] - x_c[idx]);
  }
}
