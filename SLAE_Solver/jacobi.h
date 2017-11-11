#ifndef JACOBI_H
#define JACOBI_H

#include<stdint.h>
#include<stdlib.h>
#include<cuda.h>

namespace solver {
  namespace jacobi {
    __device__ double abs(double number);

    __global__ void solve(double* A, double* b,
			  double* x_c, double* x_n,
			  uint32_t n, float rel);

    __global__ void compute_error (double* x_c, double* x_n,
				   double* x_e, uint32_t n);

  }
}
#endif /* JACOBI_H */
