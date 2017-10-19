#ifndef JACOBI_H
#define JACOBI_H

#include<stdint.h>
#include<stdlib.h>
#include<cuda.h>

namespace solver{
  namespace jacobi {
    __global__ void solve(long double* A, long double* b,
			  long double* x_c, long double* x_n,
			  uint32_t n, uint32_t rel);

    __global__ void compute_error (long double* x_c, long double* x_n,
				   long double* x_e, n);

  }
}
#endif /* JACOBI_H */
