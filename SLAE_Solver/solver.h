#ifndef SOLVER_H
#define SOLVER_H

#include<stdlib.h>
#include<jacobi.h>

namespace solver {
void solve(long double* A, long double* b,
	     long double* x_c, uint32_t niter,
	     float tol, float rel);
}
#endif /* SOLVER_H */
