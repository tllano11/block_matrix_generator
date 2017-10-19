#include<solver.h>

using namespace solver;

void solver::solve(long double* A, long double* b,
		   long double* x_c, uint32_t niter,
		   float tol, float rel){
  int tpb = 32;
  int bpg = bpg = len(A) + (tpb - 1) / tpb;
  long double x_n[sizeof(x_c)/sizeof(long double*)];
  long double x_e[sizeof(x_c)/sizeof(long double*)];
}
