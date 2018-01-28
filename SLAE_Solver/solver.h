#pragma once

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "jacobi.h"

extern int rows_A, cols_A;
extern double rel;
extern int bpg;

template <class T> T* cuda_allocate (int size);

template <class T> T* to_device(T* src, int size);

void launch_jacobi(double* A, double* gpu_A, double* gpu_b,
		   double* gpu_x_c, double* gpu_x_n, double* gpu_x_e,
		   int rows_gpu, int total_iters);

void solve(double* A, double* b, int niter, double tol);
