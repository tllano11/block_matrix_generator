#pragma once

#include<stdint.h>
#include<cuda.h>

// A __device__ function can only be called from __global__ functions.
__device__ double gpu_abs(double number);

// __global__ functions are executed in the GPU.
__global__ void run_jacobi(double* A, double* b,
			   double* x_c, double* x_n,
			   int rows, int cols,
			   int first_row_block, double rel, double* param);

__global__ void compute_error (double* x_c, double* x_n,
			       double* x_e, int n);
