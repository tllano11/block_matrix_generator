#pragma once

#include<stdint.h>
#include<cuda.h>

// A __device__ function can only be called from __global__ functions.
__device__ double abs(double number);

// __global__ functions are executed in the GPU.
__global__ void run_jacobi(double* A, double* b,
			   double* x_c, double* x_n,
			   uint32_t rows, uint32_t cols,
			   uint32_t first_row_block, double rel);

__global__ void compute_error (double* x_c, double* x_n,
			       double* x_e, uint32_t n);
