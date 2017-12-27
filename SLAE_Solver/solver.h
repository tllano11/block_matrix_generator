#pragma once

#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <cuda_runtime_api.h>
//#include "jacobi.h"

template <class T> T* cuda_allocate (int size);

template <class T> T* to_device(T* src, int size);

extern "C++" void solve(double* A, double* b, int matrix_size,
			  int vector_size, int niter,
			  float tol, float rel);

void print_data(double* matrix, long rows, long cols);
