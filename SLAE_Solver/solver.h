#pragma once

#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
//#include "jacobi.h"

extern int rows, cols;
extern float rel;
int bpg;

template <class T> T* cuda_allocate (int size);

template <class T> T* to_device(T* src, int size);

void solve(double* A, double* b, uint32_t niter, float tol);

void print_data(double* matrix, long rows, long cols);
