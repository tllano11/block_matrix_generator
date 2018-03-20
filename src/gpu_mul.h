#pragma once

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

using namespace std;

const int Tile_size = 2;

__global__ void matrixMul(double *A, double *B, double *C, long rows, long cols);

void generate_b_gpu(double *hostA, double *hostX, double *hostB, long cols, long rows);
