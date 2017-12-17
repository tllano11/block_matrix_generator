#include<iostream>
#include<assert.h>

__global__ void multiply(double* m, double* v, double* r,
		       uint32_t rows, uint32_t cols) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows) {
    uint32_t begin = idx*cols;
    uint32_t end = begin + cols;
    uint32_t j = 0;

    for (uint32_t i = begin; i < end; ++i, ++j) {
      r[idx] += m[i] * v[j];
    }
  }
}

int main() {
  double a[6] = {1,-1,2,0,-3,1};
  double x[3] = {2,1,0};
  double b[2] = {0,0};
  uint32_t rows = 2;
  uint32_t cols = 3;

  double *a_g, *x_g, *b_g;

  assert(cudaSuccess == cudaMalloc((void **) &a_g, rows*cols*sizeof(double)));
  assert(cudaSuccess == cudaMalloc((void **) &x_g, rows*sizeof(double)));
  assert(cudaSuccess == cudaMalloc((void **) &b_g, rows*sizeof(double)));

  assert(cudaSuccess == cudaMemcpy(a_g, a, rows*cols*sizeof(double), cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(x_g, x, rows*sizeof(double), cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(b_g, b, rows*sizeof(double), cudaMemcpyHostToDevice));

  int tpb = 32;
  int bpg = rows*cols + (tpb - 1) / tpb;
  multiply <<< bpg, tpb  >>> (a_g, x_g, b_g, rows, cols);

  assert(cudaSuccess == cudaMemcpy(b, b_g, rows*sizeof(double), cudaMemcpyDeviceToHost));

  std::cout << b[0] << b[1]  << "\n";

}
