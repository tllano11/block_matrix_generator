#include "solver.h"

using namespace solver;
using namespace std;

template <class T> T* solver::cuda_allocate (int size) {
  T* device_ptr;
  assert(cudaSuccess == cudaMalloc((void **) &device_ptr, size*sizeof(T)));
  return device_ptr;
}

template <class T> T* solver::to_device(T* src, int size) {
  T* dst = cuda_allocate<double>(size);
  assert(cudaSuccess == cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyHostToDevice));
  return dst;
}

void solver::solve(double* A, double* b,
		   int matrix_size, int vector_size,
		   double* x_c, uint32_t niter,
		   float tol, float rel){

  int tpb = 32;
  int bpg = matrix_size + (tpb - 1) / tpb;

  // Pointers to host memory
  double* x_n = new double[vector_size]; // x next
  double* x_e = new double[vector_size]; // x error
  // Pointers to GPU memory
  double* gpu_x_c = cuda_allocate<double>(vector_size); // x current
  double* gpu_x_n = cuda_allocate<double>(vector_size); // x next
  double* gpu_x_e = cuda_allocate<double>(vector_size); // x error
  double* gpu_A = to_device<double>(A, matrix_size);
  double* gpu_b = to_device<double>(b, vector_size);
  // Control whether the algorithm fails or succeeds
  uint32_t count = 0;
  float error = tol + 1;

  while ( (error > tol) && (count < niter) ) {
    if (count % 2) {
      jacobi::solve <<< bpg, tpb >>> (gpu_A, gpu_b, gpu_x_c, gpu_x_n, vector_size, rel);
      jacobi::compute_error <<< bpg, tpb >>> (gpu_x_c, gpu_x_n, gpu_x_e, vector_size);
    } else {
      jacobi::solve <<< bpg, tpb >>> (gpu_A, gpu_b, gpu_x_n, gpu_x_c, vector_size, rel);
      jacobi::compute_error <<< bpg, tpb >>> (gpu_x_n, gpu_x_c, gpu_x_e, vector_size);
    }
    assert(cudaSuccess == cudaMemcpy(x_e, gpu_x_e, vector_size*sizeof(double), cudaMemcpyDeviceToHost));
    error = (float) *(std::max_element(x_e, x_e + vector_size));
    count++;
  }

  if (error < tol) {
    if (count % 2) {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_n, vector_size*sizeof(double), cudaMemcpyDeviceToHost));
    } else {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_c, vector_size*sizeof(double), cudaMemcpyDeviceToHost));
    }
    std::cout << "Jacobi succeeded in " << count << " iterations with an error of "
	 << error << std::endl;
  } else {
    std::cout << "Jacobi failed." << std::endl;
  }
}

int main() {
  double A[] = {4, -1, -1, -2, 6, 1, -1, 1, 7};
  double b[] = {3, 9, -6};
  double x_c[] = {0, 0, 0};
  int niter = 1000;
  float rel = 1;
  float tol = 0.01;

  solver::solve(A, b, 9, 3, x_c, niter, tol, rel);
}
