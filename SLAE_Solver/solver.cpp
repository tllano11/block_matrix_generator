#include "solver.h"

using namespace solver;
using namespace std;

const int double_size = sizeof(double);
// Number of vectors, besides A, to allocate GPU memory to
const int gpu_vector_count = 4;

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

void solve(double* A, double* b,
		   int rows, int cols, uint32_t niter,
		   float tol, float rel){

  int tpb = 32;
  int bpg = matrix_size + (tpb - 1) / tpb;

  int A_slots = rows * cols;

  // Pointers to host memory
  double* x_c = new double[cols]; // x current

  // Initialize the current solution to whatever value (zero in this case)
  // is necessary to overwrite any junk data present in the memory space
  // allocated for gpu_x_c
  for(int i = 0; i < cols; ++i){
    x_c[i] = 0;
  }

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int gpu_mem = props.totalGlobalMem;
  int gpu_mem_for_A = gpu_mem - (gpu_vector_count * cols) - double_size;
  // Max rows to allocate for A
  int rows_gpu = gpu_mem_for_A / cols;

  // Pointers to GPU memory
  double* gpu_x_c = to_device<double>(x_c, cols);
  double* gpu_x_n = cuda_allocate<double>(cols);
  double* gpu_x_e = cuda_allocate<double>(cols);
  double* gpu_max_err = cuda_allocate<double>(1);
  double* gpu_A = cuda_allocate<double>(rows_gpu);
  double* gpu_b = to_device<double>(b, cols);

  // Control whether the algorithm fails or succeeds
  uint32_t count = 0;
  float error = tol + 1;

  while ( (error > tol) && (count < niter) ) {
    if (count % 2) {
      run_jacobi <<< bpg, tpb >>> (gpu_A, gpu_b, gpu_x_n, gpu_x_c, cols, rel);
      compute_error <<< bpg, tpb >>> (gpu_x_c, gpu_x_n, gpu_x_e, cols);
    } else {
      run_jacobi <<< bpg, tpb >>> (gpu_A, gpu_b, gpu_x_c, gpu_x_n, cols, rel);
      compute_error <<< bpg, tpb >>> (gpu_x_n, gpu_x_c, gpu_x_e, cols);
    }
    assert(cudaSuccess == cudaMemcpy(x_e, gpu_x_e, cols*sizeof(double), cudaMemcpyDeviceToHost));
    error = (float) *(std::max_element(x_e, x_e + cols));
    count++;
  }

  if (error < tol) {
    if (count % 2) {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_n, cols*sizeof(double), cudaMemcpyDeviceToHost));
    } else {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_c, cols*sizeof(double), cudaMemcpyDeviceToHost));
    }
    std::cout << "Jacobi succeeded in " << count << " iterations with an error of "
	      << error << std::endl;
  } else {
    std::cout << "Jacobi failed." << std::endl;
  }

  assert(cudaSuccess == cudaFree(gpu_A));
  assert(cudaSuccess == cudaFree(gpu_b));
  assert(cudaSuccess == cudaFree(gpu_x_n));
  assert(cudaSuccess == cudaFree(gpu_x_e));
  assert(cudaSuccess == cudaFree(gpu_x_c));
  delete[] x_n;
  delete[] x_e;
  delete[] x_c;
}

/*
int main() {
  double A[] = {4, -1, -1, -2, 6, 1, -1, 1, 7};
  //double A[] = {2, 1, 5, 7};
  double b[] = {3, 9, -6};
  //double b[] = {11, 13};
  double x_c[] = {0, 0, 0};
  int niter = 1000;
  float rel = 1;
  float tol = 0.0001;

  solver::solve(A, b, 9, 3, niter, tol, rel);

  return 0;
}
*/
