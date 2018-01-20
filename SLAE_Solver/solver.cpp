#include "solver.h"

using namespace std;

const int double_size = sizeof(double);
// Number of vectors, besides A, to allocate GPU memory to
const int gpu_vector_count = 4;
// Threads per block
const int  tpb = 32;

template <class T> T* cuda_allocate (int size) {
  T* device_ptr;
  assert(cudaSuccess == cudaMalloc((void **) &device_ptr, size*sizeof(T)));
  return device_ptr;
}

template <class T> T* to_device(T* src, int size) {
  T* dst = cuda_allocate<double>(size);
  assert(cudaSuccess == cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyHostToDevice));
  return dst;
}

void launch_jacobi(double* A, double* gpu_A, double* gpu_b,
		   double* gpu_x_c, double* gpu_x_n, double* gpu_x_e,
		   int rows_gpu, int total_iters) {
  double* A_ptr = A;
  int shift;
  for (int i = 0; i < total_iters; ++i) {
    A_ptr = A_ptr + i * rows_gpu * cols;
    if (i == total_iters - 1) {
      rows_gpu = rows - i * rows_gpu;
    }
    assert(cudaSuccess == cudaMemcpy(gpu_A, A_ptr, rows_gpu*cols*double_size, cudaMemcpyHostToDevice));
    run_jacobi <<< bpg, tpb >>> (gpu_A, gpu_b, gpu_x_c, gpu_x_n, cols, rel);
    shift = i * rows_gpu;
    compute_error <<< bpg, tpb >>> (gpu_x_n + shift,
				    gpu_x_c + shift,
				    gpu_x_e + shift,
				    rows_gpu);
  }
}

void solve(double* A, double* b, uint32_t niter, double tol){
  int A_slots = rows * cols;
  bpg = cols + (tpb - 1) / tpb;

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
  double* max_err_gpu = cuda_allocate<double>(1);

  // Initialize cublas
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Control whether the algorithm fails or succeeds
  uint32_t count = 0;
  double error = tol + 1;
  double* max_err = &error;

  int total_iters = ceil(rows/rows_gpu);

  while ( (*max_err > tol) && (count < niter) ) {
    if (count % 2) {
      launch_jacobi(A, gpu_A, gpu_b, gpu_x_c, gpu_x_n, gpu_x_e, rows_gpu, total_iters);
    } else {
      launch_jacobi(A, gpu_A, gpu_b, gpu_x_n, gpu_x_c, gpu_x_e, rows_gpu, total_iters);
    }
    assert(CUBLAS_STATUS_SUCCESS == cublasIsamax(handle, cols, gpu_x_e, 1, max_err_gpu));
    assert(cudaSuccess == cudaMemcpy(max_err, max_err_gpu, double_size, cudaMemcpyDeviceToHost));
    count++;
  }

  if (*max_err < tol) {
    if (count % 2) {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_n, cols*double_size, cudaMemcpyDeviceToHost));
    } else {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_c, cols*double_size, cudaMemcpyDeviceToHost));
    }
    std::cout << "Jacobi succeeded in " << count << " iterations with an error of "
	      << *max_err << std::endl;
  } else {
    std::cout << "Jacobi failed." << std::endl;
  }

  assert(cudaSuccess == cudaFree(gpu_A));
  assert(cudaSuccess == cudaFree(gpu_b));
  assert(cudaSuccess == cudaFree(gpu_x_n));
  assert(cudaSuccess == cudaFree(gpu_x_e));
  assert(cudaSuccess == cudaFree(gpu_x_c));
  cublasDestroy(handle);
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

  solve(A, b, 9, 3, niter, tol, rel);

  return 0;
}
*/
