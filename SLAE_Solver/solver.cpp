#include "solver.h"

using namespace std;

const int double_size = sizeof(double);
// Number of vectors, besides A, to allocate GPU memory to
const int gpu_vector_count = 4;
// Threads per block
const int  tpb = 32;
int bpg;

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

void print_vector(double* vector, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j){
      cout << vector[i * cols + j] << " ";
    }
    cout << endl;
  }
}

void launch_jacobi(double* A, double* gpu_A, double* gpu_b,
		   double* gpu_x_c, double* gpu_x_n, double* gpu_x_e,
		   int rows_gpu, int total_iters) {
  int first_row_block;
  int shift;
  for (int i = 0; i < total_iters; ++i) {
    first_row_block = i * rows_gpu;
    shift = first_row_block * cols_A;

    if (i == total_iters - 1) {
      rows_gpu = rows_A - i * rows_gpu;
    }

    assert(cudaSuccess == cudaMemcpy(gpu_A, A + shift, rows_gpu*cols_A*double_size, cudaMemcpyHostToDevice));

    run_jacobi <<< bpg, tpb >>> (gpu_A, gpu_b + first_row_block,
				 gpu_x_c + first_row_block, gpu_x_n + first_row_block,
				 rows_gpu, cols_A,
				 first_row_block, rel);

    compute_error <<< bpg, tpb >>> (gpu_x_c + first_row_block,
				    gpu_x_n + first_row_block,
				    gpu_x_e + first_row_block,
				    rows_gpu);

  }
}

void solve(double* A, double* b, int niter, double tol){

#ifdef DEBUG
  cout << string(50, '*') << endl;
  cout << "GPU Matrix A: " << endl;
  print_vector(A, rows_A, cols_A);
  cout << string(50, '*') << endl;
  cout << "GPU Vector b: " << endl;
  cout << string(50, '*') << endl;
  print_vector(b, cols_A, 1);
#endif //DEBUG

  bpg = cols_A + (tpb - 1) / tpb;

  // Pointers to host memory
  double* x_c = new double[cols_A]; // x current

  // Initialize the current solution to whatever value (zero in this case)
  // is necessary to overwrite any junk data present in the memory space
  // allocated for gpu_x_c
  for(int i = 0; i < cols_A; ++i){
    x_c[i] = 0;
  }

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int gpu_mem = props.totalGlobalMem;
  int gpu_mem_for_A = gpu_mem - (gpu_vector_count * cols_A);
  // Max rows to allocate for A
  int rows_gpu = 2; //gpu_mem_for_A / (cols_A * double_size);

  // Pointers to GPU memory
  double* gpu_x_c = to_device<double>(x_c, cols_A);
  double* gpu_x_n = cuda_allocate<double>(cols_A);
  double* gpu_x_e = cuda_allocate<double>(rows_A);
  double* gpu_A = cuda_allocate<double>(rows_gpu * cols_A);
  double* gpu_b = to_device<double>(b, rows_A);

  // Initialize cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  // Will store index of max_err
  int* max_err_incx = new int();

  // Control whether the algorithm fails or succeeds
  int count = 0;
  double error = tol + 1;
  double* max_err = &error;

  int total_iters = ceil(rows_A/rows_gpu);
  double* x_e = new double[rows_A];

  while ( (*max_err > tol) && (count < niter) ) {
    if ((count % 2) == 0) {
      launch_jacobi(A, gpu_A, gpu_b, gpu_x_c, gpu_x_n , gpu_x_e, rows_gpu, total_iters);
    } else {
      launch_jacobi(A, gpu_A, gpu_b, gpu_x_n, gpu_x_c, gpu_x_e, rows_gpu, total_iters);
    }
    assert(CUBLAS_STATUS_SUCCESS == cublasIdamax(handle, cols_A, (const double*)gpu_x_e, 1, max_err_incx));
    assert(cudaSuccess == cudaMemcpy(max_err, gpu_x_e + (*max_err_incx - 1), double_size, cudaMemcpyDeviceToHost));
    count++;
  }

  if (*max_err < tol) {
    if ((count % 2) == 0) {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_n, cols_A*double_size, cudaMemcpyDeviceToHost));
    } else {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_c, cols_A*double_size, cudaMemcpyDeviceToHost));
    }
    cout << "Jacobi succeeded in " << count << " iterations with an error of "
	      << *max_err << endl;

    print_vector(x_c, rows_A, 1);
  } else {
    cout << "Jacobi failed." << endl;
  }

  assert(cudaSuccess == cudaFree(gpu_A));
  assert(cudaSuccess == cudaFree(gpu_b));
  assert(cudaSuccess == cudaFree(gpu_x_n));
  assert(cudaSuccess == cudaFree(gpu_x_e));
  assert(cudaSuccess == cudaFree(gpu_x_c));
  cublasDestroy(handle);
  delete max_err_incx;
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
