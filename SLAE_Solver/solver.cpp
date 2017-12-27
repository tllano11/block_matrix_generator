#include "solver.h"
#include "jacobi.h"

using namespace std;

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
/*
void print_data(double* matrix, long rows, long cols) {
  for (long i = 0; i < rows; ++i) {
    for (long j = 0; j < cols; ++j){
      cout << matrix[i * cols + j] << " ";
    }
    cout << endl;
  }
}*/

extern "C++" void solve(double* A, double* b, int matrix_size,
		   int vector_size, int niter,
		   float tol, float rel){

  int tpb = 32;
  int bpg = matrix_size + (tpb - 1) / tpb;

  // Pointers to host memory
  double* x_c = new double[vector_size]; // x current
  double* x_n = new double[vector_size]; // x next
  double* x_e = new double[vector_size]; // x error

  for(int i = 0; i < vector_size; ++i){
    x_c[i] = 0;
  }

  // Pointers to GPU memory
  double* gpu_x_c = to_device<double>(x_c, vector_size); // x current
  double* gpu_x_n = cuda_allocate<double>(vector_size); // x next
  double* gpu_x_e = cuda_allocate<double>(vector_size); // x error
  double* gpu_A = to_device<double>(A, matrix_size);
  double* gpu_b = to_device<double>(b, vector_size);
  // Control whether the algorithm fails or succeeds
  int count = 0;
  float error = tol + 1;

  while ( (error > tol) && (count < niter) ) {
    if (count % 2) {
      solve <<< bpg, tpb >>> (gpu_A, gpu_b, gpu_x_n, gpu_x_c, vector_size, rel);
      compute_error <<< bpg, tpb >>> (gpu_x_c, gpu_x_n, gpu_x_e, vector_size);
    } else {
      solve <<< bpg, tpb >>> (gpu_A, gpu_b, gpu_x_c, gpu_x_n, vector_size, rel);
      compute_error <<< bpg, tpb >>> (gpu_x_n, gpu_x_c, gpu_x_e, vector_size);
    }
    assert(cudaSuccess == cudaMemcpy(x_e, gpu_x_e, vector_size*sizeof(double), cudaMemcpyDeviceToHost));
    error = (float) *(max_element(x_e, x_e + vector_size));
    count++;
  }

  if (error < tol) {
    if (count % 2) {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_n, vector_size*sizeof(double), cudaMemcpyDeviceToHost));
    } else {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_c, vector_size*sizeof(double), cudaMemcpyDeviceToHost));
    }
    cout << "Jacobi succeeded in " << count << " iterations with an error of "
	      << error << endl;
    print_data(x_c, 3 , 1);
  } else {
    cout << "Jacobi failed." << endl;
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

  solve(A, b, 9, 3, niter, tol, rel);

  return 0;
}
*/
