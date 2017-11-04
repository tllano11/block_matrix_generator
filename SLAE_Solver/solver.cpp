#include<solver.h>

using namespace solver;

template <class T> T* solver::cuda_allocate (int size) {
  T* device_ptr;
  assert(cudaSuccess == cudaMalloc((void **) &device_ptr, size*sizeof(T)));
  return device_ptr;
}

template <class T> T* solver::to_device(T* src, int size) {
  T* dst = cuda_allocate(size);
  assert(cudaSuccess == cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyHostToDevice));
  return dst;
}

void solver::solve(long double* A, long double* b,
		   long double* x_c, uint32_t niter,
		   float tol, float rel){
  int tpb = 32;
  int bpg = bpg = len(A) + (tpb - 1) / tpb;

  int vector_size = sizeof(x_c)/sizeof(long double*);
  int matrix_size = sizeof(A)/sizeof(long double*);
  // Pointers to host memory
  long double* x_n = new long double[vector_size]; // x next
  long double* x_e = new long double[vector_size]; // x error
  // Pointers to GPU memory
  long double* gpu_x_c = cuda_allocate<long double>(vector_size); // x current
  long double* gpu_x_n = cuda_allocate<long double>(vector_size); // x next
  long double* gpu_x_e = cuda_allocate<long double>(vector_size); // x error
  long double* gpu_A = to_device<long double>(A, matrix_size);
  long double* gpu_b = to_device<long double>(b, vector_size);
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
    assert(cudaSuccess == cudaMemcpy(x_e, gpu_x_e, vector_size*sizeof(long double), cudaMemcpyDeviceToHost));
    error = (float) *(max_element(x_e, x_e + vector_size));
    count++;
  }

  if (error < tol) {
    if (count % 2) {
      assert(cudaSuccess == cudaMemcpy(x_n, gpu_x_n, vector_size*sizeof(long double), cudaMemcpyDeviceToHost));
    } else {
      assert(cudaSuccess == cudaMemcpy(x_c, gpu_x_c, vector_size*sizeof(long double), cudaMemcpyDeviceToHost));
    }
    cout << "Jacobi succeeded in " << count << " iterations with an error of "
	 << error << endl;
  } else {
    cout << "Jacobi failed." << endl;
  }
}
