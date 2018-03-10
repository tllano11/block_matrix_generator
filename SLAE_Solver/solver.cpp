#include "solver.h"

using namespace std;

const int double_size = sizeof(double);
// Number of vectors, besides A, to allocate GPU memory to
const int gpu_vector_count = 4;
// Threads per block
const int  tpb = 32;
// Cuda events used to measure computation time for kernels
cudaEvent_t start,stop;
int bpg;

template <class T> T* cuda_allocate (int size) {
  T* device_ptr;
  cudaError_t err = cudaMalloc((void **) &device_ptr, size*sizeof(T));
  gassert(err);
  return device_ptr;
}

template <class T> T* to_device(T* src, int size) {
  T* dst = cuda_allocate<double>(size);
  gassert(cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyHostToDevice));
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

void print_telapsed(cudaEvent_t start, cudaEvent_t stop, string msg) {
  float telapsed;
  gassert(cudaEventElapsedTime(&telapsed,start,stop));
  cout <<"Elapsed " << msg << " Time = " << telapsed << " ms" << endl;
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

    gassert(cudaMemcpy(gpu_A, A + shift, rows_gpu*cols_A*double_size, cudaMemcpyHostToDevice));

    gassert(cudaEventRecord(start));
    run_jacobi <<< bpg, tpb >>> (gpu_A, gpu_b,
				 gpu_x_c, gpu_x_n,
				 rows_gpu, cols_A,
				 first_row_block, rel);
    gassert(cudaEventRecord(stop));
    gassert(cudaEventSynchronize(stop));
    print_telapsed(start, stop, "run_jacobi");

    gassert(cudaEventRecord(start));
    compute_error <<< bpg, tpb >>> (gpu_x_c + first_row_block,
				    gpu_x_n + first_row_block,
				    gpu_x_e + first_row_block,
				    rows_gpu);
    gassert(cudaEventRecord(stop));
    gassert(cudaEventSynchronize(stop));
    print_telapsed(start, stop, "compute_error");

#ifdef DEBUG
    cout << string(50, '+') << endl;
    cout << "First row: " << first_row_block << endl;
    cout << "subiter: " << i << endl;
    cout << string(50, 'A') << endl;
    print_vector(A+shift, rows_gpu, cols_A);
#endif //DEBUG
  }
}

void solve(double* A, double* b, double* x_ptr, int niter, double tol){

#ifdef DEBUG
  cout << string(50, '*') << endl;
  cout << "GPU Matrix A: " << endl;
  print_vector(A, rows_A, cols_A);
  cout << string(50, '*') << endl;
  cout << "GPU Vector b: " << endl;
  cout << string(50, '*') << endl;
  print_vector(b, cols_A, 1);
#endif //DEBUG

  //bpg = cols_A + (tpb - 1) / tpb;

  // Pointers to host memory
  double* x_c = new double[cols_A]; // x current
  double* x_n = new double[cols_A];

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int gpu_mem = props.totalGlobalMem;
  int gpu_mem_for_A = gpu_mem - (gpu_vector_count * cols_A);
  gpu_mem_for_A = gpu_mem_for_A * 0.8;
  // Max rows to allocate for A
  int rows_gpu = gpu_mem_for_A / (cols_A * double_size);

  if(rows_gpu < 1){
    cerr << "The matrix is too BIG; not even one row fits in the memory" << endl;
    exit(1);
  } else if(rows_gpu > rows_A){
    rows_gpu = rows_A;
  }

  //cout << "Rows GPU: " << rows_gpu << endl;

  bpg = ceil(rows_gpu / (double)tpb);

  // Pointers to GPU memory
  double* gpu_x_c = to_device<double>(x_c, cols_A);
  double* gpu_x_n = to_device<double>(x_n, cols_A);
  double* gpu_x_e = cuda_allocate<double>(rows_A);
  double* gpu_A = cuda_allocate<double>(rows_gpu * cols_A);
  double* gpu_b = to_device<double>(b, rows_A);

  // Initialize cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  // Will store index of max_err
  int* max_err_incx = new int();

  //Create events to measure time
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Control whether the algorithm fails or succeeds
  int count = 0;
  double error = tol + 1;
  double* max_err = &error;

  int total_iters = ceil(rows_A/(double)rows_gpu);
  //cout << "Total iters: " << total_iters << endl;

#ifdef DEBUG
  double* x_e = new double[rows_A];
#endif //DEBUG

  while ( (*max_err > tol) && (count < niter) ) {
    if ((count % 2) == 0) {
      launch_jacobi(A, gpu_A, gpu_b, gpu_x_c, gpu_x_n , gpu_x_e, rows_gpu, total_iters);
    } else {
      launch_jacobi(A, gpu_A, gpu_b, gpu_x_n, gpu_x_c, gpu_x_e, rows_gpu, total_iters);
    }
    gassert(cudaEventRecord(start));
    assert(CUBLAS_STATUS_SUCCESS == cublasIdamax(handle, cols_A, (const double*)gpu_x_e, 1, max_err_incx));
    gassert(cudaEventRecord(stop));
    gassert(cudaEventSynchronize(stop));
    print_telapsed(start, stop, "cublas");
    gassert(cudaMemcpy(max_err, gpu_x_e + (*max_err_incx - 1), double_size, cudaMemcpyDeviceToHost));
    count++;

#ifdef DEBUG
    gassert(cudaMemcpy(x_e, gpu_x_e, cols_A*double_size, cudaMemcpyDeviceToHost));
    cout << string(50, 'E') << endl;
    print_vector(x_e, rows_A, 1);
    cout << "Max Err: " << *max_err << endl;
#endif //DEBUG
  }

  if (*max_err < tol) {
    cout << "Jacobi succeeded in " << count << " iterations with an error of "
	 << *max_err << endl;
    if ((count % 2) == 0) {
      gassert(cudaMemcpy(x_c, gpu_x_n, cols_A*double_size, cudaMemcpyDeviceToHost));
    } else {
      gassert(cudaMemcpy(x_c, gpu_x_c, cols_A*double_size, cudaMemcpyDeviceToHost));
    }
    //print_vector(x_c, rows_A, 1);
    double jacobi_err[rows_A];
    vdSub(rows_A, x_ptr, x_c, jacobi_err);
    double jacobi_norm = cblas_dnrm2(rows_A, jacobi_err, 1);
    double x_norm = cblas_dnrm2(rows_A, x_ptr, 1);
    double rel_jacobi_err = jacobi_norm / x_norm;
    cout << "Jacobi relative error is: " << rel_jacobi_err << endl;

  } else {
    cout << "Jacobi failed." << endl;
  }

  gassert(cudaEventDestroy(start));
  gassert(cudaEventDestroy(stop));
  gassert(cudaFree(gpu_A));
  gassert(cudaFree(gpu_b));
  gassert(cudaFree(gpu_x_n));
  gassert(cudaFree(gpu_x_e));
  gassert(cudaFree(gpu_x_c));
  cublasDestroy(handle);
  delete max_err_incx;
  delete x_c;
}

void solve_mkl(double* A, double* b, int n, double* x) {
  // The number of columns of b
  int nrhs = 1;
  // A's dimension
  int lda = n;
  // b's dimension
  int ldb = nrhs;
  // Will contain the pivot indices
  int ipiv[n];
  // Wheter mkl failed or succeeded
  int info;

  // Solve system
  double start = dsecnd();
  info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, lda, ipiv, b, ldb);
  cout << "Elapsed mkl_dgesv Time = " << dsecnd() - start << endl;

  cout << "\nMKL results:\n" << endl;
  if(info > 0) {
    cout << "The solution could not be computed." << endl;
  } else {
    //print_vector(b, n, 1);

    double err_v[n];
    double err_abs[n];
    // Compute err_v = x - b
    vdSub(n, x, b, err_v);
    //cout << "\nMKL error: \n" << endl;
    //print_vector(err_v, n, 1);
    //compute err_abs = | err_v |
    vdAbs(n, err_v, err_abs);
    //cout << "\nMKL error abs: \n" << endl;
    //print_vector(err_abs, n, 1);
    int index = cblas_idamax(n, err_abs, 1);
    cout << "\nMKL succeeded with an error of: " << err_abs[index] << endl;
  }
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
