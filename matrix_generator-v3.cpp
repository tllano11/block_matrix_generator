#include <iostream>
#include <random>
#include <limits>
#include <cstdio>
#include <ctype.h>
#include <stdlib.h>
#include <getopt.h>
#include <cmath>
#include <sys/sysinfo.h>
#include <thread>
#include <math.h>
#include <mutex>
#include <condition_variable>
#include "solver.h"

#define ERROR 1
#define SUCCESS 0
#define MASTER 0

using namespace std;

double *A_ptr, *x_ptr, *b_ptr, rel;
int rows_A, cols_A, vector_size, delta, thread_counter;
mutex print_mutex, mult_mutex, mtx;
condition_variable cv;

void print_data(double* vector, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j){
      cerr << vector[i * cols + j] << " ";
    }
    cerr << endl;
  }
}

/**
 * Fills A matrix with random numbers, at the end the matrix will be diagonally dominant.
 **/
void generate_A(double* A_submatrix, int rows_per_thread, int inital_row) {
  random_device rd;
  mt19937_64 mt(rd());
  uniform_real_distribution<double> urd(-cols_A, cols_A);
  double accum;

  for (int i = 0; i < rows_per_thread; ++i) {
    accum = 0;
    for (int j = 0; j < cols_A; ++j) {
      A_submatrix[i * cols_A + j] = urd(mt);
      accum += abs(A_submatrix[i * cols_A + j]);
    }
    // The value of the diagonal will be the sum of the elements of the row plus the delta given by the user.
    double* diagonal = &A_submatrix[i * cols_A + inital_row];
    *diagonal = accum - *diagonal + delta;
    inital_row++;
  }
}

void generate_x(double* x_vector, int n){
  random_device rd;
  mt19937_64 mt(rd());
  uniform_real_distribution<double> urd(-vector_size, vector_size);

  for(int i = 0; i < n; ++i){
    x_vector[i] = urd(mt);
  }
}

void generate_b(double* b_subvector, double* A_submatrix, int rows_per_thread){
  for(int i = 0; i < rows_per_thread; ++i){
    b_subvector[i] = 0;
    for(int j = 0; j < cols_A; ++j){
      b_subvector[i] += A_submatrix[ i * cols_A + j] * x_ptr[j];
      //cout << "b[" << i << "]+= " << A_submatrix[ i * cols_A + j] << "*" << x_ptr[j] << endl;
    }
  }
}

void barrier(){
  mult_mutex.lock();
  thread_counter--;
  mult_mutex.unlock();
  if(thread_counter == 0){
    unique_lock<mutex> lck(mult_mutex);
    cv.notify_all();
  }else{
    unique_lock<mutex> lck(mult_mutex);
    while(thread_counter != 0){
      cv.wait(lck);
    }
  }
}

void generate_system(int rows_per_thread, int number_threads, int thread_id){
  int initial_thread_row = thread_id * rows_per_thread;
  if(thread_id == number_threads - 1){
    rows_per_thread = rows_A - initial_thread_row;
  }
  /*print_mutex.lock();
  cout << "thread_id: " << thread_id << " initial_thread_row: " << initial_thread_row << " rows_per_thread: " << rows_per_thread << endl;
  print_mutex.unlock();*/
  double *A_ptr_local = A_ptr + initial_thread_row * cols_A;
  generate_A(A_ptr_local, rows_per_thread, initial_thread_row);
  double *x_ptr_local = x_ptr + initial_thread_row;
  generate_x(x_ptr_local, rows_per_thread);
  barrier();
  double *b_ptr_local = b_ptr + initial_thread_row;
  generate_b(b_ptr_local, A_ptr_local, rows_per_thread);
}

int main(int argc, char** argv){

  int opt, rows_per_thread, filename_length, number_threads, niter;
  double tol;
  char* filename;
  //lck = new unique_lock(mult_mutex);

  //TODO Check arguments parser
  if(argc < 2){
    cerr << "Not enough arguments" << endl;
    return 1;
  }else{
    while ((opt = getopt(argc, argv, "n:f:d:t:r:i:e:h")) != EOF) {
      switch (opt) {
      case 'n':
	cols_A = stoi(optarg);
	rows_A = stoi(optarg);
	vector_size = stoi(optarg);
	break;
      case 'f':
	filename = optarg;
	break;
      case 't':
	number_threads = stoi(optarg);
	thread_counter = stoi(optarg);
	break;
      case 'r':
	rel = stod(optarg);
	break;
      case 'i':
	niter = stoi(optarg);
	break;
      case 'e':
	tol = stod(optarg);
	break;
      case 'd':
	delta = stof(optarg);
	break;
      case 'h':
	cout << "\nUsage:\n"
	     << "\r\t-t <Number of threads>\n"
	     << "\r\t-s <Matrix (NxN) size>\n"
	     << "\r\t-f <Output filename>\n"
	     << "\r\t-d <delta value>\n"
	     << endl;
	return 0;
      case '?':
	cerr << "Use option -h to display a help message." << endl;
	return 1;
      default:
	cerr << "Use option -h to display a help message." << endl;
	return 1;
      }
    }
  }

  A_ptr = new double[rows_A * cols_A];
  b_ptr = new double[vector_size];
  x_ptr = new double[vector_size];
  rows_per_thread = floor(rows_A / number_threads);

  thread system_threads[number_threads];
  for(int i = 0; i < number_threads; i++){
    system_threads[i] = thread(generate_system, rows_per_thread, number_threads, i);
  }

  for(int i = 0; i < number_threads; i++){
    system_threads[i].join();
  }

#ifdef DEBUG
  cout << string(50, '*') << endl;
  cout << "Matrix A: " << endl;
  cout << string(50, '*') << endl;
  print_data(A_ptr, rows_A, cols_A);
  cout << string(50, '*') << endl;
  cout << "Vector x: " << endl;
  cout << string(50, '*') << endl;
  print_data(x_ptr, vector_size, 1);
  cout << string(50, '*') << endl;
  cout << "Vector b: " << endl;
  cout << string(50, '*') << endl;
  print_data(b_ptr, vector_size, 1);
#endif //DEBUG

  solve(A_ptr, b_ptr, niter, tol);
  //print_data(x_ptr, vector_size, 1);
}
