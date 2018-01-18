#include<iostream>
#include<random>
#include<limits>
#include<cstdio>
#include<ctype.h>
#include<stdlib.h>
#include<getopt.h>
#include<cmath>
#include<sys/sysinfo.h>
#include<cstring>
#include<thread>
#include<math.h>
#include<sstream>
//#include "solver.h"

#define ERROR 1
#define SUCCESS 0
#define MASTER 0

using namespace std;

double *A_ptr, *x_ptr, *b_ptr; 
int rows_A, cols_A, vector_size, delta;

// Each thread is in charge of a part of the multiplication
void generate_b(double* b_subvector, double* A_submatrix, double* x_vector, int rows_per_thread){
  for(int i = 0; i < rows_per_thread; ++i){
    b_subvector[i] = 0;
    for(int j = 0; j < cols_A; ++j){
      b_subvector[i] += A_submatrix[ i * cols_A + j] * x_vector[j];
    }
  }
}

void print_data(double* A_submatrix, long rows_per_thread, long n) {
  for (long i = 0; i < rows_per_thread; ++i) {
    for (long j = 0; j < n; ++j){
      cout << A_submatrix[i * n + j] << " ";
    }
    cout << endl;
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
  uniform_real_distribution<double> urd(-n, n);

  for(int i = 0; i < n; ++i){
    x_vector[i] = urd(mt);
  }
}

void generate_system(int rows_per_thread, int number_threads, int thread_id){
  //stringstream ss;
  //ss << this_thread::get_id();
  //int thread_id = stoull(ss.str());
  int initial_row = thread_id * rows_per_thread;
  if(thread_id == number_threads - 1){
    rows_per_thread = rows_A - initial_row;
  }
  cout << "thread_id: " << thread_id << " initial_row: " << initial_row << " rows_per_thread: " << rows_per_thread << endl;
  double *A_ptr_local = &A_ptr[0] + thread_id * rows_per_thread * cols_A;
  //generate_A(A_ptr_local, rows_per_thread, initial_row);
  double *x_ptr_local = &x_ptr[0] + thread_id * rows_per_thread;
  //generate_x(x_ptr_local, rows_per_thread);
  double *b_ptr_local = &b_ptr[0] + thread_id * rows_per_thread;
  //generate_b(b_ptr_local, A_ptr_local, x_ptr_local, rows_per_thread);
}

int main(int argc, char** argv){

  int opt, rows_per_thread, filename_length, number_threads;
  char* filename;
  
  //TODO Check arguments parser
  if(argc < 2){
    cerr << "Not enough arguments" << endl;
  }else{
    while ((opt = getopt(argc, argv, "s:f:d:t:h")) != EOF) {
      switch (opt) {
        case 's':
	        cols_A = stoi(optarg);
	        rows_A = stoi(optarg);
	        vector_size = stoi(optarg);
	        break;
        case 'f':
	        filename = optarg;
	        break;
        case 't':
          number_threads = stoi(optarg);
         break;
        case 'h':
        	cout << "\nUsage:\n"
	             << "\r\t-t <Number of threads>\n"
	             << "\r\t-s <Matrix (NxN) size>\n"
	             << "\r\t-f <Output filename>\n"
	             << "\r\t-d <delta value>\n"
	             << endl;
	        break;
        case 'd':
	        delta = stof(optarg);
	        break;
        case '?':
	        cerr << "Use option -h to display a help message." << endl;
	        break;
       default:
	        cerr << "Use option -h to display a help message." << endl;
      }
    }
  }    
  rows_per_thread = floor(rows_A/number_threads);
  thread system_threads[number_threads];
  for(int i = 0; i < number_threads; i++){
    system_threads[i] = thread(generate_system, rows_per_thread, number_threads, i);
  }

  for(int i = 0; i < number_threads; i++){
    system_threads[i].join();
  }
}
