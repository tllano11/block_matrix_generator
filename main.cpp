#include<iostream>
#include<stdio.h>
#include<ctype.h>
#include<stdlib.h>
#include<getopt.h>
#include<mpi.h>
#include<math.h>
#include<sys/sysinfo.h>

#define ERROR 1
#define SUCCESS 0

using namespace std;

int main (int argc, char** argv) {
  int opt, num_procs, rank;
  struct sysinfo mem_info;
  long matrix_size;
  double mem_percentage;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  while ((opt = getopt(argc, argv, "p:s:h")) != EOF) {
    switch (opt) {
    case 'p':
      mem_percentage = stof(optarg);
      break;
    case 's':
      matrix_size = stol(optarg);
      break;
    case 'h':
      cout << "\n\nUsage:\n"
	   << "\r\t-p <Percentage of RAM to use>\n"
	   << "\r\t-s <Matrix (NxN) size>\n"
	   << endl;
      break;
    case '?':
      cerr << "Use option -h to display a help message.";
    default:
      MPI_Abort(MPI_COMM_WORLD, ERROR);
    }
  }

  sysinfo(&mem_info);
  double mem_to_use = mem_info.freeram * mem_percentage;
  long row_size = matrix_size * sizeof(long double);
  int total_rows = floor(mem_to_use /row_size);
  int rows_per_proc = floor(total_rows / num_procs);

#ifdef DEBUG
  cout << "Total RAM to use = " << mem_to_use << endl;
  cout << "Single row size in RAM = " << row_size << endl;
  cout << "Rows to produce per iteration = " << total_rows << endl;
  cout << "Rows to produce per processor = " << rows_per_proc << endl;
#endif

  if (matrix_size < rows_per_proc) {
    cerr << "This program is intendend to produce large matrices, "
  	 << "which can not be loaded into RAM without swapping. "
  	 << "Therefore a matrix of size " << matrix_size
  	 << "is not suitable to produce given your system "
  	 << "resources.";
    abort();

  } else if (rows_per_proc < 1) {
    cerr << "Not even a single row of a " << matrix_size << "x"
  	 << matrix_size << " matrix can be loaded without into"
  	 << "your system RAM without swapping.";
    abort();
  }

  //cout << (matrix_size / (long) total_rows) << endl;

  //long double* data = new long double[rows_per_proc * matrix_size];
  //delete data;


  MPI_Finalize();
  return SUCCESS;
}
