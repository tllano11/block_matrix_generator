#include<iostream>
#include<random>
#include<limits>
#include<stdio.h>
#include<ctype.h>
#include<stdlib.h>
#include<getopt.h>
#include<mpi.h>
#include<cmath>
#include<sys/sysinfo.h>
#include<cstring>

#define ERROR 1
#define SUCCESS 0
#define MASTER 0

using namespace std;

void fill_with_random(long double* data, long rows_per_proc, long matrix_size, long inital_row, long double delta) {
  random_device rd;
  mt19937_64 mt(rd());

  uniform_real_distribution<long double> urd(-matrix_size, matrix_size);
  long double accum;

  for (long i = 0; i < rows_per_proc; ++i) {
    accum = 0;
    for (long j = 0; j < matrix_size; ++j) {
      data[i * matrix_size + j] = urd(mt);
      accum += abs(data[i * matrix_size + j]);
    }
    long double* diagonal = &data[i * matrix_size + inital_row];
    *diagonal = accum - *diagonal + delta;
    inital_row++;
  }
}

void print_data(long double* data, long rows_per_proc, long n) {
  for (long i = 0; i < rows_per_proc; ++i) {
    for (long j = 0; j < n; ++j){
      cout << data[i * n + j] << " ";
    }
    cout << endl;
  }
}

int main (int argc, char** argv) {
  int opt, num_procs, rank, rows_per_iter, rows_per_proc, max_rows_per_iter, filename_length;
  struct sysinfo mem_info;
  long matrix_size, row_size;
  double mem_percentage;
  long double delta;
  MPI_Offset my_offset, my_current_offset;
  MPI_File mpi_file;
  MPI_Status status;
  char* filename;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //TODO Check arguments parser
  if (rank == MASTER) {
    if (argc < 2){
      cerr << "Not enough arguments" << endl;
      exit(0);
    }

    while ((opt = getopt(argc, argv, "p:s:f:d:h")) != EOF) {
      switch (opt) {
      case 'p':
	mem_percentage = stof(optarg);
	break;
      case 's':
	matrix_size = stol(optarg);
	break;
      case 'f':
        filename = optarg;
        break;
      case 'h':
	cout << "\nUsage:\n"
	     << "\r\t-p <Percentage of RAM to use>\n"
	     << "\r\t-s <Matrix (NxN) size>\n"
	     << "\r\t-f <Output filename>\n"
	     << endl;
	MPI_Finalize();
	exit(0);
	break;
      case 'd':
	delta = stof(optarg);
	break;
      case '?':
	cerr << "Use option -h to display a help message." << endl;
	MPI_Finalize();
	exit(0);
	break;
      default:
	cerr << "Use option -h to display a help message." << endl;
	MPI_Finalize();
	exit(0);
      }
    }

    filename_length = strlen(filename) + 1;
    sysinfo(&mem_info);
    double mem_to_use = mem_info.freeram * mem_percentage;
    row_size = matrix_size * sizeof(long double);
    max_rows_per_iter = floor(mem_to_use /row_size);
    rows_per_proc = floor(max_rows_per_iter / num_procs);
    rows_per_iter = rows_per_proc * num_procs;
#ifdef DEBUG
    cout << string(50, '*') << endl;
    cout << "Total RAM to use = " << mem_to_use << endl;
    cout << "Single row size in RAM = " << row_size << endl;
    cout << "Rows to produce per processor = " << rows_per_proc << endl;
    cout << "Rows to produce per iteration = " << rows_per_iter << endl;
    cout << string(50, '*') << endl;
#endif // DEBUG

    if (matrix_size < rows_per_proc) {
      cerr << "This program is intendend to produce large matrices, "
	   << "which can not be loaded into RAM without swapping. "
	   << "Therefore a matrix of size " << matrix_size
	   << " is not suitable to produce given your system "
	   << "resources." << endl;
      MPI_Abort(MPI_COMM_WORLD, ERROR);

    } else if (rows_per_proc < 1) {
      cerr << "Not even a single row of a " << matrix_size << "x"
	   << matrix_size << " matrix can be loaded into "
	   << "your system RAM without swapping." << endl;
      MPI_Abort(MPI_COMM_WORLD, ERROR);
    }
  }
  MPI_Bcast(&row_size, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&matrix_size, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&rows_per_iter, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&rows_per_proc, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&filename_length, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&delta, 1, MPI_LONG_DOUBLE, MASTER, MPI_COMM_WORLD);

  if (rank != 0 ){
    filename = new char[filename_length];
  }

  MPI_Bcast(filename, filename_length, MPI_CHAR, MASTER, MPI_COMM_WORLD);

#ifdef DEBUG
  cout << "Rank " << rank << " received row_size = "
       << row_size << endl;
  cout << "Rank " << rank << " received rows_per_iter = "
       << rows_per_iter << endl;
  cout << "Rank " << rank << " received rows_per_proc = "
       << rows_per_proc << endl;
  cout << "Rank " << rank << " received matrix_size = "
       << matrix_size << endl;
  cout << "Rank " << rank << " Filename = "
       << filename << endl;
#endif //DEBUG

  long initial_it_row = rank * rows_per_proc;
  long final_it_row;
  long data_size = rows_per_proc * matrix_size;
  long double* data = new long double[data_size];
  my_offset = (long long)rank * (long long)sizeof(long double) * (long long)data_size;

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);

  while (initial_it_row < matrix_size) {
    MPI_File_seek(mpi_file, my_offset, MPI_SEEK_SET);
    MPI_File_get_position(mpi_file, &my_current_offset);

#ifdef DEBUG
    cout << "Rank: " << rank << " My Current Offset: " << my_current_offset << endl;
    //cout << "Rank: " << rank << " My inital row: " << initial_it_row << endl;
#endif //DEBUG

    final_it_row = initial_it_row + rows_per_proc - 1;
    if (final_it_row > matrix_size) {
      delete data;
      final_it_row = matrix_size;
      rows_per_proc = final_it_row - initial_it_row;
      cout << rows_per_proc << endl;
      data_size = matrix_size * rows_per_proc;
      data = new long double[data_size];
    }

    fill_with_random(data, rows_per_proc, matrix_size, initial_it_row, delta);
    print_data(data, rows_per_proc, matrix_size);
    MPI_File_write(mpi_file, data, data_size, MPI_LONG_DOUBLE, &status);

#ifdef DEBUG
    MPI_File_get_position(mpi_file, &my_current_offset);
    cout << "Rank: " << rank << " My Final Offset: " << my_current_offset << endl;
#endif //DEBUG

    //cout << "Rank: " << rank << " After Write" << endl;
    //print_data(data, data_size);
    initial_it_row += rows_per_iter;
    my_offset += (long long)rows_per_iter * (long long)sizeof(long double) * (long long)matrix_size;
  }
  MPI_File_close(&mpi_file);


  MPI_Finalize();
  return SUCCESS;
}
