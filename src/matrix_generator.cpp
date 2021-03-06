#include<iostream>
#include<random>
#include<limits>
#include<cstdio>
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

void fill_with_random(long double* data, long rows_per_proc, long cols_num, long inital_row, long double delta) {
  random_device rd;
  mt19937_64 mt(rd());
  uniform_real_distribution<long double> urd(-cols_num, cols_num);
  long double accum;

  for (long i = 0; i < rows_per_proc; ++i) {
    accum = 0;
    for (long j = 0; j < cols_num; ++j) {
      data[i * cols_num + j] = urd(mt);
      accum += abs(data[i * cols_num + j]);
    }
    long double* diagonal = &data[i * cols_num + inital_row];
    *diagonal = accum - *diagonal + delta;
    inital_row++;
  }
}

void generate_x_vector(long double* x_vector, long n){
  random_device rd;
  mt19937_64 mt(rd());
  uniform_real_distribution<long double> urd(-n, n);

  for(long i = 0; i < n; ++i){
    x_vector[i] = urd(mt);
  }
}

void generate_b_vector(long double* b_vector, long double* data, long double* x_vector, long cols_num, long rows_per_proc){
  for(long i = 0; i < rows_per_proc; ++i){
    b_vector[i] = 0;
    for(long j = 0; j < cols_num; ++j){
      b_vector[i] += data[ i * cols_num + j] * x_vector[j];
    }
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
  long cols_num, row_size;
  double mem_percentage;
  long double delta;
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
	cols_num = stol(optarg);
	break;
      case 'f':
        filename = optarg;
        break;
      case 'h':
	cout << "\nUsage:\n"
	     << "\r\t-p <Percentage of RAM to use>\n"
	     << "\r\t-s <Matrix (NxN) size>\n"
	     << "\r\t-f <Output filename>\n"
	     << "\r\t-d <delta value>\n"
	     << endl;
	MPI_Abort(MPI_COMM_WORLD, 0);
	break;
      case 'd':
	delta = stof(optarg);
	break;
      case '?':
	cerr << "Use option -h to display a help message." << endl;
	MPI_Abort(MPI_COMM_WORLD, 0);
	break;
      default:
	cerr << "Use option -h to display a help message." << endl;
	MPI_Abort(MPI_COMM_WORLD, 0);
      }
    }

    filename_length = strlen(filename) + 1;
    sysinfo(&mem_info);
    double mem_to_use = mem_info.freeram * mem_percentage;
    row_size = cols_num * sizeof(long double);

    /* Substract the size occupied by the x vector times the number of process,
       because they will need it to perform the matrix multiplication of its own submatrix.*/
    mem_to_use -= row_size * num_procs;

    // Substract one additional row_size the allocate space for the b vector.
    mem_to_use -= row_size;

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

    if (cols_num < max_rows_per_iter) {
      cerr << "This program is intendend to produce large matrices, "
	   << "which can not be loaded into RAM without swapping. "
	   << "Therefore a matrix of size " << cols_num
	   << " is not suitable to produce given your system "
	   << "resources." << endl;
      MPI_Abort(MPI_COMM_WORLD, ERROR);

    } else if (rows_per_proc < 1) {
      cerr << "Not even a single row of a " << cols_num << "x"
	   << cols_num << " matrix can be loaded into "
	   << "your system RAM without swapping." << endl;
      MPI_Abort(MPI_COMM_WORLD, ERROR);
    }
  }
  MPI_Bcast(&row_size, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&cols_num, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&rows_per_iter, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&rows_per_proc, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&filename_length, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&delta, 1, MPI_LONG_DOUBLE, MASTER, MPI_COMM_WORLD);

  if (rank != MASTER){
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
  cout << "Rank " << rank << " received cols_num = "
       << cols_num << endl;
  cout << "Rank " << rank << " Filename = "
       << filename << endl;
#endif //DEBUG

  MPI_Offset A_offset, A_current_offset, b_offset, b_current_offset;
  MPI_File A_file, b_file;
  MPI_Status A_status, b_status;

  int b_filename_length = filename_length + 2;
  char b_filename[b_filename_length];
  strcpy(b_filename, "b_");
  strcat(b_filename, filename);

  long initial_it_row = rank * rows_per_proc;
  long final_it_row;
  long long rows_per_iter_size;
  long data_size = rows_per_proc * cols_num;
  long double* data = new long double[ data_size ];
  long double* b_vector = new long double[ cols_num ];

  double start_time, end_time, delta_time, longest_time;

  A_offset = (long long)rank * (long long)sizeof(long double) * (long long)data_size;
  b_offset = (long long)rank * (long long)sizeof(long double) * (long long)rows_per_proc;

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &A_file);
  MPI_File_open(MPI_COMM_WORLD, b_filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &b_file);

  start_time = MPI_Wtime();
  long double* x_vector = new long double[ cols_num ];
  if(rank == MASTER){
    generate_x_vector(x_vector, cols_num);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(x_vector, cols_num, MPI_LONG_DOUBLE, MASTER, MPI_COMM_WORLD);

  end_time = MPI_Wtime();

  delta_time = end_time - start_time;

  MPI_Reduce(&delta_time, &longest_time, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);

  if(rank == MASTER){
    printf("The longest time to produce and get x vector %f secs\n", longest_time);
  }

  MPI_Finalize();
  return 0;

  if(rank == MASTER){
    int x_filename_length = filename_length + 2;
    char x_filename[x_filename_length];
    strcpy(x_filename, "x_");
    strcat(x_filename, filename);

    if(FILE* f1 = fopen(x_filename, "wb")) {
      fwrite(x_vector, sizeof(long double), cols_num, f1);
      fclose(f1);
    }

    //print_data(x_vector, cols_num, 1);
  }

  while (initial_it_row < cols_num) {
    MPI_File_seek(A_file, A_offset, MPI_SEEK_SET);
    MPI_File_get_position(A_file, &A_current_offset);

    MPI_File_seek(b_file, b_offset, MPI_SEEK_SET);
    MPI_File_get_position(b_file, &b_current_offset);

#ifdef DEBUG
    //cout << "Rank: " << rank << " My Current Offset: " << A_current_offset << endl;
    //cout << "Rank: " << rank << " My inital row: " << initial_it_row << endl;
    cout << "Rank: " << rank << endl;
#endif //DEBUG

    final_it_row = initial_it_row + rows_per_proc - 1;
    if (final_it_row > cols_num) {
      delete data;
      final_it_row = cols_num;
      rows_per_proc = final_it_row - initial_it_row;
      data_size = cols_num * rows_per_proc;
      data = new long double[data_size];
    }

    fill_with_random(data, rows_per_proc, cols_num, initial_it_row, delta);
    //print_data(data, rows_per_proc, cols_num);

    generate_b_vector(b_vector, data, x_vector, cols_num, rows_per_proc);

    // cout << string(50, '*') << endl;
    //print_data(b_vector, rows_per_proc, 1);

    MPI_File_write(A_file, data, data_size, MPI_LONG_DOUBLE, &A_status);
    MPI_File_write(b_file, b_vector, rows_per_proc, MPI_LONG_DOUBLE, &b_status);

#ifdef DEBUG
    MPI_File_get_position(A_file, &A_current_offset);
    cout << "Rank: " << rank << " My Final Offset: " << A_current_offset << endl;
#endif //DEBUG

    initial_it_row += rows_per_iter;
    rows_per_iter_size = (long long)rows_per_iter * (long long)sizeof(long double);
    A_offset += rows_per_iter_size * (long long)cols_num;
    b_offset += rows_per_iter_size;
  }

  MPI_File_close(&A_file);
  MPI_File_close(&b_file);

  delete[] data;
  if(rank != MASTER){
    delete[] filename;
  }
  delete[] x_vector;
  delete[] b_vector;

  MPI_Finalize();
  return SUCCESS;
}
