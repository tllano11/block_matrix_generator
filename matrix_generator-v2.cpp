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

void fill_with_random(double* A_submatrix, long rows_per_proc, long cols_num, long inital_row, double delta) {
  random_device rd;
  mt19937_64 mt(rd());
  uniform_real_distribution<double> urd(-cols_num, cols_num);
  double accum;

  for (long i = 0; i < rows_per_proc; ++i) {
    accum = 0;
    for (long j = 0; j < cols_num; ++j) {
      A_submatrix[i * cols_num + j] = urd(mt);
      accum += abs(A_submatrix[i * cols_num + j]);
    }
    double* diagonal = &A_submatrix[i * cols_num + inital_row];
    *diagonal = accum - *diagonal + delta;
    inital_row++;
  }
}

void generate_x_vector(double* x_vector, int n){
  random_device rd;
  mt19937_64 mt(rd());
  uniform_real_distribution<double> urd(-n, n);

  for(int i = 0; i < n; ++i){
    x_vector[i] = urd(mt);
  }
}

void generate_b_subvector(double* b_subvector, double* A_submatrix, double* x_vector, long cols_num, long rows_per_proc){
  for(long i = 0; i < rows_per_proc; ++i){
    b_subvector[i] = 0;
    for(long j = 0; j < cols_num; ++j){
      b_subvector[i] += A_submatrix[ i * cols_num + j] * x_vector[j];
    }
  }
}

void print_data(double* A_submatrix, long rows_per_proc, long n) {
  for (long i = 0; i < rows_per_proc; ++i) {
    for (long j = 0; j < n; ++j){
      cout << A_submatrix[i * n + j] << " ";
    }
    cout << endl;
  }
}

void generate_b_gpu(double *hostA, double *hostX, double *hostB, long cols, long rows);

int main (int argc, char** argv) {
  int opt, num_procs, rank, rows_per_iter, rows_per_proc, max_rows_per_iter,
    filename_length, exit, ret_code;
  struct sysinfo mem_info;
  long cols_num, row_size;
  double mem_percentage;
  double delta;
  char* filename;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //TODO Check arguments parser
  if (rank == MASTER) {
    exit = false;
    ret_code = 0;

    if (argc < 2){
      cerr << "Not enough arguments" << endl;
      exit = true;
      ret_code = 1;
    } else {
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
	  exit = true;
	  break;
	case 'd':
	  delta = stof(optarg);
	  break;
	case '?':
	  cerr << "Use option -h to display a help message." << endl;
	  exit = true;
	  ret_code = 1;
	  break;
	default:
	  cerr << "Use option -h to display a help message." << endl;
	  exit = true;
	  ret_code = 1;
	}
      }
    }

    if (! exit) {
      filename_length = strlen(filename) + 1;
      sysinfo(&mem_info);
      double mem_to_use = mem_info.freeram * mem_percentage;
      row_size = cols_num * sizeof(double);

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
	cerr << "The matrix is too SMALL; you are trying to produce a matrix that fits in the memory" << endl;
	exit = true;
	ret_code = 1;

      } else if (rows_per_proc < 1) {
	cerr << "The matrix is too BIG; not even one row fits in the memory " << endl;
	exit = true;
	ret_code = 1;
      }
    }
  }

  MPI_Bcast(&exit, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&ret_code, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  if (exit) {
    MPI_Finalize();
    return ret_code;
  }

  MPI_Bcast(&row_size, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&cols_num, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&rows_per_iter, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&rows_per_proc, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&filename_length, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&delta, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

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
  long A_submatrix_size = rows_per_proc * cols_num;
  double* A_submatrix = new double[ A_submatrix_size ];
  double* b_subvector = new double[ rows_per_proc ];

  double start_time, end_time, delta_time, longest_time;

  A_offset = (long long)rank * (long long)sizeof(double) * (long long)A_submatrix_size;
  b_offset = (long long)rank * (long long)sizeof(double) * (long long)rows_per_proc;

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &A_file);
  MPI_File_open(MPI_COMM_WORLD, b_filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &b_file);

  start_time = MPI_Wtime();
  double* x_vector = new double[ cols_num ];

  int pos_by_proc = round(cols_num/(double)num_procs);
  //int initial_pos = rank * pos_by_proc;
  int* displacements = new int[ num_procs ];
  int* recvcounts = new int[ num_procs ];

  for(int i = 0; i < num_procs; ++i){
    displacements[i] = i * pos_by_proc;

    if (i == num_procs - 1){
      pos_by_proc = cols_num - displacements[i];
    }

    recvcounts[i] = pos_by_proc;
  }

  //printf("Rank: %d pos_proc: %d init_pos: %d\n", rank, recvcounts[rank], displacements[rank]);

  double* x_subvector = new double[ recvcounts[rank] ];

  generate_x_vector(x_subvector, recvcounts[rank]);
  //print_data(x_subvector, recvcounts[rank], 1);

  MPI_Allgatherv(x_subvector, recvcounts[rank], MPI_DOUBLE, x_vector, recvcounts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
  end_time = MPI_Wtime();

  delta_time = end_time - start_time;

  if(rank == MASTER){
    int x_filename_length = filename_length + 2;
    char x_filename[x_filename_length];
    strcpy(x_filename, "x_");
    strcat(x_filename, filename);

    if(FILE* f1 = fopen(x_filename, "wb")) {
      fwrite(x_vector, sizeof(double), cols_num, f1);
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
    //cout << "Rank: " << rank << " My initial row: " << initial_it_row << endl;
    cout << "Rank: " << rank << endl;
#endif //DEBUG

    final_it_row = initial_it_row + rows_per_proc - 1;
    if (final_it_row > cols_num) {
      delete[] A_submatrix;
      final_it_row = cols_num;
      rows_per_proc = final_it_row - initial_it_row;
      A_submatrix_size = cols_num * rows_per_proc;
      A_submatrix = new double[A_submatrix_size];
    }

    fill_with_random(A_submatrix, rows_per_proc, cols_num, initial_it_row, delta);
    //print_data(A_submatrix, rows_per_proc, cols_num);

    //generate_b_subvector(b_subvector, A_submatrix, x_vector, cols_num, rows_per_proc);
    generate_b_gpu(A_submatrix, x_vector, b_subvector, cols_num, rows_per_proc);

    MPI_File_write(A_file, A_submatrix, A_submatrix_size, MPI_DOUBLE, &A_status);
    MPI_File_write(b_file, b_subvector, rows_per_proc, MPI_DOUBLE, &b_status);

#ifdef DEBUG
    MPI_File_get_position(A_file, &A_current_offset);
    cout << "Rank: " << rank << " My Final Offset: " << A_current_offset << endl;
#endif //DEBUG

    initial_it_row += rows_per_iter;
    rows_per_iter_size = (long long)rows_per_iter * (long long)sizeof(double);
    A_offset += rows_per_iter_size * (long long)cols_num;
    b_offset += rows_per_iter_size;
  }

  MPI_File_close(&A_file);
  MPI_File_close(&b_file);

  delete[] A_submatrix;
  if(rank != MASTER){
    delete[] filename;
  }
  delete[] x_vector;
  delete[] b_subvector;
  delete[] x_subvector;

  MPI_Reduce(&delta_time, &longest_time, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);

  if(rank == MASTER){
    printf("The longest time to produce and get x vector %f secs\n", longest_time);
  }

  MPI_Finalize();
  return SUCCESS;
}
