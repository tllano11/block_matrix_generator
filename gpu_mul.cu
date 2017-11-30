#include "gpu_mul.h"

// Compute C = A * B
//*************************************************************
//Kernel for shared memory/ Tiled execution
__global__ void matrixMul(double *A, double *B, double *C, long rows, long cols){


  __shared__ float sA[Tile_size][Tile_size];   // Tile size to store elements in shared memory
  __shared__ float sB[Tile_size][Tile_size];

  int Row = blockDim.y*blockIdx.y + threadIdx.y; //To generate ids of threads.
  int Col = blockDim.x*blockIdx.x + threadIdx.x;
  float Cvalue = 0.0;
  sA[threadIdx.y][threadIdx.x] = 0.0;
  sB[threadIdx.y][threadIdx.x] = 0.0;

  for (int k = 0; k < (((cols - 1)/ Tile_size) + 1); k++){
    if ( (Row < rows) && (threadIdx.x + (k*Tile_size)) < cols){//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
      sA[threadIdx.y][threadIdx.x] = A[(Row*cols) + threadIdx.x + (k*Tile_size)];
    } else{
      sA[threadIdx.y][threadIdx.x] = 0.0;
    }

    // 1 is the amount of B columns
    if ( Col < 1 && (threadIdx.y + k*Tile_size) < rows){//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
      sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*Tile_size) + Col];
    } else{
      sB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    for (int j = 0; j < Tile_size; ++j){//Multiplying Elements present in tile
      Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
  }

  if (Row < rows && Col < 1){//Saving Final result into Matrix C
    C[Row + Col] = Cvalue;
  }
}
//*************************************************************
void Print_Mat(int Row,int Col,float * Mat){//Function To print the Matrix

  for(int i=0; i < Row; ++i){
    for(int j=0; j < Col; ++j){
      cout << Mat[i * Row + j] << " ";
    }
    cout << endl;
  }
}//Function close
//*************************************************************

//Normal CPU Matrix Multiplication
void matMultiplyOnHost(float * A, float * B, float * C, int numARows,
                        int numAColumns, int numBRows, int numBColumns,
                        int numCRows, int numCColumns){
  for (int i=0; i < numARows; i ++){
    for (int j = 0; j < numAColumns; j++){
      C[i*numCColumns + j ] = 0.0;
      for (int k = 0; k < numCColumns; k++){
	C[i*numCColumns + j ] += A[i*numAColumns + k] * B [k*numBColumns + j];
      }
    }
  }
  return;
}
//*************************************************************
void generate_b_gpu(double *hostA, double *hostX, double *hostB, long cols, long rows) {

  double *deviceA;
  double *deviceB;
  double *deviceX;

  // Allocating GPU memory
  assert(cudaSuccess == cudaMalloc((void **)&deviceA, sizeof(double)*cols*rows));
  assert(cudaSuccess == cudaMalloc((void **)&deviceB, sizeof(double)*rows));
  assert(cudaSuccess == cudaMalloc((void **)&deviceC, sizeof(double)*rows));

    // Copy memory to the GPU
  assert(cudaSuccess == cudaMemcpy(deviceA, hostA, sizeof(double)*cols*rows, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(deviceB, hostB, sizeof(float)*rows, cudaMemcpyHostToDevice));

  // Initialize the grid and block dimensions

  dim3 dimGrid((1/Tile_size) + 1, (rows/Tile_size) + 1, 1);//Number of Blocks required
  dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block

  //@@ Launch the GPU Kernel here
  matrixMul<<<dimGrid, dimBlock>>>(deviceA, deviceX, deviceB, rows, cols);

  cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

  cudaDeviceSynchronize();//To synchronize the device

  // Copy the results in GPU memory back to the CPU
  assert(cudaSuccess == cudaMemcpy(hostB, deviceB, sizeof(float)*rows, cudaMemcpyDeviceToHost));

  cout << hostB[0] << endl;

  //matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //printf("\nMatrix C From Host\n");
  //Print_Mat(numCRows,numCColumns,hostComputedC);//Function Call

  printf("\n Number of Blocks Created:%d \n",((1/Tile_size) + 1)*((1/Tile_size) + 1));
  printf("\n Number of Threads Per Block: %d \n",(Tile_size*Tile_size));

  // Free the GPU memory
  assert(cudaSuccess == cudaFree(deviceA));
  assert(cudaSuccess == cudaFree(deviceB));
  assert(cudaSuccess == cudaFree(deviceX));
  //Free the Pointer Memory
  free(hostA);
  free(hostB);
  free(hostX);

  return 0;
}