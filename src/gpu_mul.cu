#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

using namespace std;

const int Tile_size = 2;

// Compute C = A * B
//*************************************************************
//Kernel for shared memory/ Tiled execution
__global__ void matrixMul(double *A, double *X, double *B, long rows, long cols){
  int tid= threadIdx.x + blockIdx.x * blockDim.x;
  double sum= 0;
  if(tid < rows){
    for(int i=0; i < cols; i++){
      sum += X[i] * A[(i * rows) + tid];
    }
    B[tid]=sum;
  }
}
//*************************************************************
void Print_Mat(int Row,int Col,double * Mat){//Function To print the Matrix

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
    for (int j= 0; j < numAColumns; j++){
      C[i*numCColumns + j ] = 0.0;
      for (int k = 0; k < numCColumns; k++){
	C[i*numCColumns + j ] += A[i*numAColumns + k] * B [k*numBColumns + j];
      }
    }
  }
}
//*************************************************************
extern "C++" void generate_b_gpu(double *hostA, double *hostX, double *hostB, long cols, long rows) {

  double *deviceA;
  double *deviceB;
  double *deviceX;

  // Allocating GPU memory
  assert(cudaSuccess == cudaMalloc((void **)&deviceA, sizeof(double)*cols*rows));
  assert(cudaSuccess == cudaMalloc((void **)&deviceB, sizeof(double)*rows));
  assert(cudaSuccess == cudaMalloc((void **)&deviceX, sizeof(double)*rows));

    // Copy memory to the GPU
  assert(cudaSuccess == cudaMemcpy(deviceA, hostA, sizeof(double)*cols*rows, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(deviceX, hostX, sizeof(float)*rows, cudaMemcpyHostToDevice));

  // Initialize the grid and block dimensions

  dim3 dimGrid((1/Tile_size) + 1, (rows/Tile_size) + 1, 1);//Number of Blocks required
  dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block

  //@@ Launch the GPU Kernel here
  matrixMul<<<dimGrid, dimBlock>>>(deviceA, deviceX, deviceB, rows, cols);

  cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

  cudaDeviceSynchronize();//To synchronize the device

  // Copy the results in GPU memory back to the CPU
  assert(cudaSuccess == cudaMemcpy(hostB, deviceB, sizeof(float)*rows, cudaMemcpyDeviceToHost));

  cout << "GPU A" << endl;
  Print_Mat(rows, cols, hostA);

  //matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //printf("\nMatrix C From Host\n");
  //Print_Mat(numCRows,numCColumns,hostComputedC);//Function Call

  //printf("\n Number of Blocks Created:%d \n",((1/Tile_size) + 1)*((1/Tile_size) + 1));
  //printf("\n Number of Threads Per Block: %d \n",(Tile_size*Tile_size));

  // Free the GPU memory
  assert(cudaSuccess == cudaFree(deviceA));
  assert(cudaSuccess == cudaFree(deviceB));
  assert(cudaSuccess == cudaFree(deviceX));
  //Free the Pointer Memory
  //free(hostA);
  //free(hostB);
  //free(hostX);
}