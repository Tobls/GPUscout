#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void copy_kernel ( int* d_in, int* d_out, int N ) { 
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
    reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
  }

  // in only one thread, process final elements (if there are any)
  int remainder = N%4;
  if (idx==N/4 && remainder!=0) {
    while(remainder) {
      int idx = N - remainder--;
      d_out[idx] = d_in[idx];
    }
  }
}

int main(int argc, char **argv) {
    uint SIZE = 512;

    int *A, *B;
    int *dA, *dB;

    A = (int *) malloc(sizeof(int) * SIZE);
    B = (int *) malloc(sizeof(int) * SIZE);

    cudaMalloc((void **) &dA, sizeof(int) * SIZE);
    cudaMalloc((void **) &dB, sizeof(int) * SIZE);

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
    }

    cudaMemcpy(dA, A, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    int threads = 128; 
    int blocks = 1;  
    copy_kernel<<<blocks, threads>>>(dA, dB, SIZE); 

    cudaMemcpy(B, dB, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

    free(A);
    free(B);
    cudaFree(dA);
    cudaFree(dB);

    return 0;
}
