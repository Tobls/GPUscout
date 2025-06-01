#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <vector>

__global__ void axpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

int main(int argc, char **argv) {
  uint SIZE = 512;

  float *A, *B;
  float *dA, *dB;

  A = (float *)malloc(sizeof(float) * SIZE);
  B = (float *)malloc(sizeof(float) * SIZE);

  cudaMalloc((void **)&dA, sizeof(float) * SIZE);
  cudaMalloc((void **)&dB, sizeof(float) * SIZE);

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }

  cudaMemcpy(dA, A, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

  int threads = 128;
  int blocks = 4;
  axpy<<<blocks, threads>>>(SIZE, 5, dA, dB);

  cudaMemcpy(B, dB, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

  free(A);
  free(B);
  cudaFree(dA);
  cudaFree(dB);

  return 0;
}
