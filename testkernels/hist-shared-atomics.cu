#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int NUM_BINS = 256;
const int NUM_PARTS = 4;

struct PixelType {
    int x;
    int y;
    int z;
};

__global__ void Hist(const PixelType *in, int width, int height, unsigned int *out)
{
  // pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // grid dimensions
  int nx = blockDim.x * gridDim.x; 
  int ny = blockDim.y * gridDim.y;

  // linear thread index within 2D block
  int t = threadIdx.x + threadIdx.y * blockDim.x; 

  // total threads in 2D block
  int nt = blockDim.x * blockDim.y; 
  
  // linear block index within 2D grid
  int g = blockIdx.x + blockIdx.y * gridDim.x;

  // initialize temporary accumulation array in shared memory
  __shared__ unsigned int smem[3 * NUM_BINS];
  for (int i = t; i < 3 * NUM_BINS; i += nt) smem[i] = 0;
  __syncthreads();

  // process pixels
  // updates our block's partial histogram in global memory
  for (int col = x; col < width; col += nx) 
    for (int row = y; row < height; row += ny) { 
      unsigned int r = (unsigned int)(256 * in[row * width + col].x);
      unsigned int g = (unsigned int)(256 * in[row * width + col].y);
      unsigned int b = (unsigned int)(256 * in[row * width + col].z);
      atomicAdd(&smem[NUM_BINS * 0 + r], 1);
      atomicAdd(&smem[NUM_BINS * 1 + g], 1);
      atomicAdd(&smem[NUM_BINS * 2 + b], 1);
    }
  __syncthreads();

  out += g * NUM_PARTS;
  for (int i = t; i < NUM_BINS; i += nt) {
    out[i + NUM_BINS * 0] = smem[i + NUM_BINS * 0];
    out[i + NUM_BINS * 1] = smem[i + NUM_BINS * 1];
    out[i + NUM_BINS * 2] = smem[i + NUM_BINS * 2];
  }
}

int main(int argc, char **argv) {
    uint SIZE = 32;

    PixelType *A;
    unsigned int *B;
    PixelType *dA;
    unsigned int *dB;

    A = (PixelType *) malloc(sizeof(PixelType) * SIZE * SIZE);
    B = (unsigned int *) malloc(sizeof(unsigned int) * SIZE * SIZE);

    cudaMalloc((void **) &dA, sizeof(PixelType) * SIZE * SIZE);
    cudaMalloc((void **) &dB, sizeof(unsigned int) * SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; i++) {
        A[i].x = 1;
        A[i].y = 1;
        A[i].z = 1;
        B[i] = 0;
    }

    cudaMemcpy(dA, A, sizeof(PixelType) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(unsigned int) * SIZE * SIZE, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    Hist<<<gridDim, blockDim>>>(dA, SIZE, SIZE, dB);

    free(A);
    cudaFree(dA);
    free(B);
    cudaFree(dB);

    return 0;
}
