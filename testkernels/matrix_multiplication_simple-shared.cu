#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void kernel(int *a, int *b, int *c, int m, int n, int k) {
    __shared__ float aTile[32][32], bTile[32][32];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    aTile[threadIdx.y][threadIdx.x] = a[row*n+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*n+col];
    __syncthreads();
    for (int i = 0; i < n; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*n+col] = sum;
}

int main(int argc, char **argv) {
    uint SIZE = 32;

    int *A, *B, *C;
    int *dA, *dB, *dC;

    A = (int *) malloc(sizeof(int) * SIZE * SIZE);
    B = (int *) malloc(sizeof(int) * SIZE * SIZE);
    C = (int *) malloc(sizeof(int) * SIZE * SIZE);

    cudaMalloc((void **) &dA, sizeof(int) * SIZE * SIZE);
    cudaMalloc((void **) &dB, sizeof(int) * SIZE * SIZE);
    cudaMalloc((void **) &dC, sizeof(int) * SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    cudaMemcpy(dA, A, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    kernel<<<gridDim, blockDim>>>(dA, dB, dC, SIZE, SIZE, SIZE);

    cudaMemcpy(C, dC, sizeof(int) * SIZE * SIZE, cudaMemcpyDeviceToHost);

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
