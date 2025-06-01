#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void test ( int* a, int* b, int* c)  {  
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  
    if (tid < 8) {  
        if (a[tid] % 2 == 0) {
            b[tid] = 1;
        }
        else {
            c[tid] = tid;
        }
    } else {
        c[tid] = a[tid];
    }
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
        A[i] = i;
        B[i] = 0;
        C[i] = 1;
    }

    cudaMemcpy(dA, A, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    test<<<gridDim, blockDim>>>(dA, dB, dC);

    free(A);
    cudaFree(dA);
    free(B);
    cudaFree(dB);
    free(C);
    cudaFree(dC);

    return 0;
}
