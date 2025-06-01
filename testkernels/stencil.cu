#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void touch2Dlinear(int* devPtr, int* outPtr, long M) {   
    int ix = blockDim.x * blockIdx.x + threadIdx.x;  
    int iy = blockDim.y * blockIdx.y + threadIdx.y;  
    int i = ix * M + iy;
    outPtr[i*M+iy] = 
        ( devPtr[(ix-1)*M+iy] + devPtr[(ix+1)*M+iy] + 
            devPtr[ix*M+(iy+1)] + devPtr[ix*M+(iy-1)] );
}

int main(int argc, char **argv) {
    uint SIZE = 32;

    int *A, *B;
    int *dA, *dB;

    A = (int *) malloc(sizeof(int) * SIZE * SIZE);
    B = (int *) malloc(sizeof(int) * SIZE * SIZE);

    cudaMalloc((void **) &dA, sizeof(int) * SIZE * SIZE);
    cudaMalloc((void **) &dB, sizeof(int) * SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; i++) {
        A[i] = i;
    }

    cudaMemcpy(dA, A, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    touch2Dlinear<<<gridDim, blockDim>>>(dA, dB, SIZE);

    cudaMemcpy(B, dB, sizeof(int) * SIZE * SIZE, cudaMemcpyDeviceToHost);

    free(B);
    cudaFree(dB);
    free(A);
    cudaFree(dA);

    return 0;
}
