#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void touch2Dlinear(cudaTextureObject_t devPtr, int* outPtr, long M) {   
    int ix = blockDim.x * blockIdx.x + threadIdx.x;  
    int iy = blockDim.y * blockIdx.y + threadIdx.y;  
    int i = ix * M + iy;
    outPtr[i*M+iy] = 
        ( tex1Dfetch<int>(devPtr, (ix-1)*M+iy) + tex1Dfetch<int>(devPtr, (ix+1)*M+iy) + 
          tex1Dfetch<int>(devPtr, ix*M+(iy-1)) + tex1Dfetch<int>(devPtr, ix*M+(iy+1)));
}

int main(int argc, char **argv) {
    uint SIZE = 32;

    int *A, *B;
    int *dB;

    A = (int *) malloc(sizeof(int) * SIZE * SIZE);
    B = (int *) malloc(sizeof(int) * SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; i++) {
        A[i] = i;
    }

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = A;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = SIZE * SIZE * sizeof(int);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex=0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

    cudaMalloc((void **) &dB, sizeof(int) * SIZE * SIZE);

    cudaMemcpy(dB, B, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    touch2Dlinear<<<gridDim, blockDim>>>(tex, dB, SIZE);

    cudaMemcpy(B, dB, sizeof(int) * SIZE * SIZE, cudaMemcpyDeviceToHost);

    cudaDestroyTextureObject(tex);

    free(B);
    cudaFree(dB);
    free(A);

    return 0;
}
