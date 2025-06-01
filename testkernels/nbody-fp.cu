#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

typedef struct { float x, y, z, vx, vy, vz; } Body;   

__global__ void bodyForce ( Body *p, float dt, int n ) {   
    int i = blockDim.x * blockIdx.x + threadIdx.x;  
    if (i < n) {   
        float Fx = 0.0f; 
        float Fy = 0.0f; 
        float Fz = 0.0f;   
        for (int j = 0; j < n; j++) { 
            float dx = p[j].x - p[i].x; 
            float dy = p[j].y - p[i].y; 
            float dz = p[j].z - p[i].z; 
            double distSqr = dx*dx + dy*dy + dz*dz + 0.2f; 
            double invDist = rsqrtf(distSqr); 
            double invDist3 = invDist * invDist * invDist; 
            Fx += dx * invDist3; 
            Fy += dy * invDist3; 
            Fz += dz * invDist3; 
        } 
        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz; 
    } 
}

int main(int argc, char **argv) {
    uint SIZE = 32;

    Body *B;
    Body *dB;

    B = (Body *) malloc(sizeof(Body) * SIZE * SIZE);

    cudaMalloc((void **) &dB, sizeof(Body) * SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; i++) {
        B[i].x = i;
        B[i].y = i + 1;
        B[i].z = i + 1;
        B[i].vx = i;
        B[i].vy = i + 1;
        B[i].vz = i + 1;
    }

    cudaMemcpy(dB, B, sizeof(Body) * SIZE * SIZE, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    bodyForce<<<gridDim, blockDim>>>(dB, 0.5f, SIZE);

    cudaMemcpy(B, dB, sizeof(Body) * SIZE * SIZE, cudaMemcpyDeviceToHost);

    free(B);
    cudaFree(dB);

    return 0;
}
