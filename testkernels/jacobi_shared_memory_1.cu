#include <cmath>
#include <cstdio>
#include <stdio.h>
#include <chrono>

__global__ void jacobi(const float * __restrict__ A, const float* b, float * __restrict__ x, float* xNew, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 

    __shared__ float sharedX[512]; 
    sharedX[row] = x[row];
    __syncthreads();

    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        if (col != row) {
            sum += A[row * N + col] * sharedX[col];
        }
    }
    xNew[row] = (b[row] - sum) / A[row * N + row];
}

int main() {
    const int N = 512;
    const int maxIterations = 10000;
    const float tolerance = 1e-5f;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point stop; 

    float *A, *b, *x, *xNew;
    A = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    x = (float *)malloc(N * sizeof(float));
    xNew = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 0.0f; 
        b[i] = static_cast<float>(i);
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100 + 1;
        }
    }
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j];
        }
        A[i * N + i] = sum + (rand() % 100 + 1);
    }

    float *dA, *db, *dx, *dxNew;
    cudaMalloc((void **)&dA, N * N * sizeof(float));
    cudaMalloc((void **)&db, N * sizeof(float));
    cudaMalloc((void **)&dx, N * sizeof(float));
    cudaMalloc((void **)&dxNew, N * sizeof(float));

    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);

    start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < N; i++) {
            xNew[i] = x[i];
        }

        cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dxNew, xNew, N * sizeof(float), cudaMemcpyHostToDevice);

        jacobi<<<1, N>>>(dA, db, dx, dxNew, N);

        cudaMemcpy(xNew, dxNew, N * sizeof(float), cudaMemcpyDeviceToHost);

        float maxError = 0.0f;
        for (int i = 0; i < N; i++) {
            maxError = fmaxf(maxError, fabs(xNew[i] - x[i]));
            x[i] = xNew[i];
        }

        if (maxError < tolerance) {
            printf("Converged after %d iterations with max error: %f\n", iter + 1, maxError);
            break;
        }
    }
    stop = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("Elapsed time: %ld ms\n", elapsed);

    free(A);
    free(b);
    free(x);
    free(xNew);
    cudaFree(dA);
    cudaFree(db);
    cudaFree(dx);
    cudaFree(dxNew);

    return 0;
}
