#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

__global__ void vector_sub(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] - b[i];
    }
}

int main(){
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);

    b   = (float*)malloc(sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    out = (float*)malloc(sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);
    cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add<<<1, 1>>>(out, a, b, N);
    vector_sub<<<1, 1>>>(out, a, b, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(out);
}

