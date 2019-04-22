#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_sub(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        out[tid] = a[tid] - b[tid];
    }
}

int main(){
    float *h_a, *h_b, *h_out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    h_a   = (float*)malloc(sizeof(float) * N);
    h_b   = (float*)malloc(sizeof(float) * N);
    h_out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        h_a[i] = 3.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel
    // <<<# of thread block per grid, # of threads per block>>>
    int block_size = 256;
    int grid_size = (N / block_size) + 1;
    vector_sub<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // TODO: implement another vector sub using resulted output and h_c
    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(h_a[i] - h_out[i] - h_b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", h_out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(h_a); 
    free(h_b); 
    free(h_out);
}
