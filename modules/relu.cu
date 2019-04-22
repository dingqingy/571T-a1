#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void relu(double *out, double *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        out[tid] = (in[tid]>0)?in[tid]:0;
    }
}

int main(){
    double *h_in *h_out;
    double *d_in, *d_out; 

    // Allocate host memory
    h_in  = (double*)malloc(sizeof(double) * N);
    h_out = (double*)malloc(sizeof(double) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        h_in[i] = (i%3==0)?1:-1;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_in, sizeof(double) * N);
    cudaMalloc((void**)&d_out, sizeof(double) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_in, h_in, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Executing kernel
    // <<<# of thread block per grid, # of threads per block>>>
    int block_size = 256;
    int grid_size = (N / block_size) + 1;
    relu<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(h_out, d_out, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // TODO: implement another vector sub using resulted output and h_c
    // Verification
    for(int i = 0; i < N; i++){
        ref = (h_in[i]>0)?h_in[i]:0;
        assert(fabs(ref - h_out[i]) < MAX_ERR);
    }
    // printf("out[0] = %f\n", h_out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Deallocate host memory
    free(h_in); 
    free(h_out);
}
