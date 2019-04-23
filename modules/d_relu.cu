#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void d_relu(double *out, double *in, double *act, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        out[tid] = (act[tid]>0)?in[tid]:0;
    }
}

int main(){
    double *h_in, *h_act, *h_out;
    double *d_in, *d_act, *d_out; 
    double ref;
    // Allocate host memory
    h_in  = (double*)malloc(sizeof(double) * N);
    h_act  = (double*)malloc(sizeof(double) * N);
    h_out = (double*)malloc(sizeof(double) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        h_in[i] = (i%3)?1.0:-1.0;
    }
    for(int i = 0; i < N; i++){
        h_act[i] = (i%2)?1.0:-1.0;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_in, sizeof(double) * N);
    cudaMalloc((void**)&d_act, sizeof(double) * N);
    cudaMalloc((void**)&d_out, sizeof(double) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_in, h_in, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, h_act, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Executing kernel
    // <<<# of thread block per grid, # of threads per block>>>
    int block_size = 256;
    int grid_size = (N / block_size) + 1;
    d_relu<<<grid_size,block_size>>>(d_out, d_in, d_act, N);
    
    // Transfer data back to host memory
    cudaMemcpy(h_out, d_out, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // TODO: implement another vector sub using resulted output and h_c
    // Verification
    for(int i = 0; i < N; i++){
        ref = (h_act[i]>0)?h_in[i]:0;
        assert(fabs(ref - h_out[i]) < MAX_ERR);
    }
    for(int i = 0; i < 10; i++){
        printf("out[%d] = %f\n",i, h_out[i]);
    }
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_in);
    cudaFree(d_act);
    cudaFree(d_out);

    // Deallocate host memory
    free(h_in); 
    free(h_act); 
    free(h_out);
}
