#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// matrix transpose: from (out_w * in_w) to (in_w * out_w)
#define IN_WIDTH 1000 
#define OUT_WIDTH 100
#define N IN_WIDTH * OUT_WIDTH

#define BLOCK_SIZE 16
#define MAX_ERR 1e-6

__global__ void matrix_transpose(float *d_out, float *d_in, int d_in_width, int d_out_width) {
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    if(cid < d_in_width && rid < d_out_width){
        d_out[cid * d_out_width + rid] = d_in[rid * d_in_width + cid];
    }
}

int main(){
    float *h_in, *h_out;
    float *d_in, *d_out; 

    // Allocate host memory
    h_in = (float*)malloc(sizeof(float) * N);
    h_out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    
    /***       TEST 1    ***/
    // for(int i = 0; i < OUT_WIDTH; i++){
    //     for(int j = 0; j < IN_WIDTH; j++){
    //         h_in[i*IN_WIDTH + j] = (float)i;
    //     }
    // }
    
    /***       TEST 2    ***/
    for (int i = 0; i< N; i++){
        h_in[i] = (float) i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_in, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Note x dim (rid) mapped to OUT_WIDTH
    dim3 dimGrid(OUT_WIDTH / BLOCK_SIZE + 1, IN_WIDTH / BLOCK_SIZE + 1);
    matrix_transpose<<<dimGrid,dimBlock>>>(d_out, d_in, IN_WIDTH, OUT_WIDTH);
    
    // Transfer data back to host memory
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < OUT_WIDTH; i++){
        for(int j = 0; j < IN_WIDTH; j++){
            assert(fabs(h_in[i * IN_WIDTH + j] - h_out[j * OUT_WIDTH + i]) < MAX_ERR);
            // printf("in[%d][%d] = %f\n", i, j, h_in[i * IN_WIDTH + j]);
        }
    }
    // for(int i = 0; i < IN_WIDTH; i++){
    //     for(int j = 0; j < OUT_WIDTH; j++){
    //         // assert(fabs(h_in[i * IN_WIDTH + j] - h_out[j * OUT_WIDTH + i]) < MAX_ERR);
    //         printf("out[%d][%d] = %f\n", i, j, h_out[i * OUT_WIDTH + j]);
    //     }
    // }
    // printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Deallocate host memory
    free(h_in); 
    free(h_out);
}
