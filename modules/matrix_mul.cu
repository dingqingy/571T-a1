#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// matrix multiplication: C = AB
// A
#define A_HEIGHT 35 
#define A_WIDTH 30 
#define A_N A_HEIGHT * A_WIDTH 
// B
#define B_HEIGHT A_WIDTH
#define B_WIDTH 35 
#define B_N B_HEIGHT * B_WIDTH

// C
#define C_HEIGHT A_HEIGHT
#define C_WIDTH B_WIDTH
#define C_N C_HEIGHT * C_WIDTH

#define BLOCK_SIZE 32
#define MAX_ERR 1e-6

__global__ void matrix_mul(float *d_C, float *d_A, float *d_B, int d_a_height, int d_a_width, int d_b_width) {
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    if(rid < d_a_height && cid < d_b_width){
    // sum: to evaluated dot product
        float sum = 0.0;
        for(int k = 0; k < d_a_width; k++){
            sum += d_A[rid * d_a_width + k] * d_B[d_b_width*k + cid];
        }
        d_C[rid * d_b_width + cid] = sum;
    }
}

float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; 
    return min + scale * ( max - min );      
}

int main(){
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    float *h_ref; // compute verified matMul
    // Allocate host memory
    h_A = (float*)malloc(sizeof(float) * A_N);
    h_B = (float*)malloc(sizeof(float) * B_N);
    h_C = (float*)malloc(sizeof(float) * C_N);
    h_ref = (float*)malloc(sizeof(float) * C_N);

    // Initialize host arrays
    
    /***       TEST 1    ***/
    // for(int i = 0; i < A_HEIGHT; i++){
    //     for(int j = 0; j < A_WIDTH; j++){
    //         h_A[i*A_WIDTH + j] = (float)i;
    //     }
    // }
    // for(int i = 0; i < B_HEIGHT; i++){
    //     for(int j = 0; j < B_WIDTH; j++){
    //         h_B[i*B_WIDTH + j] = (float)i;
    //     }
    // }
    
    /***       TEST 2    ***/
    for (int i = 0; i< A_N; i++){
        h_A[i] = float_rand(-1, 1);
    }
    for (int i = 0; i< B_N; i++){
        h_B[i] = float_rand(-1, 1);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, sizeof(float) * A_N);
    cudaMalloc((void**)&d_B, sizeof(float) * B_N);
    cudaMalloc((void**)&d_C, sizeof(float) * C_N);

    // Transfer data from host to device memory
    cudaMemcpy(d_A, h_A, sizeof(float) * A_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * B_N, cudaMemcpyHostToDevice);

    // Executing kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Note C_mat row maps to x dimension, and col maps to y dimension
    dim3 dimGrid(C_HEIGHT / BLOCK_SIZE + 1, C_WIDTH / BLOCK_SIZE + 1);
    // dim3 dimGrid(2, 1);
    matrix_mul<<<dimGrid,dimBlock>>>(d_C, d_A, d_B, A_HEIGHT, A_WIDTH, B_WIDTH);
    
    // Transfer data back to host memory
    cudaMemcpy(h_C, d_C, sizeof(float) * C_N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < C_HEIGHT; i++){
        for(int j = 0; j < C_WIDTH; j++){
            float sum = 0.0;
            for(int k = 0; k < A_WIDTH; k++){
                sum += h_A[i*A_WIDTH+k] * h_B[k*B_WIDTH + j];
            }
            h_ref[i * C_WIDTH + j] = sum;
            assert(fabs(h_ref[i*C_WIDTH + j] - h_C[i * C_WIDTH + j]) < MAX_ERR);
            // printf("h_c[%d][%d] = %f\n", i, j, h_C[i * C_WIDTH + j]);
            // printf("h_ref[%d][%d] = %f\n", i, j, h_ref[i * C_WIDTH + j]);
        }
    }
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Deallocate host memory
    free(h_A); 
    free(h_B);
    free(h_C);
}
