#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// matrix multiplication: C = AB
// A
#define A_HEIGHT 1024 
#define A_WIDTH 1024 
#define A_N A_HEIGHT * A_WIDTH 
// B
#define B_HEIGHT A_WIDTH
#define B_WIDTH 1024 
#define B_N B_HEIGHT * B_WIDTH

// C
#define C_HEIGHT A_HEIGHT
#define C_WIDTH B_WIDTH
#define C_N C_HEIGHT * C_WIDTH

#define BLOCK_SIZE 32
#define MAX_ERR 1e-6

__global__ void matrix_mul_shared(double *d_C, double *d_A, double *d_B, int d_a_height, int d_a_width, int d_b_width) {
    // global position in the C (output) matrix
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    // block position
    int c_block = blockIdx.y;
    int r_block = blockIdx.x;

    // thread position in the block
    int c_thread = threadIdx.y;
    int r_thread = threadIdx.x;

    __shared__ double d_A_sub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double d_B_sub[BLOCK_SIZE][BLOCK_SIZE];

    double sum = 0.0;

    // iterate over tiles across horizontal direction of A
    for(k = 0; k<(d_a_width+1)/BLOCK_SIZE + 1; k++){
        // load d_A[rid, k*BLOCK_SIZE+c_thread] into d_A_sub[r_thread][c_thread]
        if((rid < d_a_height) && (k*BLOCK_SIZE+c_thread < d_a_width)){
            d_A_sub[r_thread][c_thread] = d_A[rid*d_a_width + k*BLOCK_SIZE+c_thread];
        } else {
            d_A_sub[r_thread][c_thread] = 0.0;
        }

        // load d_B[k*BLOCK_SIZE + r_thread, cid] into d_B_sub[r_thread][c_thread]
        if((k*BLOCK_SIZE+r_thread < d_a_width) && (cid < d_b_width)){
            d_B_sub[r_thread][c_thread] = d_B[(k*BLOCK_SIZE+r_thread)*d_b_width + cid];
        } else {
            d_B_sub[r_thread][c_thread] = 0.0;
        }

        __syncthreads();

        // dot product within a tile
        for(i = 0; i<d_a_width; i++){
            sum += d_A_sub[r_thread][i] * d_B_sub[i][c_thread];
        }

        __syncthreads();
    }

    // write sum back
    if(rid < d_a_height && cid < d_b_width)
        d_C[rid * d_b_width + cid] = sum;
}

// float float_rand( float min, float max )
// {
//     float scale = rand() / (float) RAND_MAX; 
//     return min + scale * ( max - min );      
// }

int main(){
    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;
    double *h_ref; // compute verified matMul
    // Allocate host memory
    h_A = (double*)malloc(sizeof(double) * A_N);
    h_B = (double*)malloc(sizeof(double) * B_N);
    h_C = (double*)malloc(sizeof(double) * C_N);
    h_ref = (double*)malloc(sizeof(double) * C_N);

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
    
    srand((unsigned int)time(NULL));
    /***       TEST 2    ***/
    for (int i = 0; i< A_N; i++){
        h_A[i] = (double)rand()/(double)(RAND_MAX);
    }
    for (int i = 0; i< B_N; i++){
        h_B[i] = (double)rand()/(double)(RAND_MAX);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, sizeof(double) * A_N);
    cudaMalloc((void**)&d_B, sizeof(double) * B_N);
    cudaMalloc((void**)&d_C, sizeof(double) * C_N);

    // Transfer data from host to device memory
    cudaMemcpy(d_A, h_A, sizeof(double) * A_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(double) * B_N, cudaMemcpyHostToDevice);

    // Executing kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Note C_mat row maps to x dimension, and col maps to y dimension
    dim3 dimGrid(C_HEIGHT / BLOCK_SIZE + 1, C_WIDTH / BLOCK_SIZE + 1);
    // dim3 dimGrid(2, 1);
    matrix_mul<<<dimGrid,dimBlock>>>(d_C, d_A, d_B, A_HEIGHT, A_WIDTH, B_WIDTH);
    
    // Transfer data back to host memory
    cudaMemcpy(h_C, d_C, sizeof(double) * C_N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < C_HEIGHT; i++){
        for(int j = 0; j < C_WIDTH; j++){
            double sum = 0.0;
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
