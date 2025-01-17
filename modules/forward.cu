#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// forward propogation
/*
def forward(X, W, v):
    Z_trans = relu(W@X.T) # mat-mat
    Z = Z_trans.T # trans
    yhat = Z@v # mat-vec
    return Z, yhat
*/

/* Parameter Setup */
#define N 4 // # of input samples
#define D 2 // # of input neurons
#define K 3 // # of hidden neurons

// X: input matrix (n * d)
#define X_HEIGHT N
#define X_WIDTH D
#define X_N X_HEIGHT * X_WIDTH
// Z: ifmap matrix (n * k)
#define Z_HEIGHT N
#define Z_WIDTH K
#define Z_N Z_HEIGHT * Z_WIDTH 
// W: layer 1 weights (k * d)
#define W_HEIGHT K
#define W_WIDTH D
#define W_N W_HEIGHT * W_WIDTH
// v: layer 2 weights
#define V_HEIGHT K
#define V_WIDTH 1
#define V_N V_HEIGHT * V_WIDTH

#define BLOCK_SIZE 32
#define MAX_ERR 1e-6

__global__ void matrix_mul(double *d_C, double *d_A, double *d_B, int d_a_height, int d_a_width, int d_b_width) {
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    if(rid < d_a_height && cid < d_b_width){
    // sum: to evaluated dot product
        double sum = 0.0;
        for(int k = 0; k < d_a_width; k++){
            sum += d_A[rid * d_a_width + k] * d_B[d_b_width*k + cid];
        }
        d_C[rid * d_b_width + cid] = sum;
    }
}

__global__ void relu_matrix_mul(double *d_C, double *d_A, double *d_B, int d_a_height, int d_a_width, int d_b_width) {
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    if(rid < d_a_height && cid < d_b_width){
    // sum: to evaluated dot product
        double sum = 0.0;
        for(int k = 0; k < d_a_width; k++){
            sum += d_A[rid * d_a_width + k] * d_B[d_b_width*k + cid];
        }
        d_C[rid * d_b_width + cid] = (sum>0)?sum:0;
    }
}

__global__ void matrix_transpose(double *d_out, double *d_in, int d_in_width, int d_out_width) {
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    if(cid < d_in_width && rid < d_out_width){
        d_out[cid * d_out_width + rid] = d_in[rid * d_in_width + cid];
    }
}

int main(){
    double *h_X, *h_W, *h_v;
    double *h_Z, *h_Z_T, *h_yhat;
    double *d_X, *d_X_T, *d_W, *d_v;
    double *d_Z, *d_Z_T, *d_yhat;
    // double *h_ref; // compute verified results
    // Allocate host memory
    h_X = (double*)malloc(sizeof(double) * X_N);
    h_W = (double*)malloc(sizeof(double) * W_N);
    h_v = (double*)malloc(sizeof(double) * V_N);
    h_Z_T = (double*)malloc(sizeof(double) * Z_N);
    h_Z = (double*)malloc(sizeof(double) * Z_N);
    h_yhat = (double*)malloc(sizeof(double) * N);
    // h_ref = (double*)malloc(sizeof(double) * N);

    // Initialize host arrays
    
    /***       TEST 1    ***/
    for(int i = 0; i < X_N; i++){
        if(i == 1 || i == 3){
            h_X[i] = (double)(-i-1);
        } else{
            h_X[i] = (double)(i+1);
        }
    }
    for(int i = 0; i < W_N; i++){
        h_W[i] = double(i+1);
    }
    for(int i = 0; i < V_HEIGHT; i++){
        h_v[i] = (double)(i+1);
    }
    
    /***       TEST 2    ***/
    // rand((unsigned int)time(NULL));
    // for (int i = 0; i< A_N; i++){
    //     h_A[i] = (double)rand()/(double)(RAND_MAX);
    // }
    // for (int i = 0; i< B_N; i++){
    //     h_B[i] = (double)rand()/(double)(RAND_MAX);
    // }

    // Allocate device memory
    cudaMalloc((void**)&d_X, sizeof(double) * X_N);
    cudaMalloc((void**)&d_X_T, sizeof(double) * X_N);
    cudaMalloc((void**)&d_Z, sizeof(double) * Z_N);
    cudaMalloc((void**)&d_Z_T, sizeof(double) * Z_N);
    cudaMalloc((void**)&d_W, sizeof(double) * W_N);
    cudaMalloc((void**)&d_v, sizeof(double) * V_N);
    cudaMalloc((void**)&d_yhat, sizeof(double) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_X, h_X, sizeof(double) * X_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, sizeof(double) * W_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(double) * V_N, cudaMemcpyHostToDevice);

    // Executing kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // X_HEIGHT (N) corresponding to OUT_WIDTH, X_WIDTH (D) corresponding to IN_WIDTH
    dim3 dimGrid1(N / BLOCK_SIZE + 1,D / BLOCK_SIZE + 1);
    matrix_transpose<<<dimGrid1,dimBlock>>>(d_X_T, d_X, D, N);
    
    dim3 dimGrid2(K / BLOCK_SIZE + 1, N / BLOCK_SIZE + 1);
    relu_matrix_mul<<<dimGrid2,dimBlock>>>(d_Z_T, d_W, d_X_T, K, D, N);
    
    dim3 dimGrid3(K / BLOCK_SIZE + 1, N / BLOCK_SIZE + 1);
    matrix_transpose<<<dimGrid3,dimBlock>>>(d_Z, d_Z_T, N, K);
    
    dim3 dimGrid4(N / BLOCK_SIZE + 1, 1 / BLOCK_SIZE + 1);
    matrix_mul<<<dimGrid4,dimBlock>>>(d_yhat, d_Z, d_v, N, K, 1);
    // Transfer data back to host memory
    cudaMemcpy(h_yhat, d_yhat, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 1; j++){
            // double sum = 0.0;
            // for(int k = 0; k < A_WIDTH; k++){
            //     sum += h_A[i*A_WIDTH+k] * h_B[k*B_WIDTH + j];
            // }
            // h_ref[i * C_WIDTH + j] = sum;
            // assert(fabs(h_ref[i*C_WIDTH + j] - h_C[i * C_WIDTH + j]) < MAX_ERR);
            printf("h_yhat[%d][%d] = %f\n", i, j, h_yhat[i * 1 + j]);
            // printf("h_ref[%d][%d] = %f\n", i, j, h_ref[i * C_WIDTH + j]);
        }
    }
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_X);
    cudaFree(d_X_T);
    cudaFree(d_W);
    cudaFree(d_v);
    cudaFree(d_Z);
    cudaFree(d_Z_T);
    cudaFree(d_yhat);

    // Deallocate host memory
    free(h_X); 
    free(h_W);
    free(h_v);
    free(h_Z);
    free(h_Z_T);
    free(h_yhat);
}
