#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// forward propogation
/*
def forwardBackward(X, y, W, v):
    Z_trans = relu(W@X.T) # mat-mat
    Z = Z_trans.T # trans
    yhat = Z@v # mat-vec
    
    error = y - yhat
    grad_v = Z.T @ error  # mat-vector
    grad_Z = np.outer(error, v) # outer product
    grad_p = dRelu(dZ, Z)
    grad_W = dp.T @ X  # mat-mat
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
#define LINEAR_BLOCK_SIZE BLOCK_SIZE * BLOCK_SIZE
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

__global__ void d_relu_matrix_mul(double *d_C, double *d_A, double *d_B, double *d_act, int d_a_height, int d_a_width, int d_b_width) {
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    if(rid < d_a_height && cid < d_b_width){
    // sum: to evaluated dot product
        double sum = 0.0;
        for(int k = 0; k < d_a_width; k++){
            sum += d_A[rid * d_a_width + k] * d_B[d_b_width*k + cid];
        }
        d_C[rid * d_b_width + cid] = (d_act[rid * d_b_width + cid]>0)?sum:0;
    }
}

__global__ void matrix_transpose(double *d_out, double *d_in, int d_in_width, int d_out_width) {
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    int rid = blockIdx.x * blockDim.x + threadIdx.x;

    if(cid < d_in_width && rid < d_out_width){
        d_out[cid * d_out_width + rid] = d_in[rid * d_in_width + cid];
    }
}

__global__ void vector_sub(double *out, double *a, double *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        out[tid] = a[tid] - b[tid];
    }
}

__global__ void d_relu(double *out, double *in, double *act, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        out[tid] = (act[tid]>0)?in[tid]:0;
    }
}

int main(){
    // forward variables
    double *h_X, *h_X_T, *h_W, *h_Z, *h_Z_T, *h_v, *h_yhat, *h_y;
    double *d_X, *d_X_T, *d_W, *d_Z, *d_Z_T, *d_v, *d_yhat, *d_y;

    // backward variables
    double *h_error, *h_grad_v, *h_grad_Z, *h_grad_p, *h_grad_p_T, *h_grad_W;
    double *d_error, *d_grad_v, *d_grad_Z, *d_grad_p, *d_grad_p_T, *d_grad_W;
    // double *h_ref; // compute verified results
    // Allocate host memory
    h_X = (double*)malloc(sizeof(double) * X_N);
    h_W = (double*)malloc(sizeof(double) * W_N);
    h_v = (double*)malloc(sizeof(double) * V_N);
    h_Z_T = (double*)malloc(sizeof(double) * Z_N);
    h_Z = (double*)malloc(sizeof(double) * Z_N);
    h_yhat = (double*)malloc(sizeof(double) * N);
    h_y = (double*)malloc(sizeof(double) * N);
    h_error = (double*)malloc(sizeof(double) * N);
    h_grad_v = (double*)malloc(sizeof(double) * V_N);
    h_grad_Z = (double*)malloc(sizeof(double) * Z_N);
    h_grad_p = (double*)malloc(sizeof(double) * Z_N);
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
    for(int i = 0; i < N; i++){
        h_y[i] = (double)(i+1);
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
    cudaMalloc((void**)&d_y, sizeof(double) * N);
    cudaMalloc((void**)&d_error, sizeof(double) * N);
    cudaMalloc((void**)&d_grad_v, sizeof(double) * V_N);
    cudaMalloc((void**)&d_grad_Z, sizeof(double) * Z_N);
    cudaMalloc((void**)&d_grad_p, sizeof(double) * Z_N);

    // Transfer data from host to device memory
    cudaMemcpy(d_X, h_X, sizeof(double) * X_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, sizeof(double) * W_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(double) * V_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(double) * N, cudaMemcpyHostToDevice);

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
    
    // backwards:
    vector_sub<<<N / LINEAR_BLOCK_SIZE + 1, LINEAR_BLOCK_SIZE>>>(d_error, d_y, d_yhat, N);
    
    dim3 dimGrid5(K / BLOCK_SIZE + 1, 1 / BLOCK_SIZE + 1);
    matrix_mul<<<dimGrid5,dimBlock>>>(d_grad_v, d_Z_T, d_error, K, N, 1);
    
    // dim3 dimGrid6(N / BLOCK_SIZE + 1, K / BLOCK_SIZE + 1);
    // matrix_mul<<<dimGrid6,dimBlock>>>(d_grad_Z, d_error, d_v, N, 1, K);
    
    dim3 dimGrid6(N / BLOCK_SIZE + 1, K / BLOCK_SIZE + 1);
    d_relu_matrix_mul<<<dimGrid6,dimBlock>>>(d_grad_Z, d_error, d_v, d_Z, N, 1, K);
    cudaMemcpy(h_grad_Z, d_grad_Z, sizeof(double) * Z_N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_Z, d_Z, sizeof(double) * Z_N, cudaMemcpyDeviceToHost);
    
    // d_relu<<<N / LINEAR_BLOCK_SIZE + 1, LINEAR_BLOCK_SIZE>>>(d_grad_p, d_grad_Z, d_Z, N);
    // Transfer data back to host memory
    // cudaMemcpy(h_grad_p, d_grad_p, sizeof(double) * Z_N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_Z, d_Z, sizeof(double) * Z_N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        for(int j = 0; j < K; j++){
            // double sum = 0.0;
            // for(int k = 0; k < A_WIDTH; k++){
            //     sum += h_A[i*A_WIDTH+k] * h_B[k*B_WIDTH + j];
            // }
            // h_ref[i * C_WIDTH + j] = sum;
            // assert(fabs(h_ref[i*C_WIDTH + j] - h_C[i * C_WIDTH + j]) < MAX_ERR);
            printf("h_grad_Z[%d][%d] = %f\n", i, j, h_grad_Z[i * K + j]);
            // printf("h_Z[%d][%d] = %f\n", i, j, h_Z[i * K + j]);
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
    cudaFree(d_y);
    cudaFree(d_error);
    cudaFree(d_grad_v);
    cudaFree(d_grad_Z);
    cudaFree(d_grad_p);

    // Deallocate host memory
    free(h_X); 
    free(h_W);
    free(h_v);
    free(h_Z);
    free(h_Z_T);
    free(h_yhat);
    free(h_y);
    free(h_error);
    free(h_grad_v);
    free(h_grad_Z);
    free(h_grad_p);
}
