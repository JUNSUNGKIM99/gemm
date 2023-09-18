#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;

}

int main(int argc, char *argv[])
{
    if( argc != 5){
        printf("Usage: ./cublas_sgemm m n k iter \n");
        printf("Current # of arguments: %d\n", argc);
        return 0;
    }
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);
    const int nIter = atoi(argv[4]);
    printf("M: %d, N: %d, K: %d\n", m, n, k);
    const int lda = m;
    const int ldb = k;
    const int ldc = m;


    double size_A = m * k;
    double mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    double size_B = n * k;
    double mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    float *d_A, *d_B, *d_C;

    unsigned int size_C = m * n;
    unsigned int mem_size_C = sizeof(float) * size_C;

    float *h_C = (float *)malloc(mem_size_C);
    float *h_CUBLAS = (float *)malloc(mem_size_C);

    CUDA_CHECK(cudaMalloc((void **)&d_A, mem_size_A));
    CUDA_CHECK(cudaMalloc((void **)&d_B, mem_size_B));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_C, mem_size_C));
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */

    CUDA_CHECK(cudaMalloc((void **)&d_A, mem_size_A));
    CUDA_CHECK(cudaMalloc((void **)&d_B, mem_size_B));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMalloc((void **)&d_C, mem_size_C));

    /*Setup to Execution*/
    //    cublasHandle_t handle;
    cudaEvent_t start, stop;
    //    CUDA_CHECK(cublasCreate(&handle));
    /* step 3: compute */
    // Warm up kernel execution
    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, NULL));

    for (int j = 0; j < nIter; j++)
    {
        CUBLAS_CHECK(
            cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
    }
    printf("Done.\n");

    CUDA_CHECK(cudaEventRecord(stop, NULL));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)m * (double)n * (double)k;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    printf("MatrixMul size, M=%d N=%d K=%d\n", m, n, k);
    printf(
        "Performance= %.4f GFlops/s, Time= %.4f msec, Size= %.0f Ops\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));\
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    free(h_A);
    free(h_B);
    free(h_C);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
