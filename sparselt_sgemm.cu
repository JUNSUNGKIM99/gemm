#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <curand_kernel.h>
#define CHECK_CUDA(func){                                               \
    cudaError_t status = (func);                                        \
        if (status != cudaSuccess){                                     \
            printf("CUDA API faild at line %d with error: %s (%d)\n",   \
            __LINE__, cudaGetErrorString(status), status);              \
            return EXIT_FAILURE;                                        \
        }                                                               \
}

#define CHECK_CUSPARSE(func){                                           \
        cusparseStatus_t status = (func);                               \
        if (status != CUSPARSE_STATUS_SUCCESS){                         \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",\
            __LINE__, cusparseGetErrorString(status), status);          \
            return EXIT_FAILURE;                                        \
        }                                                               \
}

int generate_random_dense_matrix(int M, int N, float **out){
    int i, j;
    float rMax = (float)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;
    int r = 0;
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            r = rand();
            float *curr = A + (j * M +i);
            float dr = (float)r;
            *curr = (dr /rMax) * 10.0;
            if (*curr != 0.0f){
                totalNnz++;
            }
        }
    }
    *out = A;
    return totalNnz;
}

void print_partial_matrix(float *M, int nrows, int ncols, int max_row, int max_col){
    int row, col;
    for (row = 0; row < max_row; row++){
        for (col = 0; col < max_col; col++){
            printf("%2.2f ", M[row*ncols+col]);
        }
        printf("...\n");
    }
    printf("...\n");
}

__global__ void initializeHalfArray(__half* array, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float randomFloat = curand_uniform(&state) * 10.0f; // 0부터 10 사이의 랜덤 값 생성
        array[idx] = static_cast<__half>(randomFloat);
    }
}
constexpr int EXIT_UNSUPPORTED = 2;

int main(int argc, char **argv) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6) &&
        !(major_cc == 8 && minor_cc == 9)) {
        printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6, 8.9 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    // Host problem definition, row-major order
    // bigger sizes may require dynamic allocations
    if( argc != 5){
        printf("Usage: ./cusparselt_sgemm m n k iter, # of rows must be at least multiple of 16\n");
        printf("Current # of arguments: %d\n", argc);
        return 0;
    }
    const int m            = atoi(argv[1]);
    const int n            = atoi(argv[2]);
    const int k            = atoi(argv[3]);
    const int nIter        = atoi(argv[4]);

    auto          order        = CUSPARSE_ORDER_ROW;
    auto          opA          = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB          = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type         = CUDA_R_16F;
    auto          compute_type = CUSPARSE_COMPUTE_16F;
    printf("M:%d, N: %d, K:%d\n", m,n,k);
    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size         = A_height * lda * sizeof(__half);
    auto     B_size         = B_height * ldb * sizeof(__half);
    auto     C_size         = C_height * ldc * sizeof(__half);
    // __half hA[m * k];
    // __half hB[k * n];
    // __half hC[m * n] = {};
    // for (int i = 0; i < m * k; i++)
    //     hA[i] = static_cast<__half>(static_cast<float>(rand() % 10));
    // for (int i = 0; i < k * n; i++)
    //     hB[i] = static_cast<__half>(static_cast<float>(rand() % 10));
    __half* hA;
    __half* hB;
    __half* hC;
    cudaMallocManaged(&hA, A_size*sizeof(__half));
    cudaMallocManaged(&hB, B_size*sizeof(__half));
    cudaMallocManaged(&hC, C_size*sizeof(__half));
    int threadsPerBlock = 256;
    int numBlocksA = (A_size + threadsPerBlock - 1) / threadsPerBlock;
    int numBlocksB = (B_size + threadsPerBlock - 1) / threadsPerBlock;

    initializeHalfArray<<<numBlocksA, threadsPerBlock>>>(hA, A_size, 42); 
    initializeHalfArray<<<numBlocksB, threadsPerBlock>>>(hB, B_size, 42); 
    float alpha = 1.0f;
    float beta  = 0.0f;
    printf("Done generation matrix\n");
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyDeviceToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyDeviceToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size, compressed_buffer_size;
    void*  dA_compressedBuffer;
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size,
                                                  &compressed_buffer_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressedBuffer,
                           compressed_buffer_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed,
                                            dA_compressedBuffer,stream) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                           dA_compressed, dB, &beta,
                                           dC, dD, nullptr,
                                           streams, num_streams) )
    // otherwise, it is possible to set it directly:
    //int alg = 0;
    //CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
    //                                        &handle, &alg_sel,
    //                                        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
    //                                        &alg, sizeof(alg)))
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))
    void* d_workspace;
    CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )
    // Perform the matrix multiplication, Warm up kernel execution 
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )

    for(int j = 0; j < nIter; j++){
        CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    // CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    // CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    // bool A_std_layout = (is_rowmajor != isA_transposed);
    // bool B_std_layout = (is_rowmajor != isB_transposed);
    // host computation
    // float hC_result[m * n];
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         float sum  = 0.0f;
    //         for (int k1 = 0; k1 < k; k1++) {
    //             auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
    //             auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
    //             sum      += static_cast<float>(hA[posA]) *  // [i][k]
    //                         static_cast<float>(hB[posB]);   // [k][j]
    //         }
    //         auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
    //         hC_result[posC] = sum;  // [i][j]
    //     }
    // }
    // // host-device comparison
    // int correct = 1;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
    //         auto device_value = static_cast<float>(hC[pos]);
    //         auto host_value   = hC_result[pos];
    //         if (device_value != host_value) {
    //             // direct floating point comparison is not reliable
    //             std::printf("(%d, %d):\t%f vs. %f\n",
    //                         i, j, host_value, device_value);
    //             correct = 0;
    //             break;
    //         }
    //     }
    // }
    // if (correct)
    //     std::printf("matmul_example test PASSED\n");
    // else
    //     std::printf("matmul_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    CHECK_CUDA( cudaFree(d_workspace) )
    CHECK_CUDA( cudaFree(dA_compressedBuffer) )
    return EXIT_SUCCESS;
}
