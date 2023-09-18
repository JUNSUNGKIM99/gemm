#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            float dr = (float)rand();
            float *curr = A + (j * M + i);
            *curr = (dr / rMax) * 10.0;
            totalNnz++;
        }
    }
    *out = A;
    return totalNnz;
}

int generate_random_sparse_matrix(int row, int col, float **out, float sparsity){
    int i, j;
    float rMax = (float)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * row * col);
    int totalNnz = 0;
    for (j = 0; j < col; j++) {
        for (i = 0; i < row; i++) {
            float dr = (float)rand();
            float *curr = A + (j * row + i);
            if (dr / rMax < sparsity) {  // Adjust the threshold (e.g., 0.3) for desired sparsity
                *curr = (dr / rMax) * 10.0;
                totalNnz++;
            } else {
                *curr = 0.0f;
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

int main(int argc, char **argv){
        if( argc != 6){
        printf("Usage: ./cusparse_sgemm m n k density nIter \n");
        printf("Current # of arguments: %d\n", argc);
        return 0;
    }
    float *A, *B, *C;
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]); 
    const int K = atoi(argv[3]);
    const float sparsity = atof(argv[4]);
    const int nIter = atoi(argv[5]);
    int A_num_rows = M;
    int A_num_cols = K;
    int A_nnz = 0;
    int B_num_rows = K;
    int B_num_cols = N; 
    int B_nnz = 0;
    int lda = A_num_cols;
    int ldb = B_num_rows;
    int ldc = A_num_rows;
    int A_size = lda * A_num_rows; 
    int B_size = ldb * B_num_cols;
    int C_size = ldc * B_num_cols;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    //--------------------------------------------------------------
    // Generate random input matrix. 
    srand(42);
    A_nnz = generate_random_sparse_matrix(M, K, &A, sparsity);  
    B_nnz = generate_random_dense_matrix(K, N, &B);
    //print_partial_matrix(A, M, K, M, K);
    //print_partial_matrix(B, K, N, K, N);
    double sparsity_A = 1 - double(A_nnz) / double((M*K));
    printf("Num_A: %d, ANnz: %d, Sparsity_A: %f, BNnz: %d\n", M*K, A_nnz, sparsity_A, B_nnz);
    //--------------------------------------------------------------

    //--------------------------------------------------------------
    // Device mem setup
    int   *d_csr_offsets, *d_csr_columns;
    float *d_csr_values,  *d_dense;
    float *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &d_dense, A_size * sizeof(float)))
    CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets, (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMemcpy(d_dense, A, A_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    // Prepare Matrix B, C 
    CHECK_CUDA( cudaMalloc((void **) &dB, B_size * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void **) &dC, C_size * sizeof(float)));
    CHECK_CUDA( cudaMemcpy(dB, B, B_size * sizeof(float), 
                            cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSAE API, Create CSR data format sparse matrix
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t tmp, matB, matC;
    cusparseSpMatDescr_t matA;
    void*                csr_buffer = NULL;
    void*                spmm_buffer = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) );
    // Create dense matrix tmp
    CHECK_CUSPARSE( cusparseCreateDnMat(&tmp, A_num_rows, A_num_cols, lda, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) );
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    //-----------------------------Create B, C matrix 
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));                                        
    //-----------------------------
    // allocate an external buffer if needed
    // Create CSR metadata and CSR sparse matrix MatrixA
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, tmp, matA,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) );
    CHECK_CUDA( cudaMalloc(&csr_buffer, bufferSize));

    // execute Dense to Sparse conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, tmp, matA,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        csr_buffer) );
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matA, &num_rows_tmp, &num_cols_tmp,
                                         &nnz));

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float)));
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matA, d_csr_offsets, d_csr_columns,
                                           d_csr_values));
    // execute Dense to Sparse conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, tmp, matA,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        csr_buffer));
    //---------------------------------------------------------------
    // Execute Spmm Operation
    cusparseSpMatDescr_t matD;
    CHECK_CUSPARSE( cusparseCreateCsr(&matD, A_num_rows, A_num_cols, 0,
                                      d_csr_offsets, d_csr_columns, d_csr_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                    handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
                                    //CUSPARSE_SPMM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG3
    CHECK_CUDA( cudaMalloc(&spmm_buffer, bufferSize))
    // Warm up kernel execution
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, spmm_buffer))
                                //CUSPARSE_SPMM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG3
    // CHECK_CUSPARSE( cusparseSpMM(handle,
    //                             CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                             CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                             &alpha, matA, matB, &beta, matC, CUDA_R_32F,
    //                             CUSPARSE_SPMM_CSR_ALG1, spmm_buffer))
    // CHECK_CUSPARSE( cusparseSpMM(handle,
    //                             CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                             CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                             &alpha, matA, matB, &beta, matC, CUDA_R_32F,
    //                             CUSPARSE_SPMM_CSR_ALG2, spmm_buffer))

    for(int j = 0; j < nIter; j++){
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, spmm_buffer))
                                //CUSPARSE_SPMM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG3
    }
    printf("Finish SpMM Operation -----\n");
    //----------------------------------------------------------
    // Transfer Metadata from Device to host
    //int A_csr_offsets[A_num_rows + 1] = {0,};
    //int A_csr_columns[nnz] = {0,};
    //float A_csr_values[nnz] = {0, };
    //CHECK_CUDA( cudaMemcpy(A_csr_offsets, d_csr_offsets,
    //                      (A_num_rows + 1) * sizeof(int),
    //                       cudaMemcpyDeviceToHost));
    //CHECK_CUDA( cudaMemcpy(A_csr_columns, d_csr_columns, nnz * sizeof(int),
    //                       cudaMemcpyDeviceToHost));
    //CHECK_CUDA( cudaMemcpy(A_csr_values, d_csr_values, nnz * sizeof(float),
    //                       cudaMemcpyDeviceToHost));   
    //C = (float *)malloc(sizeof(float) * M * N);
    //CHECK_CUDA( cudaMemcpy(C, dC, C_size * sizeof(float), cudaMemcpyDeviceToHost))
    //print_partial_matrix(C,M,N,M,N);
    //----------------------------------------------------------
    //-----------Print offset, columns, values------------//
    // printf("\nOffset\n");
    // for (int i = 0; i < A_num_rows + 1; i++) {
    //     printf("%d ", A_csr_offsets[i]);
    // }
    // printf("\ncolumns\n");
    // for (int i = 0; i < nnz; i++) {
    //     printf("%d ", A_csr_columns[i]);
    // }
    // printf("\nvalues\n");
    // for (int i = 0; i < nnz; i++) {
    //     printf("%f ", A_csr_values[i]);
    // }             
    // printf("\n");
    //---------------------------------------------------//

    //----------------------------------------------------//
    free(A);
    free(B);
    free(C);
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(tmp) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matD) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // device memory deallocation
    CHECK_CUDA( cudaFree(csr_buffer) )
    CHECK_CUDA( cudaFree(spmm_buffer))
    CHECK_CUDA( cudaFree(d_csr_offsets) )
    CHECK_CUDA( cudaFree(d_csr_columns) )
    CHECK_CUDA( cudaFree(d_csr_values) )
    CHECK_CUDA( cudaFree(d_dense) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return EXIT_SUCCESS;
}
