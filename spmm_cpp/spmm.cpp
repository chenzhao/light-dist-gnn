#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "cusparse.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}



#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

typedef const at::Tensor& T;

void spmm_cusparse_coo(T A_row_idx, T A_col_idx, T A_values, int32_t A_row, int32_t A_col, T B, T C, float alpha, float beta, int use_half) {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC; // mat from torch is row major
    if (use_half){
        CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_row, A_col, A_values.size(0), A_row_idx.data_ptr<int>(), A_col_idx.data_ptr<int>(), A_values.data_ptr<at::Half>(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )

        CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B.size(0), B.size(1), B.size(1), B.data_ptr<at::Half>(), CUDA_R_16F, CUSPARSE_ORDER_ROW) )
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C.size(0), C.size(1), C.size(1), C.data_ptr<at::Half>(), CUDA_R_16F, CUSPARSE_ORDER_ROW) )
    }else{
        CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_row, A_col, A_values.size(0), A_row_idx.data_ptr<int>(), A_col_idx.data_ptr<int>(), A_values.data_ptr<float>(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

        CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B.size(0), B.size(1), B.size(1), B.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW) )
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C.size(0), C.size(1), C.size(1), C.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    }

    CHECK_CUSPARSE(cusparseSpMM(at::cuda::getCurrentCUDASparseHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_COO_ALG4, NULL));  //CUSPARSE_MM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG2 , CUSPARSE_SPMM_COO_ALG4


   CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
   CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
   CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
}


void spmm_cusparse_csr(T A_row_offsets, T A_col_idx, T A_values, int32_t A_row, int32_t A_col, T B, T C, float alpha, float beta, int use_half) {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC; // mat from torch is row major

    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
        
    cusparseHandle_t     handle = NULL;

    CHECK_CUSPARSE( cusparseCreate(&handle) )

    if (use_half){
        CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_row, A_col, A_values.size(0), A_row_offsets.data_ptr<int>(), A_col_idx.data_ptr<int>(), A_values.data_ptr<at::Half>(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )

        CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B.size(0), B.size(1), B.size(1), B.data_ptr<at::Half>(), CUDA_R_16F, CUSPARSE_ORDER_ROW) )
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C.size(0), C.size(1), C.size(1), C.data_ptr<at::Half>(), CUDA_R_16F, CUSPARSE_ORDER_ROW) )

    }else{
        CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_row, A_col, A_values.size(0), A_row_offsets.data_ptr<int>(), A_col_idx.data_ptr<int>(), A_values.data_ptr<float>(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

        CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B.size(0), B.size(1), B.size(1), B.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW) )
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C.size(0), C.size(1), C.size(1), C.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    }

    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    CHECK_CUSPARSE(cusparseSpMM(at::cuda::getCurrentCUDASparseHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2 , dBuffer));  //CUSPARSE_MM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG2 , CUSPARSE_SPMM_COO_ALG4

   CHECK_CUDA( cudaFree(dBuffer) )

   CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
   CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
   CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )

    CHECK_CUSPARSE( cusparseCreate(&handle) )
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_cusparse_coo", &spmm_cusparse_coo, "SpMM wrapper for cusparse coo");
    m.def("spmm_cusparse_csr", &spmm_cusparse_csr, "SpMM wrapper for cusparse csr");
}
