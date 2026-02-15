#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>

#include "cudaKernels.hpp"


#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cudaError_t status = call; \ 
        if (status != cudaSuccess) { \
            printf("FAILED: CUDA API returned error = %d, details: " #msg "\n", status); \
        } \
    } while(0);


#define CUSP_CALL_AND_CHECK(call, msg) \
    do { \
        cusparseStatus_t status = call; \ 
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            printf("FAILED: CUSPARSE call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, msg) \
    do { \
        cudssStatus_t status = call; \ 
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
        } \
    } while(0);


cudaStream_t getStreamFromHandle(cublasHandle_t handle); 
cublasPointerMode_t getPointerMode(cublasHandle_t handle);
cublasStatus_t betaWeightsUpdateLauncher(
    cublasHandle_t handle,
    int n,
    const double* x,
    int incx,
    const double* y,
    int incy,
    double* z,
    int incz,
    int pivot_idx
);

void replaceColumnCSC(
    cusparseHandle_t handle,
    const cusparseSpMatDescr_t matA,  
    int col_idx,
    const double* d_new_col_values, 
    const int* d_new_col_row_ind, int new_col_nnz,
    cusparseSpMatDescr_t* matB,      
    cudaDataType dataType 
); 