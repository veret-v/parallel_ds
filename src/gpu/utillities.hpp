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


#define CUBLAS_CALL_AND_CHECK(call, msg) \
    do { \
        cublasStatus_t status = call; \ 
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("FAILED: CUBLAS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, msg) \
    do { \
        cudssStatus_t status = call; \ 
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
        } \
    } while(0);


cudaStream_t utilCublasGetStreamFromHandle(cublasHandle_t handle); 
cublasPointerMode_t utilCublasGetPointerMode(cublasHandle_t handle);
cudaStream_t utilCusparseGetStreamFromHandle(cusparseHandle_t handle); 
cusparsePointerMode_t utilCusparseGetPointerMode(cusparseHandle_t handle);


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


cublasStatus_t btranOrFtran(
    cublasHandle_t handle,
    double *y,
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int size,
    const int col_len,
    const bool transpose
);


cusparseStatus_t spmvUpdateInc(
    cusparseHandle_t handle,
    const bool sparse_x,
    int nnz,
    int major_dim,
    double* y,    
    const double* x,  
    const int* x_idx, 
    const double* val, 
    const int* id, 
    const int* ptr,
    const int* need_ptrs,
    const double alpha,
    const double beta
); 


cusparseStatus_t spmvUpdateSet(
    cusparseHandle_t handle,
    int nnz,
    int major_dim,
    double* y, 
    const int* set_id,  
    const int set_id_size,  
    const double* x,  
    const double* z, 
    const double* val, 
    const int* id, 
    const int* ptr,
    const int* need_ptrs,
    const double alpha,
    const double beta
); 