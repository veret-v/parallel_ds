#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio> 

#define WARP_SIZE 32


__global__ void betaWeightsUpdateKernel(
    int n,
    const double* x,
    int incx,
    const double* y,
    int incy,
    double* z,
    int incz,
    int pivot_idx
);


template<int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void betaWeightsUpdateKernelOpt(
    int n,
    const double* x,
    int incx,
    const double* y,
    int incy,
    double* z,
    int incz,
    int pivot_idx
);


__global__ void spmvUpdateKernel(    
    double* sol,
    const double* vec1,
    const double* vec2,
    const double* val_csr,
    const int* id_csr,
    const int* ptr_csr,
    const int* cols_idx,
    double* buff,
    const int cols_idx_size,
    const int nnz,
    const int major_dim,
    const double alpha,
    const double beta,
    const bool set
);


__global__ void applyEtaMatKernel(    
    double *y, 
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int eta_num,
    const int col_len
);


__global__ void applyPFIKernel(    
    double *y,
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int size,
    const int col_len
);


__global__ void applyEtaMatTKernel(   
    double *y, 
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int eta_num,
    const int col_len
);


__global__ void applyPFITKernel(    
    double *y,
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int size,
    const int col_len
);
