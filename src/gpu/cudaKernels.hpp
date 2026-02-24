#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    double *y,
    const double *x,
    const double *val_csr,
    const int *id_csr,
    const int *ptr_csr,
    const int* need_ptrs,
    const int nnz,
    const int major_dim,
    const double alpha,
    const double beta
);


__global__ void spmvUpdateSpKernel(    
    double* y,
    const double* x,
    const int* x_idx, 
    const double* val_csr,
    const int* id_csr,
    const int* ptr_csr,
    const int* need_ptrs,
    const int nnz,
    const int major_dim,
    const double alpha,
    const double beta
);


__global__ void spmvUpdateSetKernel(    
    double *y,
    const int *set_id,
    const int set_id_size,
    const double *x,
    const double *z,
    const double *val_csr,
    const int *id_csr,
    const int *ptr_csr,
    const int* need_ptrs,
    const int nnz,
    const int major_dim,
    const double alpha,
    const double beta
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
