#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>


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


__global__ void replaceColumnKernel(
    int m, 
    int n,
    double* values1, 
    int* row_id1, 
    int* col_ptr1,                           
    int col_idx,
    const double* new_col_values, 
    const int* new_col_row_ind, 
    int new_col_nnz,
    double* new_values, 
    int* new_row_ind, 
    int* new_col_ptr
);