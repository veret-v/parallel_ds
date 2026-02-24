#include "utillities.hpp"


cudaStream_t utilCublasGetStreamFromHandle(cublasHandle_t handle) {
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    return stream;
}


cublasPointerMode_t utilCublasGetPointerMode(cublasHandle_t handle) {
    cublasPointerMode_t mode;
    cublasGetPointerMode(handle, &mode);
    return mode;
}


cudaStream_t utilCusparseGetStreamFromHandle(cusparseHandle_t handle) {
    cudaStream_t stream;
    cusparseGetStream(handle, &stream);
    return stream;
}


cusparsePointerMode_t utilCusparseGetPointerMode(cusparseHandle_t handle) {
    cusparsePointerMode_t mode;
    cusparseGetPointerMode(handle, &mode);
    return mode;
}


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
)
{
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;
    if (!x || !y || !z) return CUBLAS_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCublasGetStreamFromHandle(handle);
    cublasPointerMode_t pointer_mode = utilCublasGetPointerMode(handle);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int block_size = (prop.major >= 7) ? 256 : 128;
    int grid_size = (n + block_size - 1) / block_size;
    int max_blocks = prop.maxGridSize[0];
    if (grid_size > max_blocks) grid_size = max_blocks;

    if (incx == 1 && incy == 1 && n >= 4096) {
        const int ELEMENTS_PER_THREAD = 4;
        int optimized_grid = (n + block_size * ELEMENTS_PER_THREAD - 1) 
                            / (block_size * ELEMENTS_PER_THREAD);
        
        betaWeightsUpdateKernelOpt<256, 4><<<optimized_grid, block_size, 0, stream>>>(
            n, x, incx, y, incy, z, incz, pivot_idx
        );
        cudaDeviceSynchronize();
    } else {
        betaWeightsUpdateKernel<<<grid_size, block_size, 0, stream>>>(
            n, x, incx, y, incy, z, incz, pivot_idx
        );
        cudaDeviceSynchronize();
    }
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    
    return CUBLAS_STATUS_SUCCESS;
}


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
)
{
    if (major_dim <= 0 || nnz <= 0) return CUSPARSE_STATUS_SUCCESS;
    if (!x || !y) return CUSPARSE_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCusparseGetStreamFromHandle(handle);
    cusparsePointerMode_t pointer_mode = utilCusparseGetPointerMode(handle);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int block_size = (prop.major >= 7) ? 256 : 128;
    int grid_size = (major_dim + block_size - 1) / block_size;
    int max_blocks = prop.maxGridSize[0];
    if (grid_size > max_blocks) grid_size = max_blocks;

    if (sparse_x)
    {
        spmvUpdateSpKernel<<<grid_size, block_size, 0, stream>>>(
            y, x, x_idx, val, id, ptr, need_ptrs, nnz, major_dim, alpha, beta
        );
        cudaDeviceSynchronize();
    }
    else
    {
        spmvUpdateKernel<<<grid_size, block_size, 0, stream>>>(
            y, x, val, id, ptr, need_ptrs, nnz, major_dim, alpha, beta
        );
        cudaDeviceSynchronize();
    }
    
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    }
    
    return CUSPARSE_STATUS_SUCCESS;
}


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
)
{
    if (major_dim <= 0 || nnz <= 0) return CUSPARSE_STATUS_SUCCESS;
    if (!x || !y) return CUSPARSE_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCusparseGetStreamFromHandle(handle);
    cusparsePointerMode_t pointer_mode = utilCusparseGetPointerMode(handle);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int block_size = (prop.major >= 7) ? 256 : 128;
    int grid_size = (major_dim + block_size - 1) / block_size;
    int max_blocks = prop.maxGridSize[0];
    if (grid_size > max_blocks) grid_size = max_blocks;

    spmvUpdateSetKernel<<<grid_size, block_size, 0, stream>>>(
        y, set_id, set_id_size, x, z, val, id, ptr, need_ptrs, nnz, major_dim, alpha, beta
    );
    cudaDeviceSynchronize();
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    }
    
    return CUSPARSE_STATUS_SUCCESS;
}


cublasStatus_t btranOrFtran(
    cublasHandle_t handle,
    double *y,
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int size,
    const int col_len,
    const bool transpose
)
{
    if (col_len <= 0) return CUBLAS_STATUS_SUCCESS;
    if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCublasGetStreamFromHandle(handle);
    cublasPointerMode_t pointer_mode = utilCublasGetPointerMode(handle);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (transpose)
    {
        applyPFITKernel<<<1, 1, 0, stream>>>(    
            y, x,
            device_values,
            device_col_id,
            size, col_len
        );
        cudaDeviceSynchronize();
    }
    else
    {
         applyPFIKernel<<<1, 1, 0, stream>>>(    
            y, x,
            device_values,
            device_col_id,
            size, col_len
        );
cudaDeviceSynchronize();
    }
    
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    
    return CUBLAS_STATUS_SUCCESS;
}