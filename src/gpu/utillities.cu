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
    int nnz,
    int minor_dim,
    int major_dim,
    int target_cols_size,
    double* sol,    
    const double* vec1,  
    const double* vec2,  
    const double* csr_val,
    const int* col_ids,
    const int* row_ptrs,
    const int* target_cols,
    const double alpha,
    const double beta,
    const bool set
)
{
    if (minor_dim <= 0 || major_dim <= 0 || nnz <= 0) return CUSPARSE_STATUS_SUCCESS;
    if (!vec1 || !vec2) return CUSPARSE_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCusparseGetStreamFromHandle(handle);
    cusparsePointerMode_t pointer_mode = utilCusparseGetPointerMode(handle);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    double* buff = nullptr;

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&buff, minor_dim*sizeof(double)),
        "cudaMalloc"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemset(buff, 0, minor_dim*sizeof(double)),
        "cudaMemset"
    );

    int block_size = (prop.major >= 7) ? 256 : 128;
    int grid_size = (major_dim + block_size - 1) / block_size;
    int max_blocks = prop.maxGridSize[0];
    if (grid_size > max_blocks) grid_size = max_blocks;

    spmvUpdateKernel<<<grid_size, block_size, 0, stream>>>(
        sol, vec1, vec2, 
        csr_val, col_ids, row_ptrs, 
        target_cols, buff, 
        target_cols_size, nnz, major_dim, 
        alpha, beta, set
    );
    
    cudaDeviceSynchronize();

    CUDA_CALL_AND_CHECK(
        cudaFree(buff),
        "cudaFree"
    );
    
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