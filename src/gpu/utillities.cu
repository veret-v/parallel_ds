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
    int m,
    int n,
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
    const bool set,
    const bool transpose
)
{
    if (m <= 0 || n <= 0 || nnz <= 0) return CUSPARSE_STATUS_SUCCESS;
    if (!vec1 || !vec2) return CUSPARSE_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCusparseGetStreamFromHandle(handle);
    cusparsePointerMode_t pointer_mode = utilCusparseGetPointerMode(handle);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    double* buff = nullptr;

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&buff, n*sizeof(double)),
        "cudaMalloc"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemset(buff, 0, n*sizeof(double)),
        "cudaMemset"
    );

    int block_size = (prop.major >= 7) ? 256 : 128;
    int grid_size = (m + block_size - 1) / block_size;
    int max_blocks = prop.maxGridSize[0];
    if (grid_size > max_blocks) grid_size = max_blocks;

    if (transpose)
    {
        spmvUpdateTKernel<<<grid_size, block_size, 0, stream>>>(
            sol, vec1, vec2, 
            csr_val, col_ids, row_ptrs, 
            target_cols, buff, 
            target_cols_size, nnz, n, 
            alpha, beta, set
        );
    }
    else
    {
        spmvUpdateKernel<<<grid_size, block_size, 0, stream>>>(
            sol, vec1, vec2, 
            csr_val, col_ids, row_ptrs, 
            target_cols, buff, 
            target_cols_size, nnz, m, 
            alpha, beta, set
        );
    }
    
    
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



cusparseStatus_t getColFromSp(
    cusparseHandle_t handle,
    int m,
    int n,
    int p,
    double* vec,    
    const double* csc_val,
    const int* col_ids,
    const int* row_ptrs
)
{
    if (m <= 0 || n <= p) return CUSPARSE_STATUS_SUCCESS;
    if (!vec) return CUSPARSE_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCusparseGetStreamFromHandle(handle);
    cusparsePointerMode_t pointer_mode = utilCusparseGetPointerMode(handle);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int block_size = (prop.major >= 7) ? 256 : 128;
    int grid_size = (m + block_size - 1) / block_size;
    int max_blocks = prop.maxGridSize[0];
    if (grid_size > max_blocks) grid_size = max_blocks;

    getColFromSpKernel<<<grid_size, block_size, 0, stream>>>(
        vec, csc_val, col_ids, row_ptrs, p
    );
       
    cudaDeviceSynchronize();

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    }
    
    return CUSPARSE_STATUS_SUCCESS;
}


cusparseStatus_t addSpColsToVec(
    cusparseHandle_t handle,
    int m,
    int n,
    int p,
    double* vec,    
    const double* csc_val,
    const int* col_ids,
    const int* row_ptrs,
    const double alpha
)
{
    if (m <= 0 || n <= p) return CUSPARSE_STATUS_SUCCESS;
    if (!vec || !csc_val || !col_ids || !row_ptrs)
        return CUSPARSE_STATUS_INVALID_VALUE;

    int start, end;
    cudaMemcpyAsync(&start, row_ptrs + p, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&end, row_ptrs + p + 1, sizeof(int), cudaMemcpyDeviceToHost);
    int nnz = end - start;
    if (nnz == 0) return CUSPARSE_STATUS_SUCCESS;

    cudaStream_t stream = utilCusparseGetStreamFromHandle(handle);
    int block = 256;
    int grid = (nnz + block - 1) / block;

    addSpColToVecKernel<<<grid, block, 0, stream>>>(
        nnz, csc_val + start, col_ids + start, vec, alpha
    );
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
    cublasSetStream(handle, stream);

    cublasPointerMode_t pointer_mode = utilCublasGetPointerMode(handle);

    cudaMemcpyAsync(y, x, col_len * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    std::vector<int> host_col_id(size);
    cudaMemcpy(host_col_id.data(), device_col_id, size * sizeof(int), cudaMemcpyDeviceToHost);

    double *dev_alpha = nullptr;
    cudaMalloc(&dev_alpha, sizeof(double));
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    for (int i = 0; i < size; ++i)
    {
        int idx = transpose ? (size - 1 - i) : i;
        int p = host_col_id[idx];                    
        const double *col = device_values + (size_t)idx * col_len;

        if (!transpose) 
        {
            cudaMemcpyAsync(dev_alpha, &y[p], sizeof(double), cudaMemcpyDeviceToDevice, stream);
            cublasDaxpy(handle, col_len, dev_alpha, col, 1, y, 1);

            cudaMemcpyAsync(&y[p], &col[p], sizeof(double), cudaMemcpyDeviceToDevice, stream);
            cublasDscal(handle, 1, dev_alpha, &y[p], 1);
        }
        else 
        {
            cublasDdot(handle, col_len, col, 1, y, 1, dev_alpha);
            cudaMemcpyAsync(&y[p], dev_alpha, sizeof(double), cudaMemcpyDeviceToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);
    cudaFree(dev_alpha);
    cublasSetPointerMode(handle, pointer_mode);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}



cublasStatus_t btranOrFtran_v2(
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
    if (col_len <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;

    cudaStream_t stream = utilCublasGetStreamFromHandle(handle);
    cublasSetStream(handle, stream);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // cudaMemcpyAsync(y, x, col_len * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    
    int max_size = min(1024, col_len);
    int block_size = 1;
    while (block_size * 2 <= max_size) block_size *= 2;
    
    etaSolveKernel<<<1, block_size, 0, stream>>>(
        y, x, device_values, device_col_id, size, col_len, transpose);
    cudaStreamSynchronize(stream);
    
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}