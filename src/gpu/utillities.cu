#include "utillities.hpp"


cudaStream_t getStreamFromHandle(cublasHandle_t handle) {
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    return stream;
}


cublasPointerMode_t getPointerMode(cublasHandle_t handle) {
    cublasPointerMode_t mode;
    cublasGetPointerMode(handle, &mode);
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

    cudaStream_t stream = getStreamFromHandle(handle);
    cublasPointerMode_t pointer_mode = getPointerMode(handle);

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
    } else {
        betaWeightsUpdateKernel<<<grid_size, block_size, 0, stream>>>(
            n, x, incx, y, incy, z, incz, pivot_idx
        );
    }
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    
    return CUBLAS_STATUS_SUCCESS;
}

