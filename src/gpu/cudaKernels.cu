#include "cudaKernels.hpp"


__global__ void betaWeightsUpdateKernel(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double z_f = z[pivot_idx];
    double x_f = x[pivot_idx];
    
    for (int i = idx; i < n; i += stride) {
        int x_idx = i * incx;
        int y_idx = i * incy;
        int z_idx = i * incz;
        z[z_idx] = (z_idx != pivot_idx) ? z[z_idx] - 2 * x[i] / x_f * y[i] + pow(x[i] / x_f, 2) * z_f : z_f / pow(x_f, 2);; 
    }
}


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
)
{
    int idx = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + threadIdx.x;

    double z_f = z[pivot_idx];
    double x_f = x[pivot_idx];
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int pos = idx + i * BLOCK_SIZE;
        if (pos < n) {
            int x_idx = pos * incx;
            int y_idx = pos * incy;
            int z_idx = pos * incz;
            z[z_idx] = (z_idx != pivot_idx) ? z[z_idx] - 2 * x[i] / x_f * y[i] + pow(x[i] / x_f, 2) * z_f : z_f / pow(x_f, 2);; 
        }
    }
};


__global__ void swapColumnKernel(
    int m, 
    int n,
    double* old_values, 
    double* old_row_ind, 
    double* old_col_ptr,                           
    int col_idx,
    double* new_col_values, 
    const int* new_col_row_ind, 
    int new_col_nnz,
    double* new_values, 
    int* new_row_ind, 
    int* new_col_ptr
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int old_col_start = old_col_ptr[col_idx];
        int old_col_end = old_col_ptr[col_idx + 1];
        int old_col_size = old_col_end - old_col_start;
        int offset = new_col_nnz - old_col_size;
        
        for (int i = 0; i <= col_idx; i++) {
            new_col_ptr[i] = old_col_ptr[i];
        }
        
        new_col_ptr[col_idx + 1] = old_col_start + new_col_nnz;
        
        for (int i = col_idx + 2; i <= n; i++) {
            new_col_ptr[i] = old_col_ptr[i] + offset;
        }
    }
    __syncthreads();
    
    // 2. Определяем границы для копирования
    int old_col_start = old_col_ptr[col_idx];
    int old_col_end = old_col_ptr[col_idx + 1];
    int left_size = old_col_start;
    int right_size = old_col_ptr[n] - old_col_end;
    
    // 3. Параллельное копирование данных
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Копируем левую часть (столбцы до заменяемого)
    if (idx < left_size) {
        new_values[idx] = old_values[idx];
        new_row_ind[idx] = old_row_ind[idx];
    }
    
    // Копируем новый столбец
    if (idx < new_col_nnz) {
        new_values[left_size + idx] = new_col_values[idx];
        new_row_ind[left_size + idx] = new_col_row_ind[idx];
    }
    
    // Копируем правую часть (столбцы после заменяемого)
    if (idx < right_size) {
        new_values[left_size + new_col_nnz + idx] = old_values[old_col_end + idx];
        new_row_ind[left_size + new_col_nnz + idx] = old_row_ind[old_col_end + idx];
    }
}