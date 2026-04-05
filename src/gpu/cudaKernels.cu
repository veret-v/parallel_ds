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
)
{
    int t_id = threadIdx.x + blockDim.x * blockIdx.x; 
    int t_warp = t_id % WARP_SIZE; 
    int warp_id = t_id / WARP_SIZE;
    int stride = blockDim.x * gridDim.x;
    int warp_stride = (blockDim.x * gridDim.x + WARP_SIZE - 1) / WARP_SIZE;

    for (int col_id = t_id; col_id < cols_idx_size; col_id += stride)
        buff[cols_idx[col_id]] = vec1[set ? cols_idx[col_id] : col_id];

    __syncthreads();

    for (int row = warp_id; row < major_dim; row += warp_stride)
    {
        double sum = 0.0;

        for (int j = ptr_csr[row] + t_warp; j < ptr_csr[row + 1]; j += WARP_SIZE)
            sum += val_csr[j] * buff[id_csr[j]];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) 
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (t_warp == 0)
        {
            sol[row] = beta * vec2[row] + alpha * sum;
            printf("Debug: row=%d, val=%f, dim=%d\n", row, sum, major_dim);
        }
    }
}


__global__ void applyEtaMatKernel(    
    double *y, 
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int eta_num,
    const int col_len
)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (i; i < col_len; i += stride)
    {
        if (i != device_col_id[eta_num])
            y[i] = x[i] + x[device_col_id[eta_num]] * device_values[i + eta_num * col_len]; 
        else
            y[i] = x[device_col_id[eta_num]] * device_values[i + eta_num * col_len]; 
    }
    
}


__global__ void applyPFIKernel(    
    double *y,
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int size,
    const int col_len
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx == 0) {  
        int numBlocks = (col_len + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int i = 0; i >= size; i++)
        {
            applyEtaMatKernel<<<numBlocks, WARP_SIZE>>>(
                y, x, device_values, 
                device_col_id, i, 
                col_len
            );
        
            __syncthreads();
        }
    }
}


__global__ void applyEtaMatTKernel(   
    double *y, 
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int eta_num,
    const int col_len
)
{
    __shared__ double sdata[WARP_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (i; i < col_len; i += stride)
        sdata[tid] = x[i] * device_values[i + eta_num * col_len];

    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) 
        atomicAdd(y + device_col_id[eta_num], sdata[0]);
}



__global__ void applyPFITKernel(    
    double *y,
    const double *x,
    const double *device_values,
    const int *device_col_id,
    const int size,
    const int col_len
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx == 0) {  
        int numBlocks = (col_len + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int i = size - 1; i >= 0; i--)
        {
            applyEtaMatTKernel<<<numBlocks, WARP_SIZE>>>(
                y, x, device_values, 
                device_col_id, i, 
                col_len
            );
        
            __syncthreads();
        }
    }
}


template __global__ void betaWeightsUpdateKernelOpt<256, 4>(
    int n, const double* x, int incx,
    const double* y, int incy,
    double* z, int incz, int pivot_idx);