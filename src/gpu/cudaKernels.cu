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
            // printf("Debug: row=%d, val=%f, dim=%d\n", row, sum, major_dim);
        }
    }
}


__global__ void spmvUpdateTKernel(    
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

    for (int row = warp_id; row < major_dim; row += warp_stride)
    {
        double sum = 0.0;

        for (int j = ptr_csr[row] + t_warp; j < ptr_csr[row + 1]; j += WARP_SIZE)
            sum += val_csr[j] * vec1[id_csr[j]];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) 
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (t_warp == 0)
        {
            buff[row] = alpha * sum;
            // printf("Debug: row=%d, val=%f, dim=%d\n", row, sum, major_dim);
        }
    }

    __syncthreads();

    for (int col_id = t_id; col_id < cols_idx_size; col_id += stride)
    {
        int col_id_it = cols_idx[col_id];
        // printf("Debug: row=%d, val=%f, dim=%d\n", col_id_it, vec2[col_id_it], major_dim);
        sol[set ? col_id_it : col_id] = buff[col_id_it] + beta * vec2[col_id_it];
    }
}

__global__ void addSpColToVecKernel(
    int nnz,
    const double* __restrict__ col_vals,
    const int* __restrict__ row_idx,
    double* __restrict__ vec,
    double alpha
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    for (tid; tid < nnz; tid += stride) {
        int row = row_idx[tid];
        double val = col_vals[tid];
        atomicAdd(vec + row, alpha * val);
    }
}


__global__ void getColFromSpKernel(    
    double* vec,
    const double* val_csc,
    const int* id_csc,
    const int* ptr_csc,
    const int col_idx
)
{
    int t_id = threadIdx.x + blockDim.x * blockIdx.x; 
    int stride = blockDim.x * gridDim.x;

    for (int row = ptr_csc[col_idx] + t_id; row < ptr_csc[col_idx + 1]; row += stride)
    {
        vec[id_csc[row]] = val_csc[row];
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
    
    for (; i < col_len; i += stride)
    {
        int col_num = device_col_id[eta_num];
        double eta_fact_val_i = x[col_num] * device_values[i + eta_num * col_len];
        y[i] = (i != col_num) ? x[i] + eta_fact_val_i : eta_fact_val_i; 
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
    extern __shared__ double sdata[];   // динамический размер (blockDim.x)
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // printf("Debug: id=%d val=%f al=%f\n", i, x[i], device_values[i + eta_num * col_len]);
    double sum = 0.0;
    for (; i < col_len; i += stride) {
        sum += x[i] * device_values[i + eta_num * col_len];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // printf("Debug: result id=%d val=%f\n", tid, sdata[0]);
        atomicAdd(y, sdata[0]);
    }
}


__global__ void negateKernel(double *val) {
    *val = -(*val);
}


template __global__ void betaWeightsUpdateKernelOpt<256, 4>(
    int n, const double* x, int incx,
    const double* y, int incy,
    double* z, int incz, int pivot_idx);



__global__ void etaSolveKernel(
    double *y,
    const double *x,
    const double *values,   
    const int *col_id,
    int size,
    int col_len,
    bool transpose
)
{
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < col_len; i += stride) {
        y[i] = x[i];
    }
    __syncthreads();

    for (int step = 0; step < size; ++step) {
        int idx = transpose ? (size - 1 - step) : step;
        int p = col_id[idx];
        const double *col = values + (size_t)idx * col_len;

        if (!transpose) {
            double alpha = y[p];
            for (int i = tid; i < col_len; i += stride) {
                double add = alpha * col[i];
                if (i == p)
                    y[i] = add;      // прямая замена, без добавки
                else
                    y[i] += add;
            }
            __syncthreads();
        } else {
            double sum = 0.0;
            for (int i = tid; i < col_len; i += stride) {
                sum += col[i] * y[i];
            }

            __shared__ double sdata[1024];  
            sdata[tid] = sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) 
            {
                y[p] = sdata[0];
            }
            __syncthreads();
        }
    }
}