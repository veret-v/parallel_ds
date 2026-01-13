#include "cudaKernels.hpp"


template<typename T>
__global__ void vectorPFIKernel(const T* PFI_matrix, T* vector, int eta_id, int size)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    T eta_factor = vector[eta_id];

    for (int i = workIndex; i < size; i += stride) 
    {
        if(i != eta_id)
            vector[i] = vector[i] + eta_factor * PFI_matrix[i];
        else
            vector[i] = vector[i] * PFI_matrix[i];
    }
}


template<typename T>
__global__ void vectorDotKernel(const T* vec1, const T* vec2, T* result, int size)
{
    __shared__ float shared[BLOCK_DIM];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float prod = (idx < size) ? vec1[idx] * vec2[idx] : 0.0f;
    shared[tid] = prod;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}


template<typename T>
__global__ void vectorAddKernel(const T* a, const T* b, T* c, int size)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < size; i += stride) 
        c[i] = a[i] + b[i];
}


template<typename T>
__global__ void vectorSubKernel(const T* a, const T* b, T* c, int size)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < size; i += stride) 
        c[i] = a[i] - b[i];
}


template<typename T>
__global__ void vectorAddKernel(const T* a, T* c, int size)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < size; i += stride) 
        c[i] += a[i];
}


template<typename T>
__global__ void vectorSubKernel(const T* a, T* c, int size)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < size; i += stride) 
        c[i] -= a[i];
}


template<typename T>
__global__ void vectorMulKernel(const T* a, const T* b, T* c, int size)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < size; i += stride) 
        c[i] = a[i] * b[i];
}


template<typename T>
__global__ void vectorScaleKernel(const T* a, double scale, T* c, int size)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < size; i += stride) 
        c[i] = a[i] * scale;
}


template<typename T>
__global__ void matrixDotKernel(const T* matrix, const T* vector, T* new_vec, 
                                int m, int n)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < m; i += stride) 
    {
        for (size_t j = 0; j < n; j++)
            new_vec[i] += matrix[i * n + j] * vector[j];
    }
}


template<typename T>
__global__ void matrixDotTKernel(const T* matrix, const T* vector, T* new_vec, 
                                int m, int n)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = workIndex; i < n; i += stride) 
    {
        for (size_t j = 0; j < m; j++)
            new_vec[i] += matrix[j * n + i] * vector[j];
    }
}


template __global__ void matrixDotKernel<double>(const double*, const double*, double*, int, int);
template __global__ void matrixDotTKernel<double>(const double*, const double*, double*, int, int);
template __global__ void vectorSubKernel<double>(const double*, const double*, double*, int);
template __global__ void vectorAddKernel<double>(const double*, const double*, double*, int);
template __global__ void vectorAddKernel<double>(const double*, double*, int);
template __global__ void vectorSubKernel<double>(const double*, double*, int);
template __global__ void vectorDotKernel<double>(const double*, const double*, double*, int);
template __global__ void vectorScaleKernel<double>(const double*, double, double*, int);
template __global__ void vectorPFIKernel<double>(const double*, double*, int, int);