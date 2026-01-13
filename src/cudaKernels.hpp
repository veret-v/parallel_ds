#pragma once

#ifdef WITH_CUDA
    #include <cuda_runtime.h>

    #define BLOCK_DIM  256
    #define BLOCKS_NUM 1

    template<typename T>
    __global__ void vectorAddKernel(const T* a, const T* b, T* c, int size);

    template<typename T>
    __global__ void vectorSubKernel(const T* a, const T* b, T* c, int size);

    template<typename T>
    __global__ void vectorAddKernel(const T* a, T* c, int size);

    template<typename T>
    __global__ void vectorSubKernel(const T* a, T* c, int size);

    template<typename T>
    __global__ void vectorMulKernel(const T* a, const T* b, T* c, int size);

    template<typename T>
    __global__ void vectorScaleKernel(const T* a, double scale, T* c, int size);
    
    template<typename T>
    __global__ void matrixDotKernel(const T* matrix, const T* vector, T* new_vec, 
                                   int m, int n);

    template<typename T>
    __global__ void matrixDotTKernel(const T* matrix, const T* vector, T* new_vec, 
                                   int m, int n);

    template<typename T>
    __global__ void vectorPFIKernel(const T* PFI_matrix, T* vector, int eta_id, int size);

    template<typename T>
    __global__ void vectorDotKernel(const T* vec1, const T* vec2, T* result, int size);

    extern template __global__ void matrixDotKernel<double>(const double*, const double*, double*, int, int);
    extern template __global__ void matrixDotTKernel<double>(const double*, const double*, double*, int, int);
    extern template __global__ void vectorSubKernel<double>(const double*, const double*, double*, int);
    extern template __global__ void vectorSubKernel<double>(const double*, double*, int);
    extern template __global__ void vectorAddKernel<double>(const double*, const double*, double*, int);
    extern template __global__ void vectorAddKernel<double>(const double*, double*, int);
    extern template __global__ void vectorDotKernel<double>(const double*, const double*, double*, int);
    extern template __global__ void vectorScaleKernel<double>(const double*, double, double*, int);
    extern template __global__ void vectorPFIKernel<double>(const double*, double*, int, int);
        
#endif 

