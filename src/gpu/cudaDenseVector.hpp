#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <CoinMpsIO.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>

#include "utillities.hpp"

#include "../common/types.hpp"


#define VECTOR_COLS 1


class CudaSparseMatrix;
class PFIfactor;


template <typename ValueType>
class CudaDenseVector
{
protected:
    ValueType* host_values   = nullptr;
    ValueType* device_values = nullptr;

    int size = 0;
        
    void allocateMemory(int size);
    void freeMemory();
    void copy(const ValueType* device_data, const ValueType* host_data);

    void checkSize(const CudaDenseVector<ValueType>& values_vector) const;
        
public:
    void updateHostMem();
    void updateDeviceMem();

    CudaDenseVector(const int size) {allocateMemory(size);};
    CudaDenseVector() : CudaDenseVector(0) {};
    ~CudaDenseVector();
        
    int getSize() const {return size;};
    int getSize() {return size;};

    void show() const;
};
