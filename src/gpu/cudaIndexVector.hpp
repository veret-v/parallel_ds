#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>

#include "utillities.hpp"
#include "cudaDenseVector.hpp"

#include "../common/types.hpp"


#define VECTOR_COLS 1


class CudaSparseMatrix;


class CudaIndexVector : public CudaDenseVector<int>
{
friend class CudaSparseMatrix;

public:
    CudaIndexVector(const int size) {allocateMemory(size);};
    CudaIndexVector(const IndexVector& vector);
    CudaIndexVector(const CudaIndexVector& vector) : CudaDenseVector<int>(vector) {};
    CudaIndexVector() : CudaIndexVector(0) {};
    ~CudaIndexVector() {freeMemory();};

    void update(const int pos, const int val);
    
    CudaIndexVector& operator=(const CudaIndexVector& values_vector);

    int& operator[](const int idx) {return host_values[idx];};
};
