#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>
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
    typedef CudaDenseVectorIterator<int> iterator;
    typedef CudaDenseVectorIterator<const int> const_iterator;

    iterator begin() {return iterator(host_values);};
    iterator end() {return iterator(host_values + size);};

    const_iterator begin() const {return const_iterator(host_values);};
    const_iterator end() const {return const_iterator(host_values + size);};


    CudaIndexVector(const int size) {allocateMemory(size);};
    CudaIndexVector(const IndexVector& vector);
    CudaIndexVector(const CudaIndexVector& vector) : CudaDenseVector<int>(vector) {};
    CudaIndexVector() : CudaIndexVector(0) {};
    ~CudaIndexVector() {freeMemory();};

    void update(const int pos, const int val);
    
    CudaIndexVector& operator=(const CudaIndexVector& values_vector);

    int& operator[](const int idx) {return host_values[idx];};
    int operator[](const int idx) const {return host_values[idx];};
};
