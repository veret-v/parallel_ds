#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <CoinMpsIO.hpp>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>

#include "utillities.hpp"

#include "../common/types.hpp"


class LUfactor;
class CudaSparseMatrix;
class CudaDenseVector;


template<typename ValueType>
class CudaDenseVectorIterator
{
    friend class CudaDenseVector;

public:
    using iterator_category = std::input_iterator_tag;
    using value_type = ValueType;
    using difference_type = std::ptrdiff_t;
    using pointer = ValueType*;
    using reference = ValueType&;

    CudaDenseVectorIterator(const CudaDenseVectorIterator &it) : p(it.p) {};
    CudaDenseVectorIterator(ValueType* p) : p(p) {};

    reference operator*() const { return *p; }
    pointer operator->() const { return p; }
    bool operator!=(CudaDenseVectorIterator const& other) const {return p != other.p;};
    bool operator==(CudaDenseVectorIterator const& other) const {return p == other.p;}; 
    CudaDenseVectorIterator& operator++() {++p;return *this;};

private:
    pointer p;
};


class CudaDenseVector
{
friend class CudaSparseMatrix;
friend class LUfactor;

protected:
    double* host_values   = nullptr;
    double* device_values = nullptr;

    size_t size = 0;

    cusparseDnVecDescr_t descr;
    
    void allocateMemory(size_t size);
    void freeMemory();
    void createDescr();
    void checkSize(const CudaDenseVector& values_vector) const;
        
public:
    typedef CudaDenseVectorIterator<double> iterator;
    typedef CudaDenseVectorIterator<const double> const_iterator;

    void updateHostMem();
    void updateDeviceMem();

    CudaDenseVector(const size_t size);
    CudaDenseVector(const ValuesVector& vector);
    CudaDenseVector(const CudaDenseVector& values_vector);
    CudaDenseVector(const CoinPackedVector& values_vector);
    CudaDenseVector() : CudaDenseVector(0) {};
    ~CudaDenseVector();
    
    CudaDenseVector& operator=(const CudaDenseVector& values_vector);
    
    double operator[](const size_t idx) const {return host_values[idx];};

    size_t getSize() const {return size;};
    size_t getSize() {return size;};

    double dot(const cublasHandle_t handle, const CudaDenseVector &values_vector) const;
    double norm(const cublasHandle_t handle) const;
   
    void axpyUpdate(const cublasHandle_t handle, const CudaDenseVector& values_vector, const double& alpha);
    void betaWeightsUpdate(const cublasHandle_t handle, const CudaDenseVector& alpha_q, const CudaDenseVector& tau, const int p_idx);

    void show() const;

    iterator begin() {return iterator(host_values);};
    iterator end() {return iterator(host_values + size);};

    const_iterator begin() const {return const_iterator(host_values);};
    const_iterator end() const {return const_iterator(host_values + size);};
};