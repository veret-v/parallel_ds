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
#include <cudss.h>

#include "utillities.hpp"
#include "cudaDenseVector.hpp"
#include "cudaIndexVector.hpp"

#include "../common/types.hpp"


#define VECTOR_COLS 1
#define EPS_Z 1e-20


class CudaSparseMatrix;
class PFIfactor;


/*
todo:
1. дескриптор надо обновлять при каждом 
   трансфере данных с кпу на гпу(сейчас он создается когда как) / 
   надо ли его вообще обновлять?

2. Добавить rvalues конструкторы через свап семантику

3. Добавить возможность выделять память только 
   на device или только на host (size = 10^6)
*/


class CudaDataDenseVector : public CudaDenseVector<double>
{
friend class CudaSparseMatrix;
friend class PFIfactor;

protected:
    cusparseDnVecDescr_t descr;
    cudssMatrix_t        cudss_descr;
    
    void createDescr();
    void destroyDescr();
    
public:
    typedef CudaDenseVectorIterator<double> iterator;
    typedef CudaDenseVectorIterator<const double> const_iterator;

    iterator begin() {return iterator(host_values);};
    iterator end() {return iterator(host_values + size);};

    const_iterator begin() const {return const_iterator(host_values);};
    const_iterator end() const {return const_iterator(host_values + size);};

    void updateVecBySparse(const int* crc_id, const double* crc_val, int nnz);
    void updateByPartialVec(const CudaDataDenseVector& values_vector, CudaIndexVector& indexes);
    void initUnitVec(const int p);
    void deleteVals(std::set<int> idxs);
    void resize(const int& new_size);

    double* getHostValues() {return host_values;};

    CudaDataDenseVector(const int size);
    CudaDataDenseVector(const ValuesVector& vector);
    CudaDataDenseVector(const CudaDataDenseVector& values_vector);
    CudaDataDenseVector() : CudaDataDenseVector(0) {};
    ~CudaDataDenseVector();
    
    CudaDataDenseVector& operator=(const CudaDataDenseVector& values_vector);
    CudaDataDenseVector& operator-();

    int countNonZero() const;

    double mean() const;
    double norm() const;

    double& operator[](const int idx) {return host_values[idx];};
    double operator[](const int idx) const {return host_values[idx];};
    
    double dot(
        const cublasHandle_t handle,
        const CudaDataDenseVector &values_vector
    ) const;
    double norm(const cublasHandle_t handle) const;
   
    void axpyUpdate(
        const cublasHandle_t handle, 
        const CudaDataDenseVector& values_vector, 
        const double& alpha
    );

    void betaWeightsUpdate(
        const cublasHandle_t handle, 
        const CudaDataDenseVector& alpha_q, 
        const CudaDataDenseVector& tau, 
        const int p_idx
    );

};


