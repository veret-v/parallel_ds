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
#include "cudaDenseVector.hpp"
#include "cudaIndexVector.hpp"

#include "../common/types.hpp"


#define VECTOR_COLS 1


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
    void updateVecBySparse(const int* crc_id, const double* crc_val, int nnz);
    void updateByPartialVec(const CudaDataDenseVector& values_vector, CudaIndexVector& indexes);
    void addSparseCol(const CudaSparseMatrix& mat, const int p, const double alpha);
    void initUnitVec(const int p);

    CudaDataDenseVector(const int size);
    CudaDataDenseVector(const ValuesVector& vector);
    CudaDataDenseVector(const CudaDataDenseVector& values_vector);
    CudaDataDenseVector() : CudaDataDenseVector(0) {};
    ~CudaDataDenseVector();
    
    CudaDataDenseVector& operator=(const CudaDataDenseVector& values_vector);
    CudaDataDenseVector& operator-();

    double& operator[](const int idx) {return host_values[idx];};
    
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
