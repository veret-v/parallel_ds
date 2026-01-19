#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>

#include <CoinMpsIO.hpp>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusparse.h>
#include <cublas_api.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include "utillities.hpp"
#include "cudaDenseVector.hpp"
#include "../types.hpp"

#define SING_EPS 1e-12


class CudaSparseMatrix
{
friend class LUfactor;

private:
    size_t m        = 0; // rows
    size_t n        = 0; // cols

    size_t major_dim = 0;
    size_t non_zero = 0;

    bool is_CSC = true;

    double* host_values   = nullptr;
    int* host_ptr     = nullptr;
    int* host_id      = nullptr;

    double* device_values = nullptr;
    int* device_ptr   = nullptr;
    int* device_id    = nullptr;

    cusparseSpMatDescr_t descr = nullptr;

    void allocateMemory(size_t non_zero_size, size_t col_starts);
    void freeMemory();
    void createDescr();
    void cscToCsr(cusparseHandle_t handle);
    
public:
    void updateDeviceMem();
    void updateHostMem();

    CudaSparseMatrix(const size_t m, const size_t n);
    CudaSparseMatrix(const Matrix& matrix);
    CudaSparseMatrix() : CudaSparseMatrix(0, 0) {};
       
    void getColumn(const size_t p, CudaDenseVector& rhs) const;
    CudaSparseMatrix& operator=(const CudaSparseMatrix& matrix);

    inline MatrixSize getSize() {return std::make_tuple(m, n);};
    inline MatrixSize getSize() const {return std::make_tuple(m, n);};

    // y = alpha*A(T)*x + beta*y
    void dotUpdate(
        cusparseHandle_t handle, 
        const CudaDenseVector& rhs, 
        CudaDenseVector& sol, 
        double beta, double alpha, bool transpose
    );

    void LUdecompose(
        cusolverSpHandle_t handle, 
        LUfactor& lu_factor
    );
    void swapColumn(
        cusparseHandle_t handle, 
        CudaSparseMatrix& A, 
        const size_t col1, const size_t col2
    );

};


class LUfactor
{
friend class CudaSparseMatrix;

protected:
    cusparseMatDescr_t descr_l;
    cusparseMatDescr_t descr_u;

    double* device_values_l = nullptr;
    int* device_ptr_l   = nullptr;
    int* device_id_l    = nullptr;

    double* device_values_u = nullptr;
    int* device_ptr_u   = nullptr;
    int* device_id_u    = nullptr;

    int* P = nullptr;
    int* Q = nullptr;

    size_t non_zero_u = 0;
    size_t non_zero_l = 0;

public:
    void setup(
        cusolverRfHandle_t handle, 
        const CudaSparseMatrix& orig
    ) const;

    void solve(
        cusolverRfHandle_t handle, 
        const CudaDenseVector& rhs, 
        CudaDenseVector& sol, 
        const bool& transpose
    );

    void update(
        cusolverRfHandle_t handle, 
        const size_t& idx, 
        const CudaDenseVector& vector
    );
};