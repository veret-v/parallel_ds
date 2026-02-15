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
#include <math.h>
#include <cudss.h>

#include "utillities.hpp"
#include "cudaDenseVector.hpp"

#include "../common/types.hpp"

#define SING_EPS 1e-12


class CudaSparseMatrix
{
private:
    size_t m        = 0; // rows
    size_t n        = 0; // cols

    size_t major_dim = 0;
    size_t non_zero  = 0;

    bool is_CSC = true;

    double* device_values = nullptr;
    int* device_ptr       = nullptr;
    int* device_id        = nullptr;

    cusparseSpMatDescr_t mat_cusp_descr;    // дескриптор для матричного умножения
    cudssMatrix_t        mat_cudss_descr;   // дескриптор не транспонированной матрицы для решений Ax = b
    cudssMatrix_t        mat_cudss_descr_T; // дескриптор транспонированной матрицы для решений A^Tx = b(из-за структуры CUDSS)

    void allocateMemory(size_t non_zero_size, size_t col_starts);
    void freeMemory();
    void createDescr();
    void cscToCsr(cusparseHandle_t handle);
    
public:
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
        cudssHandle_t handle, 
        cudssConfig_t config, 
        cudssData_t data,
        cudssHandle_t handle_T, 
        cudssConfig_t config_T, 
        cudssData_t data_T
    );

    void solve(
        cudssHandle_t handle, 
        cudssConfig_t config, 
        cudssData_t data,
        const CudaDenseVector& rhs, 
        CudaDenseVector& sol, 
        const bool& transpose
    );

    void swapColumn(
        cusparseHandle_t handle, 
        CudaSparseMatrix& A, 
        const size_t col1, const size_t col2
    );

};


class PFIfactor
{
private:
    size_t n = 0;

    double* device_values = nullptr;
    int* device_ptr       = nullptr;

public:
    PFIfactor();
    ~PFIfactor();

    void addEtaMatrix();
    void applyPFI();
};