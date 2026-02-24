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
#include "cudaDataDenseVector.hpp"
#include "cudaIndexVector.hpp"


#include "../common/types.hpp"

#define SING_EPS 1e-12



//----------------------------------------------------------------------------------------
// Sparse matrix class, store csr an csc representation simultaniously for CUDSS solver.
//----------------------------------------------------------------------------------------
class CudaSparseMatrix
{
private:
    int m        = 0; // rows
    int n        = 0; // cols

    int major_dim = 0;
    int non_zero  = 0;

    bool CSR_exist = false;

    double* device_values_csc = nullptr;
    int* device_ptr_csc       = nullptr;
    int* device_id_csc        = nullptr;

    double* device_values_csr = nullptr;
    int* device_ptr_csr       = nullptr;
    int* device_id_csr        = nullptr;

    cusparseSpMatDescr_t mat_cusp_descr;    // дескриптор для матричного умножения
    cudssMatrix_t        mat_cudss_descr;   // дескриптор не транспонированной матрицы для решений Ax = b
    cudssMatrix_t        mat_cudss_descr_T; // дескриптор транспонированной матрицы для решений A^Tx = b(из-за структуры CUDSS)

    void allocateMemory(int non_zero_size, int col_starts);
    void copy(
        double* device_values, int* device_ptr, int* device_id, 
        const double* values_new, const int* ptr_new, const int* id_new,
        cudaMemcpyKind copy_type
    );
    void freeMemory();

    void createDescr();
    void genCsr(cusparseHandle_t& handle);
    
public:
    CudaSparseMatrix(const int m, const int n);
    CudaSparseMatrix(const Matrix& matrix, cusparseHandle_t& cusp_handle);
    CudaSparseMatrix(const CudaSparseMatrix& matrix);
    CudaSparseMatrix() : CudaSparseMatrix(0, 0) {};
    
    void getColumn(const int p, CudaDataDenseVector& rhs) const;
    void initI(const int n); // заполняет как единичную матрицу(нужно в начале алгоритма, так как всегда выбираем последние n столбцов)

    CudaSparseMatrix& operator=(const CudaSparseMatrix& matrix);
    CudaSparseMatrix& operator=(CoinPackedMatrix& matrix);

    inline MatrixSize getSize() {return std::make_tuple(m, n);};
    inline MatrixSize getSize() const {return std::make_tuple(m, n);};

    // sol = alpha*A(T)*vec1 + beta*vec2
    void dotUpdate(
        cusparseHandle_t& handle, 
        const CudaDataDenseVector& vec1, 
        const CudaIndexVector& vec1_idx,
        const bool sparse_vec1,
        const CudaDataDenseVector& vec2, 
        CudaDataDenseVector& sol,
        const CudaIndexVector& need_ptrs,
        const int size_ptrs,
        const double beta,
        const double alpha,
        const CudaIndexVector& set_idx,
        const SpmvOptions method
    );

    void LUdecompose(
        cudssHandle_t& handle, 
        cudssConfig_t& config, 
        cudssData_t& data,
        cudssHandle_t& handle_T, 
        cudssConfig_t& config_T, 
        cudssData_t& data_T
    );

    void solve(
        cudssHandle_t& handle, 
        cudssConfig_t& config, 
        cudssData_t& data,
        const CudaDataDenseVector& rhs, 
        CudaDataDenseVector& sol, 
        const bool& transpose
    );
};
