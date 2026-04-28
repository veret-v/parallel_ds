#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>

#include <CoinMpsIO.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>
#include <math.h>
#include <cudss.h>

#include "utillities.hpp"
#include "cudaDataDenseVector.hpp"
#include "cudaIndexVector.hpp"

#include "../common/types.hpp"

#include "../cpu/matrix.hpp"


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

    bool CSС_exist   = false;
    bool memory_init = false;
    bool descr_exist = false;

    double* device_values_csc = nullptr;
    int* device_ptr_csc       = nullptr;
    int* device_id_csc        = nullptr;

    double* device_values_csr = nullptr;
    int* device_ptr_csr       = nullptr;
    int* device_id_csr        = nullptr;

    cusparseSpMatDescr_t mat_cusp_descr;    // дескриптор для матричного умножения
    cudssMatrix_t        mat_cudss_descr;   // дескриптор не транспонированной матрицы для решений Ax = b
    cudssMatrix_t        mat_cudss_descr_T; // дескриптор транспонированной матрицы для решений A^Tx = b(из-за структуры CUDSS)

    void allocateMemory(int non_zero_size, int m, int n);
    void copy(
        double* device_values, int* device_ptr, int* device_id, 
        const double* values_new, const int* ptr_new, const int* id_new,
        cudaMemcpyKind copy_type
    );
    void freeMemory();
    void destroyDesr();

    void copyCsrToHost(
        std::vector<double>& elem_csr,
        std::vector<int>& row_ptr,
        std::vector<int>& col_id
    );
    void updateDataByHost(
        cusparseHandle_t& handle, 
        const std::vector<double>& elem_csr,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_id
    );
    
public:
    CudaSparseMatrix(const int m, const int n);
    CudaSparseMatrix(const CoinPackedMatrix& matrix);
    CudaSparseMatrix(const CudaSparseMatrix& matrix);
    CudaSparseMatrix(Matrix& matrix);
    CudaSparseMatrix() : CudaSparseMatrix(0, 0) {};
    ~CudaSparseMatrix();

    void genCsc(cusparseHandle_t& handle);
    void genCsr(cusparseHandle_t& handle);
    void createDescr();

    int calcNonzeroInColumn(const int& p) const;
    
    void getColumn(cusparseHandle_t& handle, const int p, CudaDataDenseVector& rhs) const;
    void addSparseCol(
        cusparseHandle_t& handle, CudaDataDenseVector& vec, 
        const IndexVector& cols, const std::vector<double>& alpha);
    void addSparseCol(
        cusparseHandle_t& handle, CudaDataDenseVector& vec, 
        const IndexVector& cols, const double& alpha);
    void resetData(
        cusparseHandle_t& handle,
        const CudaSparseMatrix& matrix, 
        const CudaIndexVector& indexes
    );
    void initI(const int n); // заполняет как единичную матрицу(нужно в начале алгоритма, так как всегда выбираем последние n столбцов)
    void stackColUnitMatrix(cusparseHandle_t& handle);
    std::set<int> deleteCols(cusparseHandle_t& handle, std::set<int> cols);

    // CudaSparseMatrix& operator=(const CudaSparseMatrix& matrix);
    CudaSparseMatrix& operator=(CoinPackedMatrix& matrix);
    CudaSparseMatrix& operator=(Matrix& matrix);

    inline MatrixSize getSize() {return std::make_tuple(m, n);};
    inline MatrixSize getSize() const {return std::make_tuple(m, n);};

    // sol = alpha*A(T)*vec1 + beta*vec2
    void dotUpdate(
        cusparseHandle_t& handle, 
        const CudaDataDenseVector& vec1, 
        const CudaDataDenseVector& vec2, 
        CudaDataDenseVector& sol,
        const double alpha,
        const double beta,
        const CudaIndexVector& col_idx,
        const SpmvOptions method,
        const bool set
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

    void show();
};
