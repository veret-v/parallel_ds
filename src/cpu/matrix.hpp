#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream> 
#include <unordered_set>
#include <set>
#include <CoinPackedMatrix.hpp>

#include "mkl.h"

#include "valuesVector.hpp"

#include "../common/types.hpp"

#define EPS_ZERO 1e-16
#define GET_ID(vec, id) (vec.size() ? vec[id] : id)
    

class Matrix
{
protected:
    int m;
    int n;

    void*   pt[64]  = {0}; 
    int iparm[64];

    int mtype   = 11;       // несимметричная матрица
    int nrhs    = 1;        // одна правая часть
    int maxfct  = 1;        // количество факторов (обычно 1)
    int mnum    = 1;        // номер фактора (обычно 1)
    int msglvl  = 0;        // без вывода сообщений
    int error   = 0;        // код ошибки

    bool factorized = false;
   
    std::vector<double> elem_csr;
    std::vector<int> row_ptr;
    std::vector<int> col_id;

    void copyCoinPackedMatrix(CoinPackedMatrix& matrix);
    void cleanLUinfo();

public:
    Matrix(const int m, const int n);
    Matrix(
        std::vector<double> elem_csr, 
        std::vector<int> row_ptr, 
        std::vector<int> col_id,
        const int m, const int n
    );
    Matrix(CoinPackedMatrix& matrix);
    Matrix() : Matrix(0, 0) {};
    ~Matrix();

    double* getNonZeroElems() {return elem_csr.data();};
    int* getRowPtrs() {return row_ptr.data();};
    int* getColIds() {return col_id.data();};

    int calcNonzeroInColumn(const int& p) const;
    int getMajorSize() const {return row_ptr.size() - 1;};
    int getNonZeroSize() const {return elem_csr.size();};
    int getNumRows() const {return m;};
    int getNumCols() const {return n;};

    double operator()(const int i, const int j) const;
    int getElemIdx(const int i, const int j) const;

    ValuesVector operator()(const int p) const;
    Matrix operator()(const IndexVector& indexes) const;
    void resetData(const Matrix& matrix, const IndexVector& indexes);
    
    Matrix& operator=(const Matrix& Matrix);
    Matrix& operator=(CoinPackedMatrix& matrix);

    inline MatrixSize getSize() const {return std::make_tuple(m, n);};
    inline std::vector<double> getElem() const {return elem_csr;};
    inline std::vector<int> getColPtrs() const {return row_ptr;};
    inline std::vector<int> getRowIds() const {return col_id;};

    void stackColUnitMatrix();
    std::set<int> deleteCols(std::set<int> cols);

    // sol = alpha*A(T)*vec1 + beta*vec2
    void dotUpdate(
        const ValuesVector& vec1, 
        const ValuesVector& vec2, 
        ValuesVector& sol, 
        const double alpha, 
        const double beta,
        const IndexVector& cols_idx,
        const SpmvOptions& method,
        const bool set
    );

    void LUdecompose();

    void solve(
        ValuesVector& rhs, 
        ValuesVector& sol,
        bool transpose
    );

    void show() const;
};

