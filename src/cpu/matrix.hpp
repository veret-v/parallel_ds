#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream> 

#include "valuesVector.hpp"

#include "../common/types.hpp"


#define GET_ID(vec, id) (vec.size() ? vec[id] : id)
    

class Matrix
{
protected:
    int m;
    int n;
   
    std::vector<double> elem_csc;
    std::vector<int> col_ptr;
    std::vector<int> row_id;

    std::vector<double> elem_csr;
    std::vector<int> row_ptr;
    std::vector<int> col_id;

    void copyCoinPackedMatrix(CoinPackedMatrix& matrix);

public:
    Matrix(const int m, const int n);
    Matrix(CoinPackedMatrix& matrix);
    Matrix() : Matrix(0, 0) {};

    double operator()(const int i, const int j) const;
    double& operator()(const int i, const int j);

    ValuesVector operator()(const int p) const;
    Matrix operator()(const IndexVector& indexes) const;
    
    Matrix& operator=(const Matrix& Matrix);
    Matrix& operator=(CoinPackedMatrix& matrix);

    inline MatrixSize getSize() const {return std::make_tuple(m, n);};
    inline std::vector<double> getElem() const {return elem_csc;};
    inline std::vector<int> getColPtrs() const {return row_ptr;};
    inline std::vector<int> getRowIds() const {return col_id;};

    void stackColUnitMatrix();
    void dotEtaMatrix(const EtaMatrix& etaMatrix);
    // sol = alpha*A(T)*vec1 + beta*vec2
    // sol = alpha*A(T)*vec1 + beta*sol
    void dotUpdate(
        const ValuesVector& vec1, 
        const IndexVector& vec1_idx,
        const ValuesVector& vec2, 
        ValuesVector& sol, 
        const double alpha, 
        const double beta,
        const IndexVector& set_idx,
        const SpmvOptions& method
    );

    void genCSRorder();
    void show() const;
};

