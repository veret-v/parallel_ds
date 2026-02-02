#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream> 

#include "valuesVector.hpp"

#include "../common/types.hpp"


class Matrix
{
protected:
    size_t m;
    size_t n;
   
    std::vector<double> elem_csc;
    std::vector<size_t> col_ptr;
    std::vector<size_t> row_id;

    std::vector<double> elem_csr;
    std::vector<size_t> row_ptr;
    std::vector<size_t> col_id;

public:
    Matrix(const size_t m, const size_t n);
    Matrix(CoinPackedMatrix& matrix);
    Matrix() : Matrix(0, 0) {};
   
    double operator()(const size_t i, const size_t j) const;
    ValuesVector operator()(const size_t p) const;
    Matrix operator()(const IndexVector& indexes) const;
    Matrix& operator=(const Matrix& Matrix);

    inline MatrixSize getSize() const {return std::make_tuple(m, n);};
    inline std::vector<double> getElem() const {return elem_csc;};
    inline std::vector<size_t> getColPtrs() const {return row_ptr;};
    inline std::vector<size_t> getRowIds() const {return col_id;};

    void stackColUnitMatrix();
    void dotEtaMatrix(const EtaMatrix& etaMatrix);
    ValuesVector dot(const ValuesVector& Vector, bool transpose);
    Matrix T() const;

    void swapColumn(Matrix& A, const size_t b_idx, const size_t a_idx);
    void swapRows(const size_t row1, const size_t row2);
    void genCSRorder();
    void show() const;
};

