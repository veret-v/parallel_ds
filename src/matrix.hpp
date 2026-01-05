#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>

#include "valuesVector.hpp"
#include "types.hpp"


class Matrix
{
private:
    size_t m;
    size_t n;
    size_t size;
    std::vector<double> elem;

public:
    Matrix(const size_t m, const size_t n);
    Matrix(const CoinPackedMatrix& matrix);
    Matrix() : Matrix(0, 0) {};

    double& operator()(const size_t i, const size_t j);
    double operator()(const size_t i, const size_t j) const;
    ValuesVector operator()(const size_t p) const;
    Matrix operator()(const IndexVector& indexes) const;
    Matrix& operator=(const Matrix& Matrix);

    inline MatrixSize getSize() {return std::make_tuple(m, n);};
    inline MatrixSize getSize() const {return std::make_tuple(m, n);};

    Matrix stackColumns(const Matrix& Matrix) const;
    Matrix T() const;
    ValuesVector dot(const ValuesVector& Vector) const;

    size_t rank() const;
    Matrix getMinor(int row, int col) const;
    double determinantRecursive() const;

    void setColumn(const size_t& idx, const ValuesVector& vector);
    void swapColumn(Matrix& A, const size_t b_idx, const size_t a_idx);

    void show() const;
};

