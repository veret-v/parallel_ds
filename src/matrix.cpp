#include "matrix.hpp"


Matrix::Matrix(const size_t m, const size_t n)
{
    this -> m = m;
    this -> n = n;
    if (m == n) size = m;
    elem = std::vector<double>(n * m);
}


double& Matrix::operator()(const size_t i, const size_t j)
{
    return elem[i * n + j];
}


double Matrix::operator()(const size_t i, const size_t j) const
{
    return elem[i * n + j];
}


ValuesVector Matrix::operator()(const size_t p) const
{
    ValuesVector new_vec(m);
    for (size_t i = 0; i < m; i++) 
            new_vec[i] = this->operator()(i, p);
    return new_vec;

}


Matrix& Matrix::operator=(const Matrix &matrix)
{
    this -> m = matrix.m;
    this -> n = matrix.n;
    this -> elem = matrix.elem;
    if (m == n) size = m;
    return *this;
}


size_t Matrix::rank() const
{
    Matrix rank_check(*this);
    size_t rank = m;
    for (size_t i = 1; i < m; i++) {
        for (size_t k = 0; k < i; k++) {
            for (size_t j = 0; j < n; j++) 
                rank_check(i, j) -= rank_check(i, k) / rank_check(k, k) * rank_check(k, j);
        }
        size_t check = 0;
        for (size_t j = 0; j < n; j++)
            if (rank_check(i, j) == 0) check += 1;
        if (check == n) rank -= 1;
    }
    return rank;
}


Matrix Matrix::T() const
{
    Matrix new_Matrix(n, m);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++)
            new_Matrix(j, i) = this->operator()(i, j);
    }
    return new_Matrix;
}

ValuesVector Matrix::dot(const ValuesVector &vector) const
{
    ValuesVector new_vec(m);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++)
            new_vec[i] += this->operator()(i, j) * vector[j];
    }
    return new_vec;
}

Matrix Matrix::stackColumns(const Matrix &matrix) const
{
    if (matrix.m != m)
    {
        throw "Incorrect size";
        exit(1);
    }

    Matrix new_Matrix(m, n + matrix.n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++)
            new_Matrix(i, j) = this->operator()(i, j);
    }
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < matrix.n; j++)
            new_Matrix(i, n + j) = matrix(i, j);
    }

    return new_Matrix;
}

void Matrix::setColumn(const size_t& idx, const ValuesVector& vector)
{
    if (m != vector.getSize())
    {
        throw "Incorrect size";
        exit(1);
    }

    for (size_t i = 0; i < m; i++) 
        operator()(i, idx) = vector[i];
}


void Matrix::swapColumn(Matrix& A, const size_t b_idx, const size_t a_idx)
{
    if (m != std::get<0>(A.getSize()))
    {
        throw "Incorrect size";
        exit(1);
    }

    double buf;
    for (size_t i = 0; i < m; i++)
    {
        buf  = operator()(i, b_idx);
        operator()(i, b_idx) = A(i, a_idx);
        A(i, a_idx) = buf;
    }
}


Matrix Matrix::operator()(const IndexVector& indexes) const
{
    Matrix new_Matrix(m, indexes.size());
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < indexes.size(); j++)
            new_Matrix(i, j) = this->operator()(i, indexes[j]);
    }
    return new_Matrix;
}


void Matrix::show() const
{
    std::cout << "Matrix" << "(" << m << "," << n << "):" << std::endl;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++)
            std::cout << this->operator()(i, j) << " ";
        std::cout << std::endl;
    }
}


Matrix Matrix::getMinor(int row, int col) const {
    Matrix minor(size - 1, size - 1);
    int minor_i = 0;
    
    for (int i = 0; i < size; i++) {
        if (i == row) continue;
        
        int minor_j = 0;
        for (int j = 0; j < size; j++) {
            if (j == col) continue;
            
            minor(minor_i, minor_j) = this->operator()(i, j);
            minor_j++;
        }
        minor_i++;
    }
    
    return minor;
}

double Matrix::determinantRecursive() const {
    if (size == 1) {
        return this->operator()(0, 0);
    }
    
    if (size == 2) {
        return this->operator()(0, 0) * this->operator()(1, 1) - this->operator()(0, 1) * this->operator()(1, 0);
    }
    
    double det = 0.0;
    int sign = 1;
    
    for (int j = 0; j < size; j++) {
        Matrix minor = getMinor(0, j);
        det += sign * this->operator()(0, j) * minor.determinantRecursive();
        sign = -sign;
    }
    
    return det;
}
