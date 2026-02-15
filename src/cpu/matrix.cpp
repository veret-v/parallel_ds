#include "matrix.hpp"


Matrix::Matrix(const size_t m, const size_t n)
{
    this -> m = m;
    this -> n = n;
    
    elem_csc = std::vector<double>(0);
    col_ptr = std::vector<size_t>(0);
    row_id = std::vector<size_t>(0);
}


Matrix::Matrix(CoinPackedMatrix& matrix)
{
    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();

    size_t non_zero_size;
    size_t ptr_size;

    if (!matrix.isColOrdered()) matrix.reverseOrdering();
    
    non_zero_size = matrix.getNumElements();
    ptr_size = matrix.getMajorDim() + 1;

    elem_csc = std::vector<double>(matrix.getElements(), matrix.getElements() + non_zero_size);
    col_ptr = std::vector<size_t>(matrix.getIndices(), matrix.getIndices() + ptr_size);
    row_id = std::vector<size_t>(matrix.getVectorStarts(), matrix.getVectorStarts() + non_zero_size);
}


double Matrix::operator()(const size_t i, const size_t j) const
{
    int left = col_ptr[j];
    int right = col_ptr[j + 1] - 1;
    
    while (left <= right) {
        int mid =  left + (right - left) / 2;
        if (row_id[mid] == i) 
            return elem_csc[mid];
        else if (row_id[mid] < i) 
            left = mid + 1;
        else 
            right = mid - 1;
    }
    return 0.0; // add unbound axception  processing
}


double& Matrix::operator()(const size_t i, const size_t j)
{
    int left = col_ptr[j];
    int right = col_ptr[j + 1] - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (row_id[mid] == i) 
            return elem_csc[mid];
        else if (row_id[mid] < i) 
            left = mid + 1;
        else 
            right = mid - 1;
    }
    return elem_csc[0]; // add unbound axception  processing
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
    
    elem_csc = matrix.elem_csc;;
    col_ptr = matrix.col_ptr;
    row_id = matrix.row_id;

    return *this;
}


ValuesVector Matrix::dot(const ValuesVector &vector, bool transpose)
{
    if (transpose)
    {
       ValuesVector new_vec(n);
       for (size_t i = 0; i < n; i++) {
            for (size_t j = col_ptr[i]; j < col_ptr[i + 1]; j++)
                new_vec[i] += elem_csc[j] * vector[row_id[j]];
        }
        return new_vec;
    }
    else
    {
        ValuesVector new_vec(m);
        for (size_t i = 0; i < n; i++) {
            if (vector[i] < 1e-12) continue;
            
            for (size_t j = col_ptr[i]; j < col_ptr[i + 1]; j++)
            {
                size_t k = row_id[j];
                new_vec[k] += elem_csc[j] * vector[i];

            }
        }
        return new_vec;
    }
}

void Matrix::stackColUnitMatrix() 
{
    for (size_t i = 0; i < m; i++)
    {
        elem_csc.push_back(1);
        col_ptr.push_back(col_ptr[i] + m);
        row_id.push_back(i);
    }
    n = n + m;
}


void Matrix::dotEtaMatrix(const EtaMatrix& etaMatrix)
{
    size_t p = std::get<1>(etaMatrix);
    ValuesVector val = std::get<0>(etaMatrix);

    for (size_t j = 0; j < m && j != p; j++)
    { 
        std::vector<size_t> buff_id;
        auto start_id = col_id.begin();
        std::merge(
            start_id + row_ptr[p], start_id + row_ptr[p + 1],
            start_id + row_ptr[j], start_id + row_ptr[j + 1],
            buff_id.begin()
        );
        std::vector<double> buff_val(buff_id.size());
        
        size_t j_it  = row_ptr[j];
        size_t p_it  = row_ptr[p];
        for (size_t i = 0; i < buff_id.size(); i++)
        {
            if (i == col_id[p_it] && i == col_id[j_it])
            {
                buff_val[i] = elem_csr[j_it] + elem_csr[p_it] * val[i];
                j_it++;
                p_it++;
            }
            else if (i == col_id[p_it])
            {
                buff_val[i] = elem_csr[p_it] * val[i];
                p_it++;
            }
            else
            {
                buff_val[i] = elem_csr[j_it];
                j_it++;
            }
        }
        auto start_val = elem_csr.begin();
        std::swap_ranges(start_id + row_ptr[j], start_id + row_ptr[j + 1],
                         buff_id.begin());
        std::swap_ranges(start_val + row_ptr[j], start_val + row_ptr[j + 1],
                         buff_val.begin());
    }   
    for (size_t i = row_ptr[p]; i < row_ptr[p + 1]; i++)
        elem_csr[i] = elem_csr[i] * val[col_id[i]];
}


void Matrix::swapColumn(Matrix& A, const size_t b_idx, const size_t a_idx)
{
    if (m != std::get<0>(A.getSize()))
    {
        throw "Incorrect size";
        exit(1);
    }

    size_t col1_start = col_ptr[b_idx];
    size_t col1_end = col_ptr[b_idx + 1];
    
    size_t col2_start = A.col_ptr[a_idx];
    size_t col2_end = A.col_ptr[a_idx + 1];

    std::swap_ranges(elem_csc.begin() + col1_start, elem_csc.begin() + col1_end,
                     A.elem_csc.begin() + col2_start);
    std::swap_ranges(row_id.begin() + col1_start, row_id.begin() + col1_end,
                     A.row_id.begin() + col2_start);
}


void Matrix::swapRows(const size_t row1, const size_t row2)
{
    size_t row1_start = row_ptr[row1];
    size_t row1_end = row_ptr[row1 + 1];
    
    size_t row2_start = row_ptr[row2];
    size_t row2_end = row_ptr[row2 + 1];

    std::swap_ranges(elem_csr.begin() + row1_start, elem_csc.begin() + row1_end,
                     elem_csc.begin() + row2_start);
    std::swap_ranges(col_id.begin() + row1_start, col_id.begin() + row1_end,
                     col_id.begin() + row2_start);
}


void Matrix::genCSRorder()
{
    std::vector<size_t> row_counts(m);
    for (size_t row_idx : row_id) 
        row_counts[row_idx]++;
    
    row_ptr = std::vector<size_t>(m + 1);
    row_ptr[0] = 0;
    for (size_t i = 0; i < m; ++i) 
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
    
    size_t nnz = elem_csc.size();

    row_ptr  = std::vector<size_t>(nnz);
    elem_csr = std::vector<double>(nnz);
    
    std::vector<size_t> next_pos = row_ptr;
    
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            size_t i = row_id[k];
            size_t pos = next_pos[i];
            
            elem_csr[pos] = elem_csc[k];
            col_id[pos] = j;
            
            next_pos[i]++;
        }
    }
}


Matrix Matrix::operator()(const IndexVector& indexes) const
{
    Matrix new_Matrix(m, indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        size_t k = indexes[i];
        for (size_t j = col_ptr[k]; j < col_ptr[k + 1]; j++)
        {
            new_Matrix.elem_csc.push_back(elem_csc[j]);
            new_Matrix.col_ptr.push_back(row_id[j]);
            new_Matrix.row_id.push_back(i);
        }
    }
    new_Matrix.row_id.push_back(indexes.size());
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

