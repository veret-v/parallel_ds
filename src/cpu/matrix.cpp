#include "matrix.hpp"


Matrix::Matrix(const int m, const int n)
{
    this -> m = m;
    this -> n = n;
    
    elem_csc = std::vector<double>(0);
    col_ptr = std::vector<int>(0);
    row_id = std::vector<int>(0);
}


Matrix::Matrix(CoinPackedMatrix& matrix)
{
    copyCoinPackedMatrix(matrix);
}


Matrix& Matrix::operator=(CoinPackedMatrix& matrix)
{
    copyCoinPackedMatrix(matrix);
    return *this;
}


void Matrix::copyCoinPackedMatrix(CoinPackedMatrix& matrix)
{
    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();

    int non_zero_size;
    int ptr_size;

    if (!matrix.isColOrdered()) matrix.reverseOrdering();
    
    non_zero_size = matrix.getNumElements();
    ptr_size = matrix.getMajorDim() + 1;

    elem_csc = std::vector<double>(matrix.getElements(), matrix.getElements() + non_zero_size);
    col_ptr = std::vector<int>(matrix.getIndices(), matrix.getIndices() + ptr_size);
    row_id = std::vector<int>(matrix.getVectorStarts(), matrix.getVectorStarts() + non_zero_size);
}


double Matrix::operator()(const int i, const int j) const
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


double& Matrix::operator()(const int i, const int j)
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


ValuesVector Matrix::operator()(const int p) const
{
    ValuesVector new_vec(m);
    for (int i = 0; i < m; i++) 
        new_vec[i] = this->operator()(i, p);
    return new_vec;

}


Matrix& Matrix::operator=(const Matrix &matrix)
{
    if (this == &matrix) 
        return *this;

    this -> m = matrix.m;
    this -> n = matrix.n;
    
    elem_csc = matrix.elem_csc;;
    col_ptr = matrix.col_ptr;
    row_id = matrix.row_id;

    return *this;
}


void Matrix::dotUpdate(
    const ValuesVector& vec1, 
    const IndexVector& vec1_idx,
    const ValuesVector& vec2, 
    ValuesVector& sol, 
    const double alpha, 
    const double beta,
    const IndexVector& set_idx,
    const SpmvOptions& method
)
{
    switch (method) 
    {
        case SpmvOptions::FULL_UPDATE:
            for (int i = 0; i < n; i++) {
                if (vec2[i] < 1e-12) continue;
                sol[i] = sol[i] * beta;
                for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
                {
                    int k = row_id[j];
                    sol[k] += elem_csc[j] * vec1[GET_ID(vec1_idx, i)] * alpha;
                }
            }
            break;
        
        case SpmvOptions::SET_UPDATE:
            for (int i_it = 0; i_it < set_idx.size(); i_it++) {
                int i = set_idx[i_it];
                if (vec2[i] < 1e-12) continue;
                sol[i] = vec2[i] * beta;
                for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
                {
                    int k = row_id[j];
                    sol[k] += elem_csc[j] * vec1[GET_ID(vec1_idx, i)] * alpha;
                }
            }
            break;

        case SpmvOptions::FULL_UPDATE_T:
            for (int i = 0; i < m; i++) {
                sol[i] = sol[i] * beta;
                for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
                    sol[i] += elem_csc[j] * vec1[GET_ID(vec1_idx, row_id[j])] * alpha;
            }
            break;

        case SpmvOptions::SET_UPDATE_T:
            for (int i_it = 0; i_it < set_idx.size(); i_it++) {
                int i = set_idx[i_it];
                sol[i] = vec2[i] * beta;
                for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
                    sol[i] += elem_csc[j] * vec1[GET_ID(vec1_idx, row_id[j])] * alpha;
            }
            break;
            
        case SpmvOptions::UNKNOWN:
            throw "Unknown spmv method";
            break;
    }
}

void Matrix::stackColUnitMatrix() 
{
    for (int i = 0; i < m; i++)
    {
        elem_csc.push_back(1);
        col_ptr.push_back(col_ptr[i] + m);
        row_id.push_back(i);
    }
    n = n + m;
}


void Matrix::dotEtaMatrix(const EtaMatrix& etaMatrix)
{
    int p = std::get<1>(etaMatrix);
    ValuesVector val = std::get<0>(etaMatrix);

    for (int j = 0; j < m && j != p; j++)
    { 
        std::vector<int> buff_id;
        auto start_id = col_id.begin();
        std::merge(
            start_id + row_ptr[p], start_id + row_ptr[p + 1],
            start_id + row_ptr[j], start_id + row_ptr[j + 1],
            buff_id.begin()
        );
        std::vector<double> buff_val(buff_id.size());
        
        int j_it  = row_ptr[j];
        int p_it  = row_ptr[p];
        for (int i = 0; i < buff_id.size(); i++)
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
    for (int i = row_ptr[p]; i < row_ptr[p + 1]; i++)
        elem_csr[i] = elem_csr[i] * val[col_id[i]];
}


void Matrix::genCSRorder()
{
    std::vector<int> row_counts(m);
    for (int row_idx : row_id) 
        row_counts[row_idx]++;
    
    row_ptr = std::vector<int>(m + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < m; ++i) 
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
    
    int nnz = elem_csc.size();

    row_ptr  = std::vector<int>(nnz);
    elem_csr = std::vector<double>(nnz);
    
    std::vector<int> next_pos = row_ptr;
    
    for (int j = 0; j < n; ++j) {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            int i = row_id[k];
            int pos = next_pos[i];
            
            elem_csr[pos] = elem_csc[k];
            col_id[pos] = j;
            
            next_pos[i]++;
        }
    }
}


Matrix Matrix::operator()(const IndexVector& indexes) const
{
    Matrix new_Matrix(m, indexes.size());
    for (int i = 0; i < indexes.size(); i++) {
        int k = indexes[i];
        for (int j = col_ptr[k]; j < col_ptr[k + 1]; j++)
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
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            std::cout << this->operator()(i, j) << " ";
        std::cout << std::endl;
    }
}

