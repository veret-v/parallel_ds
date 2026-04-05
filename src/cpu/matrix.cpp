#include "matrix.hpp"


Matrix::Matrix(const int m, const int n)
{
    this -> m = m;
    this -> n = n;
    
    elem_csr = std::vector<double>(0);
    row_ptr = std::vector<int>(0);
    col_id = std::vector<int>(0);
}


Matrix::Matrix(CoinPackedMatrix& matrix)
{
    copyCoinPackedMatrix(matrix);
}


Matrix::Matrix(
    std::vector<double> elem_csr, 
    std::vector<MKL_INT> row_ptr, 
    std::vector<MKL_INT> col_id,
    const int m, const int n
)
{
    this -> m = m;
    this -> n = n;

    this->elem_csr = elem_csr;
    this->row_ptr = row_ptr;
    this->col_id = col_id;
}


Matrix& Matrix::operator=(CoinPackedMatrix& matrix)
{
    copyCoinPackedMatrix(matrix);
    return *this;
}


void Matrix::cleanLUinfo()
{
    MKL_INT phase = -1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &n, nullptr, nullptr, nullptr,
            nullptr, &nrhs, iparm, &msglvl,
            nullptr, nullptr, &error);
    if (error != 0) {
        std::cerr << "Ошибка при освобождении памяти: " << error << std::endl;
    }
}


Matrix::~Matrix()
{
    if (factorized)
        cleanLUinfo();
}


void Matrix::copyCoinPackedMatrix(CoinPackedMatrix& matrix)
{
    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();

    int non_zero_size;
    int ptr_size;

    non_zero_size = matrix.getNumElements();
    ptr_size      = matrix.getMajorDim() + 1;

    elem_csr = std::vector<double>(matrix.getElements(), matrix.getElements() + non_zero_size);
    row_ptr  = std::vector<int>(matrix.getVectorStarts(), matrix.getVectorStarts() + ptr_size);
    col_id   = std::vector<int>(matrix.getIndices(), matrix.getIndices() + non_zero_size);
}


double Matrix::operator()(const int i, const int j) const
{
    int left = row_ptr[i];
    int right = row_ptr[i + 1] - 1;
    while (left <= right) {
        int mid =  left + (right - left) / 2;
        if (col_id[mid] == j) 
            return elem_csr[mid];
        else if (col_id[mid] < j) 
            left = mid + 1;
        else 
            right = mid - 1;
    }
    return 0.0; 
}


int Matrix::getElemIdx(const int i, const int j) const
{
    int left = row_ptr[i];
    int right = row_ptr[i + 1] - 1;
    
    while (left <= right) {
        int mid =  left + (right - left) / 2;
        if (col_id[mid] == j) 
            return mid;
        else if (col_id[mid] < j) 
            left = mid + 1;
        else 
            right = mid - 1;
    }
    return -1; 
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

    if(factorized)
        cleanLUinfo();

    this -> m = matrix.m;
    this -> n = matrix.n;
    
    elem_csr = matrix.elem_csr;;
    row_ptr = matrix.row_ptr;
    col_id = matrix.col_id;

    return *this;
}


void Matrix::dotUpdate(
    const ValuesVector& vec1, 
    const ValuesVector& vec2, 
    ValuesVector& sol, 
    const double alpha, 
    const double beta,
    const IndexVector& cols_idx,
    const SpmvOptions& method,
    const bool set
)
{
    ValuesVector buff(n);
    
    switch (method) 
    {
        
        case SpmvOptions::UPDATE_T:
            for (int i = 0; i < m; i++) {
                for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
                {
                    int k = col_id[j];
                    buff[k] += elem_csr[j] * vec1[i] * alpha;
                }
            }

            for (int i = 0; i < cols_idx.size(); i++) {
                int i_it = cols_idx[i];
                sol[set ? i_it : i] = buff[i_it] + vec2[i_it] * beta;
            }
            break;

      
        case SpmvOptions::UPDATE:
           for (size_t i = 0; i < cols_idx.size(); i++)
                buff[cols_idx[i]] = vec1[set ? cols_idx[i] : i];
            
            for (int i = 0; i < m; i++) {
                sol[i] = vec2[i] * beta;
                for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
                    sol[i] += elem_csr[j] * buff[col_id[j]] * alpha;
            }
            break;
            
        case SpmvOptions::UNKNOWN:
            throw "Unknown spmv method";
            break;
    }
}


void Matrix::stackColUnitMatrix() 
{
    std::vector<double> new_elem_csr(elem_csr.size() + m);
    std::vector<int> new_col_id(col_id.size() + m);
   
    int ptr_curr = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            new_elem_csr[ptr_curr] = elem_csr[j];
            new_col_id[ptr_curr] = col_id[j];
            ptr_curr++;
        }
        new_elem_csr[ptr_curr] = 1;
        new_col_id[ptr_curr] = n + i;
        row_ptr[i] += i;
        ptr_curr++;
    }
    row_ptr[m] += m;
    col_id = new_col_id;
    elem_csr = new_elem_csr;
    n = n + m;
}


Matrix Matrix::operator()(const IndexVector& indexes) const
{
    std::vector<double> new_elem_csr;
    std::vector<MKL_INT> new_row_ptr;
    std::vector<MKL_INT> new_col_id;

    int row_start_ptr = 0;
    new_row_ptr.push_back(row_start_ptr);
    
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < indexes.size(); j++)
        {
            double val = this->operator()(i, indexes[j]);
            if (fabs(val) > EPS_ZERO)
            {
                new_elem_csr.push_back(val);
                new_col_id.push_back(j);
                row_start_ptr += 1;
            }
        }
        new_row_ptr.push_back(row_start_ptr);
    }
    
    Matrix new_Matrix(
        new_elem_csr, 
        new_row_ptr, 
        new_col_id, 
        m, indexes.size()
    );
    return new_Matrix;
}


void Matrix::resetData(const Matrix& matrix, const IndexVector& indexes)
{
    std::vector<double> new_elem_csr;
    std::vector<MKL_INT> new_row_ptr;
    std::vector<MKL_INT> new_col_id;

    int row_start_ptr = 0;
    new_row_ptr.push_back(row_start_ptr);

    n = indexes.size();
    m = matrix.m;
    
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < indexes.size(); j++)
        {
            double val = matrix(i, indexes[j]);
            if (fabs(val) > EPS_ZERO)
            {
                new_elem_csr.push_back(val);
                new_col_id.push_back(j);
                row_start_ptr += 1;
            }
        }
        new_row_ptr.push_back(row_start_ptr);
    }
    
    elem_csr = new_elem_csr;
    col_id = new_col_id;
    row_ptr = new_row_ptr;
}



std::set<int> Matrix::deleteCols(std::set<int> cols)
{
    std::vector<double> new_elem_csr;
    std::vector<MKL_INT> new_row_ptr;
    std::vector<MKL_INT> new_col_id;
    
    std::set<int> excpet_rows;

    int row_start_ptr = 0;
    new_row_ptr.push_back(row_start_ptr);
    

    for (int i = 0; i < m; i++)
    {
        bool row_exists = false;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            if (cols.find(col_id[j]) == cols.end())
            {
                int dist = 0;
                auto it = cols.lower_bound(col_id[j]);
                if (it != cols.begin()) dist = std::distance(cols.begin(), it);

                new_elem_csr.push_back(elem_csr[j]);
                new_col_id.push_back(col_id[j] - dist);
                row_start_ptr += 1;
                row_exists = true;
            }
        }
        if (row_exists) 
            new_row_ptr.push_back(row_start_ptr);
        else   
            excpet_rows.insert(i);
    }

    elem_csr = new_elem_csr;
    col_id = new_col_id;
    row_ptr = new_row_ptr;
    n -= cols.size();
    m -= excpet_rows.size();

    return excpet_rows;
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


void Matrix::LUdecompose()
{
    if (factorized)
        cleanLUinfo();
        
    factorized = true;

    mtype   = 11;       
    nrhs    = 1;       
    maxfct  = 1;        
    mnum    = 1;       
    msglvl  = 0;      
    error   = 0;      

    pardisoinit(pt, &mtype, iparm);

    iparm[0]  = 1;    // нет итеративного уточнения
    iparm[7]  = 2;    // макс. число итераций (не используется)
    iparm[9]  = 13;   // порог для перестановок (по умолч.)
    iparm[10] = 1;    // включить масштабирование
    iparm[12] = 1;    // улучшенная точность (и для транспонирования)
    iparm[17] = -1;   // вывод числа ненулей в факторах
    iparm[34] = 1;    // индексация с 1 (как в наших массивах)

    MKL_INT phase = 12;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &n, elem_csr.data(), row_ptr.data(), col_id.data(),
            nullptr, &nrhs, iparm, &msglvl,
            nullptr, nullptr, &error);

    if (error != 0) {
        std::cerr << "Ошибка при факторизации: " << error << std::endl;
    }
}

void Matrix::solve(
    ValuesVector& rhs, 
    ValuesVector& sol,
    bool transpose
)
{
    iparm[11] = transpose;   // решаем A*x = b
    MKL_INT phase = 33;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &n, elem_csr.data(), row_ptr.data(), col_id.data(),
            nullptr, &nrhs, iparm, &msglvl,
            rhs.data.data(), sol.data.data(), &error);
    if (error != 0) {
        std::cerr << "Ошибка при решении A*x=b: " << error << std::endl;
    }
}


int Matrix::calcNonzeroInColumn(const int& p) const
{
    int count = 0;
    for (int i = 0; i < m; i++)
        count += fabs(operator()(i, p)) < EPS_ZERO ? 0 : 1;
    return count;
}
