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

    csr_built_ = true;
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


void Matrix::genSparseReprs()
{
    const MKL_INT nnz = static_cast<MKL_INT>(elem_csr.size());
    const MKL_INT m_  = static_cast<MKL_INT>(m);
    const MKL_INT n_  = static_cast<MKL_INT>(n);

    if (!csr_built_ && nnz > 0) 
    {
        std::vector<MKL_INT> rows_start(m_ + 1);
        std::vector<MKL_INT> rows_end(m_);
        std::vector<MKL_INT> col_idx(nnz);

        for (MKL_INT i = 0; i <= m_; ++i) rows_start[i] = static_cast<MKL_INT>(row_ptr[i]);
        for (MKL_INT i = 0; i < m_;  ++i) rows_end[i]   = static_cast<MKL_INT>(row_ptr[i+1]);
        for (MKL_INT j = 0; j < nnz; ++j) col_idx[j]    = static_cast<MKL_INT>(col_id[j]);

        sparse_status_t status = mkl_sparse_d_create_csr(
            &csr_handle_,
            SPARSE_INDEX_BASE_ZERO,
            m_, n_,
            rows_start.data(),
            rows_end.data(),
            col_idx.data(),
            elem_csr.data()  
        );
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create CSR handle");
        }

        csr_descr_.type = SPARSE_MATRIX_TYPE_GENERAL;
        csr_descr_.mode = SPARSE_FILL_MODE_FULL;
        csr_descr_.diag = SPARSE_DIAG_NON_UNIT;

        mkl_sparse_optimize(csr_handle_);
        csr_built_ = true;
    } else if (nnz == 0) {
        csr_handle_ = nullptr;
        csr_built_ = true;  // помечаем как созданный (пустой)
    }

    if (!csc_built_ && nnz > 0) {
        csc_col_ptr_.assign(n_ + 1, 0);
        for (MKL_INT i = 0; i < m_; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
                MKL_INT col = static_cast<MKL_INT>(col_id[j]);
                ++csc_col_ptr_[col + 1];
            }
        }
        for (MKL_INT j = 1; j <= n_; ++j) {
            csc_col_ptr_[j] += csc_col_ptr_[j-1];
        }

        std::vector<MKL_INT> current = csc_col_ptr_;
        csc_row_idx_.resize(nnz);
        csc_values_.resize(nnz);

        for (MKL_INT i = 0; i < m_; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
                MKL_INT col = static_cast<MKL_INT>(col_id[j]);
                MKL_INT pos = current[col]++;
                csc_row_idx_[pos] = i;
                csc_values_[pos] = elem_csr[j];
            }
        }

        sparse_status_t status = mkl_sparse_d_create_csc(
            &csc_handle_,
            SPARSE_INDEX_BASE_ZERO,
            m_, n_,
            csc_col_ptr_.data(),
            csc_col_ptr_.data() + 1,
            csc_row_idx_.data(),
            csc_values_.data()
        );
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create CSC handle");
        }

        csc_descr_.type = SPARSE_MATRIX_TYPE_GENERAL;
        csc_descr_.mode = SPARSE_FILL_MODE_FULL;
        csc_descr_.diag = SPARSE_DIAG_NON_UNIT;

        mkl_sparse_optimize(csc_handle_);
        csc_built_ = true;
    } else if (nnz == 0) {
        csc_handle_ = nullptr;
        csc_built_ = true;
    }
}


Matrix::~Matrix()
{
    if (factorized)
        cleanLUinfo();
}


void Matrix::copyCoinPackedMatrix(CoinPackedMatrix& matrix)
{
    if (matrix.isColOrdered()) {
        matrix.reverseOrdering();   
    }

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
    if (csr_built_)
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

    if (csc_built_)
    {
        int left = csc_col_ptr_[j];
        int right = csc_col_ptr_[j + 1] - 1;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (csc_row_idx_[mid] == i)
                return csc_values_[mid];
            else if (csc_row_idx_[mid] < i)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return 0.0;
    }
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
    ValuesVector& vec1, 
    ValuesVector& vec2, 
    ValuesVector& sol, 
    const double alpha, const double beta,
    const IndexVector& cols_idx,
    const SpmvOptions& method,
    const bool set)
{
    ValuesVector buff(n);   // инициализирован нулями

    switch (method) 
    {
        case SpmvOptions::UPDATE_T:
        {
            for (auto col_idx : cols_idx) {
                for (int i = csc_col_ptr_[col_idx]; i < csc_col_ptr_[col_idx + 1]; i++) {
                    int row_idx = csc_row_idx_[i];
                    buff[col_idx] += csc_values_[i] * vec1[row_idx] * alpha;
                }
                
            }
            
            for (size_t i = 0; i < cols_idx.size(); ++i) {
                int idx = cols_idx[i];
                sol[set ? idx : i] = buff[idx] + vec2[idx] * beta;
            }
            break;
        }


        case SpmvOptions::UPDATE_T_SP_COL_WISE:
        {
            #pragma omp parallel for schedule(dynamic, 8)
            for (auto col_idx : cols_idx) {
                for (int i = csc_col_ptr_[col_idx]; i < csc_col_ptr_[col_idx + 1]; i++) {
                    int row_idx = csc_row_idx_[i];
                    buff[col_idx] += csc_values_[i] * vec1[row_idx];
                } 
            }
            
            #pragma omp parallel for
            for (size_t i = 0; i < cols_idx.size(); ++i) {
                int idx = cols_idx[i];
                sol[i] = buff[idx];
            }
            break;
        }

        case SpmvOptions::UPDATE_T_SP_ROW_WISE:
        {
            const std::vector<int>& sp_idxs = vec1.getSpIdx();
          
            #pragma omp parallel for schedule(dynamic, 8)
            for (auto row_idx : sp_idxs) {
                for (int j = row_ptr[row_idx]; j < row_ptr[row_idx + 1]; j++)
                {
                    int col_idx = col_id[j];
                    #pragma omp atomic
                    buff[col_idx] += elem_csr[j] * vec1[row_idx];
                }
            }

            #pragma omp parallel for
            for (size_t i = 0; i < cols_idx.size(); ++i) {
                int idx = cols_idx[i];
                sol[i] = buff[idx];
            }
            break;
        }

        case SpmvOptions::UPDATE:
        {
            for (size_t i = 0; i < cols_idx.size(); i++)
                buff[cols_idx[i]] = vec1[set ? cols_idx[i] : i];
            
            for (int i = 0; i < m; i++) {
                sol[i] = vec2[i] * beta;
                for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
                    sol[i] += elem_csr[j] * buff[col_id[j]] * alpha;
            }
            break;
        }

        default:
            throw std::runtime_error("Unknown SpmvOptions");
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
    std::vector<double> new_elem_csc;
    std::vector<MKL_INT> new_col_ptr;
    std::vector<MKL_INT> new_row_id;

    new_elem_csc.reserve(matrix.csc_values_.size());
    new_col_ptr.reserve(matrix.csc_col_ptr_.size());
    new_row_id.reserve(matrix.csc_row_idx_.size());

    int col_start_ptr = 0;
    new_col_ptr.push_back(col_start_ptr);

    n = indexes.size();
    m = matrix.m;
    
    for (auto i : indexes)
    {
        size_t start = matrix.csc_col_ptr_[i];
        size_t end   = matrix.csc_col_ptr_[i + 1];

        new_elem_csc.insert(new_elem_csc.end(), matrix.csc_values_.begin() +start, matrix.csc_values_.begin() + end);
        new_row_id.insert(new_row_id.end(), matrix.csc_row_idx_.begin() + start, matrix.csc_row_idx_.begin() + end);
        col_start_ptr = new_row_id.size();
        new_col_ptr.push_back(col_start_ptr);
    }
    
    csc_col_ptr_ = std::move(new_col_ptr);
    csc_row_idx_ = std::move(new_row_id);
    csc_values_ = std::move(new_elem_csc);

    csc_built_ = true;
    csr_built_ = false;
}



void Matrix::deleteCols(std::set<int> cols)
{
    std::vector<double> new_elem_csr;
    std::vector<MKL_INT> new_row_ptr;
    std::vector<MKL_INT> new_col_id;

    new_elem_csr.reserve(elem_csr.size());
    new_col_id.reserve(elem_csr.size());
    new_row_ptr.reserve(col_id.size());
    
    int row_start_ptr = 0;
    new_row_ptr.push_back(row_start_ptr);
    

    for (int i = 0; i < m; i++)
    {
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
            }
        }
        new_row_ptr.push_back(row_start_ptr);
    }

    elem_csr = new_elem_csr;
    col_id = new_col_id;
    row_ptr = new_row_ptr;
    n -= cols.size();
}


void Matrix::deleteRows(std::set<int> rows)
{
    std::vector<double> new_elem_csr;
    std::vector<MKL_INT> new_row_ptr;
    std::vector<MKL_INT> new_col_id;

    new_elem_csr.reserve(elem_csr.size());
    new_col_id.reserve(elem_csr.size());
    new_row_ptr.reserve(col_id.size());
    
    int row_start_ptr = 0;
    new_row_ptr.push_back(row_start_ptr);
    

    for (int i = 0; i < m; i++)
    {
        if (!rows.count(i))
        {
            new_elem_csr.insert(new_elem_csr.end(), elem_csr.begin() + row_ptr[i], elem_csr.begin() + row_ptr[i + 1]);
            new_col_id.insert(new_col_id.end(), col_id.begin() + row_ptr[i], col_id.begin() + row_ptr[i + 1]);
            row_start_ptr = new_col_id.size();
            new_row_ptr.push_back(row_start_ptr);
        }
    }

    elem_csr = new_elem_csr;
    col_id = new_col_id;
    row_ptr = new_row_ptr;
    m -= rows.size();
}


void Matrix::show() const
{
    std::cout << "Matrix" << "(" << m << "," << n << "):" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
        {
            std::cout << this->operator()(i, j) << " ";
        }
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
    iparm[36] = 1;

    MKL_INT phase = 12;

    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &n, csc_values_.data(), csc_col_ptr_.data(), csc_row_idx_.data(),
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
    iparm[11] = !transpose;   // решаем A*x = b
    MKL_INT phase = 33;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &n, csc_values_.data(), csc_col_ptr_.data(), csc_row_idx_.data(),
            nullptr, &nrhs, iparm, &msglvl,
            rhs.data.data(), sol.data.data(), &error);
    if (error != 0) {
        std::cerr << "Ошибка при решении A*x=b: " << error << std::endl;
    }
}


double Matrix::dotCol(
    ValuesVector& vec1, 
    int col_id
)
{
    double res = 0;
    for (int i = csc_col_ptr_[col_id]; i < csc_col_ptr_[col_id + 1]; i++)
    {
        int row_id = csc_row_idx_[i];
        res += csc_values_[i] * vec1[row_id];
    }
    return res;
}


int Matrix::calcNonzeroInColumn(const int& p) const
{
    int count = 0;
    for (int i = 0; i < m; i++)
        count += fabs(operator()(i, p)) < EPS_ZERO ? 0 : 1;
    return count;
}


void Matrix::addSparseCol(ValuesVector& res, const IndexVector& indexes, const ValuesVector& multiplier, double extra_mult) const
{
    for (auto id : indexes)
    {
        for (int i = csc_col_ptr_[id]; i < csc_col_ptr_[id + 1]; i++)
        {
            int row_id = csc_row_idx_[i];
            res[row_id] += csc_values_[i] * multiplier[id] * extra_mult;
        }
    }
}


void Matrix::addSparseColParallel(ValuesVector& res, const IndexVector& indexes, const ValuesVector& multiplier, double extra_mult) const
{
    for (auto id : indexes)
    {
        #pragma omp parallel for
        for (int i = csc_col_ptr_[id]; i < csc_col_ptr_[id + 1]; i++)
        {
            int row_id = csc_row_idx_[i];
            res[row_id] += csc_values_[i] * multiplier[id] * extra_mult;
        }
    }
}


void Matrix::addSparseCol(ValuesVector& res, const int& id, double extra_mult) const
{
    #pragma omp parallel for
    for (int i = csc_col_ptr_[id]; i < csc_col_ptr_[id + 1]; i++)
    {
        int row_id = csc_row_idx_[i];
        res[row_id] += csc_values_[i] * extra_mult;
    }
}
