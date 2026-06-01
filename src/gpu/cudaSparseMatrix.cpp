#include "cudaSparseMatrix.hpp"


void CudaSparseMatrix::allocateMemory(int non_zero_size, int m, int n)
{
    memory_init = true; 

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_values_csc, non_zero_size*sizeof(double)),
        "cudaMalloc for device_values_csc"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_ptr_csc, (n + 1) * sizeof(double)),
        "cudaMalloc for device_ptr_csc"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_id_csc, non_zero_size*sizeof(double)),
        "cudaMalloc for device_id_csc"
    );

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_values_csr, non_zero_size * sizeof(double)),
        "cudaMalloc for scr_val"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_ptr_csr, (m + 1) * sizeof(int)),
        "cudaMalloc for scr_ptr"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_id_csr, non_zero_size * sizeof(int)),
        "cudaMalloc for scr_id"
    );
}


void CudaSparseMatrix::freeMemory()
{
    memory_init = false; 
    CSС_exist   = false;

    CUDA_CALL_AND_CHECK(cudaFree(device_values_csc),
                        "cudaFree for device_values_csc");
    CUDA_CALL_AND_CHECK(cudaFree(device_ptr_csc),
                        "cudaFree for device_ptr_csc")
    CUDA_CALL_AND_CHECK(cudaFree(device_id_csc),
                        "cudaFree for device_id_csc")
    
    CUDA_CALL_AND_CHECK(cudaFree(device_values_csr),
                        "cudaFree for device_values_csr");
    CUDA_CALL_AND_CHECK(cudaFree(device_ptr_csr),
                        "cudaFree for device_ptr_csr")
    CUDA_CALL_AND_CHECK(cudaFree(device_id_csr),
                        "cudaFree for device_id_csr")

    device_values_csc = nullptr;
    device_ptr_csc    = nullptr;
    device_id_csc     = nullptr;

    device_values_csr = nullptr;
    device_ptr_csr    = nullptr;
    device_id_csr     = nullptr;
}


void CudaSparseMatrix::copy(
    double* device_values, int* device_ptr, int* device_id, 
    const double* values_new, const int* ptr_new, const int* id_new,
    cudaMemcpyKind copy_type
)
{
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, 
            values_new, 
            non_zero*sizeof(double), 
            copy_type),
        "cudaMemcpy for device_values_csc"
    );    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr, 
            ptr_new, 
            major_dim*sizeof(int), 
            copy_type),
        "cudaMemcpy for device_ptr_csc"
    );  
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id, 
            id_new, 
            non_zero*sizeof(int), 
            copy_type),
        "cudaMemcpy for device_id_csc"
    ); 
}


void CudaSparseMatrix::destroyDesr()
{
    descr_exist = false;

    CUSP_CALL_AND_CHECK(
        cusparseDestroySpMat(mat_cusp_descr), 
        "cusparseDestroySpMat"
    );

    CUDSS_CALL_AND_CHECK(
        cudssMatrixDestroy(mat_cudss_descr),
        "cudssMatrixDestroy"
    );

     CUDSS_CALL_AND_CHECK(
        cudssMatrixDestroy(mat_cudss_descr_T),
        "cudssMatrixDestroy"
    );
}


void CudaSparseMatrix::createDescr()
{
    if (descr_exist)
    {
        std::cout << "ERROR: descriptors exists" << std::endl;
        exit(1);
    }

    if (!CSС_exist)
    {
        std::cout << "ERROR: can't create descr without CSC" << std::endl;
        exit(1);
    }
    
    descr_exist = true;

    CUSP_CALL_AND_CHECK(
        cusparseCreateCsr(
            &mat_cusp_descr,
            m, n, non_zero,
            device_ptr_csr,
            device_id_csr,
            device_values_csr,
            CUSPARSE_INDEX_32I,  
            CUSPARSE_INDEX_32I,  
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F),
        "cusparseCreateCsr"
    );  

    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateCsr(
            &mat_cudss_descr,
            m, n, non_zero,
            device_ptr_csr,
            NULL,
            device_id_csr,
            device_values_csr,
            CUDA_R_32I, 
            CUDA_R_64F,
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO
        ), 
        "cudssMatrixCreateCsr"
    );

    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateCsr(
            &mat_cudss_descr_T,
            n, m, non_zero,
            device_ptr_csc,
            NULL,
            device_id_csc,
            device_values_csc,
            CUDA_R_32I, 
            CUDA_R_64F,
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO
        ), 
        "cudssMatrixCreateCsr"
    );
}


CudaSparseMatrix::~CudaSparseMatrix()
{
    if (memory_init) 
        freeMemory();

    if (descr_exist) 
        destroyDesr();
}


CudaSparseMatrix::CudaSparseMatrix(const int m, const int n)
{
    this -> m = m;
    this -> n = n;
}


CudaSparseMatrix& CudaSparseMatrix::operator=(Matrix& matrix)
{ 
    if (memory_init) 
        freeMemory();

    if (descr_exist) 
        destroyDesr();

    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();

    major_dim = matrix.getMajorSize() + 1;
    non_zero  = matrix.getNonZeroSize();

    allocateMemory(non_zero, m, n);
    copy(
        device_values_csr, device_ptr_csr, device_id_csr, 
        matrix.getNonZeroElems(), matrix.getRowPtrs(), matrix.getColIds(), 
        cudaMemcpyHostToDevice
    );

    return *this;
}


CudaSparseMatrix::CudaSparseMatrix(Matrix& matrix)
{
    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();

    major_dim = matrix.getMajorSize() + 1;
    non_zero  = matrix.getNonZeroSize();

    allocateMemory(non_zero, m, n);
    copy(
        device_values_csr, device_ptr_csr, device_id_csr, 
        matrix.getNonZeroElems(), matrix.getRowPtrs(), matrix.getColIds(), 
        cudaMemcpyHostToDevice
    );
}


CudaSparseMatrix::CudaSparseMatrix(const CoinPackedMatrix& matrix)
{
    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();

    major_dim = n;
    non_zero  = matrix.getNumElements();

    allocateMemory(non_zero, m, n);
    copy(
        device_values_csr, device_ptr_csr, device_id_csr, 
        matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
        cudaMemcpyHostToDevice
    );
}


CudaSparseMatrix::CudaSparseMatrix(const CudaSparseMatrix& matrix)
{
    this -> m = matrix.m;
    this -> n = matrix.n;

    major_dim = matrix.major_dim;
    non_zero  = matrix.non_zero;

    allocateMemory(non_zero, m, n);

    copy(
        device_values_csc, device_ptr_csc, device_id_csc,
        matrix.device_values_csc, matrix.device_ptr_csc, matrix.device_id_csc, 
        cudaMemcpyDeviceToDevice
    );
}


// CudaSparseMatrix& CudaSparseMatrix::operator=(const CudaSparseMatrix& matrix)
// {
//     if (this == &matrix) 
//         return *this;

//     if (memory_init) 
//         freeMemory();

//     if (descr_exist) 
//         destroyDesr();

//     this -> m = matrix.m;
//     this -> n = matrix.n;

//     CSС_exist = true;

//     major_dim = matrix.major_dim;
//     non_zero  = matrix.non_zero;

//     allocateMemory(non_zero, m, n);

//     copy(
//         device_values_csc, device_ptr_csc, device_id_csc,
//         matrix.device_values_csc, matrix.device_ptr_csc, matrix.device_id_csc, 
//         cudaMemcpyDeviceToDevice
//     );

//     copy(
//         device_values_csr, device_ptr_csr, device_id_csr,
//         matrix.device_values_csr, matrix.device_ptr_csr, matrix.device_id_csr, 
//         cudaMemcpyDeviceToDevice
//     );
    
//     createDescr();

//     return *this;
// }


void CudaSparseMatrix::dotUpdate(
    cusparseHandle_t& handle, 
    const CudaDataDenseVector& vec1, 
    const CudaDataDenseVector& vec2, 
    CudaDataDenseVector& sol,
    const double alpha,
    const double beta,
    const CudaIndexVector& col_idx,
    const SpmvOptions method,
    const bool set
) 
{
    if (!CSС_exist)
    {
        std::cout << "WARNING: csc doesnt exist" << std::endl;
        return;
    }
    
    switch (method) 
    {
        case SpmvOptions::UPDATE:
            CUSP_CALL_AND_CHECK(
                spmvUpdateInc(
                    handle,
                    non_zero, m, n,
                    col_idx.getSize(),
                    sol.device_values,
                    vec1.device_values,
                    vec2.device_values,
                    device_values_csr,
                    device_id_csr,
                    device_ptr_csr,
                    col_idx.device_values,
                    alpha,
                    beta,
                    set,
                    false
                ), "cusparseSpMV"
            );
            break;

        case SpmvOptions::UPDATE_T:
            CUSP_CALL_AND_CHECK(
                spmvUpdateInc(
                    handle,
                    non_zero, m, n,
                    col_idx.getSize(),
                    sol.device_values,
                    vec1.device_values,
                    vec2.device_values,
                    device_values_csc,
                    device_id_csc,
                    device_ptr_csc,
                    col_idx.device_values,
                    alpha,
                    beta,
                    set,
                    true
                ), "cusparseSpMV"
            );
            break;

        case SpmvOptions::UNKNOWN:
            throw "Unknown spmv method";
            break;
    }
}


void CudaSparseMatrix::getColumn(cusparseHandle_t& handle, const int p, CudaDataDenseVector& rhs) const
{
    if (m != rhs.getSize())
    {
        throw "Incorrect size";
        exit(1);
    }

    CUDA_CALL_AND_CHECK(
        cudaMemset(rhs.device_values, 0, rhs.getSize()*sizeof(double)),
        "cudaMemset"
    );

    getColFromSp(handle, m, n, p, rhs.device_values, device_values_csc, device_id_csc, device_ptr_csc);
}


void CudaSparseMatrix::genCsc(cusparseHandle_t& handle)
{
    if (CSС_exist)
    {
        std::cout << "WARNING: csc exist" << std::endl;
        return;
    }
    
    size_t bufferSize = 0;
    void* d_buffer = nullptr;

    CUSP_CALL_AND_CHECK(
        cusparseCsr2cscEx2_bufferSize(
            handle,
            m, n, non_zero,
            device_values_csr, device_ptr_csr, device_id_csr,
            device_values_csc, device_ptr_csc, device_id_csc,
            CUDA_R_64F,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            &bufferSize
        ), "cusparseCsr2cscEx2_bufferSize"
    );
    
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&d_buffer, bufferSize),
        "cudaMalloc for d_buffer"
    );
    
    CUSP_CALL_AND_CHECK(
        cusparseCsr2cscEx2(
            handle,
            m, n, non_zero,
            device_values_csr, device_ptr_csr, device_id_csr,
            device_values_csc, device_ptr_csc, device_id_csc,
            CUDA_R_64F,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            d_buffer
        ), "cusparseCsr2cscEx2_bufferSize"
    );

    CUDA_CALL_AND_CHECK(
        cudaFree(d_buffer),
        "cudaFree for d_buffer"
    );

    CSС_exist = true;
}


void CudaSparseMatrix::genCsr(cusparseHandle_t& handle)
{
    size_t bufferSize = 0;
    void* d_buffer = nullptr;

    CUSP_CALL_AND_CHECK(
        cusparseCsr2cscEx2_bufferSize(
            handle,
            n, m, non_zero,
            device_values_csc, device_ptr_csc, device_id_csc,
            device_values_csr, device_ptr_csr, device_id_csr, 
            CUDA_R_64F,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            &bufferSize
        ), "cusparseCsr2cscEx2_bufferSize"
    );
    
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&d_buffer, bufferSize),
        "cudaMalloc for d_buffer"
    );
    
    CUSP_CALL_AND_CHECK(
        cusparseCsr2cscEx2(
            handle,
            n, m, non_zero,
            device_values_csc, device_ptr_csc, device_id_csc,
            device_values_csr, device_ptr_csr, device_id_csr, 
            CUDA_R_64F,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            d_buffer
        ), "cusparseCsr2cscEx2_bufferSize"
    );

    CUDA_CALL_AND_CHECK(
        cudaFree(d_buffer),
        "cudaFree for d_buffer"
    );
}


void CudaSparseMatrix::LUdecompose(
    cudssHandle_t& handle, 
    cudssConfig_t& config, 
    cudssData_t& data,
    cudssHandle_t& handle_T, 
    cudssConfig_t& config_T, 
    cudssData_t& data_T
)
{
    cudssMatrix_t x, rhs; // у функции cudssExecute универсальная сигнатура для всех операций, это просто заглушки
    if (!CSС_exist)
    {
        std::cout << "WARNING: csc doesnt exist" << std::endl;
        return;
    }

    if (!descr_exist)
    {
        std::cout << "WARNING: descr doesn't exist" << std::endl;
        return;
    };

    CUDSS_CALL_AND_CHECK(
        cudssExecute(
            handle, 
            CUDSS_PHASE_ANALYSIS, 
            config, 
            data,
            mat_cudss_descr,
            x, rhs),
        "cudssExecute:CUDSS_PHASE_ANALYSIS"
    );

    CUDSS_CALL_AND_CHECK(
        cudssExecute(
            handle, 
            CUDSS_PHASE_FACTORIZATION, 
            config, 
            data,
            mat_cudss_descr,
            x, rhs),
        "cudssExecute:CUDSS_PHASE_FACTORIZATION"
    );

    CUDSS_CALL_AND_CHECK(
        cudssExecute(
            handle_T, 
            CUDSS_PHASE_ANALYSIS, 
            config_T, 
            data_T,
            mat_cudss_descr_T,
            x, rhs),
        "cudssExecute:CUDSS_PHASE_ANALYSIS"
    );

    CUDSS_CALL_AND_CHECK(
        cudssExecute(
            handle_T, 
            CUDSS_PHASE_FACTORIZATION, 
            config_T, 
            data_T,
            mat_cudss_descr_T,
            x, rhs),
        "cudssExecute:CUDSS_PHASE_FACTORIZATION"
    );
}


 void CudaSparseMatrix::solve(
    cudssHandle_t& handle, 
    cudssConfig_t& config, 
    cudssData_t& data,
    const CudaDataDenseVector& rhs, 
    CudaDataDenseVector& sol, 
    const bool& transpose
)
{
    if (transpose)
    {
        CUDSS_CALL_AND_CHECK(
            cudssExecute(
                handle, 
                CUDSS_PHASE_SOLVE, 
                config, 
                data,
                mat_cudss_descr_T,
                sol.cudss_descr, rhs.cudss_descr),
            "cudssExecute:CUDSS_PHASE_SOLVE transpose"
        );
    } 
    else
    {
        CUDSS_CALL_AND_CHECK(
            cudssExecute(
                handle, 
                CUDSS_PHASE_SOLVE, 
                config, 
                data,
                mat_cudss_descr,
                sol.cudss_descr, rhs.cudss_descr),
            "cudssExecute:CUDSS_PHASE_SOLVE"
        );
    }
}


CudaSparseMatrix& CudaSparseMatrix::operator=(
    CoinPackedMatrix& matrix
)
{
   if (memory_init) 
        freeMemory();

    if (descr_exist) 
        destroyDesr();

    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();


    non_zero = matrix.getNumElements();

    allocateMemory(non_zero, m, n);

    if (matrix.isColOrdered()) 
    {
        major_dim = matrix.getMajorDim() + 1;
        copy(
            device_values_csc, device_ptr_csc, device_id_csc,
            matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
            cudaMemcpyHostToDevice
        );

        matrix.reverseOrdering();

        major_dim = matrix.getMajorDim() + 1;
        copy(
            device_values_csr, device_ptr_csr, device_id_csr,
            matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
            cudaMemcpyHostToDevice
        );        

        return *this;
    }

    major_dim = matrix.getMajorDim() + 1;
    copy(
        device_values_csr, device_ptr_csr, device_id_csr,
        matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
        cudaMemcpyHostToDevice
    );   
    
    matrix.reverseOrdering();

    major_dim = matrix.getMajorDim() + 1;
    copy(
        device_values_csc, device_ptr_csc, device_id_csc,
        matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
        cudaMemcpyHostToDevice
    );

    return *this;
}


void CudaSparseMatrix::initI(const int n)
{
    if (memory_init) 
        freeMemory();

    if (descr_exist) 
        destroyDesr();

    this -> m = n;
    this -> n = n;

    major_dim = n + 1;
    non_zero  = n;

    CSС_exist = true;

    allocateMemory(non_zero, m, n);

    int* buff = nullptr;
    double* buff_vals = nullptr;
    CUDA_CALL_AND_CHECK(
        cudaMallocHost(
            &buff, (n + 1) * sizeof(int)
        ),
        "cudaMallocHost for buff"
    );

    CUDA_CALL_AND_CHECK(
        cudaMallocHost(
            &buff_vals, (n + 1) * sizeof(double)
        ),
        "cudaMallocHost for buff_vals"
    );

    for (int i = 0; i < n + 1; i++)
    {
        buff[i] = i;
        buff_vals[i] = 1.0;
    }

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values_csr, buff_vals, n*sizeof(double), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_values_csr"
    )
    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values_csc, buff_vals, n*sizeof(double), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_values_csc"
    )

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id_csc, buff, n*sizeof(int), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_id_csc"
    )
    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id_csr, buff, n*sizeof(int), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_id_csr"
    )

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr_csc, buff, (n + 1)*sizeof(int), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_ptr_csc"
    )
    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr_csr, buff, (n + 1)*sizeof(int), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_ptr_csr"
    )

    CUDA_CALL_AND_CHECK(
        cudaFreeHost(buff),
        "cudaMallocHost for buff"
    );
}


void CudaSparseMatrix::copyCsrToHost(
    std::vector<double>& elem_csr,
    std::vector<int>& row_ptr,
    std::vector<int>& col_id
)
{
    copy(
        elem_csr.data(), row_ptr.data(), col_id.data(), 
        device_values_csr, device_ptr_csr, device_id_csr,
        cudaMemcpyDeviceToHost
    );    
}


void CudaSparseMatrix::updateDataByHost(
    const std::vector<double>& elem_csr,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_id
)
{
    if (memory_init) 
        freeMemory();

    if (descr_exist) 
        destroyDesr();

    allocateMemory(elem_csr.size(), m, n);
    copy(
        device_values_csr, device_ptr_csr, device_id_csr,
        elem_csr.data(), row_ptr.data(), col_id.data(),  
        cudaMemcpyHostToDevice
    );   
}


void CudaSparseMatrix::stackColUnitMatrix()
{
    std::vector<double> new_elem_csr(non_zero + m);
    std::vector<int> new_col_id(non_zero + m);

    std::vector<double> elem_csr(non_zero);
    std::vector<int> row_ptr(m + 1);
    std::vector<int> col_id(non_zero);

    copyCsrToHost(elem_csr, row_ptr, col_id);
   
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
    col_id = std::move(new_col_id);
    elem_csr = std::move(new_elem_csr);
    n = n + m;

    major_dim = m + 1;
    non_zero += m;

    updateDataByHost(elem_csr, row_ptr, col_id);
}
    

std::set<int> CudaSparseMatrix::deleteCols(std::set<int> cols)
{
    std::vector<double> new_elem_csr;
    std::vector<int> new_row_ptr;
    std::vector<int> new_col_id;

    std::vector<double> elem_csr(non_zero);
    std::vector<int> row_ptr(m + 1);
    std::vector<int> col_id(non_zero);

    std::set<int> excpet_rows;

    copyCsrToHost(elem_csr, row_ptr, col_id);
    
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

    updateDataByHost(elem_csr, row_ptr, col_id);

    return excpet_rows;
}


void CudaSparseMatrix::resetData(
    cusparseHandle_t& handle,
    const CudaSparseMatrix& matrix, 
    const CudaIndexVector& indexes
)
{
    std::vector<std::pair<int, int>> ranges(indexes.getSize());
    int new_size = 0;

    if (memory_init) 
        freeMemory();

    if (descr_exist) 
        destroyDesr();

    for (size_t i = 0; i < indexes.getSize(); i++)
    {
        CUDA_CALL_AND_CHECK(
            cudaMemcpy(
                &ranges[i].first, matrix.device_ptr_csc + indexes[i], 
                sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy"
        );
        CUDA_CALL_AND_CHECK(
            cudaMemcpy(
                &ranges[i].second, matrix.device_ptr_csc + indexes[i] + 1, 
                sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy"
        );
        new_size += ranges[i].second - ranges[i].first;
    }

    non_zero = new_size;
    this->m = matrix.m;            
    this->n = indexes.getSize(); 

    allocateMemory(new_size, m, n);

    int curr_size = 0;

    for (size_t i = 0; i < indexes.getSize(); i++)
    {
        CUDA_CALL_AND_CHECK(
            cudaMemcpy(
                device_values_csc + curr_size, matrix.device_values_csc + ranges[i].first, 
                (ranges[i].second - ranges[i].first) * sizeof(double), cudaMemcpyDeviceToDevice),
            "cudaMemcpy"
        );
        CUDA_CALL_AND_CHECK(
            cudaMemcpy(
                device_id_csc + curr_size, matrix.device_id_csc + ranges[i].first, 
                (ranges[i].second - ranges[i].first) * sizeof(int), cudaMemcpyDeviceToDevice),
            "cudaMemcpy"
        );
        CUDA_CALL_AND_CHECK(
            cudaMemcpy(
                device_ptr_csc + i, &curr_size, 
                sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy"
        );
        curr_size += ranges[i].second - ranges[i].first;
    }
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr_csc + indexes.getSize(), &curr_size, 
            sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy"
    );

    CSС_exist = true;

    genCsr(handle);
}


int CudaSparseMatrix::calcNonzeroInColumn(const int& p) const
{
    int count = 0;
    int col_start, col_end;

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            &col_start, device_ptr_csc + p, 
            sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            &col_end, device_ptr_csc + p + 1, 
            sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );

    int col_size = col_end - col_start;
    double* buff = nullptr;

    CUDA_CALL_AND_CHECK(
        cudaMallocHost(&buff, col_size*sizeof(double)),
        "cudaMallocHost"
    )
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            buff, device_values_csc + col_start, 
            col_size*sizeof(double), cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );

    for (int i = 0; i < col_size; i++)
        count += (fabs(buff[i]) < EPS_Z) ? 0 : 1;

    CUDA_CALL_AND_CHECK(
        cudaFreeHost(buff),
        "cudaFreeHost"
    )

    return count;
}


void CudaSparseMatrix::addSparseCol(
    cusparseHandle_t& handle, CudaDataDenseVector& vec, 
    const IndexVector& cols, const std::vector<double>& alpha, double multiplier
)
{
    if (m != vec.getSize())
        throw std::runtime_error("Incorrect size");

    for (const auto& i : cols) {
        addSpColsToVec(handle, m, n, i,
                       vec.device_values,
                       device_values_csc,
                       device_id_csc,
                       device_ptr_csc,
                       multiplier * alpha[i]);
    }
    cudaDeviceSynchronize();
}


void CudaSparseMatrix::addSparseCol(
    cusparseHandle_t& handle, CudaDataDenseVector& vec, 
    const IndexVector& cols, const double& alpha
)
{
    if (m != vec.getSize())
        throw std::runtime_error("Incorrect size");

    for (const auto& i : cols) {
        addSpColsToVec(handle, m, n, i,
                       vec.device_values,
                       device_values_csc,
                       device_id_csc,
                       device_ptr_csc,
                       alpha);

    }
    cudaDeviceSynchronize();
}
 

// debug 
void CudaSparseMatrix::show()
{
    std::vector<double> elem_csr(non_zero);
    std::vector<int> row_ptr(m + 1);
    std::vector<int> col_id(non_zero);

    copyCsrToHost(elem_csr, row_ptr, col_id);
    Matrix buff(elem_csr, row_ptr, col_id, m, n);

    buff.show();
}