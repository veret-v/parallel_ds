#include "cudaSparseMatrix.hpp"


void CudaSparseMatrix::allocateMemory(int non_zero_size, int col_starts)
{
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_values_csc, non_zero_size*sizeof(double)),
        "cudaMalloc for device_values_csc"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_ptr_csc, col_starts*sizeof(double)),
        "cudaMalloc for device_ptr_csc"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_id_csc, non_zero_size*sizeof(double)),
        "cudaMalloc for device_id_csc"
    );

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_values_csr, non_zero * sizeof(double)),
        "cudaMalloc for scr_val"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_ptr_csr, (m + 1) * sizeof(int)),
        "cudaMalloc for scr_ptr"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_id_csr, non_zero * sizeof(int)),
        "cudaMalloc for scr_id"
    );
}


void CudaSparseMatrix::freeMemory()
{
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


void CudaSparseMatrix::createDescr()
{
    CUSP_CALL_AND_CHECK(
        cusparseCreateCsr(
            &mat_cusp_descr,
            m, n, non_zero,
            device_ptr_csc,
            device_id_csc,
            device_values_csc,
            CUSPARSE_INDEX_32I,  
            CUSPARSE_INDEX_32I,  
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F),
        "cusparseCreateCsc"
    );  

    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateCsr(
            &mat_cudss_descr,
            m, n, non_zero,
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

    freeMemory();
}


CudaSparseMatrix::CudaSparseMatrix(const int m, const int n)
{
    this -> m = m;
    this -> n = n;
}


CudaSparseMatrix::CudaSparseMatrix(const Matrix& matrix, cusparseHandle_t& cusp_handle)
{
    this -> m = std::get<0>(matrix.getSize());
    this -> n = std::get<1>(matrix.getSize());

    std::vector<double> elements_buf = matrix.getElem();
    std::vector<int> col_ptr_buf = matrix.getColPtrs();
    std::vector<int> row_id_buf = matrix.getRowIds();

    major_dim = n;
    non_zero  = elements_buf.size();

    allocateMemory(non_zero, major_dim + 1);

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values_csc, 
            elements_buf.data(),
            non_zero*sizeof(double), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy for device_values_csc"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr_csc, 
            col_ptr_buf.data(), 
            major_dim*sizeof(int), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy for device_ptr_csc"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id_csc, 
            row_id_buf.data(), 
            non_zero*sizeof(int), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy for device_id_csc"
    );

    genCsr(cusp_handle);

    createDescr();
}


CudaSparseMatrix::CudaSparseMatrix(const CudaSparseMatrix& matrix)
{
    this -> m = matrix.m;
    this -> n = matrix.n;

    major_dim = matrix.major_dim;
    non_zero  = matrix.non_zero;

    allocateMemory(non_zero, major_dim + 1);

    copy(
        device_values_csc, device_ptr_csc, device_id_csc,
        matrix.device_values_csc, matrix.device_ptr_csc, matrix.device_id_csc, 
        cudaMemcpyDeviceToDevice
    );
   
    createDescr();
}


CudaSparseMatrix& CudaSparseMatrix::operator=(const CudaSparseMatrix& matrix)
{
    if (this == &matrix) {
        return *this;
    }

    if (device_values_csc  != nullptr || 
        device_ptr_csc != nullptr || 
        device_id_csc  != nullptr) 
        freeMemory();

    if (mat_cusp_descr != nullptr) cusparseDestroySpMat(mat_cusp_descr);

    this -> m = matrix.m;
    this -> n = matrix.n;

    major_dim = matrix.major_dim;
    non_zero  = matrix.non_zero;

    allocateMemory(non_zero, major_dim + 1);

    copy(
        device_values_csc, device_ptr_csc, device_id_csc,
        matrix.device_values_csc, matrix.device_ptr_csc, matrix.device_id_csc, 
        cudaMemcpyDeviceToDevice
    );

    copy(
        device_values_csr, device_ptr_csr, device_id_csr,
        matrix.device_values_csr, matrix.device_ptr_csr, matrix.device_id_csr, 
        cudaMemcpyDeviceToDevice
    );
    
    createDescr();

    return *this;
}


void CudaSparseMatrix::dotUpdate(
    cusparseHandle_t& handle, 
    const CudaDataDenseVector& vec1, 
    const CudaIndexVector& vec1_idx,
    const bool sparse_vec1,
    const CudaDataDenseVector& vec2, 
    CudaDataDenseVector& sol,
    const CudaIndexVector& need_ptrs,
    const int size_ptrs,
    const double beta,
    const double alpha,
    const CudaIndexVector& set_idx,
    const SpmvOptions method
) 
{
    switch (method) 
    {
        case SpmvOptions::FULL_UPDATE:
            CUSP_CALL_AND_CHECK(
                spmvUpdateInc(
                    handle,
                    sparse_vec1,
                    non_zero,
                    size_ptrs,
                    sol.device_values,
                    vec1.device_values,
                    vec1_idx.device_values,
                    device_values_csr,
                    device_id_csr,
                    device_ptr_csr,
                    need_ptrs.device_values,
                    alpha,
                    beta
                ), "cusparseSpMV"
            );
            break;
        
        case SpmvOptions::SET_UPDATE:
            CUSP_CALL_AND_CHECK(
                spmvUpdateSet(
                    handle,
                    non_zero,
                    size_ptrs,
                    sol.device_values,
                    set_idx.device_values,
                    set_idx.getSize(),
                    vec1.device_values,
                    vec2.device_values,
                    device_values_csr,
                    device_id_csr,
                    device_ptr_csr,
                    need_ptrs.device_values,
                    alpha,
                    beta
                ), "cusparseSpMV"
            );
            break;

        case SpmvOptions::FULL_UPDATE_T:
            CUSP_CALL_AND_CHECK(
                spmvUpdateInc(
                    handle,
                    sparse_vec1,
                    non_zero,
                    size_ptrs,
                    sol.device_values,
                    vec1.device_values,
                    vec1_idx.device_values,
                    device_values_csc,
                    device_id_csc,
                    device_ptr_csc,
                    need_ptrs.device_values,
                    alpha,
                    beta
                ), "cusparseSpMV"
            );
            break;

        case SpmvOptions::SET_UPDATE_T:
            CUSP_CALL_AND_CHECK(
                spmvUpdateSet(
                    handle,
                    non_zero,
                    size_ptrs,
                    sol.device_values,
                    set_idx.device_values,
                    set_idx.getSize(),
                    vec1.device_values,
                    vec2.device_values,
                    device_values_csc,
                    device_id_csc,
                    device_ptr_csc,
                    need_ptrs.device_values,
                    alpha,
                    beta
                ), "cusparseSpMV"
            );
            break;
            
        case SpmvOptions::UNKNOWN:
            throw "Unknown spmv method";
            break;
    }
}


void CudaSparseMatrix::getColumn(const int p, CudaDataDenseVector& rhs) const
{
    if (m != rhs.getSize())
    {
        throw "Incorrect size";
        exit(1);
    }

    double* buff_res = NULL;
    double* buff_val = NULL;
    int* buff_rows   = NULL;

    int col_size = device_ptr_csc[p + 1] - device_ptr_csc[p];

    CUDA_CALL_AND_CHECK(
        cudaMallocHost(&buff_res, n*sizeof(double)),
        "cudaMallocHost : buff_res"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemset(buff_res, 0, n*sizeof(double)),
        "cudaMemset : buff_res"
    );

    CUDA_CALL_AND_CHECK(
        cudaMallocHost(&buff_val, col_size*sizeof(double)),
        "cudaMallocHost : buff_val"
    );
    CUDA_CALL_AND_CHECK(
        cudaMallocHost(&buff_rows, col_size*sizeof(int)),
        "cudaMallocHost : buff_rows"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values_csc + device_ptr_csc[p], buff_val, 
            col_size * sizeof(int), cudaMemcpyDeviceToHost), 
        "cudaMemcpy : buff_val"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id_csc + device_ptr_csc[p], buff_rows, 
            col_size * sizeof(int), cudaMemcpyDeviceToHost), 
        "cudaMemcpy : buff_rows"
    );

    rhs.updateVecBySparse(buff_rows, buff_val, col_size);

    CUDA_CALL_AND_CHECK(
        cudaFreeHost(buff_rows),
        "cudaFree : buff_rows"
    );
    CUDA_CALL_AND_CHECK(
        cudaFreeHost(buff_val),
        "cudaFree : buff_val"
    );    
}


void CudaSparseMatrix::genCsr(cusparseHandle_t& handle)
{
    if (CSR_exist)
    {
        throw "Already scr";
        exit(1);
    }
    
    size_t bufferSize = 0;
    void* d_buffer = nullptr;

    CUSP_CALL_AND_CHECK(
        cusparseCsr2cscEx2_bufferSize(
            handle,
            m, n, non_zero,
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
            m, n, non_zero,
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

    CSR_exist = true;
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
                mat_cudss_descr_T,
                sol.cudss_descr, rhs.cudss_descr),
            "cudssExecute:CUDSS_PHASE_SOLVE"
        );
    }
}


CudaSparseMatrix& CudaSparseMatrix::operator=(
    CoinPackedMatrix& matrix
)
{
    if (device_values_csc  != nullptr || 
        device_ptr_csc != nullptr || 
        device_id_csc  != nullptr) 
        freeMemory();

    if (mat_cusp_descr != nullptr) cusparseDestroySpMat(mat_cusp_descr);

    this -> m = matrix.getNumRows();
    this -> n = matrix.getNumCols();

    int non_zero_size;
    int ptr_size;
    
    non_zero_size = matrix.getNumElements();
    ptr_size = matrix.getMajorDim();

    allocateMemory(non_zero_size, ptr_size);

    if (matrix.isColOrdered()) 
    {
        copy(
            device_values_csc, device_ptr_csc, device_id_csc,
            matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
            cudaMemcpyHostToDevice
        );

        matrix.reverseOrdering();

        copy(
            device_values_csr, device_ptr_csr, device_id_csr,
            matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
            cudaMemcpyHostToDevice
        );        

        return *this;
    }

    copy(
        device_values_csr, device_ptr_csr, device_id_csr,
        matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
        cudaMemcpyHostToDevice
    );   
    
    matrix.reverseOrdering();

    copy(
        device_values_csc, device_ptr_csc, device_id_csc,
        matrix.getElements(), matrix.getVectorStarts(), matrix.getIndices(), 
        cudaMemcpyHostToDevice
    );

    return *this;
}


void CudaSparseMatrix::initI(const int n)
{
     if (device_values_csc  != nullptr || 
        device_ptr_csc != nullptr || 
        device_id_csc  != nullptr) 
        freeMemory();

    if (mat_cusp_descr != nullptr) cusparseDestroySpMat(mat_cusp_descr);

    this -> m = n;
    this -> n = n;

    major_dim = n;
    non_zero  = n;

    allocateMemory(non_zero, major_dim + 1);

    int* buff;
    CUDA_CALL_AND_CHECK(
        cudaMallocHost(
            &buff, (n + 1) * sizeof(int)
        ),
        "cudaMallocHost for buff"
    );

    for (size_t i = 0; i < n + 1; i++)
        buff[i] = i - 1;
    

    CUDA_CALL_AND_CHECK(
        cudaMemset(
            device_values_csr, 1, n*sizeof(double)
        ),
        "cudaMemset for device_values_csr"
    )

    CUDA_CALL_AND_CHECK(
        cudaMemset(
            device_values_csc, 1, n*sizeof(double)
        ),
        "cudaMemset for device_values_csr"
    )

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id_csc, buff, n, cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_id_csc"
    )
    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id_csr, buff, n, cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_id_csr"
    )

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr_csc, buff, n + 1, cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_ptr_csc"
    )
    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr_csr, buff, n + 1, cudaMemcpyHostToDevice
        ),
        "cudaMemcpy for device_ptr_csr"
    )

    CUDA_CALL_AND_CHECK(
        cudaFree(buff),
        "cudaMallocHost for buff"
    );

    createDescr();
}