#include "cudaSparseMatrix.hpp"


void CudaSparseMatrix::allocateMemory(size_t non_zero_size, size_t col_starts)
{
    CUDA_CALL_AND_CHECK(cudaMalloc(&device_values, non_zero_size*sizeof(double)),
                        "cudaMalloc for device_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&device_ptr, col_starts*sizeof(double)),
                        "cudaMalloc for device_ptr");
    CUDA_CALL_AND_CHECK(cudaMalloc(&device_id, non_zero_size*sizeof(double)),
                        "cudaMalloc for device_id");
}


void CudaSparseMatrix::freeMemory()
{
    CUDA_CALL_AND_CHECK(cudaFree(device_values),
                        "cudaFree for device_values");
    CUDA_CALL_AND_CHECK(cudaFree(device_ptr),
                        "cudaFree for device_ptr")
    CUDA_CALL_AND_CHECK(cudaFree(device_id),
                        "cudaFree for device_id")

    device_values = nullptr;
    device_ptr = nullptr;
    device_id = nullptr;
}


void CudaSparseMatrix::createDescr()
{
    CUSP_CALL_AND_CHECK(
        cusparseCreateCsr(
            &mat_cusp_descr,
            m, n, non_zero,
            device_ptr,
            device_id,
            device_values,
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
            device_ptr,
            NULL,
            device_id,
            device_values,
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

    freeMemory();
}


CudaSparseMatrix::CudaSparseMatrix(const size_t m, const size_t n)
{
    this -> m = m;
    this -> n = n;
}


CudaSparseMatrix::CudaSparseMatrix(const Matrix& matrix)
{
    this -> m = std::get<0>(matrix.getSize());
    this -> n = std::get<1>(matrix.getSize());

    std::vector<double> elements_buf = matrix.getElem();
    std::vector<size_t> col_ptr_buf = matrix.getColPtrs();
    std::vector<size_t> row_id_buf = matrix.getRowIds();

    major_dim = n;
    non_zero  = elements_buf.size();

    allocateMemory(non_zero, major_dim + 1);

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, 
            elements_buf.data(),
            non_zero*sizeof(double), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy for device_values"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr, 
            col_ptr_buf.data(), 
            major_dim*sizeof(int), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy for device_ptr"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id, 
            row_id_buf.data(), 
            non_zero*sizeof(int), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy for device_id"
    );

    createDescr();
}


CudaSparseMatrix::CudaSparseMatrix(const CudaSparseMatrix& matrix)
{
    this -> m = matrix.m;
    this -> n = matrix.n;

    major_dim = matrix.major_dim;
    non_zero  = matrix.non_zero;

    allocateMemory(non_zero, major_dim + 1);

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, 
            matrix.device_values, 
            non_zero*sizeof(double), 
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy for device_values"
    );    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr, 
            matrix.device_ptr, 
            major_dim*sizeof(int), 
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy for device_ptr"
    );  
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id, 
            matrix.device_id, 
            non_zero*sizeof(int), 
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy for device_id"
    );  

    createDescr();
}


CudaSparseMatrix& CudaSparseMatrix::operator=(const CudaSparseMatrix& matrix)
{
    if (this == &matrix) {
        return *this;
    }

    if (device_values  != nullptr || 
        device_ptr != nullptr || 
        device_id  != nullptr) 
        freeMemory();

    if (mat_cusp_descr != nullptr) cusparseDestroySpMat(mat_cusp_descr);

    this -> m = matrix.m;
    this -> n = matrix.n;

    major_dim = matrix.major_dim;
    non_zero  = matrix.non_zero;

    allocateMemory(non_zero, major_dim + 1);

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, 
            matrix.device_values, 
            non_zero*sizeof(double), 
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy for device_values"
    );    
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_ptr, 
            matrix.device_ptr, 
            major_dim*sizeof(int), 
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy for device_ptr"
    );  
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_id, 
            matrix.device_id, 
            non_zero*sizeof(int), 
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy for device_id"
    );  

    createDescr();

    return *this;
}


void CudaSparseMatrix::dotUpdate(
    cusparseHandle_t handle, 
    const CudaDenseVector& rhs, 
    CudaDenseVector& sol,
    double beta,
    double alpha,
    bool transpose
) 
{
    cusparseOperation_t oper_type;
    if (transpose)
        oper_type = CUSPARSE_OPERATION_TRANSPOSE;
    else
        oper_type = CUSPARSE_OPERATION_NON_TRANSPOSE;

    void* dBuffer = nullptr;
    size_t bufferSize = 0;

    CUSP_CALL_AND_CHECK(
        cusparseSpMV_bufferSize(
            handle,
            oper_type,
            &alpha,
            mat_cusp_descr,
            rhs.descr,
            &beta,
            sol.descr,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSize
        ), "cusparseSpMV_bufferSize"
    );

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&dBuffer, bufferSize),
        "cudaMalloc for dotUpdate"
    );

    CUSP_CALL_AND_CHECK(
        cusparseSpMV(
            handle,
            oper_type,
            &alpha,
            mat_cusp_descr,
            rhs.descr,
            &beta,
            sol.descr,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            dBuffer
        ), "cusparseSpMV"
    );

    CUDA_CALL_AND_CHECK(
        cudaFree(dBuffer),
        "cudaFree for dotUpdate"
    );
}


void CudaSparseMatrix::swapColumn(
    cusparseHandle_t handle, 
    CudaSparseMatrix& A, 
    const size_t col1, const size_t col2)
{
    if (m != std::get<0>(A.getSize()))
    {
        throw "Incorrect size";
        exit(1);
    }

    int start1 = device_ptr[col1];
    int end1 = device_ptr[col1 + 1];
    int size1 = end1 - start1;
    
    int start2 = A.device_ptr[col2];
    int end2 = A.device_ptr[col2 + 1];
    int size2 = end2 - start2;


}


void CudaSparseMatrix::getColumn(const size_t p, CudaDenseVector& rhs) const
{
    if (m != rhs.getSize())
    {
        throw "Incorrect size";
        exit(1);
    }


}


void CudaSparseMatrix::cscToCsr(cusparseHandle_t handle)
{
    if (!is_CSC)
    {
        throw "Already scr";
        exit(1);
    }
    
    size_t bufferSize = 0;
    void* d_buffer = nullptr;

    double* scr_val = nullptr;
    int* scr_ptr    = nullptr;
    int* scr_id     = nullptr;

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&scr_val, non_zero * sizeof(double)),
        "cudaMalloc for scr_val"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&scr_ptr, (m + 1) * sizeof(int)),
        "cudaMalloc for scr_ptr"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&scr_id, non_zero * sizeof(int)),
        "cudaMalloc for scr_id"
    );

    CUSP_CALL_AND_CHECK(
        cusparseCsr2cscEx2_bufferSize(
            handle,
            m, n, non_zero,
            device_values, device_ptr, device_id,
            scr_val, scr_ptr, scr_id,
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
            device_values, device_ptr, device_id,
            scr_val, scr_ptr, scr_id,
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

    freeMemory();

    major_dim = m;

    allocateMemory(non_zero, major_dim + 1);

    device_values = scr_val;
    device_ptr    = scr_ptr;
    device_id     = scr_id;

    scr_val = nullptr;
    scr_ptr = nullptr;
    scr_id  = nullptr;
}


void CudaSparseMatrix::LUdecompose(
    cudssHandle_t handle, 
    cudssConfig_t config, 
    cudssData_t data,
    cudssHandle_t handle_T, 
    cudssConfig_t config_T, 
    cudssData_t data_T
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
    cudssHandle_t handle, 
    cudssConfig_t config, 
    cudssData_t data,
    const CudaDenseVector& rhs, 
    CudaDenseVector& sol, 
    const bool& transpose
)
{
    if (transpose)
    {
        CUDSS_CALL_AND_CHECK(
            cudssExecute(
                handle, 
                CUDSS_PHASE_ANALYSIS, 
                config, 
                data,
                mat_cudss_descr_T,
                sol.cudss_descr, rhs.cudss_descr),
            "cudssExecute:CUDSS_PHASE_ANALYSIS"
        );
    } 
    else
    {
        CUDSS_CALL_AND_CHECK(
            cudssExecute(
                handle, 
                CUDSS_PHASE_FACTORIZATION, 
                config, 
                data,
                mat_cudss_descr_T,
                sol.cudss_descr, rhs.cudss_descr),
            "cudssExecute:CUDSS_PHASE_FACTORIZATION"
        );
    }
}
