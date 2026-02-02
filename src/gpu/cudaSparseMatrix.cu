#include "cudaSparseMatrix.hpp"


void CudaSparseMatrix::allocateMemory(size_t non_zero_size, size_t col_starts)
{
    cudaMallocHost(&host_values, non_zero_size*sizeof(double));
    cudaMallocHost(&host_ptr, col_starts*sizeof(double));
    cudaMallocHost(&host_id, non_zero_size*sizeof(double));

    cudaMalloc(&device_values, non_zero_size*sizeof(double));
    cudaMalloc(&device_ptr, col_starts*sizeof(double));
    cudaMalloc(&device_id, non_zero_size*sizeof(double));
}


void CudaSparseMatrix::freeMemory()
{
    cudaFree(device_values);
    cudaFree(device_ptr);
    cudaFree(device_id);

    cudaFreeHost(host_values);
    cudaFreeHost(host_ptr);
    cudaFreeHost(host_id);

    device_values = nullptr;
    device_ptr = nullptr;
    device_id = nullptr;

    host_values = nullptr;
    host_ptr = nullptr;
    host_id = nullptr;
}


void CudaSparseMatrix::createDescr()
{
    cusparseCreateCsc(
        &descr,
        m, n, non_zero,
        (void*)device_ptr,
        (void*)device_id,
        (void*)device_id,
        CUSPARSE_INDEX_32I,  
        CUSPARSE_INDEX_32I,  
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    );  
}


CudaSparseMatrix::~CudaSparseMatrix()
{
    freeMemory();
    cusparseDestroySpMat(descr);
}


void CudaSparseMatrix::updateDeviceMem()
{
    cudaMemcpy(device_values, host_values, non_zero*sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(device_ptr, host_ptr, major_dim*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(device_id, host_id, non_zero*sizeof(int), cudaMemcpyDefault);
}


void CudaSparseMatrix::updateHostMem()
{
    cudaMemcpy(host_values, device_values, non_zero*sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(host_ptr, device_ptr, major_dim*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(host_id, device_id, non_zero*sizeof(int), cudaMemcpyDefault);
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

    cudaMemcpy(device_values, elements_buf.data(), non_zero*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ptr, col_ptr_buf.data(), major_dim*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_id, row_id_buf.data(), non_zero*sizeof(int), cudaMemcpyHostToDevice);

    createDescr();
}


CudaSparseMatrix::CudaSparseMatrix(const CudaSparseMatrix& matrix)
{
    this -> m = matrix.m;
    this -> n = matrix.n;

    major_dim = matrix.major_dim;
    non_zero  = matrix.non_zero;

    allocateMemory(non_zero, major_dim + 1);

    cudaMemcpy(device_values, matrix.device_values, non_zero*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(device_ptr, matrix.device_ptr, major_dim*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(device_id, matrix.device_id, non_zero*sizeof(int), cudaMemcpyDeviceToDevice);

    createDescr();
}


CudaSparseMatrix& CudaSparseMatrix::operator=(const CudaSparseMatrix& matrix)
{
    if (this == &matrix) {
        return *this;
    }

    if (host_values    != nullptr || 
        host_ptr   != nullptr || 
        host_id    != nullptr || 
        device_values  != nullptr || 
        device_ptr != nullptr || 
        device_id  != nullptr) 
        freeMemory();

    if (descr != nullptr) cusparseDestroySpMat(descr);

    this -> m = matrix.m;
    this -> n = matrix.n;

    major_dim = matrix.major_dim;
    non_zero  = matrix.non_zero;

    allocateMemory(non_zero, major_dim + 1);

    cudaMemcpy(host_values, matrix.host_values, non_zero*sizeof(double), cudaMemcpyHostToHost);
    cudaMemcpy(host_ptr, matrix.host_ptr, major_dim*sizeof(int), cudaMemcpyHostToHost);
    cudaMemcpy(host_id, matrix.host_id, non_zero*sizeof(int), cudaMemcpyHostToHost);

    cudaMemcpy(device_values, matrix.device_values, non_zero*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(device_ptr, matrix.device_ptr, major_dim*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(device_id, matrix.device_id, non_zero*sizeof(int), cudaMemcpyDeviceToDevice);

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

    cusparseSpMV_bufferSize(
        handle,
        oper_type,
        &alpha,
        descr,
        rhs.descr,
        &beta,
        sol.descr,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize
    );

    cudaMalloc(&dBuffer, bufferSize);

    cusparseSpMV(
        handle,
        oper_type,
        &alpha,
        descr,
        rhs.descr,
        &beta,
        sol.descr,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer
    );

    cudaFree(dBuffer);
}




void CudaSparseMatrix::swapColumn(cusparseHandle_t handle, CudaSparseMatrix& A, const size_t col1, const size_t col2)
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


void CudaSparseMatrix::LUdecompose(cusolverSpHandle_t handle, LUfactor& lu_factor)
{
    if (m != n)
    {
        throw "Already scr";
        exit(1);
    }
    csrluInfo_t info = NULL;
    cusparseMatDescr_t descr_lu = NULL;

    cusolverSpCreateCsrluInfo(&info);

    size_t internalDataInBytes, workspaceInBytes;

    cusparseCreateMatDescr(&descr_lu);
    cusparseSetMatType(descr_lu, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_lu, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpDcsrluBufferInfo(
        handle, n, non_zero, descr_lu,
        device_values, device_ptr, device_id,
        info, &internalDataInBytes, &workspaceInBytes
    );

    cusolverSpXcsrluAnalysis(
        handle, n, non_zero, descr_lu,
        device_ptr, device_id, info
    );

    int singularity = 0;
    void* d_buffer = nullptr;
    cudaMalloc(&d_buffer, workspaceInBytes);
    
    cusolverSpDcsrluFactor(
        handle, n, non_zero, descr_lu,
        device_values, device_ptr, device_id,
        info, SING_EPS, d_buffer
    );

    int nnzL = 0, nnzU = 0;

    cusolverSpXcsrluNnz(
        handle, &nnzL, 
        &nnzU, info
    );

    cusolverSpDcsrluExtract(
        handle, 
        lu_factor.P, lu_factor.Q,
        lu_factor.descr_l, 
        lu_factor.device_values_l, lu_factor.device_ptr_l, lu_factor.device_id_l,
        lu_factor.descr_u, 
        lu_factor.device_values_u, lu_factor.device_ptr_u, lu_factor.device_id_u,
        info, d_buffer
    );
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

    cudaMalloc(&scr_val, non_zero * sizeof(double));
    cudaMalloc(&scr_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&scr_id, non_zero * sizeof(int));

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
    );
    
    cudaMalloc(&d_buffer, bufferSize);
    
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
    );

    cudaFree(d_buffer);

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


void LUfactor::setup(
    cusolverRfHandle_t rf_handle, 
    const CudaSparseMatrix& orig
) const
{
    cusolverRfSetupDevice(
        orig.n, orig.non_zero,
        orig.device_ptr, orig.device_id, orig.device_values, 
        non_zero_l, 
        device_ptr_l, device_id_l, device_values_l, 
        non_zero_u, 
        device_ptr_u, device_id_u, device_values_u, 
        P, Q, rf_handle
    );

    cusolverRfAnalyze(rf_handle);
    cusolverRfRefactor(rf_handle);
}


void LUfactor::solve(
    cusolverRfHandle_t handle, 
    const CudaDenseVector& rhs, 
    CudaDenseVector& sol, 
    const bool& transpose
) 
{
    cusolverRfSolve(
        handle,
         P, Q, 1, 
         sol.device_values, sol.size, 
         rhs.device_values, rhs.size
    );
    cusolverRfSolve
}


void LUfactor::update(
    cusolverRfHandle_t handle, 
    const size_t& idx, 
    const CudaDenseVector& vector
)
{
}    

