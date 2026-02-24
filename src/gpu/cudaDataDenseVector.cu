#include "cudaDataDenseVector.hpp"


void CudaDataDenseVector::destroyDescr()
{
     CUSP_CALL_AND_CHECK(
        cusparseDestroyDnVec(descr),
        "cusparseDestroyDnVec"
    );

    CUDSS_CALL_AND_CHECK(
        cudssMatrixDestroy(cudss_descr), 
        "cudssMatrixDestroy"
    );
}


CudaDataDenseVector::CudaDataDenseVector(const int size)
{
    allocateMemory(size);
    createDescr();
}


CudaDataDenseVector::CudaDataDenseVector(const ValuesVector& vector)
{
    allocateMemory(vector.getSize());

    for (size_t i = 0; i < size; i++)
        host_values[i] = vector[i];

    updateDeviceMem();
    createDescr();
}


CudaDataDenseVector::CudaDataDenseVector(const CudaDataDenseVector& values_vector)
{
    allocateMemory(values_vector.getSize());
    copy(values_vector.device_values, values_vector.host_values);
    createDescr();
}


CudaDataDenseVector::~CudaDataDenseVector()
{
    destroyDescr();
    freeMemory();
}


double CudaDataDenseVector::dot(const cublasHandle_t handle, const CudaDataDenseVector& values_vector) const
{
    checkSize(values_vector);
    double result;
    CUBLAS_CALL_AND_CHECK(
        cublasDdot(
            handle, 
            size, 
            device_values, 
            1, 
            values_vector.device_values, 
            1, &result),
        "cublasDdot"
    );
    return result;
}


double CudaDataDenseVector::norm(const cublasHandle_t handle) const
{
    double result;
    CUBLAS_CALL_AND_CHECK(
        cublasDnrm2(
            handle, size, 
            device_values, 1, 
            &result
        ),
        "cublasDnrm2"
    );
    return result;
}


void CudaDataDenseVector::axpyUpdate(
    const cublasHandle_t handle, 
    const CudaDataDenseVector& values_vector, 
    const double& alpha
)
{
    checkSize(values_vector);
    CUBLAS_CALL_AND_CHECK(
        cublasDaxpy(
            handle, size, 
            &alpha, 
            values_vector.device_values, 
            1, device_values, 1
        ),
        "cublasDaxpy"
    );
}


void CudaDataDenseVector::betaWeightsUpdate(
    const cublasHandle_t handle, 
    const CudaDataDenseVector& alpha_q, 
    const CudaDataDenseVector& tau, 
    const int p_idx
)
{
    checkSize(alpha_q);
    checkSize(tau);
    CUBLAS_CALL_AND_CHECK(
        betaWeightsUpdateLauncher(
            handle, 
            size, 
            alpha_q.device_values, 
            1, tau.device_values, 
            1, device_values, 
            1, p_idx
        ),
        "betaWeightsUpdateLauncher"
    );
}


void CudaDataDenseVector::updateVecBySparse(
    const int* crc_id, 
    const double* crc_val,
    int nnz
)
{
    CUDA_CALL_AND_CHECK(
        cudaMemset(host_values, 0, size*sizeof(double)),
        "cudaMemset"
    );

    for (size_t i = 0; i < nnz; i++)
        host_values[crc_id[i]] = crc_val[i];
}


CudaDataDenseVector& CudaDataDenseVector::operator=(const CudaDataDenseVector& values_vector)
{
    if (this == &values_vector) 
        return *this;
    
    if (host_values != nullptr || 
        device_values != nullptr) 
        freeMemory();

    if (descr != nullptr) destroyDescr();

    allocateMemory(values_vector.getSize());

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            host_values, values_vector.host_values, 
            size*sizeof(double), cudaMemcpyHostToHost),
        "cudaMemcpy"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, values_vector.device_values, 
            size*sizeof(double), cudaMemcpyDeviceToDevice),
        "cudaMemcpy"
    );

    createDescr();

    return *this;
}


void CudaDataDenseVector::createDescr()
{
    int64_t nrows = size, ncols = size;
    size_t ldb = ncols, ldx = nrows;

    CUSP_CALL_AND_CHECK(
        cusparseCreateDnVec(
            &descr, 
            size, 
            device_values, 
            CUDA_R_64F),
        "cusparseCreateDnVec"
    );

    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateDn(
            &cudss_descr, 
            size, VECTOR_COLS, ldb, 
            device_values, 
            CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn"
    );
}


 void CudaDataDenseVector::updateByPartialVec(
    const CudaDataDenseVector& values_vector, 
    CudaIndexVector& indexes
)
{
    for (int i = 0; i < indexes.getSize(); i++)
        host_values[i] = values_vector.host_values[indexes[i]];
}


void CudaDataDenseVector::addSparseCol(
    const CudaSparseMatrix& mat, 
    const int p, const double alpha
)
{
    CudaDataDenseVector buff; 
    mat.getColumn(p, buff);

    for (int i = 0; i < size; i++)
        host_values[i] += alpha * buff[i];
}


void CudaDataDenseVector::initUnitVec(const int p)
{
    CUDA_CALL_AND_CHECK(
        cudaMemset(host_values, 0, size*sizeof(double)),
        "cudaMemset"
    );

    host_values[p] = 1;
}


CudaDataDenseVector& CudaDataDenseVector::operator-()
{
    CudaDataDenseVector buff(size); 
    for (int i = 0; i < size; i++)
        buff.host_values[i] = -host_values[i];

    return buff;
}