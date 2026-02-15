#include "cudaDenseVector.hpp"


void CudaDenseVector::allocateMemory(size_t size)
{
    this->size = size;

    CUDA_CALL_AND_CHECK(
        cudaMallocHost(&host_values, size*sizeof(double)),
        "cudaMallocHost"
    )
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_values, size*sizeof(double)),
        "cudaMalloc"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemset(device_values, 0, size*sizeof(double)),
        "cudaMemset"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemset(host_values, 0, size*sizeof(double)),
        "cudaMemset"
    );
}


void CudaDenseVector::updateDeviceMem()
{
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, host_values, 
            size*sizeof(double), cudaMemcpyDefault),
        "cudaMemcpy"
    );
}


void CudaDenseVector::updateHostMem()
{
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            host_values, device_values, 
            size*sizeof(double), cudaMemcpyDefault),
        "cudaMemcpy"
    );
}


void CudaDenseVector::freeMemory()
{
    cudaFree(device_values);
    cudaFreeHost(host_values);

    device_values = nullptr;
    host_values = nullptr;

    this->size = 0;
}


void CudaDenseVector::createDescr()
{
    int64_t nrows = size, ncols = size;
    int ldb = ncols, ldx = nrows;

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


void CudaDenseVector::checkSize(const CudaDenseVector& values_vector) const
{
    if (values_vector.size != size)
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
}


CudaDenseVector::~CudaDenseVector()
{
    CUSP_CALL_AND_CHECK(
        cusparseDestroyDnVec(descr),
        "cusparseDestroyDnVec"
    );

    CUDSS_CALL_AND_CHECK(
        cudssMatrixDestroy(cudss_descr), 
        "cudssMatrixDestroy"
    );

    freeMemory();
}


CudaDenseVector::CudaDenseVector(const size_t size)
{
    allocateMemory(size);
    createDescr();
}


CudaDenseVector::CudaDenseVector(const ValuesVector& vector)
{
    allocateMemory(vector.getSize());

    for (size_t i = 0; i < size; i++)
        host_values[i] = vector[i];

    updateDeviceMem();
    createDescr();
}


CudaDenseVector::CudaDenseVector(const CudaDenseVector& values_vector)
{
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
}


CudaDenseVector& CudaDenseVector::operator=(const CudaDenseVector& values_vector)
{
    if (this == &values_vector) 
        return *this;
    
    if (host_values != nullptr || 
        device_values != nullptr) 
        CudaDenseVector::~CudaDenseVector();

    if (descr != nullptr) CudaDenseVector::~CudaDenseVector();

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


double CudaDenseVector::dot(const cublasHandle_t handle, const CudaDenseVector& values_vector) const
{
    checkSize(values_vector);
    double result;
    cublasDdot(handle, size, device_values, 1, values_vector.device_values, 1, &result);
    return result;
}


double CudaDenseVector::norm(const cublasHandle_t handle) const
{
    double result;
    cublasDnrm2(handle, size, device_values, 1, &result);
    return result;
}


void CudaDenseVector::axpyUpdate(const cublasHandle_t handle, const CudaDenseVector& values_vector, const double& alpha)
{
    checkSize(values_vector);
    cublasDaxpy(handle, size, &alpha, values_vector.device_values, 1, device_values, 1);
}


void CudaDenseVector::betaWeightsUpdate(const cublasHandle_t handle, const CudaDenseVector& alpha_q, const CudaDenseVector& tau, const int p_idx)
{
    checkSize(alpha_q);
    checkSize(tau);
    betaWeightsUpdateLauncher(handle, size, alpha_q.device_values, 1, tau.device_values, 1, device_values, 1, p_idx);
}


void CudaDenseVector::show() const
{
    std::cout << "Vector(" << getSize() << "):" << std::endl;
    for (size_t i = 0; i < getSize(); i++)
        std::cout << operator[](i) << " ";
    std::cout << std::endl;
}

