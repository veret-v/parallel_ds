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

    for (size_t i = 0; i < size; i++)
        host_values[i] = 0;
    updateDeviceMem();

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
    
    size = values_vector.getSize();
    allocateMemory(size);
            
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
            size, VECTOR_COLS, size, 
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

void CudaDataDenseVector::setValues(
    const CudaDataDenseVector& values_vector, 
    CudaIndexVector& indexes
)
{
    for (int i = 0; i < indexes.getSize(); i++)
        host_values[indexes[i]] = values_vector.host_values[i];
}


void CudaDataDenseVector::initUnitVec(const int p)
{
    CUDA_CALL_AND_CHECK(
        cudaMemset(host_values, 0, size*sizeof(double)),
        "cudaMemset"
    );

    host_values[p] = 1;
}


void CudaDataDenseVector::multiplyHostData(const double& alpha)
{
    for (int i = 0; i < size; i++)
        host_values[i] = alpha * host_values[i];
}


CudaDataDenseVector& CudaDataDenseVector::operator-()
{
    CudaDataDenseVector buff(*this);  
    for (int i = 0; i < size; i++)
        buff.host_values[i] = -buff.host_values[i];
    return buff;
}


std::vector<double> CudaDataDenseVector::operator-(const CudaDataDenseVector& values_vector)
{
    if (values_vector.getSize() != size)
    {
        std::cout << "ERROR: size != values_vector.size" << std::endl;
        exit(1);
    }

    std::vector<double> new_vec(size);
    for (int i = 0; i < size; i++)
    {
        new_vec[i] = host_values[i] - values_vector.host_values[i];
    }

    return new_vec;
}


double CudaDataDenseVector::mean() const
{
    double sum = 0;
    for (int i = 0; i < getSize(); i++)
        sum +=  host_values[i];
    return sum / getSize();
}


double CudaDataDenseVector::norm() const
{
    double sum = 0;
    for (int i = 0; i < getSize(); i++)
        sum += pow(host_values[i], 2);
    return sum;
}


int CudaDataDenseVector::countNonZero() const
{
    double count = 0;
    for (int i = 0; i < getSize(); i++)
        count += (fabs(host_values[i]) < EPS_Z) ? 0 : 1;
    return count;
}


void CudaDataDenseVector::deleteVals(std::set<int> idxs)
{
    const int new_size = getSize() - idxs.size();
    double* new_data = NULL;
    double* device_new_data = NULL;

    cudaMallocHost(&new_data, new_size*sizeof(double));
    cudaMalloc(&device_new_data, new_size*sizeof(double));

    int new_pos = 0;
    for (int i = 0; i < getSize(); i++)
        if (idxs.find(i) == idxs.end()) new_data[new_pos++] = host_values[i];

    std::swap(host_values, new_data);
    std::swap(device_values, device_new_data);

    cudaFreeHost(new_data);
    cudaFree(device_new_data);

    size = new_size;

    updateDeviceMem();
}


void CudaDataDenseVector::resize(const int& new_size)
{
    if (new_size < size)
    {
        std::cout << "ERROR: new_size must be  > size" << std::endl;
        exit(1);
    }
    else if (new_size > size)
    {
        double* new_data = NULL;
        double* device_new_data = NULL;

        cudaMallocHost(&new_data, new_size*sizeof(double));
        cudaMalloc(&device_new_data, new_size*sizeof(double));

        for (int i = 0; i < getSize(); i++)
            new_data[i] = host_values[i];

        std::swap(host_values, new_data);
        std::swap(device_values, device_new_data);

        cudaFreeHost(new_data);
        cudaFree(device_new_data);

        size = new_size;

        updateDeviceMem();
    }
}