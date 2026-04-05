#include "cudaDenseVector.hpp"


template <typename ValueType>
void CudaDenseVector<ValueType>::allocateMemory(int size)
{
    this->size = size;

    CUDA_CALL_AND_CHECK(
        cudaMallocHost(&host_values, size*sizeof(ValueType)),
        "cudaMallocHost"
    )
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_values, size*sizeof(ValueType)),
        "cudaMalloc"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemset(device_values, 0, size*sizeof(ValueType)),
        "cudaMemset"
    );
    CUDA_CALL_AND_CHECK(
        cudaMemset(host_values, 0, size*sizeof(ValueType)),
        "cudaMemset"
    );
}


template <typename ValueType>
void CudaDenseVector<ValueType>::updateDeviceMem()
{
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, host_values, 
            size*sizeof(ValueType), cudaMemcpyHostToDevice),
        "cudaMemcpy"
    );
}


template <typename ValueType>
void CudaDenseVector<ValueType>::updateHostMem()
{
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            host_values, device_values, 
            size*sizeof(ValueType), cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );
}


template <typename ValueType>
void CudaDenseVector<ValueType>::freeMemory()
{
    cudaFree(device_values);
    cudaFreeHost(host_values);

    device_values = nullptr;
    host_values = nullptr;

    this->size = 0;
}


template <typename ValueType>
void CudaDenseVector<ValueType>::checkSize(const CudaDenseVector<ValueType>& values_vector) const
{
    if (values_vector.size != size)
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
}


template <typename ValueType>
CudaDenseVector<ValueType>::~CudaDenseVector()
{
    freeMemory();
}


template <typename ValueType>
void CudaDenseVector<ValueType>::show() const
{
    std::cout << "Vector(" << getSize() << "):" << std::endl;
    for (size_t i = 0; i < getSize(); i++)
        std::cout << host_values[i] << " ";
    std::cout << std::endl;
}


template <typename ValueType>
void CudaDenseVector<ValueType>::copy(
    const ValueType* device_data, const ValueType* host_data
)
{
     CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            host_values, host_data, 
            size*sizeof(ValueType), cudaMemcpyHostToHost),
        "cudaMemcpy"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values, device_data, 
            size*sizeof(ValueType), cudaMemcpyDeviceToDevice),
        "cudaMemcpy"
    );
}


template class CudaDenseVector<int>;
template class CudaDenseVector<double>;