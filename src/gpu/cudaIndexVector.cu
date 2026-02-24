#include "cudaIndexVector.hpp"


CudaIndexVector& CudaIndexVector::operator=(const CudaIndexVector& values_vector)
{
    if (this == &values_vector) 
        return *this;
    
    if (host_values != nullptr || 
        device_values != nullptr) 
        CudaDenseVector::~CudaDenseVector();

    allocateMemory(values_vector.getSize());
    copy(values_vector.device_values, values_vector.host_values);

    return *this;
}


CudaIndexVector::CudaIndexVector(const IndexVector& vector)
{
    allocateMemory(vector.size());

    for (size_t i = 0; i < size; i++)
        host_values[i] = vector[i];

    updateDeviceMem();
}


void CudaIndexVector::update(const int pos, const int val)
{
    host_values[pos] = val;

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values + pos, &val, 
            sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy"
    ); 
}