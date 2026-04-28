#include "PFIfactor.hpp"


void PFIfactor::allocateMemory()
{
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_values, capacity*col_len*sizeof(double)),
        "cudaMalloc : device_values"
    );
    CUDA_CALL_AND_CHECK(
        cudaMalloc(&device_col_id, capacity*sizeof(int)),
        "cudaMalloc : device_col_id"
    );
}

void PFIfactor::freeMemory()
{
    CUDA_CALL_AND_CHECK(cudaFree(device_values),
                        "cudaFree : device_values");
    CUDA_CALL_AND_CHECK(cudaFree(device_col_id),
                        "cudaFree : device_col_id");
}


PFIfactor::PFIfactor(const int col_len, const int capacity)
{
    this->col_len  = col_len;
    this->capacity = capacity;

    allocateMemory();
}


PFIfactor::~PFIfactor()
{
    freeMemory();
}


void PFIfactor::addEtaMatrix(const int q, CudaDataDenseVector& vals)
{
    int* buff_q = nullptr;
    
    CUDA_CALL_AND_CHECK(
        cudaMallocHost(&buff_q, sizeof(int)),
        "cudaMalloc : device_values"
    );

    buff_q[0] = q;

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_values + size * col_len, 
            vals.host_values, 
            col_len*sizeof(double), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy : vals.host_values"
    );

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            device_col_id + size, 
            buff_q, 
            sizeof(int), 
            cudaMemcpyHostToDevice),
        "cudaMemcpy : buff_q"
    );

    CUDA_CALL_AND_CHECK(cudaFreeHost(buff_q),
                        "cudaFree : buff_q");

    size += 1;
}


void PFIfactor::applyPFI( 
    const cublasHandle_t& handle, 
    const CudaDataDenseVector& rhs, 
    CudaDataDenseVector& sol, 
    const bool& transpose
)
{   
    CUBLAS_CALL_AND_CHECK(
        btranOrFtran(
            handle,
            sol.device_values,
            rhs.device_values,
            device_values,
            device_col_id,
            size,
            col_len,
            transpose),
        "applyPFI : "
    );
}