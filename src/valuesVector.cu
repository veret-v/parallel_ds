#include "valuesVector.hpp"


void ValuesVector::allocateMemory(size_t size)
{
    vector_size = size;

    cudaMallocHost(&host_mem, size*sizeof(double));
    cudaMalloc(&device_mem, size*sizeof(double));

    cudaMemset(device_mem, 0, size*sizeof(double));
    cudaMemset(host_mem, 0, size*sizeof(double));
}


void ValuesVector::updateDeviceMem()
{
    cudaMemcpy(device_mem, host_mem, vector_size*sizeof(double), cudaMemcpyDefault);
}


void ValuesVector::updateHostMem()
{
    cudaMemcpy(host_mem, device_mem, vector_size*sizeof(double), cudaMemcpyDefault);
}


void ValuesVector::updateDeviceMem() const
{
    cudaMemcpy(device_mem, host_mem, vector_size*sizeof(double), cudaMemcpyDefault);
}


void ValuesVector::freeMemory()
{
    cudaFree(device_mem);
    cudaFreeHost(host_mem);
    device_mem = nullptr;
    host_mem = nullptr;
}


ValuesVector::~ValuesVector()
{
    freeMemory();
}


ValuesVector::ValuesVector(const size_t size)
{
    
    allocateMemory(size);
}


ValuesVector::ValuesVector(const std::vector<double> vector)
{
    allocateMemory(vector.size());
    for (size_t i = 0; i < vector_size; i++)
        host_mem[i] = vector[i];
}


ValuesVector::ValuesVector(const ValuesVector& values_vector)
{
    allocateMemory(values_vector.getSize());
    cudaMemcpy(host_mem, values_vector.host_mem, vector_size*sizeof(double), cudaMemcpyHostToHost);
}


void ValuesVector::pushBack(const double& value)
{
    ValuesVector buff(*this);
    freeMemory();
    allocateMemory(vector_size + 1);
    for (size_t i = 0; i < vector_size; i++)
        host_mem[i] = (i != vector_size - 1) ? buff[i] : value;
}


ValuesVector::iterator ValuesVector::begin()
{
    return iterator(host_mem);
}


ValuesVector::iterator ValuesVector::end()
{
    return iterator(host_mem + vector_size);
}


ValuesVector::const_iterator ValuesVector::begin() const
{
    return const_iterator(host_mem);
}


ValuesVector::const_iterator ValuesVector::end() const
{
    return const_iterator(host_mem + vector_size);
}


ValuesVector ValuesVector::operator-(const ValuesVector &values_vector) const
{
    updateDeviceMem();
    values_vector.updateDeviceMem();
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    ValuesVector new_vec(values_vector.getSize());
    vectorSubKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(device_mem, values_vector.device_mem, new_vec.device_mem, vector_size);
    cudaDeviceSynchronize(); 
    new_vec.updateHostMem();
    return new_vec;
}


ValuesVector ValuesVector::operator-() const
{
    updateDeviceMem();
    ValuesVector new_vec(getSize());
    vectorSubKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(device_mem, new_vec.device_mem, vector_size);
    cudaDeviceSynchronize(); 
    new_vec.updateHostMem();
    return new_vec;
}


double ValuesVector::dot(const ValuesVector &values_vector) const
{
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    updateDeviceMem();
    values_vector.updateDeviceMem();

    double dot_val = 0;
    double* d_block_results;
    double* dot_val_dev_final;
    cudaMalloc(&d_block_results, BLOCKS_NUM * sizeof(double));
    cudaMalloc(&dot_val_dev_final, sizeof(double));

    vectorDotKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(device_mem, values_vector.device_mem, d_block_results, vector_size);
    cudaDeviceSynchronize(); 
    if (BLOCKS_NUM > 1) {
        vectorDotKernel<<<1, BLOCK_DIM>>>(d_block_results, d_block_results, dot_val_dev_final, BLOCKS_NUM);
        cudaDeviceSynchronize(); 
    } else {
        cudaMemcpy(dot_val_dev_final, d_block_results, sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    cudaMemcpy(&dot_val, dot_val_dev_final, sizeof(double), cudaMemcpyDeviceToHost);

    return dot_val;
}

void ValuesVector::PFIupdate(const ValuesVector& PFI_vec, const size_t PFI_idx)
{
    if (PFI_vec.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }

    updateDeviceMem();
    PFI_vec.updateDeviceMem();
    vectorPFIKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(PFI_vec.device_mem, device_mem, PFI_idx, vector_size);
    cudaDeviceSynchronize(); 
    updateHostMem();
}


double ValuesVector::mean() const
{
    double sum = 0;
    for (auto a_i : *this)
        sum +=  a_i;
    return sum / getSize();
}


double ValuesVector::norm() const
{
    double sum = 0;
    for (auto a_i : *this)
        sum += pow(a_i, 2);
    return sum;
}


size_t ValuesVector::countNonZero() const
{
    double count = 0;
    for (auto a_i : *this)
        count += (a_i == 0) ? 0 : 1;
    return count;
}

ValuesVector& ValuesVector::operator=(const ValuesVector& values_vector)
{
    if (this == &values_vector) {
        return *this;
    }
   
    freeMemory();
    allocateMemory(values_vector.getSize());
    cudaMemcpy(host_mem, values_vector.host_mem, vector_size*sizeof(double), cudaMemcpyHostToHost);
    return *this;
}


ValuesVector& ValuesVector::operator-=(const ValuesVector& values_vector)
{
    updateDeviceMem();
    values_vector.updateDeviceMem();
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    vectorSubKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(values_vector.device_mem, device_mem, vector_size);
    cudaDeviceSynchronize(); 
    updateHostMem();
    return *this;   
}


ValuesVector& ValuesVector::operator+=(const ValuesVector& values_vector)
{
    updateDeviceMem();
    values_vector.updateDeviceMem();
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    vectorAddKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(values_vector.device_mem, device_mem, vector_size);
    cudaDeviceSynchronize(); 
    updateHostMem();
    return *this;   
}


ValuesVector ValuesVector::operator+(const ValuesVector& values_vector) const
{
    updateDeviceMem();
    values_vector.updateDeviceMem();
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    ValuesVector new_vec(values_vector.getSize());
    vectorAddKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(device_mem, values_vector.device_mem, new_vec.device_mem, vector_size);
    cudaDeviceSynchronize(); 
    new_vec.updateHostMem();
    return new_vec;
}


ValuesVector ValuesVector::operator*(const double& value) const
{
    updateDeviceMem();
    ValuesVector new_vec(getSize());
    vectorScaleKernel<<<BLOCKS_NUM, BLOCK_DIM>>>(device_mem, value, new_vec.device_mem, vector_size);
    cudaDeviceSynchronize(); 
    new_vec.updateHostMem();
    return new_vec;
}



ValuesVector ValuesVector::operator()(const IndexVector& indexes) const
{
    ValuesVector new_vector(indexes.size());
    
    for (size_t i = 0; i < indexes.size(); i++)
        new_vector[i] = operator[](indexes[i]);
    
    return new_vector;
}


void ValuesVector::show() const
{
    std::cout << "Vector(" << getSize() << "):" << std::endl;
    for (size_t i = 0; i < getSize(); i++)
        std::cout << operator[](i) << " ";
    std::cout << std::endl;
}


void ValuesVector::setValues(const ValuesVector& values_vector, const IndexVector& index_vector)
{
    for (size_t i = 0; i < index_vector.size(); i++)
    {
        size_t j = index_vector[i];
        operator[](j) = values_vector[i];
    } 
}


ValuesVector ValuesVector::operator()(const size_t start, const size_t stop) const
{
    ValuesVector result(stop - start);
    for (size_t i = start; i < stop; i++)
        result[i] = operator[](i);
    return result;
}

