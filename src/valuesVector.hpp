#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <CoinMpsIO.hpp>
#ifdef WITH_CUDA
    #include "cudaKernels.hpp"
#endif


typedef std::vector<size_t> IndexVector;

class Matrix;


#ifdef WITH_CUDA
    template<typename ValueType>
    class VectorIterator
    {
        friend class ValuesVector;
   
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = ValueType;
        using difference_type = std::ptrdiff_t;
        using pointer = ValueType*;
        using reference = ValueType&;

        VectorIterator(const VectorIterator &it) : p(it.p) {};
        VectorIterator(ValueType* p) : p(p) {};

        reference operator*() const { return *p; }
        pointer operator->() const { return p; }
        bool operator!=(VectorIterator const& other) const {return p != other.p;};
        bool operator==(VectorIterator const& other) const {return p == other.p;}; 
        VectorIterator& operator++() {++p;return *this;};
    private:
        pointer p;
    };
#endif


class ValuesVector
{
friend class Matrix;

protected:
    #ifdef WITH_CUDA
        double* host_mem   = nullptr;
        double* device_mem = nullptr;
        size_t vector_size = 0;
        
        void allocateMemory(size_t size);
        void freeMemory();
        void updateHostMem();
        void updateDeviceMem();
        void updateDeviceMem() const;
    #else
        std::vector<double> data;
    #endif


public:
    ValuesVector(const size_t size);
    ValuesVector(const std::vector<double> vector);
    ValuesVector(const ValuesVector& values_vector);
    ValuesVector(const CoinPackedVector& values_vector);
    ValuesVector() : ValuesVector(0) {};
    
    ValuesVector& operator=(const ValuesVector& values_vector);
    ValuesVector& operator-=(const ValuesVector& values_vector);
    ValuesVector& operator+=(const ValuesVector& values_vector);
    ValuesVector operator-(const ValuesVector& values_vector) const;
    ValuesVector operator-() const;
    ValuesVector operator+(const ValuesVector& values_vector) const;
    ValuesVector operator*(const double& value) const;
    ValuesVector operator()(const IndexVector& indexes) const;
    ValuesVector operator()(const size_t start, const size_t stop) const;
    
    #ifdef WITH_CUDA
        typedef VectorIterator<double> iterator;
        typedef VectorIterator<const double> const_iterator;

        double operator[](const size_t idx) const {return host_mem[idx];};
        double& operator[](const size_t idx) {return host_mem[idx];};

        size_t getSize() const {return vector_size;};
        size_t getSize() {return vector_size;};
        
        const std::vector<double>& getData();

        void PFIupdate(const ValuesVector& PFI_vec, const size_t PFI_idx);

        ~ValuesVector();
    #else
        typedef std::vector<double>::iterator iterator;
        typedef std::vector<double>::const_iterator const_iterator;

        double operator[](const size_t idx) const {return data[idx];};
        double& operator[](const size_t idx) {return data[idx];};

        size_t getSize() const {return data.size();};
        size_t getSize() {return data.size();};

        const std::vector<double>& getData() {return data;};
    #endif
    
    size_t countNonZero() const;

    double dot(const ValuesVector &values_vector) const;
    double mean() const;
    double norm() const;
   
    void setValues(const ValuesVector& values_vector, const IndexVector& index_vector);
    void pushBack(const double& value);

    void show() const;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;
};
