#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <CoinMpsIO.hpp>


typedef std::vector<int> IndexVector;

class Matrix;


class ValuesVector
{
friend class Matrix;

protected:
    std::vector<double> data;
 
public:
    ValuesVector(const int size);
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
    ValuesVector operator()(const int start, const int stop) const;
    
   
    typedef std::vector<double>::iterator iterator;
    typedef std::vector<double>::const_iterator const_iterator;

    double operator[](const int idx) const {return data[idx];};
    double& operator[](const int idx) {return data[idx];};

    int getSize() const {return data.size();};
    int getSize() {return data.size();};

    const std::vector<double>& getData() {return data;};

    
    int countNonZero() const;

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
