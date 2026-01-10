#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <CoinMpsIO.hpp>


typedef std::vector<size_t> IndexVector;


class ValuesVector
{
private:
    std::vector<double> data;

public:
    typedef std::vector<double>::iterator iterator;
    typedef std::vector<double>::const_iterator const_iterator;

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
    
    double operator[](const size_t idx) const {return data[idx];};
    double& operator[](const size_t idx) {return data[idx];};

    size_t getSize() const {return data.size();};
    size_t getSize() {return data.size();};
    size_t countNonZero() const;
    const std::vector<double>& getData() {return data;};
    
    double dot(const ValuesVector &values_vector) const;
    double mean() const;
    double norm() const;
    // double minAbs() const {std::min_element(data.begin(), data.end(), [](double a, double b) {return fabs(a) > fabs(b);});};
    // double maxAbs() const {std::max_element(data.begin(), data.end(), [](double a, double b) {return fabs(a) < fabs(b);});};

    void setValues(const ValuesVector& values_vector, const IndexVector& index_vector);
    void pushBack(const double& value);

    void show() const;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;
};


